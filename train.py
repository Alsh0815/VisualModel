import os
import csv
import math
import random
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.luan.model import TripleStreamModel
from dataset import VisualDataset, AspectRatioBatchSampler


def parse_resolution(res_str):
    """'128' -> (128, 128), '128x192' -> (128, 192) のように解像度文字列をパースする"""
    if res_str is None or res_str.strip() == "":
        return None
    res_str = res_str.lower().strip()
    if "x" in res_str:
        h, w = map(int, res_str.split("x"))
        return (h, w)
    val = int(res_str)
    return (val, val)


def get_cosine_alpha_cumprod(t, max_steps=1000, s=0.008):
    """特異点を回避するためのオフセット付きコサインスケジュール"""
    t_norm = t.float() / max_steps
    f_t = torch.cos((t_norm + s) / (1.0 + s) * math.pi / 2) ** 2
    f_0 = torch.cos(torch.tensor(s / (1.0 + s) * math.pi / 2, device=t.device)) ** 2
    return f_t / f_0


def save_metrics_csv_and_plot(metrics_history, save_dir):
    """これまでの各種指標の推移をCSVと対数グラフ画像として保存"""
    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "iteration", "loss", "grad_norm_A", "grad_norm_B", "grad_norm_C"]
        )
        for row in metrics_history:
            writer.writerow(
                [
                    row["epoch"],
                    row["iteration"],
                    f"{row['loss']:.6f}",
                    f"{row['grad_A']:.6f}",
                    f"{row['grad_B']:.6f}",
                    f"{row['grad_C']:.6f}",
                ]
            )

    epochs = [row["epoch"] for row in metrics_history]
    losses = [max(row["loss"], 1e-7) for row in metrics_history]
    grad_A = [max(row["grad_A"], 1e-8) for row in metrics_history]
    grad_B = [max(row["grad_B"], 1e-8) for row in metrics_history]
    grad_C = [max(row["grad_C"], 1e-8) for row in metrics_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(
        epochs,
        losses,
        label="Min-SNR Weighted MSE Loss",
        color="tab:red",
        linewidth=1.2,
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("Loss (Log Scale)")
    ax1.set_title("Training Loss & Gradient Norms History")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, grad_A, label="Stream A (Global)", color="tab:blue", alpha=0.8)
    ax2.plot(
        epochs, grad_B, label="Stream B (Relational)", color="tab:green", alpha=0.8
    )
    ax2.plot(epochs, grad_C, label="Stream C (Texture)", color="tab:orange", alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Grad Norm (Log Scale)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "metrics_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)


def ddim_sample_batch(
    model, contexts, shape, device, max_steps=1000, capture_steps=None
):
    """
    DDIM (eta=0) 決定論的サンプラーにより、任意の解像度 shape=(B, C, H, W) で画像を生成する
    """
    B, C, H, W = shape
    x = torch.randn(shape, device=device)
    progression_tensors = []

    for i in reversed(range(max_steps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        alpha_cumprod_t = get_cosine_alpha_cumprod(t, max_steps=max_steps).view(
            B, 1, 1, 1
        )

        if i > 0:
            t_prev = torch.full((B,), i - 1, device=device, dtype=torch.long)
            alpha_cumprod_t_prev = get_cosine_alpha_cumprod(
                t_prev, max_steps=max_steps
            ).view(B, 1, 1, 1)
        else:
            alpha_cumprod_t_prev = torch.ones((B, 1, 1, 1), device=device)

        predicted_x0 = model(x, contexts, t)
        predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)

        if i > 0:
            # 予測された x0 から推定ノイズ pred_eps を逆算して DDIM 遷移
            pred_eps = (x - torch.sqrt(alpha_cumprod_t) * predicted_x0) / torch.sqrt(
                torch.clamp(1.0 - alpha_cumprod_t, min=1e-5)
            )
            x = (
                torch.sqrt(alpha_cumprod_t_prev) * predicted_x0
                + torch.sqrt(torch.clamp(1.0 - alpha_cumprod_t_prev, min=1e-5))
                * pred_eps
            )
        else:
            x = predicted_x0

        if capture_steps is not None and i in capture_steps:
            step_img = torch.clamp((x[0].detach().cpu() + 1.0) / 2.0, 0.0, 1.0)
            progression_tensors.append(step_img)

    return x, progression_tensors


def generate_evaluation_samples(
    model, dataset, device, save_dir, max_steps=1000, eval_res=None
):
    """
    データセットからランダムに3つ選び:
    1. comparison_grid.png (eval_res指定時はその解像度で推論、未指定時は各画像のバケット解像度で推論)
    2. denoising_progression.png (1枚目の推論ステップごとの変化)
    """
    model.eval()
    num_samples = 3

    if len(dataset) >= num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
    else:
        indices = [random.choice(range(len(dataset))) for _ in range(num_samples)]

    stride = max(1, max_steps // 10)
    capture_steps = set(
        [max_steps - 1] + list(range(max_steps - stride, 0, -stride)) + [0]
    )

    orig_vis_list = []
    gen_vis_list = []
    progression_tensors = []

    with torch.no_grad():
        for s_idx, ds_idx in enumerate(indices):
            img, ctx = dataset[ds_idx]
            img = img.unsqueeze(0).to(device)  # [1, 3, H_orig, W_orig]
            ctx = ctx.unsqueeze(0).to(device)  # [1, M, D]

            if eval_res is not None:
                target_h, target_w = eval_res
                orig_resized = F.interpolate(
                    img, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            else:
                _, _, target_h, target_w = img.shape
                orig_resized = img

            c_steps = capture_steps if s_idx == 0 else None
            gen_x, prog = ddim_sample_batch(
                model,
                ctx,
                shape=(1, 3, target_h, target_w),
                device=device,
                max_steps=max_steps,
                capture_steps=c_steps,
            )

            if s_idx == 0:
                progression_tensors = prog

            orig_vis_list.append(
                torch.clamp((orig_resized[0].cpu() + 1.0) / 2.0, 0.0, 1.0)
            )
            gen_vis_list.append(torch.clamp((gen_x[0].cpu() + 1.0) / 2.0, 0.0, 1.0))

        # マルチ解像度で3枚のサイズが異なる場合、比較グリッド表示用に1枚目のサイズへ揃えて並べる
        grid_h, grid_w = orig_vis_list[0].shape[1], orig_vis_list[0].shape[2]
        orig_aligned = [
            (
                F.interpolate(
                    t.unsqueeze(0),
                    size=(grid_h, grid_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                if t.shape[1:] != (grid_h, grid_w)
                else t
            )
            for t in orig_vis_list
        ]
        gen_aligned = [
            (
                F.interpolate(
                    t.unsqueeze(0),
                    size=(grid_h, grid_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                if t.shape[1:] != (grid_h, grid_w)
                else t
            )
            for t in gen_vis_list
        ]

        comparison_tensors = torch.stack(orig_aligned + gen_aligned, dim=0)
        comparison_grid = make_grid(
            comparison_tensors, nrow=3, padding=4, pad_value=1.0
        )
        save_image(comparison_grid, os.path.join(save_dir, "comparison_grid.png"))

        if progression_tensors:
            progression_stack = torch.stack(progression_tensors, dim=0)
            progression_grid = make_grid(
                progression_stack,
                nrow=len(progression_tensors),
                padding=4,
                pad_value=1.0,
            )
            save_image(
                progression_grid, os.path.join(save_dir, "denoising_progression.png")
            )

    model.train()


def run_training(
    subset,
    datasets_root="datasets",
    batch_size=2,
    target_iterations=50000,
    timesteps=1000,
    base_size=128,
    train_res=None,
    eval_res=None,
    multi_aspect=True,
    lr=1e-3,
    save_every_epochs=50,
):
    print(f"--- Starting Production Training (Subset: {subset}) ---")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", timestamp_str)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Checkpoint Root Directory: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TripleStreamModel(in_channels=3, hidden_dim_C=64).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction="none")

    dataset = VisualDataset(
        subset=subset,
        datasets_root=datasets_root,
        base_size=base_size,
        fixed_res=train_res,
        multi_aspect=multi_aspect,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"データセットが空です。先に prepare_text.py --subset {subset} を実行してください。"
        )

    batch_sampler = AspectRatioBatchSampler(
        dataset.bucket_to_indices, batch_size=batch_size, shuffle=True
    )
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    iters_per_epoch = len(dataloader)
    epochs = math.ceil(target_iterations / iters_per_epoch)
    active_buckets = {k: len(v) for k, v in dataset.bucket_to_indices.items()}
    print(f"Active Aspect Ratio Buckets (H, W): {active_buckets}")
    print(
        f"Total Images: {len(dataset)} | Batch Size: {batch_size} | Timesteps: {timesteps} | "
        f"Epochs: {epochs} (Total Iters: {epochs * iters_per_epoch})"
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    global_iter = 0
    metrics_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        grad_norm_A = 0.0
        grad_norm_B = 0.0
        grad_norm_C = 0.0

        for images, contexts in dataloader:
            images = images.to(device)
            contexts = contexts.to(device)
            B = images.size(0)

            optimizer.zero_grad()

            t = torch.randint(0, timesteps, (B,), device=device).long()
            alpha_cumprod = get_cosine_alpha_cumprod(t, max_steps=timesteps).view(
                B, 1, 1, 1
            )

            noise = torch.randn_like(images)
            noisy_images = (
                torch.sqrt(alpha_cumprod) * images
                + torch.sqrt(1.0 - alpha_cumprod) * noise
            )

            predicted_x0 = model(noisy_images, contexts, t)

            # Min-SNR-gamma (gamma=5.0) による重み付け Loss
            mse_per_sample = criterion(predicted_x0, images).mean(dim=[1, 2, 3])
            snr = alpha_cumprod.view(-1) / torch.clamp(
                1.0 - alpha_cumprod.view(-1), min=1e-5
            )
            snr_weight = torch.clamp(snr, max=5.0)
            loss = (mse_per_sample * snr_weight).mean()

            loss.backward()

            # 勾配爆発（スパイク）を防ぐための勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            grad_norm_A = (
                model.attn_A.in_proj_weight.grad.norm().item()
                if hasattr(model, "attn_A")
                and model.attn_A.in_proj_weight.grad is not None
                else 0.0
            )
            grad_norm_B = (
                model.mix_B[0].weight.grad.norm().item()
                if hasattr(model, "mix_B") and model.mix_B[0].weight.grad is not None
                else 0.0
            )
            grad_norm_C = (
                model.cross_attn_C.in_proj_weight.grad.norm().item()
                if hasattr(model, "cross_attn_C")
                and model.cross_attn_C.in_proj_weight.grad is not None
                else 0.0
            )

            optimizer.step()
            epoch_loss += loss.item()
            global_iter += 1

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)

        metrics_history.append(
            {
                "epoch": epoch,
                "iteration": global_iter,
                "loss": avg_loss,
                "grad_A": grad_norm_A,
                "grad_B": grad_norm_B,
                "grad_C": grad_norm_C,
            }
        )

        if epoch % save_every_epochs == 0 or epoch == 1 or epoch == epochs:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch:04d}/{epochs}] (Iter: {global_iter}) | "
                f"Loss: {avg_loss:.6f} | LR: {current_lr:.2e}"
            )
            print(
                f"  Grad Norms -> Stream A: {grad_norm_A:.4f} | "
                f"Stream B: {grad_norm_B:.4f} | Stream C: {grad_norm_C:.4f}"
            )

            ckpt_subdir = os.path.join(run_dir, f"epoch{epoch}_iter{global_iter}")
            os.makedirs(ckpt_subdir, exist_ok=True)

            ckpt_path = os.path.join(ckpt_subdir, "model.pth")
            torch.save(model.state_dict(), ckpt_path)

            save_metrics_csv_and_plot(metrics_history, ckpt_subdir)

            generate_evaluation_samples(
                model,
                dataset,
                device,
                ckpt_subdir,
                max_steps=timesteps,
                eval_res=eval_res,
            )
            print(f"  -> Saved checkpoint, metrics, and samples to {ckpt_subdir}")

    print("--- Training Finished ---")
    final_path = os.path.join(run_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model weights to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triple-Stream 本番用学習スクリプト")
    parser.add_argument(
        "--subset", "-s", type=str, required=True, help="学習対象のサブフォルダ名"
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="datasets",
        help="データセットのルートフォルダ",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="バッチサイズ (デフォルト: 2)"
    )
    parser.add_argument(
        "--target-iters", type=int, default=50_000, help="目標イテレーション数"
    )
    parser.add_argument(
        "--timesteps", "-t", type=int, default=1000, help="ノイズのタイムステップ数"
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=128,
        help="マルチアスペクト比バケットの基準解像度 (例: 128, 256, 512)",
    )
    parser.add_argument(
        "--train-res",
        type=str,
        default=None,
        help="学習解像度を固定したい場合に指定 (例: '128' または '128x192')。未指定時はマルチアスペクト比で学習",
    )
    parser.add_argument(
        "--eval-res",
        type=str,
        default=None,
        help="チェックポイント推論時の出力解像度を指定 (例: '128x192')。未指定時は元画像のバケットサイズを使用",
    )
    parser.add_argument(
        "--no-multi-aspect",
        action="store_true",
        help="指定するとマルチアスペクト比を無効化し、base-sizeの正方形に固定する",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="初期学習率 (デフォルト: 1e-3)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="チェックポイント保存間隔（エポック単位）",
    )
    args = parser.parse_args()

    run_training(
        subset=args.subset,
        datasets_root=args.datasets_root,
        batch_size=args.batch_size,
        target_iterations=args.target_iters,
        timesteps=args.timesteps,
        base_size=args.base_size,
        train_res=parse_resolution(args.train_res),
        eval_res=parse_resolution(args.eval_res),
        multi_aspect=not args.no_multi_aspect,
        lr=args.lr,
        save_every_epochs=args.save_every,
    )
