import os
import re
import copy
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


class EMAModel:
    """モデルの重みの指数移動平均（EMA）を管理するクラス"""

    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


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


def load_metrics_csv(csv_path):
    """既存の metrics.csv から履歴リストを復元する"""
    history = []
    if not os.path.exists(csv_path):
        return history
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append(
                {
                    "epoch": int(row["epoch"]),
                    "iteration": int(row["iteration"]),
                    "loss": float(row["loss"]),
                    "grad_A": float(row["grad_norm_A"]),
                    "grad_B": float(row["grad_norm_B"]),
                    "grad_C": float(row["grad_norm_C"]),
                }
            )
    return history


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
        label="Min-SNR Weighted Hybrid Loss",
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
    1. comparison_grid.png
    2. denoising_progression.png
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
            img = img.unsqueeze(0).to(device)
            ctx = ctx.unsqueeze(0).to(device)

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


def resolve_resume_paths(resume_arg):
    """
    --resume に渡されたパス（ディレクトリ、model.pth、checkpoint.pt）を解析し、
    (ckpt_file_path, ckpt_subdir, parent_run_dir) を返す
    """
    resume_arg = os.path.normpath(resume_arg.strip())
    if not os.path.exists(resume_arg):
        raise FileNotFoundError(f"指定された再開パスが存在しません: {resume_arg}")

    if os.path.isdir(resume_arg):
        ckpt_subdir = resume_arg
        full_ckpt = os.path.join(ckpt_subdir, "checkpoint.pt")
        weights_only = os.path.join(ckpt_subdir, "model.pth")
        if os.path.exists(full_ckpt):
            ckpt_file = full_ckpt
        elif os.path.exists(weights_only):
            ckpt_file = weights_only
        else:
            raise FileNotFoundError(
                f"ディレクトリ内に checkpoint.pt または model.pth が見つかりません: {ckpt_subdir}"
            )
    else:
        ckpt_file = resume_arg
        ckpt_subdir = os.path.dirname(ckpt_file)
        # model.pth が指定されていても、同階層に完全状態の checkpoint.pt があれば優先利用
        sibling_full = os.path.join(ckpt_subdir, "checkpoint.pt")
        if os.path.basename(ckpt_file) == "model.pth" and os.path.exists(sibling_full):
            ckpt_file = sibling_full

    subdir_name = os.path.basename(ckpt_subdir)
    if re.match(r"^epoch\d+_iter\d+$", subdir_name):
        parent_run_dir = os.path.dirname(ckpt_subdir)
    else:
        parent_run_dir = None

    return ckpt_file, ckpt_subdir, parent_run_dir


def run_training(
    subset,
    datasets_root="datasets",
    checkpoints_root="checkpoints",
    batch_size=2,
    target_iterations=50000,
    timesteps=1000,
    base_size=128,
    train_res=None,
    eval_res=None,
    multi_aspect=True,
    lr=1e-3,
    save_every_epochs=50,
    resume=None,
):
    print(f"--- Starting Production Training (Subset: {subset}) ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TripleStreamModel(in_channels=3, hidden_dim_C=64).to(device)
    ema = EMAModel(model, decay=0.9995)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    start_epoch = 0
    global_iter = 0
    metrics_history = []
    run_dir = None

    # --- Resume 処理 ---
    if resume is not None and resume.strip() != "":
        ckpt_file, ckpt_subdir, parent_run_dir = resolve_resume_paths(resume)
        print(f"Resuming training from: {ckpt_file}")
        ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)

        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            # 1. 完全なチェックポイント (checkpoint.pt) からの復元
            model.load_state_dict(ckpt_data["model_state_dict"])
            ema.ema_model.load_state_dict(
                ckpt_data.get("ema_state_dict", ckpt_data["model_state_dict"])
            )
            if "optimizer_state_dict" in ckpt_data:
                optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt_data:
                scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
                scheduler.T_max = epochs
            start_epoch = int(ckpt_data.get("epoch", 0))
            global_iter = int(
                ckpt_data.get("global_iter", start_epoch * iters_per_epoch)
            )
            metrics_history = ckpt_data.get("metrics_history", [])
            if not metrics_history:
                metrics_history = load_metrics_csv(
                    os.path.join(ckpt_subdir, "metrics.csv")
                )
        else:
            # 2. 既存の重みファイル (model.pth) からの復元
            model.load_state_dict(ckpt_data)
            ema.ema_model.load_state_dict(ckpt_data)

            csv_path = os.path.join(ckpt_subdir, "metrics.csv")
            metrics_history = load_metrics_csv(csv_path)

            if metrics_history:
                start_epoch = metrics_history[-1]["epoch"]
                global_iter = metrics_history[-1]["iteration"]
            else:
                m = re.search(r"epoch(\d+)_iter(\d+)", os.path.basename(ckpt_subdir))
                if m:
                    start_epoch = int(m.group(1))
                    global_iter = int(m.group(2))
                else:
                    print(
                        "Warning: エポック/イテレーション番号が特定できないため Epoch 0 から再開します。"
                    )

            # スケジューラを再開エポックまで進める
            for _ in range(start_epoch):
                scheduler.step()

        if parent_run_dir is not None:
            run_dir = parent_run_dir

        print(
            f"  -> Restored Progress: Epoch {start_epoch}/{epochs} | "
            f"Global Iter: {global_iter}/{epochs * iters_per_epoch} | "
            f"Current LR: {scheduler.get_last_lr()[0]:.2e}"
        )

    if run_dir is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(checkpoints_root, timestamp_str)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Checkpoint Root Directory: {run_dir}")

    active_buckets = {k: len(v) for k, v in dataset.bucket_to_indices.items()}
    print(f"Active Aspect Ratio Buckets (H, W): {active_buckets}")
    print(
        f"Total Images: {len(dataset)} | Batch Size: {batch_size} | Timesteps: {timesteps} | "
        f"Save Every: {save_every_epochs} epochs | "
        f"Target Epochs: {epochs} (Total Iters: {epochs * iters_per_epoch})"
    )

    if start_epoch >= epochs:
        print(
            f"すでに目標エポック数 ({start_epoch}/{epochs}) に到達しています。"
            f"さらに学習を継続する場合は --target-iters に {global_iter} より大きい値を指定してください。"
        )
        return

    for epoch in range(start_epoch + 1, epochs + 1):
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

            snr = alpha_cumprod.view(-1) / torch.clamp(
                1.0 - alpha_cumprod.view(-1), min=1e-5
            )
            snr_weight = torch.clamp(snr, max=5.0)

            mse_loss = F.mse_loss(predicted_x0, images, reduction="none").mean(
                dim=[1, 2, 3]
            )
            l1_loss = F.l1_loss(predicted_x0, images, reduction="none").mean(
                dim=[1, 2, 3]
            )

            combined_loss = 0.5 * mse_loss + 0.5 * l1_loss
            loss = (combined_loss * snr_weight).mean()

            loss.backward()

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
            ema.update(model)
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

        if (
            epoch % save_every_epochs == 0
            or epoch == start_epoch + 1
            or epoch == epochs
        ):
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

            # 1. 推論互換用のEMA重みファイル (model.pth)
            ckpt_path = os.path.join(ckpt_subdir, "model.pth")
            torch.save(ema.ema_model.state_dict(), ckpt_path)

            # 2. Resume用の完全状態ファイル (checkpoint.pt)
            full_ckpt_path = os.path.join(ckpt_subdir, "checkpoint.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_iter": global_iter,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metrics_history": metrics_history,
                },
                full_ckpt_path,
            )

            save_metrics_csv_and_plot(metrics_history, ckpt_subdir)

            generate_evaluation_samples(
                ema.ema_model,
                dataset,
                device,
                ckpt_subdir,
                max_steps=timesteps,
                eval_res=eval_res,
            )
            print(f"  -> Saved checkpoint, metrics, and samples to {ckpt_subdir}")

    print("--- Training Finished ---")
    final_path = os.path.join(run_dir, "model_final.pth")
    torch.save(ema.ema_model.state_dict(), final_path)
    print(f"Saved final EMA model weights to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triple-Stream 本番用学習スクリプト")
    parser.add_argument(
        "--subset", "-s", type=str, required=True, help="学習対象のサブフォルダ名"
    )
    parser.add_argument(
        "--datasets_root",
        "--datasets-root",
        dest="datasets_root",
        type=str,
        default="datasets",
        help="データセットのルートフォルダ",
    )
    parser.add_argument(
        "--checkpoints_root",
        "--checkpoints-root",
        dest="checkpoints_root",
        type=str,
        default="checkpoints",
        help="チェックポイントの保存先ルートフォルダ (デフォルト: checkpoints)",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=None,
        help="学習を再開するチェックポイントのパス (model.pth, checkpoint.pt, または epochX_iterY フォルダ)",
    )
    parser.add_argument(
        "--save-every",
        "--save_every",
        dest="save_every",
        type=int,
        default=200,
        help="チェックポイント保存間隔（エポック単位、デフォルト: 50）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="バッチサイズ (デフォルト: 2)"
    )
    parser.add_argument(
        "--target-iters", type=int, default=100_000, help="目標イテレーション数"
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
        help="学習解像度を固定したい場合に指定 (例: '128' または '128x192')",
    )
    parser.add_argument(
        "--eval-res",
        type=str,
        default=None,
        help="チェックポイント推論時の出力解像度を指定 (例: '128x192')",
    )
    parser.add_argument(
        "--no-multi-aspect",
        action="store_true",
        help="指定するとマルチアスペクト比を無効化し、base-sizeの正方形に固定する",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="初期学習率 (デフォルト: 1e-3)"
    )
    args = parser.parse_args()

    run_training(
        subset=args.subset,
        datasets_root=args.datasets_root,
        checkpoints_root=args.checkpoints_root,
        batch_size=args.batch_size,
        target_iterations=args.target_iters,
        timesteps=args.timesteps,
        base_size=args.base_size,
        train_res=parse_resolution(args.train_res),
        eval_res=parse_resolution(args.eval_res),
        multi_aspect=not args.no_multi_aspect,
        lr=args.lr,
        save_every_epochs=args.save_every,
        resume=args.resume,
    )
