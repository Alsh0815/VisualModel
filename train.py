import os
import csv
import math
import random
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.luan.model import TripleStreamPoC
from dataset import PoCDataset


def get_cosine_alpha_cumprod(t, max_steps=1000, s=0.008):
    """特異点を回避するためのオフセット付きコサインスケジュール"""
    t_norm = t.float() / max_steps
    f_t = torch.cos((t_norm + s) / (1.0 + s) * math.pi / 2) ** 2
    f_0 = torch.cos(torch.tensor(s / (1.0 + s) * math.pi / 2, device=t.device)) ** 2
    return f_t / f_0


def save_metrics_csv_and_plot(metrics_history, save_dir):
    """これまでの各種指標の推移をCSVとグラフ画像として保存"""
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
    losses = [row["loss"] for row in metrics_history]
    grad_A = [row["grad_A"] for row in metrics_history]
    grad_B = [row["grad_B"] for row in metrics_history]
    grad_C = [row["grad_C"] for row in metrics_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, losses, label="MSE Loss", color="tab:red", linewidth=1.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss & Gradient Norms History")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, grad_A, label="Stream A (Global)", color="tab:blue", alpha=0.8)
    ax2.plot(
        epochs, grad_B, label="Stream B (Relational)", color="tab:green", alpha=0.8
    )
    ax2.plot(epochs, grad_C, label="Stream C (Texture)", color="tab:orange", alpha=0.8)
    ax1.set_yscale("log")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Grad Norm")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "metrics_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)


def generate_evaluation_samples(model, dataset, device, save_dir, max_steps=1000):
    """
    データセットからランダムに3つ選び:
    1. 上段に元画像3枚、下段に推論画像3枚の比較画像 (comparison_grid.png) を保存
    2. そのうち1つについて、推論ステップ毎の変化 (denoising_progression.png) を保存
    """
    model.eval()
    num_samples = 3

    if len(dataset) >= num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
    else:
        indices = [random.choice(range(len(dataset))) for _ in range(num_samples)]

    orig_images = []
    contexts = []
    for idx in indices:
        img, ctx = dataset[idx]
        orig_images.append(img)
        contexts.append(ctx)

    orig_images = torch.stack(orig_images).to(device)  # [3, 3, H, W]
    contexts = torch.stack(contexts).to(device)  # [3, M, D]

    _, C, H, W = orig_images.shape

    # max_steps に応じて約10分割（計11枚）になるようキャプチャステップを動的に決定
    stride = max(1, max_steps // 10)
    capture_steps = set(
        [max_steps - 1] + list(range(max_steps - stride, 0, -stride)) + [0]
    )
    progression_tensors = []

    with torch.no_grad():
        x = torch.randn(num_samples, C, H, W, device=device)

        for i in reversed(range(max_steps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            alpha_cumprod_t = get_cosine_alpha_cumprod(t, max_steps=max_steps).view(
                num_samples, 1, 1, 1
            )

            if i > 0:
                t_prev = torch.full(
                    (num_samples,), i - 1, device=device, dtype=torch.long
                )
                alpha_cumprod_t_prev = get_cosine_alpha_cumprod(
                    t_prev, max_steps=max_steps
                ).view(num_samples, 1, 1, 1)
                noise = torch.randn_like(x)
            else:
                alpha_cumprod_t_prev = torch.ones((num_samples, 1, 1, 1), device=device)
                noise = torch.zeros_like(x)

            alpha_t = alpha_cumprod_t / alpha_cumprod_t_prev
            beta_t = 1.0 - alpha_t
            beta_t = torch.clamp(beta_t, max=0.999)
            alpha_t = 1.0 - beta_t

            predicted_x0 = model(x, contexts, t)
            predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)

            if i > 0:
                mean = (
                    torch.sqrt(alpha_cumprod_t_prev) * beta_t / (1.0 - alpha_cumprod_t)
                ) * predicted_x0 + (
                    torch.sqrt(alpha_t)
                    * (1.0 - alpha_cumprod_t_prev)
                    / (1.0 - alpha_cumprod_t)
                ) * x

                variance = (
                    (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t) * beta_t
                )
                x = mean + torch.sqrt(torch.clamp(variance, min=1e-5)) * noise
            else:
                x = predicted_x0

            if i in capture_steps:
                step_img = (x[0].detach().cpu() + 1.0) / 2.0
                step_img = torch.clamp(step_img, 0.0, 1.0)
                progression_tensors.append(step_img)

        orig_vis = torch.clamp((orig_images.cpu() + 1.0) / 2.0, 0.0, 1.0)
        gen_vis = torch.clamp((x.cpu() + 1.0) / 2.0, 0.0, 1.0)

        comparison_tensors = torch.cat([orig_vis, gen_vis], dim=0)
        comparison_grid = make_grid(
            comparison_tensors, nrow=3, padding=4, pad_value=1.0
        )
        comparison_path = os.path.join(save_dir, "comparison_grid.png")
        save_image(comparison_grid, comparison_path)

        progression_stack = torch.stack(progression_tensors, dim=0)
        progression_grid = make_grid(
            progression_stack, nrow=len(progression_tensors), padding=4, pad_value=1.0
        )
        progression_path = os.path.join(save_dir, "denoising_progression.png")
        save_image(progression_grid, progression_path)

    model.train()


def run_conditional_training(
    subset,
    datasets_root="datasets",
    batch_size=2,
    target_iterations=10000,
    timesteps=1000,
):
    print(
        f"--- Starting Conditional Generation Overfitting Test (Subset: {subset}) ---"
    )

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", timestamp_str)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Checkpoint Root Directory: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TripleStreamPoC(in_channels=3, hidden_dim_C=64).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.MSELoss(reduction="none")

    dataset = PoCDataset(subset=subset, datasets_root=datasets_root, image_size=128)
    if len(dataset) == 0:
        raise RuntimeError(
            f"データセットが空です。先に prepare_text.py --subset {subset} を実行して .pt ファイルを生成してください。"
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    iters_per_epoch = len(dataloader)
    epochs = math.ceil(target_iterations / iters_per_epoch)
    print(
        f"Total Images: {len(dataset)} | Batch Size: {batch_size} | Timesteps: {timesteps} | Epochs: {epochs} (Total Iters: {epochs * iters_per_epoch})"
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

            # 指定された timesteps の範囲 [0, timesteps - 1] でサンプリング
            t = torch.randint(0, timesteps, (B,), device=device).long()

            alpha_cumprod = get_cosine_alpha_cumprod(t, max_steps=timesteps)
            alpha_cumprod = alpha_cumprod.view(B, 1, 1, 1)

            noise = torch.randn_like(images)
            noisy_images = (
                torch.sqrt(alpha_cumprod) * images
                + torch.sqrt(1.0 - alpha_cumprod) * noise
            )

            predicted_x0 = model(noisy_images, contexts, t)

            # サンプルごとのMSEを計算: [B, C, H, W] -> [B]
            mse_per_sample = criterion(predicted_x0, images).mean(dim=[1, 2, 3])

            # SNR = alpha_cumprod / (1 - alpha_cumprod)
            snr = alpha_cumprod.view(-1) / torch.clamp(
                1.0 - alpha_cumprod.view(-1), min=1e-5
            )

            # Min-SNR-gamma (gamma=5.0) による重み付け: 高SNR時の過剰な重みを抑えつつ低ノイズ域を重視
            snr_weight = torch.clamp(snr, max=5.0)
            loss = (mse_per_sample * snr_weight).mean()

            loss.backward()

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

        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch:04d}/{epochs}] (Iter: {global_iter}) | Loss: {avg_loss:.6f}"
            )
            print(
                f"  Grad Norms -> Stream A: {grad_norm_A:.4f} | Stream B: {grad_norm_B:.4f} | Stream C: {grad_norm_C:.4f}"
            )

            ckpt_subdir = os.path.join(run_dir, f"epoch{epoch}_iter{global_iter}")
            os.makedirs(ckpt_subdir, exist_ok=True)

            ckpt_path = os.path.join(ckpt_subdir, "model.pth")
            torch.save(model.state_dict(), ckpt_path)

            save_metrics_csv_and_plot(metrics_history, ckpt_subdir)

            generate_evaluation_samples(
                model, dataset, device, ckpt_subdir, max_steps=timesteps
            )
            print(f"  -> Saved checkpoint, metrics, and samples to {ckpt_subdir}")

    print("--- Training Finished ---")
    final_path = os.path.join(run_dir, "poc_model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model weights to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="条件付き生成モデルの学習スクリプト")
    parser.add_argument(
        "--subset",
        "-s",
        type=str,
        required=True,
        help="datasetsフォルダ内の学習対象サブフォルダ名 (例: poc_set1)",
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="datasets",
        help="データセットのルートフォルダ (デフォルト: datasets)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="バッチサイズ (デフォルト: 2)"
    )
    parser.add_argument(
        "--target-iters",
        type=int,
        default=3_000_000,
        help="目標イテレーション数（デフォルト: 3,000,000）",
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=1000,
        help="ノイズの学習タイムステップ数（デフォルト: 1000）",
    )
    args = parser.parse_args()

    run_conditional_training(
        subset=args.subset,
        datasets_root=args.datasets_root,
        batch_size=args.batch_size,
        target_iterations=args.target_iters,
        timesteps=args.timesteps,
    )
