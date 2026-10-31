import os
import re
import json
import copy
import csv
import math
import time
import random
import argparse
from datetime import datetime
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from models.luan.model import TripleStreamModel
from dataset import VisualDataset, AspectRatioBatchSampler


_JP_FONT_PROP = None


def get_japanese_font_prop():
    """
    環境に日本語フォントが無い場合でも、NotoSansJPを自動取得して
    文字化け（□□□）を100%防ぐ FontProperties を返す
    """
    global _JP_FONT_PROP
    if _JP_FONT_PROP is not None:
        return _JP_FONT_PROP

    # 1. システム内の既存フォントを探索
    search_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    ]
    for p in search_paths:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            _JP_FONT_PROP = fm.FontProperties(fname=p)
            return _JP_FONT_PROP

    # 2. 見つからない場合はキャッシュに NotoSansJP-Regular.otf を自動ダウンロード
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fonts")
    os.makedirs(cache_dir, exist_ok=True)
    font_path = os.path.join(cache_dir, "NotoSansJP-Regular.otf")

    if not os.path.exists(font_path):
        url = "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
        try:
            print("Downloading Japanese font for heatmap labels...")
            urllib.request.urlretrieve(url, font_path)
        except Exception as e:
            print(f"Warning: 日本語フォントの自動取得に失敗しました ({e})")
            return None

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        _JP_FONT_PROP = fm.FontProperties(fname=font_path)
        return _JP_FONT_PROP

    return None


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
    if res_str is None or res_str.strip() == "":
        return None
    res_str = res_str.lower().strip()
    if "x" in res_str:
        h, w = map(int, res_str.split("x"))
        return (h, w)
    val = int(res_str)
    return (val, val)


def parse_auto_int(val):
    if val is None:
        return None
    if isinstance(val, str):
        val_str = val.strip().lower()
        if val_str in ("", "auto", "none", "0"):
            return None
        val = int(val_str)
    return val if val > 0 else None


def round_to_nice_step(val):
    if val <= 1:
        return 1
    nice_steps = [
        1,
        2,
        5,
        10,
        20,
        25,
        50,
        100,
        200,
        250,
        500,
        1000,
        2000,
        2500,
        5000,
        10000,
    ]
    for s in nice_steps:
        if val <= s:
            return s
    return int(round(val / 5000.0)) * 5000


def estimate_target_iterations(dataset, batch_size, timesteps=1000):
    num_images = max(1, len(dataset))
    total_pixels = sum(
        h * w * len(indices) for (h, w), indices in dataset.bucket_to_indices.items()
    )
    avg_pixels = total_pixels / num_images if total_pixels > 0 else 256 * 256
    area_ratio = avg_pixels / (128.0 * 128.0)
    res_factor = max(1.0, area_ratio**0.35)
    base_iters_128 = 35000.0 + 9000.0 * math.sqrt(num_images)
    batch_factor = (2.0 / max(1, batch_size)) ** 0.25
    timestep_factor = (timesteps / 1000.0) ** 0.5
    raw_iters = base_iters_128 * res_factor * batch_factor * timestep_factor
    return max(20000, int(round(raw_iters / 5000.0)) * 5000)


def estimate_save_every_epochs(epochs, iters_per_epoch):
    total_iters = epochs * iters_per_epoch
    target_ckpt_count = min(25, max(10, total_iters // 4000))
    raw_epochs = max(1, epochs / max(1, target_ckpt_count))
    return min(epochs, round_to_nice_step(raw_epochs))


def estimate_log_every_epochs(epochs, iters_per_epoch, save_every_epochs):
    by_iters = max(1, round(150.0 / max(1, iters_per_epoch)))
    by_save = max(1, save_every_epochs // 10)
    raw_log = min(by_iters, by_save)
    return min(save_every_epochs, round_to_nice_step(raw_log))


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_cosine_alpha_cumprod(t, max_steps=1000, s=0.008):
    t_norm = t.float() / max_steps
    f_t = torch.cos((t_norm + s) / (1.0 + s) * math.pi / 2) ** 2
    f_0 = torch.cos(torch.tensor(s / (1.0 + s) * math.pi / 2, device=t.device)) ** 2
    return f_t / f_0


# =====================================================================
# 追加評価指標: PSNR / SSIM の計算関数 (値域 [0.0, 1.0] の画像テンソル対象)
# =====================================================================
@torch.no_grad()
def compute_psnr(img1, img2):
    """[C, H, W] または [B, C, H, W] (値域 0〜1) のPSNR(dB)を算出"""
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 50.0
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def compute_ssim(img1, img2, window_size=11):
    """[1, C, H, W] (値域 0〜1) のSSIM (0〜1) をガウシアン窓で算出"""
    # 型不一致 (BFloat16 / Half と Float32 の混在) を防ぐため float() に統一
    img1 = img1.float()
    img2 = img2.float()
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    C = img1.size(1)
    coords = (
        torch.arange(window_size, dtype=img1.dtype, device=img1.device)
        - window_size // 2
    )
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = window.expand(C, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean().item())


# =====================================================================
# CSV読み書き & 3段グラフ描画 (Loss / Grad Norm / PSNR & SSIM)
# =====================================================================
def load_metrics_csv(csv_path):
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
                    "mse_loss": float(row.get("mse_loss", row["loss"])),
                    "l1_loss": float(row.get("l1_loss", row["loss"])),
                    "grad_A": float(row["grad_norm_A"]),
                    "grad_B": float(row["grad_norm_B"]),
                    "grad_C": float(row["grad_norm_C"]),
                    "lr": float(row.get("lr", 0.0)),
                    "eval_psnr": (
                        float(row["eval_psnr"])
                        if row.get("eval_psnr") not in (None, "", "None")
                        else None
                    ),
                    "eval_ssim": (
                        float(row["eval_ssim"])
                        if row.get("eval_ssim") not in (None, "", "None")
                        else None
                    ),
                }
            )
    return history


def save_metrics_csv_and_plot(metrics_history, save_dir):
    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "iteration",
                "loss",
                "mse_loss",
                "l1_loss",
                "grad_norm_A",
                "grad_norm_B",
                "grad_norm_C",
                "lr",
                "eval_psnr",
                "eval_ssim",
            ]
        )
        for row in metrics_history:
            writer.writerow(
                [
                    row["epoch"],
                    row["iteration"],
                    f"{row['loss']:.6f}",
                    f"{row.get('mse_loss', row['loss']):.6f}",
                    f"{row.get('l1_loss', row['loss']):.6f}",
                    f"{row['grad_A']:.6f}",
                    f"{row['grad_B']:.6f}",
                    f"{row['grad_C']:.6f}",
                    f"{row.get('lr', 0.0):.2e}",
                    (
                        f"{row['eval_psnr']:.2f}"
                        if row.get("eval_psnr") is not None
                        else ""
                    ),
                    (
                        f"{row['eval_ssim']:.4f}"
                        if row.get("eval_ssim") is not None
                        else ""
                    ),
                ]
            )

    epochs = [row["epoch"] for row in metrics_history]
    losses = [max(row["loss"], 1e-7) for row in metrics_history]
    mse_losses = [
        max(row.get("mse_loss", row["loss"]), 1e-7) for row in metrics_history
    ]
    l1_losses = [max(row.get("l1_loss", row["loss"]), 1e-7) for row in metrics_history]
    grad_A = [max(row["grad_A"], 1e-8) for row in metrics_history]
    grad_B = [max(row["grad_B"], 1e-8) for row in metrics_history]
    grad_C = [max(row["grad_C"], 1e-8) for row in metrics_history]

    eval_pts = [
        (row["epoch"], row["eval_psnr"], row["eval_ssim"])
        for row in metrics_history
        if row.get("eval_psnr") is not None and row.get("eval_ssim") is not None
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    # 1段目: Hybrid Loss / Raw MSE / Raw L1
    ax1.plot(
        epochs, losses, label="Min-SNR Hybrid Loss", color="tab:red", linewidth=1.2
    )
    ax1.plot(
        epochs,
        l1_losses,
        label="Raw L1 (Edge/Detail)",
        color="tab:purple",
        alpha=0.6,
        linewidth=0.9,
    )
    ax1.plot(
        epochs,
        mse_losses,
        label="Raw MSE (Global Color)",
        color="tab:brown",
        alpha=0.6,
        linewidth=0.9,
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("Loss (Log Scale)")
    ax1.set_title("Training Losses, Stream Gradient Norms & Image Quality Metrics")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # 2段目: Stream A / B / C 勾配ノルム
    ax2.plot(epochs, grad_A, label="Stream A (Global)", color="tab:blue", alpha=0.8)
    ax2.plot(
        epochs, grad_B, label="Stream B (Relational)", color="tab:green", alpha=0.8
    )
    ax2.plot(epochs, grad_C, label="Stream C (Texture)", color="tab:orange", alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_ylabel("Grad Norm (Log Scale)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    # 3段目: 推論画像の画質評価指標 (PSNR & SSIM)
    ax3_right = ax3.twinx()
    if eval_pts:
        ev_epochs = [p[0] for p in eval_pts]
        ev_psnr = [p[1] for p in eval_pts]
        ev_ssim = [p[2] for p in eval_pts]

        l1 = ax3.plot(
            ev_epochs,
            ev_psnr,
            marker="o",
            color="tab:cyan",
            linewidth=1.5,
            label="Eval PSNR (dB)",
        )
        l2 = ax3_right.plot(
            ev_epochs,
            ev_ssim,
            marker="s",
            color="tab:pink",
            linewidth=1.5,
            linestyle="--",
            label="Eval SSIM",
        )
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc="lower right")

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("PSNR (dB)", color="tab:cyan")
    ax3_right.set_ylabel("SSIM (0.0 - 1.0)", color="tab:pink")
    ax3_right.set_ylim(0.0, 1.05)
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "metrics_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)


# =====================================================================
# プロンプト文字列・トークン復元 & Cross-Attention ヒートマップ生成
# =====================================================================
_CACHED_TOKENIZER = None


def get_tokenizer():
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is None:
        try:
            from transformers import AutoTokenizer

            _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(
                "cl-tohoku/bert-base-japanese-v3"
            )
        except Exception as e:
            print(f"Warning: AutoTokenizerのロードに失敗しました ({e})")
            _CACHED_TOKENIZER = False
    return _CACHED_TOKENIZER if _CACHED_TOKENIZER is not False else None


def load_sample_caption_and_tokens(dataset, ds_idx, seq_len):
    """
    dataset.samples[ds_idx] から直接画像パスを取得し、
    同階層の <base_name>.txt および <base_name>.tokens.json を読み込む
    """
    sample_tuple = dataset.samples[ds_idx]
    img_path = sample_tuple[0]  # (img_path, text_path, best_bucket) の先頭要素
    base_no_ext = os.path.splitext(img_path)[0]
    target_dir = os.path.dirname(img_path)
    base_name = os.path.basename(base_no_ext)

    tensors_dir = os.path.join(target_dir, "tensors")

    # 1. 個別の .txt ファイルを読み込む
    caption = None
    txt_path = f"{base_no_ext}.txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
    else:
        # captions.json フォールバック
        json_path = os.path.join(target_dir, "captions.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            caption = data.get(os.path.basename(img_path)) or data.get(base_name)

    # 2. prepare_text.py が保存した .tokens.json があれば優先使用
    tok_json_path = os.path.join(tensors_dir, f"{base_name}.tokens.json")
    if os.path.exists(tok_json_path):
        with open(tok_json_path, "r", encoding="utf-8") as f:
            raw_tokens = json.load(f)
        return caption, raw_tokens

    # 3. .tokens.json が無い場合はその場でトークナイズ
    if caption:
        tokenizer = get_tokenizer()
        if tokenizer is not None:
            inputs = tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                max_length=seq_len,
                truncation=True,
            )
            ids = inputs["input_ids"][0].tolist()
            return caption, tokenizer.convert_ids_to_tokens(ids)

        # 万が一 fugashi/unidic-lite 未導入で tokenizer が動かない場合でも
        # .txt の文字列を句読点・助詞で分割して日本語ラベルを作る
        fallback_words = [
            w
            for w in re.split(
                r"[、。，．・\s 「」（）()のにをはがとでやもへ]+", caption
            )
            if w
        ]
        # [CLS] が index 0 なので index 1 から順に割り当て
        raw_tokens = ["[CLS]"] + fallback_words[: seq_len - 1]
        return caption, raw_tokens

    return None, None


def build_token_display_items(dataset, ds_idx, seq_len):
    """
    ヒートマップに表示する [(単語ラベル, トークンインデックスリスト), ...] を構築する
    """
    caption, raw_tokens = load_sample_caption_and_tokens(dataset, ds_idx, seq_len)

    if not raw_tokens:
        return [(f"Token #{i}", [i]) for i in range(min(seq_len, 14))]

    merged_words = []
    i = 0
    limit = min(len(raw_tokens), seq_len)
    while i < limit:
        tok = raw_tokens[i]
        if tok in ("[CLS]", "[SEP]", "[PAD]", "[UNK]"):
            i += 1
            continue
        word = tok[2:] if tok.startswith("##") else tok
        idxs = [i]
        j = i + 1
        while j < limit and raw_tokens[j].startswith("##"):
            word += raw_tokens[j][2:]
            idxs.append(j)
            j += 1
        i = j
        merged_words.append((word, idxs))

    # 【修正】助詞・記号に加え、ヒートマップのノイズになる文末表現・補助動詞・抽象語を除外
    skip_set = {
        "、",
        "。",
        "，",
        "．",
        "・",
        "：",
        "；",
        "！",
        "？",
        "（",
        "）",
        "(",
        ")",
        "「",
        "」",
        "『",
        "』",
        "【",
        "】",
        "の",
        "に",
        "を",
        "は",
        "が",
        "と",
        "で",
        "や",
        "も",
        "へ",
        "な",
        "た",
        "て",
        "し",
        "から",
        "まで",
        "より",
        "ほど",
        "くらい",
        "など",
        "といった",
        "ような",
        "ように",
        "よう",
        "ます",
        "です",
        "でした",
        "ました",
        "いる",
        "いた",
        "いて",
        "おり",
        "ある",
        "あっ",
        "れる",
        "られる",
        "せる",
        "させる",
        "され",
        "でき",
        "ない",
        "ず",
        "ん",
        "こと",
        "もの",
        "ため",
        "ところ",
        "あたり",
        "感じ",
        "印象",
        "雰囲気",
        "全体",
        "特徴",
        "特徴的",
        "見せ",
        "見え",
        "浮かべ",
        "浮かぶ",
        "写り",
        "込ん",
        "組み合わさり",
        "あしらわ",
        "身につけ",
        "添え",
        "持っ",
    }
    filtered = [
        (w, idxs)
        for w, idxs in merged_words
        if w not in skip_set
        and len(w.strip()) > 0
        and not re.fullmatch(r"[ぁ-ん]{1}", w)
    ]

    if len(filtered) < min(3, len(merged_words)):
        return (
            merged_words
            if merged_words
            else [(f"Token #{k}", [k]) for k in range(limit)]
        )

    return filtered


@torch.no_grad()
def save_attention_heatmap(
    orig_vis, gen_vis, attn_weights_C, dataset, ds_idx, save_dir, max_display_tokens=14
):
    if attn_weights_C is None:
        return

    font_prop = get_japanese_font_prop()
    _, HW, seq_len = attn_weights_C.shape
    _, H, W = gen_vis.shape

    attn_map_2d = attn_weights_C[0].transpose(0, 1).contiguous()  # [Seq_Len, H*W]
    if HW == H * W:
        attn_map_2d = attn_map_2d.view(seq_len, H, W)
    else:
        side_h = int(round(math.sqrt(HW * H / W)))
        side_w = max(1, HW // max(1, side_h))
        attn_map_2d = attn_map_2d.view(1, seq_len, side_h, side_w)
        attn_map_2d = F.interpolate(
            attn_map_2d, size=(H, W), mode="bilinear", align_corners=False
        )[0]

    token_items = build_token_display_items(dataset, ds_idx, seq_len)

    # 空間方向だけでなくトークン方向にも平均値を引いて「全トークン共通で反応しやすい背景バイアス」を除去
    valid_token_indices = [
        idx for _, idxs in token_items for idx in idxs if idx < seq_len
    ]
    if valid_token_indices:
        spatial_baseline = attn_map_2d[valid_token_indices].mean(dim=0, keepdim=True)
        rel_attn_map = torch.clamp(attn_map_2d - spatial_baseline, min=0.0)
    else:
        rel_attn_map = attn_map_2d

    scored_maps = []
    for word, idxs in token_items:
        valid_idxs = [idx for idx in idxs if idx < seq_len]
        if not valid_idxs:
            continue
        raw_tok_map = attn_map_2d[valid_idxs].mean(dim=0)
        rel_tok_map = rel_attn_map[valid_idxs].mean(dim=0)
        smoothed = F.avg_pool2d(
            rel_tok_map.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2
        )[0, 0]

        peak_raw = float(raw_tok_map.max().item())
        q95 = float(torch.quantile(smoothed.flatten(), 0.95).item())
        q50 = float(torch.quantile(smoothed.flatten(), 0.50).item())
        contrast_score = q95 - q50
        scored_maps.append(
            (contrast_score, word, valid_idxs[0], smoothed.cpu().numpy(), peak_raw)
        )

    if not scored_maps:
        return

    scored_maps.sort(key=lambda x: x[0], reverse=True)
    selected = scored_maps[:max_display_tokens]
    selected.sort(key=lambda x: x[2])

    # 【修正】ピークが極小のトークンが真っ赤に誇張されるのを防ぐため、全体の上位値を基準の下限にする
    global_ref_max = max([float(np.percentile(m[3], 99.0)) for m in selected] + [1e-6])

    orig_np = orig_vis.permute(1, 2, 0).cpu().numpy()
    gen_np = gen_vis.permute(1, 2, 0).cpu().numpy()
    gen_gray = np.dot(gen_np[..., :3], [0.299, 0.587, 0.114])
    gen_gray_rgb = np.stack([gen_gray] * 3, axis=-1)

    total_panels = 2 + len(selected)
    ncols = 4
    nrows = math.ceil(total_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.6))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gen_np)
    axes[1].set_title("Generated (Sample #1)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    cmap = plt.get_cmap("jet")

    for idx, (score, word, tok_pos, hmap, peak_raw) in enumerate(selected):
        ax = axes[2 + idx]
        h_min = float(hmap.min())
        # 個別ピークと全体基準のハイブリッド分母で正規化（弱いトークンは控えめな色になる）
        denom = max(
            float(np.percentile(hmap, 99.5)) - h_min, global_ref_max * 0.5, 1e-6
        )
        h_norm = np.clip((hmap - h_min) / denom, 0.0, 1.0)

        h_emph = np.power(h_norm, 1.3)
        heat_rgb = cmap(h_emph)[..., :3]

        alpha = (0.12 + 0.68 * h_emph)[..., None]
        blended = np.clip(
            (1.0 - alpha) * (gen_gray_rgb * 0.85) + alpha * heat_rgb, 0.0, 1.0
        )

        ax.imshow(blended)
        title_text = f"#{tok_pos}「{word}」\n(peak: {peak_raw:.3f})"
        if font_prop is not None:
            ax.set_title(title_text, fontproperties=font_prop, fontsize=10)
        else:
            ax.set_title(title_text, fontsize=10)
        ax.axis("off")

    for k in range(total_panels, len(axes)):
        axes[k].axis("off")

    plt.suptitle(
        "Stream C Cross-Attention Heatmap per Prompt Token (Sample #1)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    heatmap_path = os.path.join(save_dir, "attention_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close(fig)


# =====================================================================
# DDIM サンプラー（最終ステップでの Cross-Attention フック取得に対応）
# =====================================================================
def ddim_sample_batch(
    model,
    contexts,
    shape,
    device,
    max_steps=1000,
    capture_steps=None,
    capture_final_attn=False,
):
    B, C, H, W = shape
    gen = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(shape, generator=gen, device=device)
    progression_tensors = []
    final_attn_weights = None

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

        # 最終ステップ (i == 0) で Stream C の Cross-Attention 重みをキャプチャ
        if capture_final_attn and i == 0 and hasattr(model, "cross_attn_C"):
            captured = {}
            orig_forward = model.cross_attn_C.forward

            def hooked_forward(query, key, value, *args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                out, weights = orig_forward(query, key, value, *args, **kwargs)
                captured["weights"] = weights.detach().cpu()
                return out, weights

            model.cross_attn_C.forward = hooked_forward
            try:
                predicted_x0 = model(x, contexts, t)
            finally:
                model.cross_attn_C.forward = orig_forward
            final_attn_weights = captured.get("weights", None)
        else:
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

    return x, progression_tensors, final_attn_weights


def generate_evaluation_samples(
    model, dataset, device, save_dir, max_steps=1000, eval_res=None
):
    """
    データセットからランダムに3つ選び:
    1. comparison_grid.png (上段: 元画像3枚, 下段: 生成画像3枚)
    2. denoising_progression.png (代表1枚目: ステップ毎の推移)
    3. attention_heatmap.png (代表1枚目: トークン毎のCross-Attentionヒートマップ)
    を生成し、3枚の平均 PSNR と SSIM を返す。
    """
    model.eval()
    num_samples = 3

    # データセットの中から毎回ランダムに3枚を選択（3枚未満の場合は重複を許容）
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
    rep_attn_weights = None
    psnr_list = []
    ssim_list = []

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

            is_rep = s_idx == 0
            gen_x, prog, attn_w = ddim_sample_batch(
                model,
                ctx,
                shape=(1, 3, target_h, target_w),
                device=device,
                max_steps=max_steps,
                capture_steps=capture_steps if is_rep else None,
                capture_final_attn=is_rep,
            )

            orig_vis = torch.clamp(
                (orig_resized[0].cpu().float() + 1.0) / 2.0, 0.0, 1.0
            )
            gen_vis = torch.clamp((gen_x[0].cpu().float() + 1.0) / 2.0, 0.0, 1.0)

            orig_vis_list.append(orig_vis)
            gen_vis_list.append(gen_vis)

            # PSNR / SSIM の算出
            psnr_list.append(compute_psnr(orig_vis, gen_vis))
            ssim_list.append(compute_ssim(orig_vis, gen_vis))

            if is_rep:
                progression_tensors = prog
                rep_attn_weights = attn_w

        # 1. comparison_grid.png の保存
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

        # 2. denoising_progression.png の保存 (代表サンプル indices[0])
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

        # 3. attention_heatmap.png の保存 (代表サンプル indices[0])
        save_attention_heatmap(
            orig_vis=orig_vis_list[0],
            gen_vis=gen_vis_list[0],
            attn_weights_C=rep_attn_weights,
            dataset=dataset,
            ds_idx=indices[0],
            save_dir=save_dir,
        )

    model.train()
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    return avg_psnr, avg_ssim


def resolve_resume_paths(resume_arg):
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
    target_iterations=None,
    timesteps=1000,
    base_size=256,
    min_short_edge=256,
    train_res=None,
    eval_res=None,
    multi_aspect=True,
    lr=5e-4,
    save_every_epochs=None,
    log_every_epochs=None,
    use_fp16=False,
    resume=None,
):
    print(f"--- Starting Production Training (Subset: {subset}) ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"Using device: {device} (Visible GPUs: {num_gpus})")

    # 1. GPUアーキテクチャに応じた自動混合精度 (L4: bfloat16 / T4: float16 + GradScaler)
    use_amp = device.type == "cuda"
    if use_amp and (torch.cuda.is_bf16_supported() and not use_fp16):
        amp_dtype = torch.bfloat16
        use_scaler = False
    else:
        amp_dtype = torch.float16
        use_scaler = use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if use_amp:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(
            f"AMP Config -> GPUs: {gpu_names} | dtype: {amp_dtype} | GradScaler: {use_scaler}"
        )

    # 生モデル (raw_model) を構築し、保存・ロード・勾配取得・EMAは常に raw_model を参照する
    raw_model = TripleStreamModel(in_channels=3).to(device)
    ema = EMAModel(raw_model, decay=0.9995)

    optimizer = optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = VisualDataset(
        subset=subset,
        datasets_root=datasets_root,
        base_size=base_size,
        min_short_edge=min_short_edge,
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

    if target_iterations is None or target_iterations <= 0:
        target_iterations = estimate_target_iterations(
            dataset, batch_size=batch_size, timesteps=timesteps
        )
        iter_mode_str = "Auto"
    else:
        iter_mode_str = "Manual"

    epochs = math.ceil(target_iterations / iters_per_epoch)

    if save_every_epochs is None or save_every_epochs <= 0:
        save_every_epochs = estimate_save_every_epochs(epochs, iters_per_epoch)
        save_mode_str = "Auto"
    else:
        save_mode_str = "Manual"

    if log_every_epochs is None or log_every_epochs <= 0:
        log_every_epochs = estimate_log_every_epochs(
            epochs, iters_per_epoch, save_every_epochs
        )
        log_mode_str = "Auto"
    else:
        log_mode_str = "Manual"

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    start_epoch = 0
    global_iter = 0
    metrics_history = []
    run_dir = None

    if resume is not None and resume.strip() != "":
        ckpt_file, ckpt_subdir, parent_run_dir = resolve_resume_paths(resume)
        print(f"Resuming training from: {ckpt_file}")
        ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)

        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            raw_model.load_state_dict(ckpt_data["model_state_dict"])
            ema.ema_model.load_state_dict(
                ckpt_data.get("ema_state_dict", ckpt_data["model_state_dict"])
            )
            if "optimizer_state_dict" in ckpt_data:
                optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt_data:
                scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
                scheduler.T_max = epochs
            if (
                use_scaler
                and "scaler_state_dict" in ckpt_data
                and ckpt_data["scaler_state_dict"] is not None
            ):
                scaler.load_state_dict(ckpt_data["scaler_state_dict"])
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
            raw_model.load_state_dict(ckpt_data)
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

            for _ in range(start_epoch):
                scheduler.step()

        # Resume時に重複したエポック行が混入していても start_epoch までにクリーンアップする
        if metrics_history:
            dedup = {}
            for row in metrics_history:
                ep = int(row["epoch"])
                if ep <= start_epoch:
                    dedup[ep] = row
            metrics_history = [dedup[k] for k in sorted(dedup.keys())]

        if parent_run_dir is not None:
            run_dir = parent_run_dir

        print(
            f"  -> Restored Progress: Epoch {start_epoch}/{epochs} | "
            f"Global Iter: {global_iter}/{epochs * iters_per_epoch} | "
            f"Current LR: {scheduler.get_last_lr()[0]:.2e}"
        )

    # 2. マルチGPU環境 (Kaggle T4x2 等) で、かつ batch_size >= 2 の場合は DataParallel で自動分散
    if num_gpus > 1 and batch_size >= num_gpus:
        print(
            f"Multi-GPU DataParallel Enabled across {num_gpus} GPUs "
            f"(Per-GPU batch size: {batch_size // num_gpus})"
        )
        forward_model = nn.DataParallel(raw_model)
    else:
        forward_model = raw_model

    if run_dir is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(checkpoints_root, timestamp_str)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Checkpoint Root Directory: {run_dir}")

    active_buckets = {k: len(v) for k, v in dataset.bucket_to_indices.items()}
    total_planned_iters = epochs * iters_per_epoch
    print(f"Active Aspect Ratio Buckets (H, W): {active_buckets}")
    print(
        f"Total Images: {len(dataset)} | Batch Size: {batch_size} | "
        f"Iters/Epoch: {iters_per_epoch} | Timesteps: {timesteps}"
    )
    print(
        f"Schedule -> Target Iters ({iter_mode_str}): {total_planned_iters} ({epochs} epochs) | "
        f"Save Every ({save_mode_str}): {save_every_epochs} ep (~{save_every_epochs * iters_per_epoch} iters) | "
        f"Log Every ({log_mode_str}): {log_every_epochs} ep (~{log_every_epochs * iters_per_epoch} iters)"
    )

    if start_epoch >= epochs:
        print(
            f"すでに目標エポック数 ({start_epoch}/{epochs}) に到達しています。"
            f"さらに学習を継続する場合は --target-iters に {global_iter} より大きい値を指定してください。"
        )
        return

    train_start_time = time.time()
    epoch_digit_width = len(str(epochs))
    iter_digit_width = len(str(total_planned_iters))

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_l1 = 0.0
        grad_norm_A_sum = 0.0
        grad_norm_B_sum = 0.0
        grad_norm_C_sum = 0.0

        for images, contexts in dataloader:
            images = images.to(device, non_blocking=True)
            contexts = contexts.to(device, non_blocking=True)
            B = images.size(0)

            optimizer.zero_grad(set_to_none=True)

            t = torch.randint(0, timesteps, (B,), device=device).long()
            alpha_cumprod = get_cosine_alpha_cumprod(t, max_steps=timesteps).view(
                B, 1, 1, 1
            )

            noise = torch.randn_like(images)
            noisy_images = (
                torch.sqrt(alpha_cumprod) * images
                + torch.sqrt(1.0 - alpha_cumprod) * noise
            )

            with torch.amp.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_amp
            ):
                # バッチ端数で B == 1 の場合は DataParallel の空デバイスエラーを避けるため raw_model で実行
                active_net = (
                    raw_model if (B < num_gpus and num_gpus > 1) else forward_model
                )
                predicted_x0 = active_net(noisy_images, contexts, t)

                snr = alpha_cumprod.view(-1) / torch.clamp(
                    1.0 - alpha_cumprod.view(-1), min=1e-5
                )
                snr_weight = torch.clamp(snr, min=0.1, max=5.0)

                mse_loss = F.mse_loss(predicted_x0, images, reduction="none").mean(
                    dim=[1, 2, 3]
                )
                l1_loss = F.l1_loss(predicted_x0, images, reduction="none").mean(
                    dim=[1, 2, 3]
                )

                combined_loss = 0.55 * mse_loss + 0.45 * l1_loss
                loss = (combined_loss * snr_weight).mean()

            # GradScaler対応の backward & unscale & 勾配クリッピング
            # (use_scaler=False の L4/BF16 環境では通常の loss.backward() / optimizer.step() と完全に等価)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)

            grad_norm_A_sum += (
                raw_model.attn_A.in_proj_weight.grad.norm().item()
                if hasattr(raw_model, "attn_A")
                and raw_model.attn_A.in_proj_weight.grad is not None
                else 0.0
            )
            grad_norm_B_sum += (
                raw_model.mix_B[0].weight.grad.norm().item()
                if hasattr(raw_model, "mix_B")
                and raw_model.mix_B[0].weight.grad is not None
                else 0.0
            )
            grad_norm_C_sum += (
                raw_model.cross_attn_C.in_proj_weight.grad.norm().item()
                if hasattr(raw_model, "cross_attn_C")
                and raw_model.cross_attn_C.in_proj_weight.grad is not None
                else 0.0
            )

            scaler.step(optimizer)
            scaler.update()
            ema.update(raw_model)

            epoch_loss += loss.item()
            epoch_mse += mse_loss.mean().item()
            epoch_l1 += l1_loss.mean().item()
            global_iter += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_mse = epoch_mse / num_batches
        avg_l1 = epoch_l1 / num_batches
        grad_norm_A = grad_norm_A_sum / num_batches
        grad_norm_B = grad_norm_B_sum / num_batches
        grad_norm_C = grad_norm_C_sum / num_batches

        is_save_step = (
            epoch % save_every_epochs == 0
            or epoch == start_epoch + 1
            or epoch == epochs
        )
        is_log_step = (epoch % log_every_epochs == 0) or is_save_step

        record = {
            "epoch": epoch,
            "iteration": global_iter,
            "loss": avg_loss,
            "mse_loss": avg_mse,
            "l1_loss": avg_l1,
            "grad_A": grad_norm_A,
            "grad_B": grad_norm_B,
            "grad_C": grad_norm_C,
            "lr": current_lr,
            "eval_psnr": None,
            "eval_ssim": None,
        }
        metrics_history.append(record)

        if is_log_step:
            elapsed = time.time() - train_start_time
            done_epochs = epoch - start_epoch
            remain_epochs = epochs - epoch
            eta_sec = (elapsed / max(1, done_epochs)) * remain_epochs
            progress_pct = (epoch / epochs) * 100.0

            print(
                f"[Progress {progress_pct:5.1f}%] "
                f"Epoch [{epoch:{epoch_digit_width}d}/{epochs}] "
                f"(Iter: {global_iter:{iter_digit_width}d}/{total_planned_iters}) | "
                f"Loss: {avg_loss:.6f} (MSE:{avg_mse:.4f}/L1:{avg_l1:.4f}) | "
                f"Grad(A/B/C): {grad_norm_A:.4f}/{grad_norm_B:.4f}/{grad_norm_C:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Elapsed: {format_duration(elapsed)} | ETA: {format_duration(eta_sec)}",
                flush=True,
            )

        if is_save_step:
            ckpt_subdir = os.path.join(run_dir, f"epoch{epoch}_iter{global_iter}")
            os.makedirs(ckpt_subdir, exist_ok=True)

            # 推論時のVRAMピークも抑えるため autocast 下で評価サンプルを生成
            with torch.amp.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_amp
            ):
                eval_psnr, eval_ssim = generate_evaluation_samples(
                    ema.ema_model,
                    dataset,
                    device,
                    ckpt_subdir,
                    max_steps=timesteps,
                    eval_res=eval_res,
                )
            record["eval_psnr"] = eval_psnr
            record["eval_ssim"] = eval_ssim

            # 推論互換用のEMA重みファイル (model.pth)
            ckpt_path = os.path.join(ckpt_subdir, "model.pth")
            torch.save(ema.ema_model.state_dict(), ckpt_path)

            # Resume用の完全状態ファイル (checkpoint.pt)
            full_ckpt_path = os.path.join(ckpt_subdir, "checkpoint.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_iter": global_iter,
                    "model_state_dict": raw_model.state_dict(),
                    "ema_state_dict": ema.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_scaler else None,
                    "metrics_history": metrics_history,
                },
                full_ckpt_path,
            )

            save_metrics_csv_and_plot(metrics_history, ckpt_subdir)

            print(
                f"  --> [Checkpoint Saved] {ckpt_subdir} | "
                f"Eval PSNR: {eval_psnr:.2f} dB | Eval SSIM: {eval_ssim:.4f}",
                flush=True,
            )

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
        type=str,
        default="auto",
        help="チェックポイント保存間隔（エポック単位、'auto' または 0 で自動決定）",
    )
    parser.add_argument(
        "--log-every",
        "--log_every",
        dest="log_every",
        type=str,
        default="auto",
        help="ターミナルへの進捗出力間隔（エポック単位、'auto' または 0 で自動決定）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="バッチサイズ (デフォルト: 2)"
    )
    parser.add_argument(
        "--target-iters",
        type=str,
        default="auto",
        help="目標イテレーション数 ('auto' または 0 でデータセット規模・解像度から自動算出)",
    )
    parser.add_argument(
        "--timesteps", "-t", type=int, default=1000, help="ノイズのタイムステップ数"
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=256,
        help="マルチアスペクト比バケットの短辺基準解像度 (デフォルト: 256)",
    )
    parser.add_argument(
        "--min-short-edge",
        type=int,
        default=256,
        help="バケットの最小短辺ピクセル数 (デフォルト: 256)",
    )
    parser.add_argument(
        "--train-res",
        type=str,
        default=None,
        help="学習解像度を固定したい場合に指定 (例: '256' または '256x384')",
    )
    parser.add_argument(
        "--eval-res",
        type=str,
        default=None,
        help="チェックポイント推論時の出力解像度を指定 (例: '256x384')",
    )
    parser.add_argument(
        "--no-multi-aspect",
        action="store_true",
        help="指定するとマルチアスペクト比を無効化し、base-sizeの正方形に固定する",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="初期学習率 (デフォルト: 5e-4)"
    )
    parser.add_argument("--use-fp16", action="store_true")
    args = parser.parse_args()

    run_training(
        subset=args.subset,
        datasets_root=args.datasets_root,
        checkpoints_root=args.checkpoints_root,
        batch_size=args.batch_size,
        target_iterations=parse_auto_int(args.target_iters),
        timesteps=args.timesteps,
        base_size=args.base_size,
        min_short_edge=args.min_short_edge,
        train_res=parse_resolution(args.train_res),
        eval_res=parse_resolution(args.eval_res),
        multi_aspect=not args.no_multi_aspect,
        lr=args.lr,
        save_every_epochs=parse_auto_int(args.save_every),
        log_every_epochs=parse_auto_int(args.log_every),
        use_fp16=args.use_fp16,
        resume=args.resume,
    )
