import os
import gc
import math
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from transformers import AutoTokenizer, AutoModel

from models.luan.model import TripleStreamModel


def parse_resolution(res_str):
    """'128' -> (128, 128), '128x192' -> (128, 192) のように解像度文字列をパースし、4の倍数に揃える"""
    if res_str is None or res_str.strip() == "":
        return (128, 128)
    res_str = res_str.lower().strip()
    if "x" in res_str:
        h, w = map(int, res_str.split("x"))
    else:
        val = int(res_str)
        h, w = val, val

    # Stream A (1/4縮小) の整合性のため4の倍数にアラインメント
    new_h = max(4, (h // 4) * 4)
    new_w = max(4, (w // 4) * 4)
    if (new_h, new_w) != (h, w):
        print(
            f"Warning: モデルのストリーム整合性のため解像度を ({h}, {w}) -> ({new_h}, {new_w}) に調整しました。"
        )
    return (new_h, new_w)


def get_cosine_alpha_cumprod(t, max_steps=1000, s=0.008):
    """特異点を回避するためのオフセット付きコサインスケジュール"""
    t_norm = t.float() / max_steps
    f_t = torch.cos((t_norm + s) / (1.0 + s) * math.pi / 2) ** 2
    f_0 = torch.cos(torch.tensor(s / (1.0 + s) * math.pi / 2, device=t.device)) ** 2
    return f_t / f_0


def resolve_weights_path(weights_arg):
    """ディレクトリが渡された場合も model.pth または checkpoint.pt を自動解決する"""
    path = os.path.normpath(weights_arg.strip())
    if not os.path.exists(path):
        raise FileNotFoundError(f"指定された重みパスが存在しません: {path}")

    if os.path.isdir(path):
        for candidate in ["model_final.pth", "model.pth", "checkpoint.pt"]:
            cand_path = os.path.join(path, candidate)
            if os.path.exists(cand_path):
                return cand_path
        raise FileNotFoundError(
            f"ディレクトリ内に重みファイルが見つかりません: {path}"
        )
    return path


def load_model_weights(model, weights_path, device):
    """単体の state_dict と Resume用 checkpoint.pt の両方に対応してEMA重みをロードする"""
    resolved_path = resolve_weights_path(weights_path)
    print(f"Loading weights from: {resolved_path}")
    ckpt = torch.load(resolved_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "ema_state_dict" in ckpt:
            model.load_state_dict(ckpt["ema_state_dict"])
            print("  -> Loaded EMA weights (ema_state_dict) from checkpoint.")
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print("  -> Loaded model_state_dict from checkpoint.")
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)


def encode_prompt(prompt, device):
    """
    prepare_text.py と完全同一の仕様でプロンプトを [1, 256, 128] テンソルに変換し、
    完了後にテキストエンコーダをメモリから解放する
    """
    model_name = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_encoder = AutoModel.from_pretrained(model_name).to(device)
    text_encoder.eval()

    # prepare_text.py と同一の固定シード射影層 (768 -> 128)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    torch.manual_seed(42)
    dim_proj = nn.Linear(768, 128).to(device)
    dim_proj.eval()

    with torch.no_grad():
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True,
        ).to(device)
        outputs = text_encoder(**inputs)
        hidden_states = outputs.last_hidden_state
        context_tensor = dim_proj(hidden_states)  # [1, 256, 128]

    # 乱数シード状態を復元
    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, device)

    # VRAM節約のためテキストエンコーダを即座に破棄
    del tokenizer, text_encoder, dim_proj, inputs, outputs, hidden_states
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return context_tensor


@torch.no_grad()
def ddim_sample(
    model,
    context_tensor,
    shape,
    device,
    max_steps=1000,
    sampling_steps=1000,
    seed=None,
    save_progression=False,
):
    """
    DDIM (eta=0) 決定論的サンプラー（ストライドによるステップ間引き対応）
    """
    B, C, H, W = shape
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, generator=generator, device=device)
    else:
        x = torch.randn(shape, device=device)

    sampling_steps = min(max(1, sampling_steps), max_steps)
    # [max_steps - 1, ..., 0] のサンプリングスケジュールを生成
    step_indices = torch.linspace(
        max_steps - 1, 0, steps=sampling_steps, dtype=torch.long
    ).tolist()

    capture_indices = set(
        [step_indices[0], step_indices[-1]]
        + step_indices[:: max(1, len(step_indices) // 10)]
    )
    progression_tensors = []

    for idx, i in enumerate(step_indices):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        alpha_cumprod_t = get_cosine_alpha_cumprod(t, max_steps=max_steps).view(
            B, 1, 1, 1
        )

        predicted_x0 = model(x, context_tensor, t)
        predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)

        if idx < len(step_indices) - 1:
            i_prev = step_indices[idx + 1]
            t_prev = torch.full((B,), i_prev, device=device, dtype=torch.long)
            alpha_cumprod_t_prev = get_cosine_alpha_cumprod(
                t_prev, max_steps=max_steps
            ).view(B, 1, 1, 1)

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

        if save_progression and i in capture_indices:
            step_img = torch.clamp((x[0].detach().cpu() + 1.0) / 2.0, 0.0, 1.0)
            progression_tensors.append(step_img)

    return x, progression_tensors


def generate_image(
    prompt: str,
    weights_path: str,
    res=(128, 128),
    timesteps: int = 1000,
    sampling_steps: int = 1000,
    num_samples: int = 1,
    seed: int = None,
    context_pt: str = None,
    output_filename: str = "generated_output.png",
    save_progression: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. コンテキストテンソルの準備 (.pt 直接指定 または プロンプトからの生成)
    if context_pt is not None and os.path.exists(context_pt):
        print(f"Loading precomputed context tensor from: {context_pt}")
        context_tensor = torch.load(
            context_pt, map_location=device, weights_only=True
        )
        if context_tensor.dim() == 2:
            context_tensor = context_tensor.unsqueeze(0)
    else:
        print("Encoding prompt with BERT...")
        context_tensor = encode_prompt(prompt, device)

    if num_samples > 1:
        context_tensor = context_tensor.expand(num_samples, -1, -1)

    # 2. TripleStreamModel のロード
    model = TripleStreamModel(in_channels=3, hidden_dim_C=64).to(device)
    load_model_weights(model, weights_path, device)
    model.eval()

    # 3. DDIM 逆拡散サンプリングの実行
    H, W = res
    print(
        f"Generating {num_samples} image(s) at {H}x{W} "
        f"(DDIM steps: {sampling_steps}/{timesteps}, seed: {seed})..."
    )
    gen_x, prog_tensors = ddim_sample(
        model=model,
        context_tensor=context_tensor,
        shape=(num_samples, 3, H, W),
        device=device,
        max_steps=timesteps,
        sampling_steps=sampling_steps,
        seed=seed,
        save_progression=save_progression,
    )

    # 4. 画像の保存
    out_dir = os.path.dirname(output_filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_vis = torch.clamp((gen_x.cpu() + 1.0) / 2.0, 0.0, 1.0)
    if num_samples == 1:
        save_image(output_vis[0], output_filename)
    else:
        nrow = min(4, num_samples)
        grid = make_grid(output_vis, nrow=nrow, padding=4, pad_value=1.0)
        save_image(grid, output_filename)
    print(f"Saved generated image to: {output_filename}")

    if save_progression and prog_tensors:
        base, ext = os.path.splitext(output_filename)
        prog_filename = f"{base}_progression{ext}"
        prog_grid = make_grid(
            torch.stack(prog_tensors, dim=0),
            nrow=len(prog_tensors),
            padding=4,
            pad_value=1.0,
        )
        save_image(prog_grid, prog_filename)
        print(f"Saved denoising progression to: {prog_filename}")


if __name__ == "__main__":
    default_prompt = (
        "目を細めて満面の笑みを浮かべる、薄いラベンダーがかった白髪とエルフ耳を持つ、"
        "3Dアニメ・ゲーム調（トゥーンシェーディング）の少女の上半身クローズアップ。\n"
        "彼女は長い白髪に、ピンクの桜の花々や、額に小さな紫色の宝石があしらわれた細いティアラ風の髪飾りを身につけています。"
        "髪の間からは特徴的な尖ったエルフ耳が覗いています。衣装は白と紫（ラベンダー）を基調とした、"
        "花や蝶の羽をモチーフにした非常に華やかなドレスで、胸元には大きなピンクと紫の花飾りが添えられています。\n"
        "ポーズは、紫色の手袋をはめた左手を顎や胸のあたりに優しく添え、右手（腕の装飾部分）を頭の後ろ側へ軽く持ち上げるような、"
        "お茶目で親しみやすい仕草をとっています。表情は、嬉しそうに目を閉じ、少し開いた口からかすかに歯が見えるほどの楽しげな笑顔を見せています。\n"
        "背景は、パステル調のブルー、パープル、ホワイトなどが柔らかくボケた室内またはフェスティバルのような場所。"
        "自然で柔らかい光が彼女の愛らしい表情とカラフルな衣装を明るく引き立てています。"
    )

    parser = argparse.ArgumentParser(description="Triple-Stream モデル 本番推論スクリプト")
    parser.add_argument(
        "--prompt", "-p", type=str, default=default_prompt, help="生成プロンプト"
    )
    parser.add_argument(
        "--context-pt",
        "-c",
        type=str,
        default=None,
        help="事前計算済みのテキスト特徴量 (.pt) を直接指定する場合のパス",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        required=True,
        help="重みファイル (model_final.pth, model.pth, checkpoint.pt) またはチェックポイントフォルダのパス",
    )
    parser.add_argument(
        "--res",
        "-r",
        type=str,
        default="128x128",
        help="生成画像の解像度 HeightxWidth (例: '128', '128x128', '160x96')",
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=1000,
        help="学習時に設定した最大タイムステップ数 (デフォルト: 1000)",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=1000,
        help="DDIM推論時のサンプリングステップ数。100〜250で高速生成、1000でフルステップ推論 (デフォルト: 1000)",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=1,
        help="同時生成する画像枚数 (2以上を指定するとグリッド画像として保存)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="初期ノイズの乱数シード (再現性確保用)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="generated_output.png",
        help="出力画像ファイル名 (デフォルト: generated_output.png)",
    )
    parser.add_argument(
        "--save-progression",
        action="store_true",
        help="指定すると逆拡散の途中経過 (*_progression.png) も合わせて保存する",
    )
    args = parser.parse_args()

    generate_image(
        prompt=args.prompt,
        weights_path=args.weights,
        res=parse_resolution(args.res),
        timesteps=args.timesteps,
        sampling_steps=args.sampling_steps if hasattr(args, "sampling_steps") else args.steps,
        num_samples=args.num_samples,
        seed=args.seed,
        context_pt=args.context_pt,
        output_filename=args.output,
        save_progression=args.save_progression,
    )