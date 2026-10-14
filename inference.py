import math
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms

from models.luan.model import TripleStreamPoC


def get_cosine_alpha_cumprod(t, max_steps=1000, s=0.008):
    """特異点を回避するためのオフセット付きコサインスケジュール"""
    t_norm = t.float() / max_steps
    f_t = torch.cos((t_norm + s) / (1.0 + s) * math.pi / 2) ** 2
    f_0 = torch.cos(torch.tensor(s / (1.0 + s) * math.pi / 2, device=t.device)) ** 2
    return f_t / f_0


def generate_image(
    prompt: str,
    weights_path: str = "poc_model.pth",
    timesteps: int = 1000,
    output_filename: str = "generated_output.png",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 学習時と同一仕様のテキストエンコーダとプロジェクション層の準備
    model_name = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_encoder = AutoModel.from_pretrained(model_name)
    text_encoder.eval()

    torch.manual_seed(42)
    dim_proj = nn.Linear(768, 128)
    dim_proj.eval()

    # 2. プロンプトのエンコードとテンソル変換
    with torch.no_grad():
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True,
        )
        outputs = text_encoder(**inputs)
        hidden_states = outputs.last_hidden_state
        context_tensor = dim_proj(hidden_states)  # [1, 256, 128]

    context_tensor = context_tensor.to(device)

    # 3. モデルのロード (学習時と合わせて hidden_dim_C=64)
    model = TripleStreamPoC(in_channels=3, hidden_dim_C=64).to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.eval()

    # 4. 初期ランダムノイズの生成
    x = torch.randn(1, 3, 128, 128, device=device)

    # 5. 推論の実行 (逆拡散プロセス / サンプリング)
    print(f"Starting reverse diffusion process ({timesteps} steps)...")
    with torch.no_grad():
        for i in reversed(range(timesteps)):
            t = torch.tensor([i], device=device).long()
            alpha_cumprod_t = get_cosine_alpha_cumprod(t, max_steps=timesteps)

            if i > 0:
                t_prev = torch.tensor([i - 1], device=device).long()
                alpha_cumprod_t_prev = get_cosine_alpha_cumprod(
                    t_prev, max_steps=timesteps
                )
                noise = torch.randn_like(x)
            else:
                alpha_cumprod_t_prev = torch.tensor([1.0], device=device)
                noise = torch.zeros_like(x)

            alpha_t = alpha_cumprod_t / alpha_cumprod_t_prev
            beta_t = 1.0 - alpha_t
            beta_t = torch.clamp(beta_t, max=0.999)
            alpha_t = 1.0 - beta_t

            predicted_x0 = model(x, context_tensor, t)
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

    # 6. 画像として保存
    output_tensor = (x.squeeze(0) + 1.0) / 2.0
    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

    to_pil = transforms.ToPILImage()
    final_image = to_pil(output_tensor.cpu())

    final_image.save(output_filename)
    print(f"Generated: {output_filename}")


if __name__ == "__main__":
    default_prompt = "目を細めて満面の笑みを浮かべる、薄いラベンダーがかった白髪とエルフ耳を持つ、3Dアニメ・ゲーム調（トゥーンシェーディング）の少女の上半身クローズアップ。\n彼女は長い白髪に、ピンクの桜の花々や、額に小さな紫色の宝石があしらわれた細いティアラ風の髪飾りを身につけています。髪の間からは特徴的な尖ったエルフ耳が覗いています。衣装は白と紫（ラベンダー）を基調とした、花や蝶の羽をモチーフにした非常に華やかなドレスで、胸元には大きなピンクと紫の花飾りが添えられています。\nポーズは、紫色の手袋をはめた左手を顎や胸のあたりに優しく添え、右手（腕の装飾部分）を頭の後ろ側へ軽く持ち上げるような、お茶目で親しみやすい仕草をとっています。表情は、嬉しそうに目を閉じ、少し開いた口からかすかに歯が見えるほどの楽しげな笑顔を見せています。\n背景は、パステル調のブルー、パープル、ホワイトなどが柔らかくボケた室内またはフェスティバルのような場所。自然で柔らかい光が彼女の愛らしい表情とカラフルな衣装を明るく引き立てています。"

    parser = argparse.ArgumentParser(description="条件付き生成モデルの推論スクリプト")
    parser.add_argument(
        "--prompt", "-p", type=str, default=default_prompt, help="生成プロンプト"
    )
    parser.add_argument(
        "--weights", "-w", type=str, default="poc_model.pth", help="重みファイルのパス"
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=1000,
        help="学習時に設定したタイムステップ数（デフォルト: 1000）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="generated_output.png",
        help="出力ファイル名",
    )
    args = parser.parse_args()

    generate_image(
        prompt=args.prompt,
        weights_path=args.weights,
        timesteps=args.timesteps,
        output_filename=args.output,
    )
