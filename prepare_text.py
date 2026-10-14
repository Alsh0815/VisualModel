import os
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


def load_captions(target_dir):
    """対象ディレクトリ内の captions.json および個別 .txt からプロンプトを読み込む"""
    captions = {}

    # 1. captions.json が存在する場合はベースとして読み込む
    json_path = os.path.join(target_dir, "captions.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                base_name = os.path.splitext(k)[0]
                captions[base_name] = v

    # 2. 画像ファイルに対応する個別の .txt ファイルがあれば優先して読み込む
    for file in sorted(os.listdir(target_dir)):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            base_name = os.path.splitext(file)[0]
            txt_path = os.path.join(target_dir, f"{base_name}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    captions[base_name] = f.read().strip()

    return captions


def create_caption_tensors(subset, datasets_root="datasets"):
    target_dir = os.path.join(datasets_root, subset)
    if not os.path.exists(target_dir):
        raise FileNotFoundError(
            f"指定されたデータセットフォルダが存在しません: {target_dir}"
        )

    captions = load_captions(target_dir)
    if not captions:
        print(
            f"Warning: {target_dir} 内に有効なプロンプト（.txt または captions.json）が見つかりませんでした。"
        )
        return

    # 日本語対応の軽量な事前学習済みモデル (BERT-base)
    model_name = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_encoder = AutoModel.from_pretrained(model_name)
    text_encoder.eval()

    # BERTの出力(768次元)を、PoCモデルの仕様(128次元)に圧縮する固定プロジェクション層
    torch.manual_seed(42)
    dim_proj = nn.Linear(768, 128)
    dim_proj.eval()

    with torch.no_grad():
        for base_name, text in captions.items():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=256,
                truncation=True,
            )

            outputs = text_encoder(**inputs)
            hidden_states = outputs.last_hidden_state

            # PoCモデルの要件 (M=4) に合わせるため、最初の4トークンを抽出
            tokens_m4 = hidden_states[:, :4, :]

            # 次元数を 768 -> 128 に圧縮し、バッチ次元を外して [4, 128] として保存
            context_tensor = dim_proj(tokens_m4).squeeze(0)

            # 画像と同じベース名で同階層に保存 (例: image_01.pt)
            save_name = f"{base_name}.pt"
            save_path = os.path.join(target_dir, save_name)
            torch.save(context_tensor, save_path)

            preview = text.replace("\n", " ")[:40]
            print(
                f"Saved {save_path} | Shape: {context_tensor.shape} | Text: {preview}..."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="プロンプトファイルから特徴量テンソル(.pt)を生成します。"
    )
    parser.add_argument(
        "--subset",
        "-s",
        type=str,
        required=True,
        help="datasetsフォルダ内の対象サブフォルダ名 (例: poc_set1)",
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default="datasets",
        help="データセットのルートフォルダ (デフォルト: datasets)",
    )
    args = parser.parse_args()

    create_caption_tensors(subset=args.subset, datasets_root=args.datasets_root)
