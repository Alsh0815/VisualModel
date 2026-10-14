import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PoCDataset(Dataset):
    def __init__(self, subset, datasets_root="datasets", image_size=64):
        self.data_dir = os.path.join(datasets_root, subset)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"データセットディレクトリが存在しません: {self.data_dir}")

        # 画像の前処理パイプライン
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),  # 0.0 ~ 1.0 に変換
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # -1.0 ~ 1.0 に正規化
            ]
        )

        self.samples = []
        for file in sorted(os.listdir(self.data_dir)):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                base_name = os.path.splitext(file)[0]
                img_path = os.path.join(self.data_dir, file)

                # 画像と同名のテンソル (<base_name>.pt) を検索
                text_filename = f"{base_name}.pt"
                text_path = os.path.join(self.data_dir, text_filename)

                # 旧仕様 (caption_XX.pt) との後方互換フォールバック
                if not os.path.exists(text_path) and "image" in base_name:
                    legacy_path = os.path.join(
                        self.data_dir, base_name.replace("image", "caption") + ".pt"
                    )
                    if os.path.exists(legacy_path):
                        text_path = legacy_path

                if os.path.exists(text_path):
                    self.samples.append((img_path, text_path))
                else:
                    print(f"Warning: 対応するテンソルファイルが見つかりません ({file} -> {text_filename})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text_path = self.samples[idx]

        # 1. 画像の読み込みと前処理 [3, H, W]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        # 2. テキストテンソルの読み込み [M=4, 128]
        context_tensor = torch.load(text_path, weights_only=True)
        if context_tensor.dim() == 3 and context_tensor.size(0) == 1:
            context_tensor = context_tensor.squeeze(0)

        return image_tensor, context_tensor