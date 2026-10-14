import os
import math
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


def generate_aspect_buckets(base_size=128, step=32, max_ratio=2.0):
    """
    base_size * base_size の総ピクセル数を維持しつつ、
    step(32)の倍数となる (H, W) アスペクト比バケット一覧を自動生成する
    """
    target_area = base_size * base_size
    buckets = set()
    min_dim = max(step, int(base_size / math.sqrt(max_ratio) // step) * step)
    max_dim = int(base_size * math.sqrt(max_ratio) // step) * step

    for h in range(min_dim, max_dim + step, step):
        w = round((target_area / h) / step) * step
        w = max(min_dim, min(max_dim, w))
        if (1.0 / max_ratio) <= (h / w) <= max_ratio:
            buckets.add((h, w))

    if not buckets:
        buckets.add((base_size, base_size))
    return sorted(list(buckets))


def resize_and_crop_to_bucket(
    img: Image.Image, target_h: int, target_w: int
) -> Image.Image:
    """画像を歪ませずに target_h x target_w を満たす最小サイズへ拡大・縮小し、中央クロップする"""
    orig_w, orig_h = img.size
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = max(target_w, int(round(orig_w * scale)))
    new_h = max(target_h, int(round(orig_h * scale)))
    img = TF.resize(
        img, [new_h, new_w], interpolation=transforms.InterpolationMode.BICUBIC
    )
    img = TF.center_crop(img, [target_h, target_w])
    return img


class VisualDataset(Dataset):
    def __init__(
        self,
        subset,
        datasets_root="datasets",
        base_size=128,
        fixed_res=None,
        multi_aspect=True,
    ):
        """
        fixed_res: (H, W) タプルが指定された場合は全画像をその解像度に固定。
        multi_aspect: True の場合、各画像の元アスペクト比に最も近いバケット解像度を割り当てる。
        """
        self.data_dir = os.path.join(datasets_root, subset)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"データセットディレクトリが存在しません: {self.data_dir}"
            )

        if fixed_res is not None:
            self.buckets = [fixed_res]
        elif multi_aspect:
            self.buckets = generate_aspect_buckets(base_size=base_size, step=32)
        else:
            self.buckets = [(base_size, base_size)]

        self.to_tensor_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.samples = []
        self.bucket_to_indices = defaultdict(list)

        for file in sorted(os.listdir(self.data_dir)):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                base_name = os.path.splitext(file)[0]
                img_path = os.path.join(self.data_dir, file)
                text_filename = f"{base_name}.pt"
                text_path = os.path.join(self.data_dir, text_filename)

                if not os.path.exists(text_path) and "image" in base_name:
                    legacy_path = os.path.join(
                        self.data_dir, base_name.replace("image", "caption") + ".pt"
                    )
                    if os.path.exists(legacy_path):
                        text_path = legacy_path

                if os.path.exists(text_path):
                    with Image.open(img_path) as im:
                        orig_w, orig_h = im.size
                    orig_aspect = orig_h / orig_w

                    # 最もアスペクト比が近いバケットを選択
                    best_bucket = min(
                        self.buckets, key=lambda b: abs((b[0] / b[1]) - orig_aspect)
                    )
                    idx = len(self.samples)
                    self.samples.append((img_path, text_path, best_bucket))
                    self.bucket_to_indices[best_bucket].append(idx)
                else:
                    print(
                        f"Warning: 対応するテンソルが見つかりません ({file} -> {text_filename})"
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text_path, target_bucket = self.samples[idx]
        target_h, target_w = target_bucket

        image = Image.open(img_path).convert("RGB")
        image = resize_and_crop_to_bucket(image, target_h, target_w)
        image_tensor = self.to_tensor_norm(image)

        context_tensor = torch.load(text_path, weights_only=True)
        if context_tensor.dim() == 3 and context_tensor.size(0) == 1:
            context_tensor = context_tensor.squeeze(0)

        return image_tensor, context_tensor


class AspectRatioBatchSampler(Sampler):
    """同一のアスペクト比バケットに属する画像同士でミニバッチを構築するサンプラー"""

    def __init__(self, bucket_to_indices, batch_size, shuffle=True):
        self.bucket_to_indices = bucket_to_indices
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batches = []
        for bucket, indices in self.bucket_to_indices.items():
            for i in range(0, len(indices), self.batch_size):
                self.batches.append(indices[i : i + self.batch_size])

    def __iter__(self):
        batches = []
        for bucket, indices in self.bucket_to_indices.items():
            idx_list = list(indices)
            if self.shuffle:
                random.shuffle(idx_list)
            for i in range(0, len(idx_list), self.batch_size):
                batches.append(idx_list[i : i + self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.batches)


# 既存コードとの互換エイリアス
PoCDataset = VisualDataset
