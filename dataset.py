import os
import math
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


def generate_aspect_buckets(
    base_size=256, step=32, max_ratio=2.0, min_short_edge=256
):
    """
    短辺を min_short_edge (デフォルト256px) 以上に保証しつつ、
    step(32)の倍数となる (H, W) アスペクト比バケット一覧を自動生成する。
    """
    short_edge = max(step, ((max(base_size, min_short_edge) + step - 1) // step) * step)
    max_long_edge = int(round(short_edge * max_ratio / step)) * step

    buckets = set()
    for long_edge in range(short_edge, max_long_edge + step, step):
        if (long_edge / short_edge) <= max_ratio:
            buckets.add((short_edge, long_edge))
            buckets.add((long_edge, short_edge))

    if not buckets:
        buckets.add((short_edge, short_edge))
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
        base_size=256,
        min_short_edge=256,
        fixed_res=None,
        multi_aspect=True,
    ):
        """
        base_size / min_short_edge: 短辺の最小基準解像度 (デフォルト: 256px)。
        fixed_res: (H, W) タプルが指定された場合は全画像をその解像度に固定。
        multi_aspect: True の場合、短辺 >= min_short_edge を満たす最も近いアスペクト比バケットを割り当てる。
        """
        self.data_dir = os.path.join(datasets_root, subset)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"データセットディレクトリが存在しません: {self.data_dir}"
            )

        effective_short = max(base_size, min_short_edge)

        if fixed_res is not None:
            self.buckets = [fixed_res]
        elif multi_aspect:
            self.buckets = generate_aspect_buckets(
                base_size=effective_short,
                step=32,
                max_ratio=2.0,
                min_short_edge=effective_short,
            )
        else:
            self.buckets = [(effective_short, effective_short)]

        self.to_tensor_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.samples = []
        self.bucket_to_indices = defaultdict(list)
        upscaled_count = 0

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
                    if min(orig_w, orig_h) < effective_short:
                        upscaled_count += 1

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

        if upscaled_count > 0:
            print(
                f"Info: 短辺が {effective_short}px 未満の画像が {upscaled_count} 枚含まれています（BICUBIC拡大を適用します）。"
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