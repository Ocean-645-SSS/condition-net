"""
数据集加载器：处理 (I⁻, I⁺, a, audio⁻, audio⁺) 元组 + 数据增强

CSV 格式 (每行一个 demonstration):
    pre_image_path, post_image_path, pre_audio_path, post_audio_path, action_text

兼容旧格式 (3 列无音频):
    pre_image_path, post_image_path, action_text

可选的改写文件 (JSON):
    {"action_text": ["variant1", "variant2", ...], ...}
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


class ConditionDataset(Dataset):
    """ConditionNET 分类数据集

    每条样本: (image, [audio], action_text) → label
        label=0: precondition   (I⁻ 配对 a)
        label=1: effect         (I⁺ 配对 a)
        label=2: unsatisfied    (跨 demo 配对)
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        paraphrase_file: str | None = None,
        augment_cross_pair: bool = True,
        cross_pair_ratio: float = 1.0,
    ):
        self.img_dir = Path(img_dir)
        self.paraphrases: dict[str, list[str]] = {}
        self.has_audio = False

        # 解析 demo
        self.demos: list[dict] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            has_audio_cols = "pre_audio" in header or "audio" in header.lower()
            self.has_audio = has_audio_cols

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    # 新版 CSV: pre_img, post_img, pre_audio, post_audio, action
                    self.demos.append({
                        "pre": parts[0], "post": parts[1],
                        "audio_pre": parts[2], "audio_post": parts[3],
                        "action": parts[4],
                    })
                    self.has_audio = True
                elif len(parts) >= 3:
                    # 旧版 CSV: pre_img, post_img, action
                    self.demos.append({
                        "pre": parts[0], "post": parts[1],
                        "audio_pre": None, "audio_post": None,
                        "action": parts[2],
                    })

        if paraphrase_file:
            with open(paraphrase_file, "r", encoding="utf-8") as f:
                self.paraphrases = json.load(f)

        n = len(self.demos)

        # 构建分类样本索引
        self.samples: list[dict] = []
        for i in range(n):
            self.samples.append({
                "img": self.demos[i]["pre"],
                "audio": self.demos[i].get("audio_pre"),
                "text": self.demos[i]["action"],
                "label": 0,
                "demo_id": i,
                "is_pre": True,
            })
            self.samples.append({
                "img": self.demos[i]["post"],
                "audio": self.demos[i].get("audio_post"),
                "text": self.demos[i]["action"],
                "label": 1,
                "demo_id": i,
                "is_pre": False,
            })

        if augment_cross_pair and n >= 2:
            num_cross = int(n * cross_pair_ratio)
            for _ in range(num_cross):
                i, j = random.sample(range(n), 2)
                self.samples.append({
                    "img": self.demos[i]["pre"],
                    "audio": self.demos[i].get("audio_pre"),
                    "text": self.demos[j]["action"],
                    "label": 2, "demo_id": -1, "is_pre": True,
                })
                self.samples.append({
                    "img": self.demos[i]["post"],
                    "audio": self.demos[i].get("audio_post"),
                    "text": self.demos[j]["action"],
                    "label": 2, "demo_id": -1, "is_pre": False,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(self.img_dir / s["img"]).convert("RGB")

        variants = self.paraphrases.get(s["text"], [s["text"]])
        paraphrase = random.choice(variants) if variants else s["text"]

        result = {
            "image": image,
            "action_text": s["text"],
            "paraphrase": paraphrase,
            "label": s["label"],
            "demo_id": s["demo_id"],
            "is_pre": s["is_pre"],
        }

        # 音频路径 (可能为 None)
        if s.get("audio"):
            result["audio_path"] = str(self.img_dir / s["audio"])
        elif self.has_audio:
            result["audio_path"] = None
        # 旧格式不添加此字段

        return result

    def get_demo_pairs(self):
        """返回所有 demo 的配对信息，用于 consistency loss（路径为绝对路径）"""
        pairs = []
        for i, demo in enumerate(self.demos):
            variants = self.paraphrases.get(demo["action"], [demo["action"]])
            pair = {
                "img_pre": str(self.img_dir / demo["pre"]),
                "img_post": str(self.img_dir / demo["post"]),
                "action": demo["action"],
                "paraphrase": random.choice(variants),
            }
            if demo.get("audio_pre"):
                pair["audio_pre"] = str(self.img_dir / demo["audio_pre"])
                pair["audio_post"] = str(self.img_dir / demo["audio_post"])
            pairs.append(pair)
        return pairs


def classification_collate(batch):
    """整理分类 batch"""
    result = {
        "images": [item["image"] for item in batch],
        "action_texts": [item["action_text"] for item in batch],
        "paraphrases": [item["paraphrase"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "demo_ids": torch.tensor([item["demo_id"] for item in batch], dtype=torch.long),
        "is_pre": torch.tensor([item["is_pre"] for item in batch], dtype=torch.bool),
    }
    if "audio_path" in batch[0]:
        result["audio_paths"] = [item.get("audio_path") for item in batch]
    return result


def consistency_collate(batch):
    """整理 consistency batch: (I⁻, I⁺, paraphrase)"""
    result = {
        "img_pre": [item["img_pre"] for item in batch],
        "img_post": [item["img_post"] for item in batch],
        "paraphrases": [item["paraphrase"] for item in batch],
    }
    if "audio_pre" in batch[0]:
        result["audio_pre_paths"] = [item.get("audio_pre") for item in batch]
        result["audio_post_paths"] = [item.get("audio_post") for item in batch]
    return result


def build_loaders(
    csv_path: str,
    img_dir: str,
    paraphrase_file: str | None = None,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    num_workers: int = 0,
    consistency_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    构建 train / val 分类 DataLoader，以及 train consistency DataLoader.
    返回: (train_loader, val_loader, consistency_loader)
    """
    full = ConditionDataset(csv_path, img_dir, paraphrase_file, augment_cross_pair=True)

    n_train = int(len(full) * train_ratio)
    n_val = len(full) - n_train
    train_ds, val_ds = random_split(full, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=classification_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=classification_collate,
    )

    consistency_loader = None
    if consistency_batch_size is None:
        consistency_batch_size = batch_size
    train_indices = train_ds.indices
    train_demo_ids = set()
    for idx in train_indices:
        demo_id = full.samples[idx]["demo_id"]
        if demo_id >= 0:
            train_demo_ids.add(demo_id)

    if len(train_demo_ids) >= 2:
        consistency_pairs = [
            full.get_demo_pairs()[i] for i in sorted(train_demo_ids)
        ]
        consistency_loader = DataLoader(
            consistency_pairs, batch_size=consistency_batch_size,
            shuffle=True, num_workers=num_workers,
            collate_fn=consistency_collate,
        )

    return train_loader, val_loader, consistency_loader


def build_loaders_from_splits(
    train_csv: str,
    val_csv: str,
    img_dir: str,
    paraphrase_file: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    consistency_batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    使用预定义的 train/val CSV 构建 DataLoader (ImperfectPour 格式).
    train_csv / val_csv 是单独的 CSV 文件。
    """
    train_ds = ConditionDataset(train_csv, img_dir, paraphrase_file, augment_cross_pair=True)
    val_ds = ConditionDataset(val_csv, img_dir, paraphrase_file, augment_cross_pair=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=classification_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=classification_collate,
    )

    consistency_loader = None
    if consistency_batch_size is None:
        consistency_batch_size = batch_size

    demo_pairs = train_ds.get_demo_pairs()
    if len(demo_pairs) >= 2:
        consistency_loader = DataLoader(
            demo_pairs, batch_size=consistency_batch_size,
            shuffle=True, num_workers=num_workers,
            collate_fn=consistency_collate,
        )

    return train_loader, val_loader, consistency_loader
