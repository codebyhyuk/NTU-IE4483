import os
import glob
from typing import Dict, Tuple, List

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES: List[str] = ["cat", "dog"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}


class SimpleImageDataset(Dataset):
    """
    Generic labeled dataset for:
        root/
          cat/
            cat.0.jpg
            ...
          dog/
            dog.0.jpg
            ...
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            paths = glob.glob(os.path.join(cls_dir, "*.jpg"))
            label = CLASS_TO_IDX[cls_name]
            for p in paths:
                self.samples.append((p, label))

        # sort by path for reproducibility
        self.samples.sort(key=lambda x: x[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class TestImageDataset(Dataset):
    """
    Unlabeled test dataset:
        root/
          0.jpg
          1.jpg
          ...
    Returns (image, filename).
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        self.paths = sorted(glob.glob(os.path.join(root, "*.jpg")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, os.path.basename(path)


def get_dataset(
    root_dir: str,
    image_size: int = 224,
) -> Dict[str, Dataset]:
    """
    root_dir 구조:
        root_dir/
          train/
            cat/
            dog/
          val/
            cat/
            dog/
          test/
            0.jpg, 1.jpg, ...

    Returns: dict with keys 'train', 'val', 'test'
    """

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    train_dataset = SimpleImageDataset(train_dir, transform=train_transform)
    val_dataset = SimpleImageDataset(val_dir, transform=eval_transform)
    test_dataset = TestImageDataset(test_dir, transform=eval_transform)

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
