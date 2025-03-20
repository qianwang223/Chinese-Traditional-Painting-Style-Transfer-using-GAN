import os
import random
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import STYLE_DIR, CONTENT_DIR

class ChinesePaintingsDataset(Dataset):
    def __init__(self, style_dir, content_dir, transform=None):
        super().__init__()
        # Collect all style images (no subfolders)
        self.style_paths = sorted(
            glob(os.path.join(style_dir, "*.*"))
        )
        # Collect all content images (recursively, including subfolders)
        self.content_paths = sorted(
            glob(os.path.join(content_dir, "**", "*.*"), recursive=True)
        )
        self.transform = transform

    def __len__(self):
        # Return the max so we can sample unpaired images from each domain
        return max(len(self.style_paths), len(self.content_paths))

    def __getitem__(self, index):
        # Random selection for unpaired data
        style_path = random.choice(self.style_paths)
        content_path = random.choice(self.content_paths)

        style_img = Image.open(style_path).convert("RGB")
        content_img = Image.open(content_path).convert("RGB")

        if self.transform:
            style_img = self.transform(style_img)
            content_img = self.transform(content_img)

        return style_img, content_img


def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
