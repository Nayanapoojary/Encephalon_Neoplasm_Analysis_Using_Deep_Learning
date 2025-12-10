#!/usr/bin/env python3
"""
Brain Tumor Segmentation Training Script
Compatible with resized 224x224 images and masks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import time
import json
from tqdm import tqdm


# ---------------- Dataset ---------------- #
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        # collect only image files that have corresponding masks
        exts = [".png", ".jpg", ".jpeg"]
        self.image_files = []
        for ext in exts:
            self.image_files.extend(self.images_dir.glob(f"*{ext}"))
            self.image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))

        self.pairs = []
        for img in self.image_files:
            mask_path = self.masks_dir / f"{img.stem}.png"
            if not mask_path.exists():
                mask_path = self.masks_dir / f"{img.stem}.jpg"
            if mask_path.exists():
                self.pairs.append((img, mask_path))

        print(f"Found {len(self.pairs)} pairs in {images_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image)
        mask = np.array(mask).astype(np.float32)
        mask = (mask > 127).astype(np.float32)  # binary mask

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask


# ---------------- Model ---------------- #
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        for f in features:
            self.downs.append(self._block(in_channels, f))
            in_channels = f

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.ups.append(self._block(f*2, f))

        self.bottleneck = self._block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = self.ups[i+1](torch.cat([skip, x], dim=1))

        return torch.sigmoid(self.final_conv(x))


# ---------------- Loss ---------------- #
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    inter = (pred * target).sum()
    return 1 - (2*inter + smooth) / (pred.sum() + target.sum() + smooth)


# ---------------- Training ---------------- #
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device), masks.to(device)
        masks = masks.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_one_epoch(model, loader, device):
    model.eval()
    total_loss, total_dice = 0, 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1)
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            pred = (outputs > 0.5).float()
            dice = 1 - dice_loss(pred, masks)
            total_loss += loss.item()
            total_dice += dice.item()
    return total_loss / len(loader), total_dice / len(loader)


# ---------------- Main ---------------- #
def main():
    data_root = Path(r"C:\Users\navee\Downloads\encephalon_neoplasm_major_project\data\preprocessed\segmentation_task")

    train_images = data_root / "train" / "images"
    train_masks = data_root / "train" / "masks"
    test_images = data_root / "test" / "images"
    test_masks = data_root / "test" / "masks"

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2(),
    ])

    train_ds = SegmentationDataset(train_images, train_masks, transform=train_tf)
    val_ds = SegmentationDataset(test_images, test_masks, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_dice = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(10):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice = validate_one_epoch(model, val_loader, device)
        dt = time.time() - t0

        print(f"Epoch {epoch+1}/10 - {dt:.1f}s | Train {train_loss:.4f} | Val {val_loss:.4f} | Dice {val_dice:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_segmentation.pth")
            print(f"  ✅ Saved new best model (Dice {best_dice:.4f})")

    with open("segmentation_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
