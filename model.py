import os
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything()

class LiverHemangiomaDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None):
        self.image_paths = []
        self.liver_masks = []
        self.hema_masks = []
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        for task_id in os.listdir(root_dir):
            task_path = os.path.join(root_dir, task_id)
            if not os.path.isdir(task_path):
                continue

            imgs = glob.glob(os.path.join(task_path, 'original_image.jpg'))
            liver = glob.glob(os.path.join(task_path, 'liver_mask.png'))
            hema = glob.glob(os.path.join(task_path, 'hemangioma_mask.png'))

            if len(imgs) != 1 or len(liver) != 1 or len(hema) != 1:
                continue

            self.image_paths.append(imgs[0])
            self.liver_masks.append(liver[0])
            self.hema_masks.append(hema[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        liver_mask = Image.open(self.liver_masks[idx]).convert('L')
        hema_mask = Image.open(self.hema_masks[idx]).convert('L')

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            liver_mask = self.mask_transform(liver_mask)
            hema_mask = self.mask_transform(hema_mask)

        # [1, H, W]
        liver_mask = (liver_mask > 0).float()
        hema_mask = (hema_mask > 0).float()

        # [2, H, W]
        mask = torch.cat([liver_mask, hema_mask], dim=0)

        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetMulti(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, dropout=0.3):
        super().__init__()

        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up3 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up1 = DoubleConv(128 + 64, 64)

        self.dropout = nn.Dropout2d(dropout)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool(c1)

        c2 = self.down2(p1)
        p2 = self.pool(c2)

        c3 = self.down3(p2)
        p3 = self.pool(c3)

        c4 = self.down4(p3)

        u3 = self.up(c4)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.up3(u3)

        u2 = self.up(u3)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.up2(u2)

        u1 = self.up(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.up1(u1)

        u1 = self.dropout(u1)
        return self.final(u1)

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        return self.bce(pred, target) + dice_loss(pred, target)

def compute_metrics(preds, masks):
    preds = preds.astype(np.float32)
    masks = masks.astype(np.float32)

    results = {}

    for i, name in enumerate(["Liver", "Hemangioma"]):
        p = preds[:, i].reshape(-1)
        m = masks[:, i].reshape(-1)

        tp = (p * m).sum()
        fp = (p * (1 - m)).sum()
        fn = ((1 - p) * m).sum()

        iou = tp / (tp + fp + fn + 1e-7)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        results[name] = {
            "IoU": iou,
            "Dice": dice,
            "Precision": precision,
            "Recall": recall,
        }

    return results

def train_model(dataset_dir, epochs=30, batch_size=8, lr=1e-3, device="cuda"):

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    dataset = LiverHemangiomaDataset(
        dataset_dir,
        image_transform=image_transform,
        mask_transform=mask_transform
    )

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False
    )

    model = UNetMulti().to(device)

    pos_weight = torch.tensor([1.0, 5.0], device=device).view(2, 1, 1)
    criterion = BCEDiceLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)

            assert preds.shape == masks.shape, f"Shape mismatch: preds={preds.shape}, masks={masks.shape}"

            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        preds_list, masks_list = [], []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                preds = model(imgs)

                assert preds.shape == masks.shape, f"Shape mismatch: preds={preds.shape}, masks={masks.shape}"

                loss = criterion(preds, masks)
                val_loss += loss.item()

                preds_bin = (torch.sigmoid(preds).cpu().numpy() > 0.5).astype(np.float32)
                masks_np = masks.cpu().numpy().astype(np.float32)

                preds_list.append(preds_bin)
                masks_list.append(masks_np)

        preds_list = np.concatenate(preds_list, axis=0)
        masks_list = np.concatenate(masks_list, axis=0)

        metrics = compute_metrics(preds_list, masks_list)

        val_loss = val_loss / max(len(val_loader), 1)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "unet_best.pth")
            print(f"Epoch {epoch + 1}: saved model (val loss: {best_loss:.4f})")

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss / max(len(train_loader), 1):.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Liver Dice: {metrics['Liver']['Dice']:.4f} | "
            f"Hema Dice: {metrics['Hemangioma']['Dice']:.4f}"
        )

if __name__ == "__main__":
    train_model("OutputNew")