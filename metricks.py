import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import numpy as np
from PIL import Image
import glob
import random
from model import LiverHemangiomaDataset, UNetMulti

def evaluate_model(model, loader, device):
    model.eval()
    preds_all = []
    masks_all = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = torch.sigmoid(model(imgs))
            preds = (outputs > 0.5).float()

            preds_all.append(preds.cpu())
            masks_all.append(masks.cpu())

    preds_all = torch.cat(preds_all, dim=0).numpy()
    masks_all = torch.cat(masks_all, dim=0).numpy()

    metrics = {}

    for i, label in enumerate(["Liver", "Hemangioma"]):
        preds_flat = preds_all[:, i].reshape(-1).astype(int)
        masks_flat = masks_all[:, i].reshape(-1).astype(int)

        precision = precision_score(masks_flat, preds_flat, zero_division=0)
        recall = recall_score(masks_flat, preds_flat, zero_division=0)

        metrics[label] = {
            "IoU": jaccard_score(masks_flat, preds_flat, zero_division=0),
            "Dice": (2 * precision * recall) / (precision + recall + 1e-7),
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score(masks_flat, preds_flat, zero_division=0)
        }

    return metrics

if __name__ == '__main__':
    dataset_dir = 'OutputNew'
    model_path = 'unet_best.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    random.seed(42)
    total = len(dataset)
    indices = list(range(total))
    random.shuffle(indices)
    split = int(0.8 * total)
    _, val_idx = indices[:split], indices[split:]
    val_dataset = Subset(dataset, val_idx)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = UNetMulti(n_channels=3, n_classes=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    metrics = evaluate_model(model, val_loader, device)

    print("=" * 40)
    for label, scores in metrics.items():
        print(f"Class: {label}")
        for metric_name, value in scores.items():
            print(f"  {metric_name}: {value:.4f}")
        print()
    print("=" * 40)