import os
import sys
import argparse
import json
import csv
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models import build_model, MODEL_FACTORY
from dataset import load_ids, GasDataset
from losses import DiceFocalLoss, dice_coefficient


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda', save_dir: Optional[Path] = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceFocalLoss()
    model.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice": []
    }

    best_dice = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Train"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion.forward(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Val"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion.forward(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                val_dice += dice_coefficient(outputs, masks).item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        # 保存最佳模型
        if save_dir is not None and val_dice > best_dice:
            save_dir.mkdir(parents=True, exist_ok=True)
            best_dice = val_dice
            best_model_path = save_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated: Dice={best_dice:.4f} -> {best_model_path}")

    # 保存训练历史
    if save_dir is not None:
        json_path = save_dir / "loss_history.json"
        csv_path = save_dir / "loss_history.csv"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_dice"])
            for epoch, train_l, val_l, dice_l in zip(history["epoch"], history["train_loss"], history["val_loss"], history["val_dice"]):
                writer.writerow([epoch, train_l, val_l, dice_l])

    #return best_model_path


# ======================
# 主函数
# ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Infrared gas segmentation training")
    parser.add_argument('--root', default='Gas_DB', help='Root directory of dataset')
    parser.add_argument('--train-file', default='train.txt', help='Relative path of train IDs file')
    parser.add_argument('--test-file', default='test.txt', help='Relative path of test IDs file')
    parser.add_argument('--resize', nargs=2, type=int, default=[512, 640], metavar=('H', 'W'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', choices=list(MODEL_FACTORY.keys()), default='unet')
    parser.add_argument('--save-dir', default='checkpoints', help='Directory to store training logs and curves')
    return parser.parse_args()

def main():
    args = parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDA not available, fall back to CPU')

    train_ids, test_ids = load_ids(os.path.join(args.root, args.train_file),
                                   os.path.join(args.root, args.test_file))

    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize(tuple(args.resize)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 验证集基础预处理（不使用数据增强）
    val_transform = transforms.Compose([
        transforms.Resize(tuple(args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = GasDataset(args.root, train_ids, transform=train_transform)
    val_dataset = GasDataset(args.root, test_ids, transform=val_transform)

    pin_memory = device.startswith('cuda')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)

    model = build_model(args.model, in_ch=1, out_ch=1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Using model: {args.model} | Params: {total_params / 1e6:.2f}M | Device: {device}")

    run_dir = Path(args.save_dir).expanduser() / f"{args.model}model"
    print(f"Saving metrics to: {run_dir}")

    train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                device=device, save_dir=run_dir)

if __name__ == "__main__":
    main()
