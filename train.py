import os
import argparse
from typing import Dict, Callable, Optional
import json
import csv
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from FasterNet.models.fasternet import Partial_conv3


# ======================
# 数据集加载
# ======================
def load_ids(train_file, test_file, blacklist_file):
    with open(train_file, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    with open(test_file, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]
    with open(blacklist_file, 'r') as f:
        blacklist = set(line.strip() for line in f.readlines())

    train_ids = [i for i in train_ids if i not in blacklist]
    test_ids = [i for i in test_ids if i not in blacklist]
    return train_ids, test_ids


class GasDataset(Dataset):
    def __init__(self, root, ids, transform=None):
        self.root = root
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img = Image.open(f"{self.root}/T/{img_id}.png").convert("L")
        label = Image.open(f"{self.root}/visual_labels/{img_id}.png").convert("L")
        label = np.array(label)
        label = (label > 0).astype(np.float32)  # 白色为1，黑色为0
        label = torch.from_numpy(label).unsqueeze(0)  # [1, H, W]

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label


# ======================
# UNet 模型
# ======================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SlimConvAttention(nn.Module):
    """Channel re-weighting block inspired by SlimConv."""

    def __init__(self, channels: int, stride: int = 1, groups: int = 1, dilation: int = 1):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("SlimConvAttention expects channels divisible by 4")

        reduce_1 = 2
        reduce_2 = 4

        self.conv_local = nn.Sequential(
            nn.Conv2d(channels // reduce_1, channels // reduce_1, kernel_size=3,
                      stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation),
            nn.BatchNorm2d(channels // reduce_1)
        )

        self.conv_enhance = nn.Sequential(
            nn.Conv2d(channels // reduce_1, channels // reduce_2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduce_2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduce_2, channels // reduce_2, kernel_size=3, stride=stride,
                      groups=groups, padding=dilation, bias=False, dilation=dilation),
            nn.BatchNorm2d(channels // reduce_2)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 32, channels, kernel_size=1),
            nn.Sigmoid()
        )

        mixed_channels = channels // reduce_1 + channels // reduce_2
        self.proj = nn.Sequential(
            nn.Conv2d(mixed_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        w_flip = torch.flip(w, dims=[1])

        gated_a = w * x
        gated_b = w_flip * x

        split_a = torch.split(gated_a, gated_a.size(1) // 2, dim=1)
        split_b = torch.split(gated_b, gated_b.size(1) // 2, dim=1)

        local = self.conv_local(split_a[0] + split_a[1])
        enhance = self.conv_enhance(split_b[0] + split_b[1])

        out = torch.cat([local, enhance], dim=1)
        out = self.proj(out)
        return out


class SlimFasterMix(nn.Module):
    """Fusion block combining FasterNet partial conv and SlimConv attention."""

    def __init__(self, channels: int, n_div: int = 4):
        super().__init__()
        self.partial = Partial_conv3(dim=channels, n_div=n_div, forward='split_cat')
        self.slim = SlimConvAttention(channels)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.partial(x)
        x = self.slim(x)
        return self.relu(self.bn(x))


class SlimFasterDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_div: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SlimFasterMix(out_ch, n_div=n_div),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SlimFasterUNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_filters=None, n_div: int = 4):
        super().__init__()
        if base_filters is None:
            base_filters = (64, 128, 256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.enc1 = SlimFasterDoubleConv(in_ch, base_filters[0], n_div=n_div)
        self.enc2 = SlimFasterDoubleConv(base_filters[0], base_filters[1], n_div=n_div)
        self.enc3 = SlimFasterDoubleConv(base_filters[1], base_filters[2], n_div=n_div)
        self.enc4 = SlimFasterDoubleConv(base_filters[2], base_filters[3], n_div=n_div)

        self.skip1 = SlimFasterMix(base_filters[0], n_div=n_div)
        self.skip2 = SlimFasterMix(base_filters[1], n_div=n_div)
        self.skip3 = SlimFasterMix(base_filters[2], n_div=n_div)

        self.bottleneck = SlimFasterMix(base_filters[3], n_div=n_div)

        self.dec3 = SlimFasterDoubleConv(base_filters[2] + base_filters[3], base_filters[2], n_div=n_div)
        self.dec2 = SlimFasterDoubleConv(base_filters[1] + base_filters[2], base_filters[1], n_div=n_div)
        self.dec1 = SlimFasterDoubleConv(base_filters[0] + base_filters[1], base_filters[0], n_div=n_div)

        self.conv_last = nn.Conv2d(base_filters[0], out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc1_mix = self.skip1(enc1)
        x = self.maxpool(enc1_mix)

        enc2 = self.enc2(x)
        enc2_mix = self.skip2(enc2)
        x = self.maxpool(enc2_mix)

        enc3 = self.enc3(x)
        enc3_mix = self.skip3(enc3)
        x = self.maxpool(enc3_mix)

        x = self.enc4(x)
        x = self.bottleneck(x)

        x = self.upsample(x)
        x = torch.cat([x, enc3_mix], dim=1)
        x = self.dec3(x)

        x = self.upsample(x)
        x = torch.cat([x, enc2_mix], dim=1)
        x = self.dec2(x)

        x = self.upsample(x)
        x = torch.cat([x, enc1_mix], dim=1)
        x = self.dec1(x)

        out = self.conv_last(x)
        return out


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        #out = torch.sigmoid(out)
        return out


MODEL_FACTORY: Dict[str, Callable[..., nn.Module]] = {
    "unet": UNet,
    "slimfaster_unet": SlimFasterUNet,
}

def dice_coefficient(logits, target, smooth=1e-5):
    """根据 logits 计算 Dice 系数"""
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# ======================
# 损失函数: BCE + Dice
# ======================
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        probs = torch.sigmoid(pred)
        pred_flat = probs.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return bce_loss + dice_loss


# ======================
# 训练 & 验证
# ======================
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda', save_dir: Optional[Path] = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss()
    model.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Train"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Val"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        json_path = save_dir / "loss_history.json"
        csv_path = save_dir / "loss_history.csv"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            for epoch, train_l, val_l in zip(history["epoch"], history["train_loss"], history["val_loss"]):
                writer.writerow([epoch, train_l, val_l])
def train_model_1(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda', save_dir: Optional[Path] = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss()
    model.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice": []
    }

    best_dice = 0.0
    best_model_path = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Train"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
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
                loss = criterion(outputs, masks)
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
    parser.add_argument('--blacklist-file', default='blacklist.txt', help='Relative path of blacklist file')
    parser.add_argument('--resize', nargs=2, type=int, default=[512, 640], metavar=('H', 'W'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', choices=list(MODEL_FACTORY.keys()), default='slimfaster_unet')
    parser.add_argument('--n-div', type=int, default=4, help='Division factor for partial conv in SlimFaster modules')
    parser.add_argument('--save-dir', default='runs', help='Directory to store training logs and curves')
    return parser.parse_args()


def build_model(name: str, in_ch: int, out_ch: int, n_div: int) -> nn.Module:
    if name == 'slimfaster_unet':
        return MODEL_FACTORY[name](in_ch=in_ch, out_ch=out_ch, n_div=n_div)
    return MODEL_FACTORY[name](in_ch=in_ch, out_ch=out_ch)


def main():
    args = parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDA not available, fall back to CPU')

    train_ids, test_ids = load_ids(os.path.join(args.root, args.train_file),
                                   os.path.join(args.root, args.test_file),
                                   os.path.join(args.root, args.blacklist_file))

    transform = transforms.Compose([
        transforms.Resize(tuple(args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = GasDataset(args.root, train_ids, transform=transform)
    val_dataset = GasDataset(args.root, test_ids, transform=transform)

    pin_memory = device.startswith('cuda')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)

    model = build_model(args.model, in_ch=1, out_ch=1, n_div=args.n_div)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Using model: {args.model} | Params: {total_params / 1e6:.2f}M | Device: {device}")

    run_dir = Path(args.save_dir).expanduser() / f"{args.model}model"
    print(f"Saving metrics to: {run_dir}")

    train_model_1(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                device=device, save_dir=run_dir)


if __name__ == "__main__":
    main()
