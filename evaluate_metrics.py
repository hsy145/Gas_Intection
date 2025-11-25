import os
import sys
import argparse
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models import build_model, MODEL_FACTORY
from dataset import load_ids, GasDataset
from utils import print_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UNet-based gas segmentation models")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to model checkpoint. Defaults to checkpoints/<model>model/best_model.pth")
    parser.add_argument("--model", choices=list(MODEL_FACTORY.keys()), default="unet_gcm",
                        help="Model architecture to evaluate")
    parser.add_argument("--data-root", type=Path, default=Path("Gas_DB"), help="Dataset root directory")
    parser.add_argument("--train-file", type=str, default="train.txt", help="Train file relative to data root")
    parser.add_argument("--test-file", type=str, default="test.txt", help="Test file relative to data root")
    parser.add_argument("--resize", type=int, nargs=2, default=(512, 640), metavar=("H", "W"),
                        help="Resize (height width) applied to inputs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Preferred compute device")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binarization")
    return parser.parse_args()


def build_dataloader(args) -> DataLoader:
    data_root = args.data_root
    train_file = data_root / args.train_file
    test_file = data_root / args.test_file

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    _, test_ids = load_ids(
        train_file=str(train_file),
        test_file=str(test_file)
    )

    if not test_ids:
        raise ValueError("No samples found in the test split after applying blacklist filtering.")

    transform = transforms.Compose([
        transforms.Resize(tuple(args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = GasDataset(str(data_root), test_ids, transform=transform)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )


def evaluate_metrics(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, threshold: float):
    model.eval()
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).bool()
            masks_bool = (masks > 0.5).bool()

            tp = torch.logical_and(preds, masks_bool).sum().item()
            tn = torch.logical_and(~preds, ~masks_bool).sum().item()
            fp = torch.logical_and(preds, ~masks_bool).sum().item()
            fn = torch.logical_and(~preds, masks_bool).sum().item()

            true_positive += tp
            true_negative += tn
            false_positive += fp
            false_negative += fn

    total = true_positive + true_negative + false_positive + false_negative
    if total == 0:
        raise ValueError("No pixels were evaluated; check dataset and preprocessing.")

    accuracy = (true_positive + true_negative) / total
    recall_den = true_positive + false_negative
    recall = true_positive / recall_den if recall_den > 0 else 0.0
    union = true_positive + false_positive + false_negative
    iou = true_positive / union if union > 0 else 1.0
    dice_den = 2 * true_positive + false_positive + false_negative
    dice = (2 * true_positive / dice_den) if dice_den > 0 else 1.0

    return (
        accuracy,
        recall,
        iou,
        dice
    )


def main():
    args = parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device == "cuda":
        print("[Warning] CUDA requested but not available. Falling back to CPU.")

    checkpoint_path = args.checkpoint or (Path("checkpoints") / f"{args.model}model" / "best_model.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataloader = build_dataloader(args)

    model = build_model(args.model, in_ch=1, out_ch=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    acc, recall, iou, dice = evaluate_metrics(model, dataloader, device, args.threshold)

    print_results(args.model,acc,recall,iou,dice)


if __name__ == "__main__":
    main()
