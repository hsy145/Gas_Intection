import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import GasDataset, load_ids  # noqa: E402
from losses import DiceBCELoss, dice_coefficient  # noqa: E402
from models import UNet, UNetGCM  # noqa: E402

EXPERIMENT_CHOICES = [
    "unet_baseline",
    "unet_gcm_full",
    "unet_gcm_no_gcm",
    "unet_gcm_no_skip",
]


def build_experiment_catalog(gcm_reduction: int) -> Dict[str, Dict[str, Callable[[int, int], torch.nn.Module]]]:
    """Return experiment builders keyed by human-readable names."""

    return {
        "unet_baseline": {
            "description": "标准 UNet 作为对照组",
            "builder": lambda in_ch, out_ch: UNet(in_ch=in_ch, out_ch=out_ch),
        },
        "unet_gcm_full": {
            "description": "启用 GCM + Skip Attention 的完整模型",
            "builder": lambda in_ch, out_ch: UNetGCM(
                in_ch=in_ch,
                out_ch=out_ch,
                use_gcm=True,
                use_skip_attention=True,
                gcm_reduction=gcm_reduction,
            ),
        },
        "unet_gcm_no_gcm": {
            "description": "去掉 GCM 仅保留 Skip Attention",
            "builder": lambda in_ch, out_ch: UNetGCM(
                in_ch=in_ch,
                out_ch=out_ch,
                use_gcm=False,
                use_skip_attention=True,
                gcm_reduction=gcm_reduction,
            ),
        },
        "unet_gcm_no_skip": {
            "description": "去掉 Skip Attention 仅保留 GCM",
            "builder": lambda in_ch, out_ch: UNetGCM(
                in_ch=in_ch,
                out_ch=out_ch,
                use_gcm=True,
                use_skip_attention=False,
                gcm_reduction=gcm_reduction,
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UNet-GCM ablation experiments")
    parser.add_argument("--root", default="Gas_DB", help="Dataset root directory")
    parser.add_argument("--train-file", default="train_gcm.txt", help="Relative path of train IDs file")
    parser.add_argument("--test-file", default="test.txt", help="Relative path of test IDs file")
    parser.add_argument("--resize", nargs=2, type=int, default=[512, 640], metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--save-dir", default="checkpoints", help="Directory to store experiment logs")
    parser.add_argument("--run-name", default="unet_gcm_ablation", help="Subdirectory name under save-dir")
    parser.add_argument("--experiments", nargs="+", default=["all"],
                        choices=EXPERIMENT_CHOICES + ["all"],
                        help="Which experiment variants to run")
    parser.add_argument("--in-channels", type=int, default=1, help="Input channel count")
    parser.add_argument("--out-channels", type=int, default=1, help="Output channel count")
    parser.add_argument("--gcm-reduction", type=int, default=16,
                        help="Reduction ratio used inside GCM channel attention")
    return parser.parse_args()


def prepare_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    train_ids, test_ids = load_ids(
        os.path.join(args.root, args.train_file),
        os.path.join(args.root, args.test_file)
    )

    transform = transforms.Compose([
        transforms.Resize(tuple(args.resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = GasDataset(args.root, train_ids, transform=transform)
    val_dataset = GasDataset(args.root, test_ids, transform=transform)

    pin_memory = args.device == "cuda" and torch.cuda.is_available()

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders


def train_single_experiment(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: str,
    save_dir: Path,
) -> Dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss()
    model.to(device)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = float("-inf")
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_path = save_dir / "best_model.pth"

    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(
            train_loader,
            desc=f"[{name}] Epoch {epoch}/{epochs} - Train",
            leave=False,
        ):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(
                val_loader,
                desc=f"[{name}] Epoch {epoch}/{epochs} - Val",
                leave=False,
            ):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                val_dice += dice_coefficient(outputs, masks).item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        improved = val_dice > best_dice or (val_dice == best_dice and val_loss < best_val_loss)
        if improved:
            best_dice = val_dice
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        print(
            f"[{name}] Epoch {epoch}: TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | ValDice={val_dice:.4f}"
        )

    # Persist history per experiment
    with (save_dir / "loss_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with (save_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice"])
        for row in zip(history["epoch"], history["train_loss"], history["val_loss"], history["val_dice"]):
            writer.writerow(row)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "best_epoch": best_epoch,
        "best_dice": best_dice,
        "best_val_loss": best_val_loss,
        "best_model_path": str(best_model_path),
        "params_million": total_params / 1e6,
        "history_dir": str(save_dir),
    }


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    experiment_catalog = build_experiment_catalog(args.gcm_reduction)

    selected_experiments = args.experiments
    if "all" in selected_experiments:
        selected_experiments = list(experiment_catalog.keys())

    loaders = prepare_dataloaders(args)
    base_save_dir = Path(args.save_dir).expanduser() / args.run_name
    base_save_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, object]] = []

    for exp_name in selected_experiments:
        spec = experiment_catalog[exp_name]
        print(f"\n========== Running {exp_name}: {spec['description']} ==========")
        model = spec["builder"](args.in_channels, args.out_channels)
        exp_dir = base_save_dir / exp_name
        result = train_single_experiment(
            exp_name,
            model,
            loaders["train"],
            loaders["val"],
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            save_dir=exp_dir,
        )
        result.update(
            {
                "experiment": exp_name,
                "description": spec["description"],
            }
        )
        summary.append(result)

    summary_json = base_save_dir / "summary.json"
    summary_csv = base_save_dir / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "description",
                "best_epoch",
                "best_dice",
                "best_val_loss",
                "params_million",
                "best_model_path",
            ]
        )
        for row in summary:
            writer.writerow(
                [
                    row["experiment"],
                    row["description"],
                    row["best_epoch"],
                    f"{row['best_dice']:.4f}",
                    f"{row['best_val_loss']:.4f}",
                    f"{row['params_million']:.3f}",
                    row["best_model_path"],
                ]
            )

    print("\n===== Ablation summary =====")
    for row in summary:
        print(
            f"{row['experiment']}: Dice={row['best_dice']:.4f} | "
            f"ValLoss={row['best_val_loss']:.4f} | Params={row['params_million']:.3f}M | "
            f"Checkpoint={row['best_model_path']}"
        )


if __name__ == "__main__":
    main()
