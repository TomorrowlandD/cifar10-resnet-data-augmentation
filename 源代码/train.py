import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from utils import (
    build_model,
    build_optimizer,
    build_scheduler,
    count_parameters,
    create_data_loaders,
    ensure_experiment_dirs,
    get_device,
    get_experiment_id,
    get_experiment_name,
    get_output_dir,
    load_config,
    set_seed,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_count += targets.size(0)
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_count += targets.size(0)
    return total_loss / total_count, total_correct / total_count


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.data_dir is not None:
        config["data"]["data_dir"] = args.data_dir

    set_seed(int(config["training"].get("seed", 42)))
    output_dir = get_output_dir(config)
    ensure_experiment_dirs(output_dir)

    exp_id = get_experiment_id(config)
    exp_name = get_experiment_name(config)
    device = get_device()
    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")

    train_loader, val_loader, _ = create_data_loaders(config)
    print(
        f"Dataset split: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}"
    )

    model = build_model(config).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    log_path = output_dir / "logs" / f"{exp_id}_train_log.csv"
    checkpoint_path = output_dir / "checkpoints" / f"{exp_name}_best.pth"
    epochs = int(config["training"]["epochs"])
    best_val_acc = 0.0

    with open(log_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "lr",
            ],
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                    "lr": f"{current_lr:.8f}",
                }
            )
            file.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "best_val_acc": best_val_acc,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                    },
                    checkpoint_path,
                )

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"best_val_acc={best_val_acc:.4f}"
            )

    print(f"Training log saved to: {Path(log_path)}")
    print(f"Best checkpoint saved to: {Path(checkpoint_path)}")


if __name__ == "__main__":
    main()
