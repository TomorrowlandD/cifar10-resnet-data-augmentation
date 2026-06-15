import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import resolve_path


DEFAULT_OUTPUT_DIR = "实验结果"
EXPERIMENTS = ["exp1", "exp2", "exp3"]


def read_log(log_path: Path):
    with open(log_path, "r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "epoch": [int(row["epoch"]) for row in rows],
        "train_loss": [float(row["train_loss"]) for row in rows],
        "val_loss": [float(row["val_loss"]) for row in rows],
        "train_acc": [float(row["train_acc"]) for row in rows],
        "val_acc": [float(row["val_acc"]) for row in rows],
    }


def plot_experiment(exp_id: str, log_path: Path, output_path: Path) -> None:
    history = read_log(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    axes[0].plot(history["epoch"], history["train_loss"], label="train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val loss")
    axes[0].set_title(f"{exp_id} loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_acc"], label="train acc")
    axes[1].plot(history["epoch"], history["val_acc"], label="val acc")
    axes[1].set_title(f"{exp_id} accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training curves.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    for exp_id in EXPERIMENTS:
        log_path = output_dir / "logs" / f"{exp_id}_train_log.csv"
        if not log_path.exists():
            print(f"Skip {exp_id}: log not found at {log_path}")
            continue
        output_path = output_dir / "curves" / f"{exp_id}_loss_acc.png"
        plot_experiment(exp_id, log_path, output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
