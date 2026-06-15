import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    CIFAR10_CLASSES,
    build_model,
    create_data_loaders,
    denormalize_image,
    ensure_experiment_dirs,
    get_device,
    get_experiment_id,
    get_experiment_name,
    get_output_dir,
    load_config,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 visualizations.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--data-dir", default=None, help="Override data directory.")
    return parser.parse_args()


@torch.no_grad()
def collect_examples(model, loader, device, limit):
    model.eval()
    correct_examples = []
    wrong_examples = []
    confusion = np.zeros((len(CIFAR10_CLASSES), len(CIFAR10_CLASSES)), dtype=np.int64)

    for images, targets in tqdm(loader, desc="visualize", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        for image, target, pred in zip(images.cpu(), targets, preds):
            confusion[int(target), int(pred)] += 1
            item = (denormalize_image(image), int(target), int(pred))
            if target == pred and len(correct_examples) < limit:
                correct_examples.append(item)
            if target != pred and len(wrong_examples) < limit:
                wrong_examples.append(item)
    return correct_examples, wrong_examples, confusion


def save_dataset_examples(loader, limit, output_path):
    images, targets = next(iter(loader))
    images = images[:limit]
    targets = targets[:limit]
    cols = 4
    rows = max(1, int(np.ceil(len(images) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=150)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, target in zip(axes.ravel(), images, targets):
        ax.imshow(denormalize_image(image).permute(1, 2, 0).numpy())
        ax.set_title(f"Label: {CIFAR10_CLASSES[int(target)]}", fontsize=9)
        ax.axis("off")

    fig.suptitle("CIFAR-10 dataset examples")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_examples(examples, title, output_path):
    cols = 4
    rows = max(1, int(np.ceil(len(examples) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=150)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")
    for ax, (image, target, pred) in zip(axes.ravel(), examples):
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title(
            f"T: {CIFAR10_CLASSES[target]}\nP: {CIFAR10_CLASSES[pred]}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_confusion_matrix(confusion, output_path):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CIFAR10_CLASSES)))
    ax.set_yticks(range(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = confusion[i, j]
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=7)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["data"]["data_dir"] = args.data_dir

    set_seed(int(config["training"].get("seed", 42)))
    output_dir = get_output_dir(config)
    ensure_experiment_dirs(output_dir)
    exp_id = get_experiment_id(config)
    exp_name = get_experiment_name(config)

    _, _, test_loader = create_data_loaders(config)
    save_dataset_examples(
        test_loader,
        args.num_examples,
        output_dir / "dataset_examples.png",
    )
    device = get_device()
    model = build_model(config).to(device)
    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else output_dir / "checkpoints" / f"{exp_name}_best.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    correct, wrong, confusion = collect_examples(
        model, test_loader, device, args.num_examples
    )
    save_examples(
        correct,
        f"{exp_id} correct examples",
        output_dir / "predictions" / "correct_examples.png",
    )
    save_examples(
        wrong,
        f"{exp_id} wrong examples",
        output_dir / "predictions" / "wrong_examples.png",
    )
    save_confusion_matrix(
        confusion,
        output_dir / "confusion_matrix" / f"{exp_id}_confusion_matrix.png",
    )
    print(f"Saved visualizations for {exp_id}")


if __name__ == "__main__":
    main()
