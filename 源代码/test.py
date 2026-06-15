import argparse

import numpy as np
import torch
from tqdm import tqdm

from utils import (
    CIFAR10_CLASSES,
    append_or_update_metrics,
    build_model,
    create_data_loaders,
    ensure_experiment_dirs,
    get_device,
    get_experiment_id,
    get_experiment_name,
    get_output_dir,
    load_config,
    set_seed,
    update_class_accuracy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument("--data-dir", default=None, help="Override data directory.")
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    for images, targets in tqdm(loader, desc="test", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(targets.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


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
    device = get_device()
    model = build_model(config).to(device)

    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else output_dir / "checkpoints" / f"{exp_name}_best.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    preds, targets = collect_predictions(model, test_loader, device)
    test_acc = float((preds == targets).mean())
    best_val_acc = float(checkpoint.get("best_val_acc", 0.0))

    per_class_acc = []
    confusion = np.zeros((len(CIFAR10_CLASSES), len(CIFAR10_CLASSES)), dtype=np.int64)
    for target, pred in zip(targets, preds):
        confusion[target, pred] += 1
    for class_index in range(len(CIFAR10_CLASSES)):
        class_total = confusion[class_index].sum()
        class_correct = confusion[class_index, class_index]
        per_class_acc.append(float(class_correct / class_total) if class_total else 0.0)

    np.savez(
        output_dir / "predictions" / f"{exp_id}_predictions.npz",
        predictions=preds,
        targets=targets,
        confusion=confusion,
    )
    append_or_update_metrics(
        output_dir / "metrics.csv",
        {
            "experiment": exp_id,
            "model": config["model"]["type"],
            "augmentation": str(bool(config["data"].get("augmentation", False))),
            "epochs": str(config["training"]["epochs"]),
            "best_val_acc": f"{best_val_acc:.4f}",
            "test_acc": f"{test_acc:.4f}",
        },
    )
    update_class_accuracy(output_dir / "class_accuracy.csv", exp_id, per_class_acc)
    print(f"{exp_id} test_acc={test_acc:.4f}, best_val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
