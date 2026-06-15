import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import ResNet18_CIFAR, SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_experiment_id(config: Dict) -> str:
    return config["experiment"]["id"]


def get_experiment_name(config: Dict) -> str:
    return config["experiment"]["name"]


def get_output_dir(config: Dict) -> Path:
    return resolve_path(config["output"]["output_dir"])


def ensure_experiment_dirs(output_dir: Path) -> None:
    for subdir in ["logs", "checkpoints", "curves", "confusion_matrix", "predictions"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms(use_augmentation: bool):
    train_ops = []
    if use_augmentation:
        train_ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return transforms.Compose(train_ops), eval_transform


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["training"]
    data_dir = resolve_path(data_cfg["data_dir"])
    use_augmentation = bool(data_cfg.get("augmentation", False))
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    seed = int(train_cfg.get("seed", 42))
    download = bool(data_cfg.get("download", True))
    val_ratio = float(data_cfg.get("val_ratio", 0.1))

    train_transform, eval_transform = get_transforms(use_augmentation)
    train_full = datasets.CIFAR10(
        root=str(data_dir), train=True, download=download, transform=train_transform
    )
    val_full = datasets.CIFAR10(
        root=str(data_dir), train=True, download=download, transform=eval_transform
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir), train=False, download=download, transform=eval_transform
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_full), generator=generator).tolist()
    val_size = int(len(indices) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def build_model(config: Dict) -> torch.nn.Module:
    model_type = config["model"]["type"]
    num_classes = int(config["model"].get("num_classes", 10))
    if model_type == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if model_type == "resnet18_cifar":
        return ResNet18_CIFAR(num_classes=num_classes)
    raise ValueError(f"Unsupported model type: {model_type}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_optimizer(config: Dict, model: torch.nn.Module):
    train_cfg = config["training"]
    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(config: Dict, optimizer):
    scheduler_name = config["training"].get("scheduler", "none").lower()
    epochs = int(config["training"]["epochs"])
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name in {"none", "null"}:
        return None
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct, targets.size(0)


def append_or_update_metrics(metrics_path: Path, row: Dict[str, str]) -> None:
    fieldnames = ["experiment", "model", "augmentation", "epochs", "best_val_acc", "test_acc"]
    rows = read_csv_rows(metrics_path)
    rows = [old for old in rows if old.get("experiment") != row["experiment"]]
    rows.append(row)
    write_csv_rows(metrics_path, fieldnames, rows)


def update_class_accuracy(class_accuracy_path: Path, experiment: str, values: List[float]) -> None:
    field = f"{experiment}_acc"
    rows = read_csv_rows(class_accuracy_path)
    row_by_class = {row["class_name"]: row for row in rows if "class_name" in row}
    for class_name, acc in zip(CIFAR10_CLASSES, values):
        row = row_by_class.setdefault(class_name, {"class_name": class_name})
        row[field] = f"{acc:.4f}"
    fieldnames = ["class_name"]
    existing_fields = set()
    for row in row_by_class.values():
        existing_fields.update(row.keys())
    for name in sorted(existing_fields):
        if name != "class_name":
            fieldnames.append(name)
    write_csv_rows(class_accuracy_path, fieldnames, list(row_by_class.values()))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)
