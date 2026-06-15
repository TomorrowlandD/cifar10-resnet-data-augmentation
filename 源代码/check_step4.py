from pathlib import Path

import torch

from utils import (
    build_model,
    build_optimizer,
    build_scheduler,
    count_parameters,
    get_output_dir,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATHS = [
    PROJECT_ROOT / "源代码" / "configs" / "exp1_cnn_baseline.yaml",
    PROJECT_ROOT / "源代码" / "configs" / "exp2_cnn_aug.yaml",
    PROJECT_ROOT / "源代码" / "configs" / "exp3_resnet18.yaml",
]
REQUIRED_TOP_LEVEL_KEYS = {"experiment", "model", "data", "training", "output"}
REQUIRED_TRAINING_KEYS = {
    "seed",
    "batch_size",
    "epochs",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "scheduler",
}


def validate_config(config_path: Path) -> dict:
    config = load_config(str(config_path))
    missing = REQUIRED_TOP_LEVEL_KEYS - set(config)
    if missing:
        raise ValueError(f"{config_path} missing top-level keys: {sorted(missing)}")

    missing_training = REQUIRED_TRAINING_KEYS - set(config["training"])
    if missing_training:
        raise ValueError(
            f"{config_path} missing training keys: {sorted(missing_training)}"
        )

    if config["data"].get("dataset") != "CIFAR10":
        raise ValueError(f"{config_path} dataset must be CIFAR10")

    output_dir = get_output_dir(config)
    if not output_dir.is_absolute():
        raise ValueError(f"{config_path} output_dir did not resolve to absolute path")

    return config


def validate_model_and_training_objects(config: dict) -> None:
    model = build_model(config)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 32, 32))
    if tuple(logits.shape) != (2, 10):
        raise ValueError(
            f"{config['experiment']['id']} forward shape mismatch: {tuple(logits.shape)}"
        )

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    param_count = count_parameters(model)
    scheduler_name = scheduler.__class__.__name__ if scheduler is not None else "None"
    print(
        f"[OK] {config['experiment']['id']} "
        f"model={config['model']['type']} "
        f"params={param_count:,} "
        f"optimizer={optimizer.__class__.__name__} "
        f"scheduler={scheduler_name}"
    )


def main() -> None:
    print("Step 4 lightweight check: configs, models, and training objects")
    for config_path in CONFIG_PATHS:
        config = validate_config(config_path)
        validate_model_and_training_objects(config)
    print("[OK] Step 4 lightweight check passed")


if __name__ == "__main__":
    main()
