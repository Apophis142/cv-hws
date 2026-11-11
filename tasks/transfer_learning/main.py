import argparse
import importlib.metadata
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .data import CLASSES, file_hash, make_datasets, prepare_splits, save_json
from .snapshot import export_source


DATASET = "waseemalastal/eurosat-rgb-dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EuroSAT RGB transfer learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the 10 class directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty experiment directory",
    )
    parser.add_argument(
        "--dataset-version",
        type=int,
        required=True,
        help="Numeric Kaggle dataset version from the Versions tab",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--val-fraction", type=float, default=0.15, help="Validation fraction"
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.15, help="Test fraction"
    )
    parser.add_argument(
        "--head-epochs", type=int, default=5, help="Classifier training epochs"
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=10, help="Fine-tuning epochs"
    )
    parser.add_argument(
        "--head-lr", type=float, default=1e-3, help="Classifier learning rate"
    )
    parser.add_argument(
        "--finetune-lr", type=float, default=1e-4, help="Fine-tuning learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--workers", type=int, default=2, help="DataLoader worker count"
    )
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda", help="Training device"
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        help="Reuse and verify train/val/test JSON manifests from a prior run",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Save dataset splits and metadata without training",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    positive = (
        args.dataset_version,
        args.head_epochs,
        args.finetune_epochs,
        args.batch_size,
    )
    if any(value <= 0 for value in positive) or args.workers < 0:
        parser.error(
            "Version, epochs and batch size must be positive; workers must be >= 0"
        )
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if not (
        0 < args.val_fraction < 1
        and 0 < args.test_fraction < 1
        and args.val_fraction + args.test_fraction < 1
    ):
        parser.error("val/test fractions must be positive and sum to less than 1")
    if not (
        math.isfinite(args.head_lr)
        and math.isfinite(args.finetune_lr)
        and 0 < args.finetune_lr < args.head_lr
    ):
        parser.error("Require 0 < finetune-lr < head-lr, both finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and >= 0")
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error(
            "output-dir must be new or empty; use a separate directory for each experiment"
        )
    if args.output_dir.is_relative_to(args.data_dir):
        parser.error("output-dir must be outside data-dir")
    return args


def environment() -> dict:
    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def run(args: argparse.Namespace) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not args.prepare_only:
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable. Select a GPU runtime, or explicitly use --device cpu"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits, dataset_summary = prepare_splits(
        args.data_dir,
        args.output_dir,
        args.seed,
        args.val_fraction,
        args.test_fraction,
        args.splits_dir,
    )
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    source = export_source(args.output_dir / "source.zip")
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "dataset": {
            "slug": DATASET,
            "version": args.dataset_version,
            "version_url": f"https://www.kaggle.com/datasets/{DATASET}/versions/{args.dataset_version}",
            "version_origin": "user-supplied Kaggle dataset version",
            **dataset_summary,
        },
        "split_files_sha256": {
            name: file_hash(args.output_dir / "splits" / f"{name}.json")
            for name in splits
        },
        "classes": list(CLASSES),
        "source": source,
        "preprocessing": {
            "train": "RandomResizedCrop(224, scale=(0.8,1), bilinear, antialias), HFlip(0.5), VFlip(0.5)",
            "val_test": "ResNet18_Weights.IMAGENET1K_V1.transforms(): resize 256, center crop 224",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "training": {
            "architecture": "resnet18",
            "weights": "IMAGENET1K_V1",
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "loss": "CrossEntropyLoss",
            "scheduler": None,
            "amp": False,
            "deterministic_algorithms": True,
            "selection": "highest val accuracy; lowest val loss breaks ties",
            "stage1_trainable": ["fc"],
            "stage2_trainable": ["layer4", "fc"],
            "frozen_batchnorm": "eval (running statistics remain fixed)",
        },
    }
    env = environment()
    save_json(args.output_dir / "environment.json", env)
    (args.output_dir / "environment.txt").write_text(
        "\n".join(env["packages"]) + "\n", encoding="utf-8"
    )
    save_json(args.output_dir / "experiment.json", metadata)
    if args.prepare_only:
        print(
            f"Prepared {dataset_summary['total_images']} images; artifacts: {args.output_dir}"
        )
        return

    from .engine import evaluate, seed_everything, train_stage
    from .model import create_model
    import torchvision

    device = torch.device(args.device)
    metadata["runtime"] = {
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_num_threads": torch.get_num_threads(),
    }
    save_json(args.output_dir / "experiment.json", metadata)
    seed_everything(args.seed)
    datasets = make_datasets(args.data_dir, splits)
    model = create_model().to(device)
    head = train_stage(model, datasets, device, args, "head_only", metadata)
    tuned = train_stage(model, datasets, device, args, "fine_tuned", metadata)
    results = {}
    for stage, checkpoint in (("head_only", head), ("fine_tuned", tuned)):
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = evaluate(model, datasets["test"], device, args, stage)
        results[stage] = {
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "best_epoch": checkpoint["epoch"],
            "val_accuracy": checkpoint["val"]["accuracy"],
        }
    delta = 100 * (
        results["fine_tuned"]["test_accuracy"] - results["head_only"]["test_accuracy"]
    )
    save_json(
        args.output_dir / "comparison.json",
        {
            "stages": results,
            "accuracy_delta_percentage_points": delta,
            "test_split_sha256": metadata["split_files_sha256"]["test"],
        },
    )
    rows = [
        "# EuroSAT RGB: результаты",
        "",
        "| Вариант | Лучшая эпоха | Val accuracy | Test accuracy | Test macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, result in results.items():
        rows.append(
            f"| {stage} | {result['best_epoch']} | {result['val_accuracy']:.4f} | "
            f"{result['test_accuracy']:.4f} | {result['test_macro_f1']:.4f} |"
        )
    rows.extend(
        [
            "",
            f"Изменение test accuracy после fine-tuning: {delta:+.2f} п.п.",
            "",
            "Оба варианта проверены на одном test. Веса выбраны только по validation.",
            "",
            "Матрицы ошибок: [head_only](head_only/confusion_matrix.png), "
            "[fine_tuned](fine_tuned/confusion_matrix.png).",
            "",
        ]
    )
    (args.output_dir / "comparison.md").write_text("\n".join(rows), encoding="utf-8")
    save_json(
        args.output_dir / "completed.json",
        {"finished_utc": datetime.now(timezone.utc).isoformat()},
    )
    print("\n".join(rows), flush=True)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
