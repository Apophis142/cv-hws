import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CLASSES, save_json
from .model import configure_stage, training_mode


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(
    dataset,
    batch_size: int,
    workers: int,
    seed: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )


def run_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None
) -> dict:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count += targets.size(0)
            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
    return {"loss": total_loss / count, "accuracy": correct / count, "samples": count}


def train_stage(model, datasets, device, args, stage, metadata):
    fine_tune = stage == "fine_tuned"
    configure_stage(model, fine_tune)
    seed_everything(args.seed)
    epochs = args.finetune_epochs if fine_tune else args.head_epochs
    lr = args.finetune_lr if fine_tune else args.head_lr
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(
        datasets["train"],
        args.batch_size,
        args.workers,
        args.seed,
        True,
        device,
    )
    val_loader = make_loader(
        datasets["val"],
        args.batch_size,
        args.workers,
        args.seed,
        False,
        device,
    )
    directory = args.output_dir / stage
    directory.mkdir()
    checkpoint_path = directory / "best.pt"
    best_key = (-1.0, float("-inf"))
    history = []
    for epoch in range(1, epochs + 1):
        training_mode(model, fine_tune)
        train = run_epoch(model, train_loader, device, optimizer)
        model.eval()
        val = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": train, "val": val, "lr": lr})
        key = (val["accuracy"], -val["loss"])
        if key > best_key:
            best_key = key
            checkpoint = {
                "model_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "stage": stage,
                "epoch": epoch,
                "val": val,
                "class_to_idx": {name: i for i, name in enumerate(CLASSES)},
                "metadata": metadata,
            }
            torch.save(checkpoint, checkpoint_path)
        save_json(directory / "history.json", history)
        print(
            f"{stage} {epoch}/{epochs}: train_acc={train['accuracy']:.4f} "
            f"val_acc={val['accuracy']:.4f} val_loss={val['loss']:.4f}",
            flush=True,
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def confusion_metrics(matrix: np.ndarray) -> dict:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = matrix.diagonal()
    precision = np.divide(
        diagonal, predicted, out=np.zeros(10, dtype=float), where=predicted != 0
    )
    recall = np.divide(
        diagonal, support, out=np.zeros(10, dtype=float), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(10, dtype=float),
        where=precision + recall != 0,
    )
    return {
        "accuracy": float(diagonal.sum() / matrix.sum()),
        "samples": int(matrix.sum()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASSES)
        },
        "class_order": list(CLASSES),
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion(matrix: np.ndarray, directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (directory / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *CLASSES])
        for name, row in zip(CLASSES, matrix):
            writer.writerow([name, *row.tolist()])
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(10),
        yticks=range(10),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="True",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    threshold = matrix.max() / 2
    for i in range(10):
        for j in range(10):
            color = "white" if matrix[i, j] > threshold else "black"
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)
    figure.tight_layout()
    figure.savefig(directory / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def evaluate(model, dataset, device, args, stage):
    loader = make_loader(
        dataset, args.batch_size, args.workers, args.seed, False, device
    )
    model.eval()
    matrix = np.zeros((10, 10), dtype=np.int64)
    directory = args.output_dir / stage
    offset = 0
    with (directory / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true_label", "predicted_label"])
        with torch.inference_mode():
            for inputs, targets in loader:
                predicted = (
                    model(inputs.to(device, non_blocking=True)).argmax(1).cpu().numpy()
                )
                actual = targets.numpy()
                np.add.at(matrix, (actual, predicted), 1)
                for truth, prediction in zip(actual, predicted):
                    writer.writerow(
                        [
                            dataset.records[offset]["path"],
                            CLASSES[truth],
                            CLASSES[prediction],
                        ]
                    )
                    offset += 1
    metrics = confusion_metrics(matrix)
    save_json(directory / "metrics.json", metrics)
    save_confusion(matrix, directory)
    return metrics
