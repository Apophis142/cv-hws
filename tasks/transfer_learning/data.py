import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def save_json(path: Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def file_hash(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path) -> list[dict]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    found = {
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    }
    if found != set(CLASSES):
        raise ValueError(
            "--data-dir must contain exactly the 10 EuroSAT class directories. "
            f"Missing: {sorted(set(CLASSES) - found)}; extra: {sorted(found - set(CLASSES))}"
        )
    records = []
    seen_hashes = {}
    for label, class_name in enumerate(CLASSES):
        paths = sorted(
            p for p in (root / class_name).rglob("*") if p.suffix.lower() in EXTENSIONS
        )
        if len(paths) < 3:
            raise ValueError(f"Need at least three images in {class_name}")
        for path in paths:
            digest = file_hash(path)
            relative = path.relative_to(root).as_posix()
            if digest in seen_hashes:
                raise ValueError(
                    f"Duplicate image content: {relative} and {seen_hashes[digest]}"
                )
            seen_hashes[digest] = relative
            records.append({"path": relative, "label": label, "sha256": digest})
    return records


def stratified_split(
    records: list[dict],
    seed: int,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label in sorted(grouped):
        group = sorted(grouped[label], key=lambda item: item["path"])
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_fraction))
        n_test = max(1, int(len(group) * test_fraction))
        if n_val + n_test >= len(group):
            raise ValueError(f"Split leaves no training examples for class {label}")
        splits["val"].extend(group[:n_val])
        splits["test"].extend(group[n_val : n_val + n_test])
        splits["train"].extend(group[n_val + n_test :])
    return {
        name: sorted(items, key=lambda item: item["path"])
        for name, items in splits.items()
    }


def prepare_splits(
    root: Path,
    output: Path,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    reuse: Path | None = None,
) -> tuple[dict, dict]:
    records = inventory(root)
    if reuse:
        splits = {
            name: json.loads((Path(reuse) / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("train", "val", "test")
        }
        flattened = [item for items in splits.values() for item in items]
        if sorted(flattened, key=lambda item: item["path"]) != records:
            raise ValueError(
                "Saved splits do not exactly match dataset paths, labels and SHA256"
            )
        if any(
            set(item["label"] for item in items) != set(range(10))
            for items in splits.values()
        ):
            raise ValueError("Every split must contain all 10 classes")
    else:
        splits = stratified_split(records, seed, val_fraction, test_fraction)
    split_dir = Path(output) / "splits"
    split_dir.mkdir()
    for name, items in splits.items():
        save_json(split_dir / f"{name}.json", items)
        (split_dir / f"{name}.txt").write_text(
            "".join(item["path"] + "\n" for item in items), encoding="utf-8"
        )
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    counts = {}
    for name, items in splits.items():
        class_counts = Counter(item["label"] for item in items)
        counts[name] = {
            CLASSES[label]: count for label, count in sorted(class_counts.items())
        }
    return splits, {
        "sha256": digest,
        "total_images": len(records),
        "class_counts": counts,
    }


class EuroSATDataset:
    def __init__(self, root: Path, records: list[dict], transform):
        self.root = Path(root)
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        item = self.records[index]
        with Image.open(self.root / item["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, item["label"]


def make_datasets(root: Path, splits: dict) -> dict:
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights
    from torchvision.transforms import InterpolationMode

    weights = ResNet18_Weights.IMAGENET1K_V1
    preset = weights.transforms()
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=preset.mean, std=preset.std),
        ]
    )
    return {
        name: EuroSATDataset(
            root, items, train_transform if name == "train" else preset
        )
        for name, items in splits.items()
    }
