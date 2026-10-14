import typing as ty
from pathlib import Path

import cv2
import numpy as np

from utils.color_transformation import bgr2rgb, rgb2bgr

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(path: ty.Union[str, Path]) -> ty.Iterator[Path]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
        return

    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                yield p


def load_image(path: ty.Union[str, Path]) -> np.ndarray:
    """Загружает изображение как RGB uint8."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return bgr2rgb(bgr)


def save_image(path: ty.Union[str, Path], img: np.ndarray) -> None:
    """Сохраняет RGB или grayscale uint8 изображение."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if img.ndim == 3:
        cv2.imwrite(str(path), rgb2bgr(img))
    elif img.ndim == 2:
        cv2.imwrite(str(path), img)
    else:
        raise ValueError(f"Unsupported image ndim={img.ndim}")
