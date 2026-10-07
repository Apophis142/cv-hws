import os
import argparse
import cv2
import numpy as np

from utils.canny import canny_operator
from utils.filtering import (
    apply_sobel_filter,
    scharr_operator,
    apply_prewitt_filter,
    apply_gauss_filter,
    apply_non_linear_filter,
)
from utils.color_transformation import bgr2rgb


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Приводит изображение к uint8 в диапазоне 0..255 для сохранения."""
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def process_image(
    img_rgb: np.ndarray,
    median_d: int = 3,
    gauss_win: int = 5,
    gauss_sigma: float = None,
    canny_thresh: tuple = (0.03, 0.11),
) -> dict:
    """
    Принимает RGB-изображение (uint8 или float 0..255).
    Возвращает словарь с 8 обработанными изображениями:
    {sobel_median, sobel_gauss, scharr_median, scharr_gauss,
     prewitt_median, prewitt_gauss, canny_median, canny_gauss}
    """
    # Предварительная фильтрация
    median_img = apply_non_linear_filter(img_rgb, "median", d=median_d)
    gauss_img = apply_gauss_filter(img_rgb, win_size=gauss_win, sigma=gauss_sigma)

    results = {
        "sobel_median": apply_sobel_filter(median_img),
        "sobel_gauss": apply_sobel_filter(gauss_img),
        "scharr_median": scharr_operator(median_img),
        "scharr_gauss": scharr_operator(gauss_img),
        "prewitt_median": apply_prewitt_filter(median_img),
        "prewitt_gauss": apply_prewitt_filter(gauss_img),
        "canny_median": canny_operator(median_img, *canny_thresh),
        "canny_gauss": canny_operator(gauss_img, *canny_thresh)
    }

    return results


def stretch_for_display(img: np.ndarray) -> np.ndarray:
    """Растягивает контраст по 1–99 перцентилям."""
    if img.dtype != np.uint8:
        arr = img.astype(np.float64)
    else:
        arr = img.astype(np.float64) / 255.0
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def log_stretch(img):
    arr = img.astype(np.float64)
    arr = np.log1p(arr)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / (hi - lo + 1e-12), 0, 1)
    return (arr * 255).astype(np.uint8)


def main(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]

    for fname in files:
        path = os.path.join(input_dir, fname)
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Не удалось прочитать {path}")
            continue

        img_rgb = bgr2rgb(img_bgr)
        results = process_image(img_rgb, median_d=7, gauss_win=5, gauss_sigma=1.0)

        base = os.path.splitext(fname)[0]
        save_dir = os.path.join(output_dir, base)
        os.makedirs(save_dir, exist_ok=True)

        for name, res in results.items():
            out_path = os.path.join(save_dir, f"{name}.png")
            if "canny" not in name:
                cv2.imwrite(out_path, to_uint8(log_stretch(res)))
            else:
                cv2.imwrite(out_path, to_uint8(res))
            print(f"Сохранено: {out_path}")

    print(f"\nГотово. Результаты в папке: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Пайплайн обработки УЗИ/КТ: фильтры + операторы выделения границ"
    )
    parser.add_argument("--input_dir", default="data/uzi_ct/raw/", help="Папка с исходными изображениями")
    parser.add_argument(
        "--output_dir", default="tasks/filters_experiments/output/", help="Папка для сохранения результатов"
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
