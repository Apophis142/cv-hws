import typing as ty

import numpy as np

import utils
from .config import ColorClassifierConfig
from .ciede import pairwise_color_distance
from .labels import LABEL_BACKGROUND, LABEL_UNDEFINED


def compute_background_mask(
    lch_pixels: np.ndarray,
    config: ColorClassifierConfig,
) -> np.ndarray:
    """True там, где пиксель ахроматический или сине-голубой (фон)."""
    blue_lo, blue_hi = config.blue_hue_range

    return (lch_pixels[..., 1] < config.achromatic_chroma_threshold) | (
            (lch_pixels[..., 2] >= blue_lo) &
            (lch_pixels[..., 2] < blue_hi)
    )


def coarse_classify_by_hue(
    lch_pixels: np.ndarray,
    bg_mask: np.ndarray,
    config: ColorClassifierConfig,
) -> np.ndarray:
    """
    Грубая классификация хроматических пикселей по hue.

    Возвращает int-массив:
        -1 : фон
        -2 : спорный (рядом с границей hue)
         k : индекс в config.target_colors
    """
    n = lch_pixels.shape[0]
    h = lch_pixels[..., 2]

    labels = np.full(n, LABEL_UNDEFINED, dtype=np.int64)
    labels[bg_mask] = LABEL_BACKGROUND

    hue_for_zone = np.where(h >= 320.0, h - 360.0, h)

    h1 = config.hue_boundaries["red_orange"]
    h2 = config.hue_boundaries["orange_yellow"]
    h3 = config.hue_boundaries["yellow_green"]
    m = config.disputed_hue_margin

    chromatic = ~bg_mask

    labels[chromatic & (hue_for_zone < h1)] = 0
    labels[chromatic & (hue_for_zone >= h1) & (hue_for_zone < h2)] = 1
    labels[chromatic & (hue_for_zone >= h2) & (hue_for_zone < h3)] = 2
    labels[chromatic & (hue_for_zone >= h3)] = 3

    disputed = chromatic & (
        (np.abs(hue_for_zone - h1) < m)
        | (np.abs(hue_for_zone - h2) < m)
        | (np.abs(hue_for_zone - h3) < m)
    )
    labels[disputed] = LABEL_UNDEFINED

    return labels


def fine_classify_by_signatures(
    lab_pixels: np.ndarray,
    mask: np.ndarray,
    signature_arrays: ty.Dict[str, np.ndarray],
    target_colors: ty.Sequence[str],
    metric: ty.Literal["euclid", "ciede2000"] = "ciede2000",
) -> ty.Tuple[np.ndarray, np.ndarray]:
    """
    Классифицировать замаскированные пиксели по ближайшей сигнатуре.

    Returns
    -------
    labels : (M,) int, индекс в target_colors
    confidence : (M,) float, отрыв лучшего от второго
    """
    if not np.any(mask):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    pixels = lab_pixels[mask].astype(np.float64, copy=False)
    n_colors = len(target_colors)

    dists = np.full((pixels.shape[0], n_colors), np.inf, dtype=np.float64)
    for c_idx, color in enumerate(target_colors):
        sigs = signature_arrays.get(color)
        if sigs is None or sigs.shape[0] == 0:
            continue
        dists[:, c_idx] = pairwise_color_distance(pixels, sigs, metric).min(axis=1)

    order = np.argsort(dists, axis=1)
    best = order[:, 0]
    best_d = dists[np.arange(dists.shape[0]), best]

    if n_colors > 1:
        second_d = dists[np.arange(dists.shape[0]), order[:, 1]]
    else:
        second_d = np.full_like(best_d, np.inf)

    return best.astype(np.int64), (second_d - best_d)


def classify_pixels(
    lab_pixels: np.ndarray,
    config: ColorClassifierConfig,
    signature_arrays: ty.Dict[str, np.ndarray],
) -> ty.Tuple[np.ndarray, np.ndarray]:
    """
    Классифицировать все пиксели изображения.

    Returns
    -------
    labels : (N,) int
        -1 фон, -2 неопределено, 0..len(target_colors)-1 — целевой цвет
    confidence : (N,) float
        отрыв лучшего цвета от второго; 0 для фона/неопределённых
    """
    n = lab_pixels.shape[0]
    lch = utils.color_transformation.lab2lch(lab_pixels)
    bg_mask = compute_background_mask(lch, config)

    if config.use_hue_prefilter:
        labels = coarse_classify_by_hue(lch, bg_mask, config)
    else:
        labels = np.full(n, LABEL_UNDEFINED, dtype=np.int64)
        labels[bg_mask] = LABEL_BACKGROUND

    confidence = np.zeros(n, dtype=np.float64)

    disputed_mask = labels == LABEL_UNDEFINED
    if np.any(disputed_mask):
        fine_labels, fine_conf = fine_classify_by_signatures(
            lab_pixels,
            disputed_mask,
            signature_arrays,
            config.target_colors,
            metric=config.fine_metric,
        )
        labels[disputed_mask] = fine_labels
        confidence[disputed_mask] = fine_conf

    return labels, confidence
