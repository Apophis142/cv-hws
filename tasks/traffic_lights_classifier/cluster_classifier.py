import typing as ty

import numpy as np


DEFAULT_BBOX_PERCENTILES: ty.Tuple[float, float] = (5.00, 95.00)


def robust_fill_ratio(mask: np.ndarray, percentiles: ty.Tuple[float, float] = DEFAULT_BBOX_PERCENTILES) -> float:
    ys, xs = np.nonzero(mask)
    area = ys.size
    if area == 0:
        return 0.0

    lo, hi = percentiles
    y_lo, y_hi = np.percentile(ys, [lo, hi])
    x_lo, x_hi = np.percentile(xs, [lo, hi])
    bbox_area = (y_hi - y_lo + 1.0) * (x_hi - x_lo + 1.0)

    if bbox_area <= 0.0:
        return 0.0

    inside = (
            (ys >= y_lo) & (ys <= y_hi) & (xs >= x_lo) & (xs <= x_hi)
    ).sum()

    return float(inside / bbox_area)


def robust_spatial_spread(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    area = ys.size
    if area == 0:
        return 0.0

    cy = np.median(ys)
    cx = np.median(xs)
    rms = float(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).mean()))
    r_eq = float(np.sqrt(area / np.pi))
    if r_eq <= 0.0:
        return 0.0

    return rms / r_eq



def cluster_compactness(
        mask,
        percentiles: ty.Tuple[float, float] = DEFAULT_BBOX_PERCENTILES
) -> ty.Dict[str, float]:
    area = mask.sum()
    if not area:
        return {
            "area": 0,
            "area_ratio": 0.0,
            "fill_ratio": 0.0,
            "spatial_spread": 0.0,
        }

    return {
        "area": area,
        "area_ratio": area / mask.size,
        "fill_ratio": robust_fill_ratio(mask, percentiles),
        "spatial_spread": robust_spatial_spread(mask),
    }


def vote_cluster_colors(
        pixel_labels: np.ndarray,
        pixel_confidence: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
        target_colors: ty.Sequence[str],
        tau: float = 0.5,
        margin: float = 0.1,
        min_valid_ratio: float = 0.0,
        min_fill_ratio: float = 0.0,
        max_spatial_spread: float = 1e18,
        total_pixels: ty.Optional[int] = None,
        image_shape: ty.Optional[ty.Tuple[int, int]] = None,
) -> ty.Dict[int, ty.Dict[str, ty.Any]]:
    """
    Определить доминирующий цвет каждого кластера по голосам пикселей.

    Доминирующий цвет принимается, если одновременно:
        votes[best] / total >= tau
        (votes[best] - votes[second]) / total >= margin
        total_valid / total >= min_valid_ratio
    Иначе — "undefined".
    """
    if total_pixels is None:
        total_pixels = cluster_labels.size
    if image_shape is None:
        image_shape = (1, cluster_labels.size)

    n_targets = len(target_colors)
    results: ty.Dict[int, ty.Dict[str, ty.Any]] = {}

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels.__eq__(cluster_id)
        total = cluster_mask.sum()

        votes = np.zeros(n_targets, dtype=np.int64)
        weighted = np.zeros(n_targets, dtype=np.float64)

        if total > 0:
            pix_labels = pixel_labels[cluster_mask]
            pix_conf = pixel_confidence[cluster_mask]
            valid = pix_labels >= 0
            for c in range(n_targets):
                c_mask = valid & (pix_labels == c)
                votes[c] = int(c_mask.sum())
                weighted[c] = float(pix_conf[c_mask].sum())

        total_valid = int(votes.sum())

        entry: ty.Dict[str, ty.Any] = {
            "total": total,
            "valid": total_valid,
            "area": total,
            "area_ratio": (total / total_pixels) if total_pixels else 0.0,
            "votes": dict(zip(target_colors, votes.tolist())),
            "weighted_votes": dict(zip(target_colors, weighted.tolist())),
            "color": "undefined",
            "ratio": 0.0,
            "margin": 0.0,
            "valid_ratio": (total_valid / total) if total > 0 else 0.0,
            "compactness": cluster_compactness(cluster_mask.reshape(image_shape)),
        }

        if total_valid > 0 and total > 0:
            order = np.argsort(votes)[::-1]
            best_idx = int(order[0])
            second_idx = int(order[1]) if n_targets > 1 else best_idx
            best_votes = int(votes[best_idx])
            second_votes = int(votes[second_idx])

            ratio = best_votes / total
            margin_val = (best_votes - second_votes) / total
            valid_ratio = total_valid / total

            entry["ratio"] = ratio
            entry["margin"] = margin_val
            entry["valid_ratio"] = valid_ratio
            entry["best_color"] = target_colors[best_idx]
            entry["second_color"] = (
                target_colors[second_idx] if n_targets > 1 else None
            )

            if (
                ratio >= tau
                and margin_val >= margin
                and valid_ratio >= min_valid_ratio
            ):
                entry["color"] = target_colors[best_idx]
            if "compactness" in entry:
                c = entry["compactness"]
                if c["fill_ratio"] < min_fill_ratio or \
                        c["spatial_spread"] > max_spatial_spread:
                    entry["color"] = "undefined"

        results[cluster_id] = entry

    return results
