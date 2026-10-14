import typing as ty

import numba as nb
import numpy as np

from .canny import canny_operator
from .filtering import apply_gauss_filter
from .geometry import line_segment_from_rho_theta


@nb.njit(cache=True)
def _hough_lines_vote(edge_points, rho, theta, diag, acc):
    n_rho, n_theta = acc.shape
    cos_theta = np.empty(n_theta, dtype=np.float64)
    sin_theta = np.empty(n_theta, dtype=np.float64)

    for t_idx in range(n_theta):
        t = t_idx * theta
        cos_theta[t_idx] = np.cos(t)
        sin_theta[t_idx] = np.sin(t)

    for k in range(edge_points.shape[0]):
        y = edge_points[k, 0]
        x = edge_points[k, 1]
        for t_idx in range(n_theta):
            r = x * cos_theta[t_idx] + y * sin_theta[t_idx]
            r_idx = int((r + diag) / rho + 0.5)
            if 0 <= r_idx < n_rho:
                acc[r_idx, t_idx] += 1


def _hough_peaks_2d(acc: np.ndarray, threshold: int) -> np.ndarray:
    padded = np.pad(
        acc, {0: (1, 1), 1: (1, 1)}, mode="constant", constant_values=-1
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=(3, 3), axis=(0, 1)
    )
    local_max = windows.max(axis=(-1, -2))

    mask = (acc.__eq__(local_max)) & (acc >= threshold)
    peaks = np.argwhere(mask)
    if peaks.size == 0:
        return np.empty((0, 3), dtype=np.intp)

    votes = acc[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(votes)[::-1]
    return np.column_stack([peaks[order], votes[order]]).astype(np.intp)


def hough_lines(
    img: np.ndarray,
    threshold1: float = 0.01,
    threshold2: float = 0.03,
    rho: float = 1.0,
    theta: float = np.pi / 180.0,
    threshold: int = 50,
    min_line_length: int = 10,
    max_lines: int = 50,
    return_edges: bool = False,
) -> ty.Union[np.ndarray, ty.Tuple[np.ndarray, np.ndarray]]:
    """
    Классическое преобразование Хафа для прямых.

    Возвращает массив формы (N, 4): x1, y1, x2, y2.
    """
    assert 0 < threshold1 < threshold2 <= 1
    assert rho > 0
    assert 0 < theta < np.pi
    assert threshold > 0
    assert min_line_length > 0
    assert max_lines > 0

    img = apply_gauss_filter(img, 5)
    edges = canny_operator(img, threshold1, threshold2)
    
    def _finish(lines: np.ndarray) -> ty.Union[np.ndarray, ty.Tuple[np.ndarray, np.ndarray]]:
        if return_edges:
            return lines, edges
        return lines

    
    edge_points = np.argwhere(edges > 0)
    if edge_points.size == 0:
        return np.empty((0, 4), dtype=np.intp)

    h, w = edges.shape
    diag = np.hypot(h, w)
    n_rho = int(2 * diag / rho) + 1
    n_theta = int(np.ceil(np.pi / theta))

    acc = np.zeros((n_rho, n_theta), dtype=np.int32)
    _hough_lines_vote(edge_points, rho, theta, diag, acc)

    peaks = _hough_peaks_2d(acc, threshold)
    lines = []

    for r_idx, t_idx, votes in peaks:
        r = r_idx * rho - diag
        t = t_idx * theta
        seg = line_segment_from_rho_theta(r, t, w, h)
        if seg.size == 0:
            continue

        x1, y1, x2, y2 = seg
        if np.hypot(x2 - x1, y2 - y1) < min_line_length:
            continue

        lines.append(seg)
        if len(lines) >= max_lines:
            break

    return _finish(np.array(lines, dtype=np.intp) if lines else np.empty((0, 4), dtype=np.intp))


@nb.njit(cache=True)
def _hough_circles_vote(edge_points, directions, radii, ab_delta, acc):
    """
    Градиентный вариант: центр окружности лежит на нормали к краю.
    Голосуем в обе стороны, так как полярность градиента Собеля
    зависит от направления перехода яркости.

    ab_delta — шаг бинаризации центра (a, b), >= 1.
    """
    h_bins, w_bins, n_radii = acc.shape
    inv_ab = 1.0 / ab_delta

    for k in range(edge_points.shape[0]):
        y = edge_points[k, 0]
        x = edge_points[k, 1]
        phi = directions[y, x]
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        for r_idx in range(n_radii):
            r = radii[r_idx]

            a = (x + r * cos_phi) * inv_ab
            b = (y + r * sin_phi) * inv_ab
            a_idx = int(a + 0.5) if a >= 0 else int(a - 0.5)
            b_idx = int(b + 0.5) if b >= 0 else int(b - 0.5)
            if 0 <= a_idx < w_bins and 0 <= b_idx < h_bins:
                acc[b_idx, a_idx, r_idx] += 1

            a = (x - r * cos_phi) * inv_ab
            b = (y - r * sin_phi) * inv_ab
            a_idx = int(a + 0.5) if a >= 0 else int(a - 0.5)
            b_idx = int(b + 0.5) if b >= 0 else int(b - 0.5)
            if 0 <= a_idx < w_bins and 0 <= b_idx < h_bins:
                acc[b_idx, a_idx, r_idx] += 1


def _hough_peaks_3d(
    acc: np.ndarray,
    radii: np.ndarray,
    threshold_frac: float,
) -> ty.Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (indices, scores):
      indices — (N, 3) intp: b_idx, a_idx, r_idx
      scores  — (N,)  float64: нормированные голоса в [0, 1]
    Оба отсортированы по убыванию scores.
    """
    expected = 2.0 * np.pi * radii.astype(np.float64)
    expected = np.maximum(expected, 1.0)

    padded = np.pad(
        acc,
        {0: (1, 1), 1: (1, 1), 2: (1, 1)},
        mode="constant",
        constant_values=-1,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=(3, 3, 3), axis=(0, 1, 2)
    )
    local_max = windows.max(axis=(-1, -2, -3))

    score = acc / expected[None, None, :]
    mask = (acc.__eq__(local_max)) & (score >= threshold_frac)

    peaks = np.argwhere(mask)
    if peaks.size == 0:
        return np.empty((0, 3), dtype=np.intp), np.empty((0,), dtype=np.float64)

    peak_scores = score[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
    order = np.argsort(peak_scores)[::-1]

    return peaks[order].astype(np.intp), peak_scores[order]

def hough_circles(
    img: np.ndarray,
    threshold1: float = 0.01,
    threshold2: float = 0.03,
    r_min: int = 0,
    r_max: int = 100,
    r_delta: int = 1,
    ab_delta: int = 1,
    threshold: float = 0.4,
    min_dist: int = 10,
    max_circles: int = 10,
    return_edges: bool = False,
) -> ty.Union[np.ndarray, ty.Tuple[np.ndarray, np.ndarray]]:
    """
    Градиентное преобразование Хафа для окружностей.

    Возвращает массив формы (N, 3): x, y, r.

    ab_delta контролирует размер аккумулятора: O((H/ab_delta) * (W/ab_delta) * n_radii).
    При больших изображениях и широком диапазоне радиусов это критично.
    threshold — минимальная доля видимой дуги (0..1). Нормировка
    голосов на 2πr делает порог инвариантным к радиусу.
    """
    assert 0 < threshold1 < threshold2 <= 1
    assert 0 <= r_min <= r_max
    assert r_delta > 0
    assert ab_delta > 0
    assert 0.0 < threshold <= 1.0
    assert threshold > 0
    assert min_dist >= 0
    assert max_circles > 0

    img = apply_gauss_filter(img, 5)
    edges, directions = canny_operator(
        img, threshold1, threshold2, return_directions=True
    )
    
    def _finish(circles: np.ndarray) -> ty.Union[np.ndarray, ty.Tuple[np.ndarray, np.ndarray]]:
        if return_edges:
            return circles, edges
        return circles

    
    edge_points = np.argwhere(edges > 0)
    if edge_points.size == 0:
        return np.empty((0, 3), dtype=np.intp)

    h, w = edges.shape
    h_bins = h // ab_delta + 1
    w_bins = w // ab_delta + 1
    radii = np.arange(r_min, r_max + 1, r_delta)
    acc = np.zeros((h_bins, w_bins, len(radii)), dtype=np.int32)

    _hough_circles_vote(edge_points, directions, radii, ab_delta, acc)

    idx, scores = _hough_peaks_3d(acc, radii, threshold_frac=threshold)
    circles = []

    for (b_idx, a_idx, r_idx), score in zip(idx, scores):
        a = int(a_idx) * ab_delta
        b = int(b_idx) * ab_delta
        r = int(radii[r_idx])

        if any(np.hypot(cx - a, cy - b) < min_dist for cx, cy, _ in circles):
            continue

        circles.append((a, b, r))
        if len(circles) >= max_circles:
            break

    return _finish(
        np.array(circles, dtype=np.intp)
        if circles
        else np.empty((0, 3), dtype=np.intp)
    )
