import numpy as np


def line_segment_from_rho_theta(
    rho: float,
    theta: float,
    w: int,
    h: int,
) -> np.ndarray:
    """
    Восстанавливает отрезок прямой внутри изображения (w, h)
    по параметрам (rho, theta). Возвращает (4,) intp: x1, y1, x2, y2,
    или пустой массив, если пересечение вырождено.
    """
    eps = 1e-9
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    points = []

    if abs(sin_t) > eps:
        y = rho / sin_t
        if 0 <= y < h:
            points.append((0.0, y))
        y = (rho - (w - 1) * cos_t) / sin_t
        if 0 <= y < h:
            points.append((float(w - 1), y))

    if abs(cos_t) > eps:
        x = rho / cos_t
        if 0 <= x < w:
            points.append((x, 0.0))
        x = (rho - (h - 1) * sin_t) / cos_t
        if 0 <= x < w:
            points.append((x, float(h - 1)))

    if len(points) < 2:
        return np.empty((0,), dtype=np.intp)

    unique = [points[0]]
    for p in points[1:]:
        if not any(
            abs(p[0] - q[0]) < eps and abs(p[1] - q[1]) < eps for q in unique
        ):
            unique.append(p)

    if len(unique) < 2:
        return np.empty((0,), dtype=np.intp)

    best: tuple[tuple[float, float], ...] | None = None
    max_dist = -1.0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            d = (unique[i][0] - unique[j][0]) ** 2 + (unique[i][1] - unique[j][1]) ** 2
            if d > max_dist:
                max_dist = d
                best = (unique[i], unique[j])

    (x1, y1), (x2, y2) = best # type: ignore
    return np.array([round(x1), round(y1), round(x2), round(y2)], dtype=np.intp)
