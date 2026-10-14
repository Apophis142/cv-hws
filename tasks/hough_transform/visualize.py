import cv2
import numpy as np


def circle_mask(
    shape: tuple,
    circles: np.ndarray,
    thickness: int = 1,
) -> np.ndarray:
    """
    Бинарная маска (H, W) uint8, 255 на контурах окружностей.

    thickness — толщина кольца в пикселях. При thickness <= 0
    окружность рисуется залитой (полезно для area-based метрик).
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    line_type = cv2.LINE_AA

    for x, y, r in circles:
        cv2.circle(
            mask,
            (int(x), int(y)),
            int(r),
            255,
            thickness if thickness > 0 else -1,
            lineType=line_type,
        )
    return mask


def overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """Накладывает цветную полупрозрачную маску на RGB-изображение."""
    vis = img.astype(np.float64).copy()
    color_arr = np.array(color, dtype=np.float64)
    mask_bool = mask > 0
    vis[mask_bool] = (1.0 - alpha) * vis[mask_bool] + alpha * color_arr
    return vis.astype(np.uint8)


def draw_circles(
    img: np.ndarray,
    circles: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    label: bool = True,
) -> np.ndarray:
    """Рисует контуры окружностей и опционально подписи."""
    vis = img.copy()
    for i, (x, y, r) in enumerate(circles):
        cv2.circle(vis, (int(x), int(y)), int(r), color, thickness)
        if label:
            cv2.putText(
                vis,
                f"#{i} r={int(r)}",
                (int(x - r), int(y - r - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return vis
