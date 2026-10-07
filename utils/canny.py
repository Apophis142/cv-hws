import numba as nb
import numpy as np

from utils.filtering import apply_sobel_filter, apply_gauss_filter


@nb.njit(cache=True, inline="always")
def _hysteresis_bfs(strong_edges, weak_edges):
    h, w = strong_edges.shape
    weak_edges_padded = np.zeros((h + 2, w + 2), dtype=np.bool_)
    weak_edges_padded[1:-1, 1:-1] = weak_edges

    stack = np.empty((h * w, 2), dtype=np.intp)
    tail = strong_edges.sum()
    stack[:tail] = np.argwhere(strong_edges)

    while tail > 0:
        tail -= 1
        i, j = stack[tail]
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and weak_edges_padded[ni + 1, nj + 1]:
                    weak_edges_padded[ni + 1, nj + 1] = False
                    strong_edges[ni, nj] = True
                    stack[tail, 0] = ni
                    stack[tail, 1] = nj
                    tail += 1

    return strong_edges


def canny_operator(img: np.ndarray, threshold1: float, threshold2: float) -> np.ndarray:
    assert 0 < threshold1 < threshold2 <= 1

    borders, directions = apply_sobel_filter(img, return_directions=True)

    mn, mx = borders.min(), borders.max()
    if mn != mx:
        borders = (borders - mn) / (mx - mn)

    directions = np.round(directions / np.radians(45)) % 4
    dir1 = directions == 0
    dir2 = directions == 1
    dir3 = directions == 2
    dir4 = directions == 3

    kernels = np.zeros(shape=(*borders.shape, 3, 3))
    kernels[dir1] = np.vstack([np.zeros(3), np.ones(3), np.zeros(3)])
    kernels[dir2] = np.fliplr(np.eye(3))
    kernels[dir3] = np.hstack([np.zeros((3, 1)), np.ones((3, 1)), np.zeros((3, 1))])
    kernels[dir4] = np.eye(3)

    borders_expanded = np.pad(borders, {0: (1, 1), 1: (1, 1)}, mode="constant", constant_values=0.0)
    neighbour_max = (
            np.lib.stride_tricks.sliding_window_view(borders_expanded, window_shape=(3, 3), axis=(0, 1)) * kernels
    ).max(axis=(-1, -2))

    suppressed = np.where(borders.__eq__(neighbour_max), borders, 0)

    strong_edges = (suppressed >= threshold2)
    weak_edges = (suppressed < threshold2) & (suppressed >= threshold1)

    strong_edges = _hysteresis_bfs(strong_edges, weak_edges)

    return strong_edges.astype(np.uint8) * 255


if __name__ == "__main__":
    import cv2
    from utils.color_transformation import bgr2rgb

    image = cv2.imread("./data/hw2/img_000036.png")

    cv2.imshow("Canny borders", canny_operator(bgr2rgb(image), 0.01, 0.03))
    cv2.waitKey(0) & 0xFF

    cv2.destroyAllWindows()
