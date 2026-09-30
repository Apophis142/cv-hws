import typing as ty

import numpy as np

import utils


def generate_gauss_kernel(win_size: int, dim: int, normalize=True, sigma: ty.Optional[float]=None) -> np.ndarray:
    r = win_size >> 1
    if sigma is None:
        sigma = 0.3 * (r - 1) + 0.8
    row = np.arange(-r, r + 1, 1)**2

    dist = np.exp(-sum([row.reshape(*([1] * k + [-1] + [1] * (dim - 1 - k))) for k in range(dim)]) / (2 * sigma**2))
    return dist / dist.sum() if normalize else dist


def window_mean_filter(
        img: np.ndarray,
        win_size: int,
) -> np.ndarray:
    r = win_size >> 1
    h, w = img.shape

    integral_img = np.cumsum(np.cumsum(img.astype(np.uint32), axis=0), axis=1)

    integral_img = np.pad(integral_img, pad_width=((1, 0), (1, 0)), mode="constant", constant_values=0)
    y, x = np.ogrid[:h, :w]

    y1, x1 = np.clip(y - r, 0, h - 1), np.clip(x - r, 0, w - 1)
    y2, x2 = np.clip(y + r, 0, h - 1), np.clip(x + r, 0, w - 1)

    window_sum = (
            integral_img[y2 + 1, x2 + 1] +
            integral_img[y1, x1] -
            integral_img[y2 + 1, x1] -
            integral_img[y1, x2 + 1]
    )

    return window_sum / ((y2 - y1 + 1) * (x2 - x1 + 1))


def apply_linear_filter(
        img: np.ndarray,
        matrix_filter: np.ndarray,
        dtype: np.dtype=np.uint8,
        apply_clipping = True
) -> np.ndarray:
    r = matrix_filter.shape[0] >> 1
    c = matrix_filter.shape[1] >> 1
    if len(img.shape) == 3:
        mod_img = np.pad(img, pad_width=((r, r), (c, c), (0, 0)), mode="edge")
    elif len(img.shape) == 2:
        mod_img = np.pad(img, pad_width=((r, r), (c, c)), mode="edge")
    else:
        raise ValueError

    if apply_clipping:
        return np.tensordot(
            np.lib.stride_tricks.sliding_window_view(mod_img, window_shape=matrix_filter.shape, axis=(0, 1)),
            matrix_filter,
            axes=([-2, -1], [0, 1])
        ).clip(0., 255.).astype(dtype)
    else:
        return np.tensordot(
            np.lib.stride_tricks.sliding_window_view(mod_img, window_shape=matrix_filter.shape, axis=(0, 1)),
            matrix_filter,
            axes=([-2, -1], [0, 1])
        )


def apply_gauss_filter(
        img: np.ndarray,
        win_size: int,
        sigma: ty.Optional[float] = None,
        dtype: np.dtype = np.uint8,
        apply_clipping: bool = True,
):
    gauss_filter = generate_gauss_kernel(win_size, 2, True, sigma)
    return apply_linear_filter(img, gauss_filter, dtype, apply_clipping)


def apply_non_linear_filter(img: np.ndarray, mode: ty.Literal['min', 'max', 'median'], d: int=3):
    r = d >> 1
    if len(img.shape) == 3:
        mod_img = np.pad(img, pad_width=((r, r), (r, r), (0, 0)), mode="edge")
    elif len(img.shape) == 2:
        mod_img = np.pad(img, pad_width=((r, r), (r, r)), mode="edge")
    else:
        raise ValueError
    mod_img = np.lib.stride_tricks.sliding_window_view(mod_img, window_shape=(d, d), axis=(0, 1))

    if mode == "max":
        return mod_img.max(axis=(-1, -2))
    elif mode == "min":
        return mod_img.min(axis=(-1, -2))
    elif mode == "median":
        return np.median(mod_img, axis=(-1, -2))


def opening_filter(img: np.ndarray, r: int=1) -> np.ndarray:
    d = (r << 1) + 1
    return apply_non_linear_filter(apply_non_linear_filter(img, "min", d), "max", d)


def closing_filter(img: np.ndarray, r: int=1) -> np.ndarray:
    d = (r << 1) + 1
    return apply_non_linear_filter(apply_non_linear_filter(img, "max", d), "min", d)


def apply_sobel_filter(
        img: np.ndarray,
        return_directions=False,
        return_gxy=False,
        channel: ty.Literal['gray', 'hsv_saturation', 'lab_luminance', 'hsv_value'] = "gray"
) -> ty.Union[np.ndarray, ty.Tuple[np.ndarray, np.ndarray]]:
    s_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    s_y = s_x.T
    img = utils.color_transformation.rgb_to_mono_channel(img, channel)

    g_x = apply_linear_filter(img, s_x, apply_clipping=False)
    g_y = apply_linear_filter(img, s_y, apply_clipping=False)
    if return_gxy:
        return g_x, g_y

    borders: np.ndarray = np.sqrt(g_x**2 + g_y**2) / np.sqrt(2)
    directions: np.ndarray = np.arctan2(g_y, g_x)

    if return_directions:
        return borders, directions
    else:
        return borders


def apply_roberts_filter(img: np.ndarray) -> np.ndarray:
    g_x = np.array([[1, 0], [0, -1]])
    g_y = np.array([[0, 1], [-1, 0]])
    img = utils.color_transformation.rgb_to_mono_channel(img, "gray") / 255

    return (np.sqrt(
        apply_linear_filter(img, g_x, img.dtype, False)**2 +
        apply_linear_filter(img, g_y, img.dtype, False)**2
    ) * 255 / np.sqrt(2)).astype(np.uint8)


def apply_prewitt_filter(img: np.ndarray) -> np.ndarray:
    g_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    g_y = g_x.T
    img = utils.color_transformation.rgb_to_mono_channel(img, "hsv_value") / 255

    return (np.sqrt(
        apply_linear_filter(img, g_x, img.dtype, False)**2 +
        apply_linear_filter(img, g_y, img.dtype, False)**2
    ) * 255 / np.sqrt(2)).astype(np.uint8)
