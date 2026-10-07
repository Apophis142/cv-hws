from functools import wraps
import typing as ty

import numpy as np
from .color_transformation import mono_channel_input
from .image_range import channel_range, preserve_format


_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
_SOBEL_Y = _SOBEL_X.T

_ROBERTS_X = np.array([[1, 0], [0, -1]])
_ROBERTS_Y = np.array([[0, 1], [-1, 0]])

_PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
_PREWITT_Y = _PREWITT_X.T

_SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
_SCHARR_Y = _SCHARR_X.T


def kernel_positive_sum(kernel: np.ndarray) -> float:
    """
    Сумма положительных коэффициентов ядра — максимум отклика при входе 0..1.
    Используется для вычисления теоретического максимума магнитуды.
    """
    return float(kernel[kernel > 0].sum())


def gradient_max_value(
        kernel_x: np.ndarray,
        kernel_y: np.ndarray,
        channel_max: float,
        norm: float = np.sqrt(2),
) -> float:
    """
    Теоретический максимум sqrt(gx^2 + gy^2) / norm
    для изображения в диапазоне [0, channel_max].
    """
    bx = channel_max * kernel_positive_sum(kernel_x)
    by = channel_max * kernel_positive_sum(kernel_y)
    return float(np.hypot(bx, by) / norm)


def match_input_format(max_value_of: ty.Callable[[float], float]) -> ty.Callable:
    """
    Приводит выход фильтра к dtype и масштабу входного изображения.

    Предполагается, что выход фильтра — неотрицательная величина
    (магнитуда градиента). Для знаковых фильтров (Лаплас и т.п.)
    нужен отдельный декоратор с явной семантикой знака.

    Параметры
    ---------
    max_value_of : callable
        Функция от channel_max (верхней границы шкалы входа),
        возвращающая теоретический максимум сырой магнитуды фильтра
        на изображении [0, channel_max].

    Поведение
    ---------
    1. Запоминает dtype и масштаб входного изображения.
    2. Вызывает исходный фильтр.
    3. Делит результат на теоретический максимум -> [0, 1].
    4. Возвращает результат в исходный диапазон и dtype.
    """
    def decorator(func: ty.Callable) -> ty.Callable:
        @wraps(func)
        def wrapper(img: np.ndarray, *args, **kwargs):
            res = func(img, *args, **kwargs)

            if not isinstance(res, np.ndarray):
                return res

            max_value = max_value_of(channel_range(img)[1])
            return preserve_format(res, img, unit_max=max_value)

        return wrapper
    return decorator


def generate_gauss_kernel(win_size: int, dim: int, normalize=True, sigma: ty.Optional[float]=None) -> np.ndarray:
    r = win_size >> 1
    if sigma is None:
        sigma = 0.3 * (r - 1) + 0.8

    row = np.arange(-r, r + 1, 1)**2
    sq_dist = sum(np.ix_(*([row] * dim)))
    dist = np.exp(-sq_dist / (2 * sigma ** 2))
    return dist / dist.sum() if normalize else dist


def window_mean_filter(
        img: np.ndarray,
        win_size: int,
) -> np.ndarray:
    r = win_size >> 1
    h, w, *_ = img.shape

    integral_img = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)

    integral_img = np.pad(integral_img, pad_width={0: (1, 0), 1: (1, 0)}, mode="constant", constant_values=0)
    y, x = np.ogrid[:h, :w]

    y1, x1 = np.clip(y - r, 0, h - 1), np.clip(x - r, 0, w - 1)
    y2, x2 = np.clip(y + r, 0, h - 1), np.clip(x + r, 0, w - 1)

    window_sum = (
            integral_img[y2 + 1, x2 + 1] +
            integral_img[y1, x1] -
            integral_img[y2 + 1, x1] -
            integral_img[y1, x2 + 1]
    )

    return window_sum / ((y2 - y1 + 1) * (x2 - x1 + 1)).reshape((h, w) + (1,) * (img.ndim - 2))


def apply_linear_filter(
        img: np.ndarray,
        matrix_filter: np.ndarray,
) -> np.ndarray:
    h, w = matrix_filter.shape
    r, c = h >> 1, w >> 1

    if img.ndim not in (2, 3):
        raise ValueError

    mod_img = np.pad(img, {0: (h - r - 1, r), 1: (w - c - 1, c)}, mode="edge")

    return np.tensordot(
        np.lib.stride_tricks.sliding_window_view(mod_img, window_shape=matrix_filter.shape, axis=(0, 1)),
        matrix_filter,
        axes=([-2, -1], [0, 1])
    )


def apply_gauss_filter(
        img: np.ndarray,
        win_size: int,
        sigma: ty.Optional[float] = None,
):
    gauss_filter = generate_gauss_kernel(win_size, 2, True, sigma)
    return apply_linear_filter(img, gauss_filter)


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
        return np.median(mod_img, axis=(-1, -2)).astype(img.dtype)
    else:
        raise ValueError(f"Unknown non-linear filter: {mode}")


def opening_filter(img: np.ndarray, r: int=1) -> np.ndarray:
    d = (r << 1) + 1
    return apply_non_linear_filter(apply_non_linear_filter(img, "min", d), "max", d)


def closing_filter(img: np.ndarray, r: int=1) -> np.ndarray:
    d = (r << 1) + 1
    return apply_non_linear_filter(apply_non_linear_filter(img, "max", d), "min", d)


@match_input_format(lambda M: gradient_max_value(_SOBEL_X, _SOBEL_Y, M))
@mono_channel_input("gray")
def apply_sobel_filter(
        img: np.ndarray,
        return_directions=False,
        return_gxy=False,
) -> ty.Union[np.ndarray, ty.Tuple[np.ndarray, np.ndarray]]:

    g_x = apply_linear_filter(img, _SOBEL_X)
    g_y = apply_linear_filter(img, _SOBEL_Y)
    if return_gxy:
        return g_x, g_y

    borders: np.ndarray = np.sqrt(g_x**2 + g_y**2) / np.sqrt(2)

    if return_directions:
        directions: np.ndarray = np.arctan2(g_y, g_x)
        return borders, directions
    else:
        return borders


@match_input_format(lambda M: gradient_max_value(_ROBERTS_X, _ROBERTS_Y, M))
@mono_channel_input("gray")
def apply_roberts_filter(img: np.ndarray) -> np.ndarray:
    return (np.sqrt(
        apply_linear_filter(img, _ROBERTS_X)**2 +
        apply_linear_filter(img, _ROBERTS_Y)**2
    ) / np.sqrt(2))


@match_input_format(lambda M: gradient_max_value(_PREWITT_X, _PREWITT_Y, M))
@mono_channel_input("gray")
def apply_prewitt_filter(img: np.ndarray) -> np.ndarray:
    return (np.sqrt(
        apply_linear_filter(img, _PREWITT_X)**2 +
        apply_linear_filter(img, _PREWITT_Y)**2
    ) / np.sqrt(2))


@match_input_format(lambda M: gradient_max_value(_SCHARR_X, _SCHARR_Y, M))
@mono_channel_input("gray")
def scharr_operator(img: np.ndarray) -> np.ndarray:
    return (np.sqrt(
        apply_linear_filter(img, _SCHARR_X) ** 2 +
        apply_linear_filter(img, _SCHARR_Y) ** 2
    ) / np.sqrt(2))
