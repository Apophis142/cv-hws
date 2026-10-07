from functools import wraps
import inspect
import typing as ty
import numpy as np


def _to_unit_range(x: np.ndarray, in_range=None, auto=False):
    arr = np.asarray(x)

    if in_range is not None:
        low, high = in_range
        return (arr.astype(np.float64) - low) / (high - low)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return (arr.astype(np.float64) - info.min) / (info.max - info.min)

    if auto and arr.size and arr.max() > 1.5:
        return arr / 255.0

    return arr


def normalize_input(param="img", auto=False) -> ty.Callable:
    def decorator(func: ty.Callable[[np.ndarray], np.ndarray]):
        sig = inspect.signature(func)
        first = param or next(iter(sig.parameters))

        @wraps(func)
        def wrapper(*args, **kwargs):
            in_range = kwargs.pop("in_range", None)

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            if first in bound.arguments:
                bound.arguments[first] = _to_unit_range(bound.arguments[first], in_range=in_range, auto=auto)

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator



def channel_range(img: np.ndarray) -> ty.Tuple[float, float]:
    """
    Возвращает (min, max) шкалы входного изображения.

    - целочисленные dtype: границы берутся из np.iinfo;
    - float: если данные явно в 0..255 — считаем шкалу 0..255,
             иначе 0..1.
    """
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        return float(info.min), float(info.max)

    if img.size and float(img.max()) > 1.5:
        return 0.0, 255.0
    return 0.0, 1.0


def preserve_format(
        arr: np.ndarray,
        ref: np.ndarray,
        unit_max: float = 1.0,
) -> np.ndarray:
    """
    Привести arr, живущий в [0, unit_max], к dtype и шкале ref.

    - unit_max=1.0, если arr уже в unit range (цветовые преобразования);
    - unit_max = теоретический максимум, если arr — сырой отклик
      фильтра (градиент, магнитуда и т.п.).
    """
    ref_min, ref_max = channel_range(ref)
    res = np.clip(arr / unit_max, 0.0, 1.0)
    res = res * (ref_max - ref_min) + ref_min
    return res.astype(ref.dtype)
