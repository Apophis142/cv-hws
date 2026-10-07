from functools import wraps
import inspect
import typing as ty
import warnings
import numpy as np

from .image_range import normalize_input, preserve_format

channelT = ty.Literal["gray", "hsv_saturation", "lab_luminance", "hsv_value"]

def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return img[..., ::-1]


rgb2bgr = bgr2rgb


@normalize_input(auto=True)
def rgb2gray(img: np.ndarray) -> np.ndarray:
    transformation_matrix = np.array([0.299, 0.587, 0.114])

    res = np.einsum('i,...i->...', transformation_matrix, img)
    return res


def rgb_to_mono_channel(
        img: np.ndarray,
        channel: channelT = "gray",
) -> np.ndarray:
    if channel == "gray":
        mono_unit = rgb2gray(img)
    elif channel == "hsv_saturation":
        mono_unit = rgb2hsv(img)[..., 1]
    elif channel == "hsv_value":
        mono_unit = rgb2hsv(img)[..., 2]
    elif channel == "lab_luminance":
        luminance = rgb2lab(img)[..., 0]
        mono_unit = luminance.clip(0.0, 100.0) / 100.0
    else:
        warnings.warn("Couldn't convert to mono channel: image have just one channel")
        return img

    return preserve_format(mono_unit, img, unit_max=1.0)


def mono_channel_input(default_channel: channelT = "gray"):
    def decorator(func):
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        params.append(inspect.Parameter(
            "channel",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=default_channel,
            annotation=channelT,
        ))

        @wraps(func)
        def wrapper(img, *args, **kwargs):
            if img.ndim != 3:
                raise ValueError(f"mono_channel_input expects HxWxC RGB image, got ndim={img.ndim}")
            channel = kwargs.pop("channel", default_channel)
            img = rgb_to_mono_channel(img, channel)
            return func(img, *args, **kwargs)

        wrapper.__signature__ = sig.replace(parameters=params)
        return wrapper
    return decorator


@normalize_input()
def rgb2hsv(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    v = np.max(img, axis=-1)
    m = np.min(img, axis=-1)
    delta = v - m

    s = np.zeros_like(delta).astype(np.float64)
    np.divide(delta, v, out=s, where=v != 0)

    mask = delta != 0
    r_big = (v == r) & mask
    g_big = (v == g) & mask
    b_big = (v == b) & mask

    h = np.zeros_like(delta).astype(np.float64)
    h[r_big] = (g[r_big] - b[r_big]) / delta[r_big] % 6.
    h[g_big] = 2. + (b[g_big] - r[g_big]) / delta[g_big]
    h[b_big] = 4. + (r[b_big] - g[b_big]) / delta[b_big]
    h *= 60.

    return np.stack([h, s, v], axis=-1)


def hsv2rgb(img: np.ndarray) -> np.ndarray:
    h, s, v = img[..., 0] % 360., img[..., 1], img[..., 2]

    c = v * s
    x = c * (1. - np.abs((h / 60) % 2. - 1.))
    m = v - c

    mask1 = (0 <= h) & (h < 60)
    mask2 = (60 <= h) & (h < 120)
    mask3 = (120 <= h) & (h < 180)
    mask4 = (180 <= h) & (h < 240)
    mask5 = (240 <= h) & (h < 300)
    mask6 = (300 <= h) & (h < 360)

    res = np.zeros_like(img)
    res[..., 0] = c * (mask1 | mask6) + x * (mask2 | mask5)
    res[..., 1] = c * (mask2 | mask3) + x * (mask1 | mask4)
    res[..., 2] = c * (mask4 | mask5) + x * (mask3 | mask6)
    res += m[..., None]

    return res


@normalize_input()
def rgb2hsl(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    v = np.max(img, axis=-1)
    m = np.min(img, axis=-1)
    delta = v - m

    l = (v + m) / 2.

    mask = delta != 0.

    s = np.zeros_like(delta)
    np.divide(delta, 1. - np.abs(2 * l - 1), out=s, where=mask)

    r_big = (v == r) & mask
    g_big = (v == g) & mask
    b_big = (v == b) & mask

    h = np.zeros_like(delta)
    h[r_big] = (g[r_big] - b[r_big]) / delta[r_big] % 6.
    h[g_big] = 2. + (b[g_big] - r[g_big]) / delta[g_big]
    h[b_big] = 4. + (r[b_big] - g[b_big]) / delta[b_big]
    h *= 60

    return np.stack([h, s, l], axis=-1)


def hsl2rgb(img: np.ndarray) -> np.ndarray:
    h, s, l = img[..., 0] % 360., img[..., 1], img[..., 2]

    c = (1. - np.abs(2. * l - 1.)) * s
    x = c * (1. - np.abs((h / 60) % 2. - 1.))
    m = l - c / 2.

    mask1 = (0 <= h) & (h < 60)
    mask2 = (60 <= h) & (h < 120)
    mask3 = (120 <= h) & (h < 180)
    mask4 = (180 <= h) & (h < 240)
    mask5 = (240 <= h) & (h < 300)
    mask6 = (300 <= h) & (h < 360)

    res = np.zeros_like(img)
    res[..., 0] = c * (mask1 | mask6) + x * (mask2 | mask5)
    res[..., 1] = c * (mask2 | mask3) + x * (mask1 | mask4)
    res[..., 2] = c * (mask4 | mask5) + x * (mask3 | mask6)
    res += m[..., None]

    return res


@normalize_input(auto=True)
def srgb2linear_rgb(img: np.ndarray) -> np.ndarray:
    mask = img <= .04045
    res = np.empty_like(img)

    res[mask] = img[mask] / 12.92
    res[~mask] = ((img[~mask] + 0.055) / 1.055)**2.4

    return res


@normalize_input(auto=True)
def linear_rgb2srgb(img: np.ndarray) -> np.ndarray:
    mask = img <= .0031308
    res = np.empty_like(img)

    res[mask] = img[mask] * 12.92
    res[~mask] = 1.055 * img[~mask] ** (1/2.4) - 0.055

    return res


@normalize_input(auto=True)
def rgb2cmy(img: np.ndarray) -> np.ndarray:
    res = 1. - img
    return res


def cmy2rgb(img: np.ndarray) -> np.ndarray:
    res = 1. - img
    return res


def cmy2cmyk(img: np.ndarray) -> np.ndarray:
    k = np.min(img, axis=-1)

    res = np.zeros_like(img)
    np.divide(img - k, 1 - k, where=k != 1., out=res)

    return np.concatenate([res, k], axis=-1)


def cmyk2cmy(img: np.ndarray) -> np.ndarray:
    k = img[..., 3]
    res = img[..., 0:3] * (1 - k) + k

    return res


def _rgb2xyz(img: np.ndarray) -> np.ndarray:
    transformation_matrix = np.array(
        [
            [.4124564, .3575761, .1804375],
            [.2126729, .7151522, .0721750],
            [.0193339, .1191920, .9503041],
        ]
    )

    res = np.einsum('ij,...j->...i', transformation_matrix, img)
    return res


def _xyz2rgb(img: np.ndarray) -> np.ndarray:
    transformation_matrix = np.array(
        [
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ]
    )

    res = np.einsum('ij,...j->...i', transformation_matrix, img)
    return res


def _xyz2lab(img: np.ndarray, white=np.array([0.95047, 1.0, 1.08883])) -> np.ndarray:
    delta = 6 / 29

    _img = img / white

    mask = _img > delta ** 3
    _img[mask] = _img[mask] ** (1. / 3.)
    _img[~mask] = _img[~mask] / (3 * delta**2) + 4 / 29

    res = np.zeros_like(img)
    res[..., 0] = 116 * _img[..., 1] - 16
    res[..., 1] = 500 * (_img[..., 0] - _img[..., 1])
    res[..., 2] = 200 * (_img[..., 1] - _img[..., 2])

    return res


def _lab2xyz(img: np.ndarray, white=np.array([0.95047, 1.0, 1.08883])) -> np.ndarray:
    y = (img[..., 0] + 16) / 116
    x = y + img[..., 1] / 500
    z = y - img[..., 2] / 200

    delta = 6 / 29

    res = np.stack([x, y, z], axis=-1)
    mask = res > delta

    res[mask] = res[mask] ** 3
    res[~mask] = 3 * delta**2 * (res[~mask] - 4 / 29)

    res = white * res

    return res


def lab2lch(img: np.ndarray) -> np.ndarray:
    l = img[..., 0]
    a = img[..., 1]
    b = img[..., 2]

    c = np.sqrt(a ** 2 + b ** 2)
    h = np.mod(np.degrees(np.arctan2(b, a)), 360.0)

    return np.stack([l, c, h], axis=-1)


def rgb2lab(img: np.ndarray, white=np.array([0.95047, 1.0, 1.08883])) -> np.ndarray:
    return _xyz2lab(_rgb2xyz(srgb2linear_rgb(img)), white=white)


def lab2rgb(img: np.ndarray, white=np.array([0.95047, 1.0, 1.08883])) -> np.ndarray:
    return linear_rgb2srgb(_xyz2rgb(_lab2xyz(img, white=white)))


@normalize_input(auto=True)
def rgb2lms(img: np.ndarray) -> np.ndarray:
    transformation_matrix = np.array(
        [
            [0.4122, 0.5363, 0.0514],
            [0.2119, 0.6807, 0.1074],
            [0.0883, 0.2817, 0.6300],
        ]
    )

    res = np.einsum('ij,...j->...i', transformation_matrix, srgb2linear_rgb(img))
    return res


def _xyz2lms(img: np.ndarray):
    transformation_matrix = np.array(
        [
            [0.8190, 0.3619, -0.1289],
            [0.0330, 0.9293,  0.0361],
            [0.0482, 0.2642,  0.6335],
        ]
    )

    res = np.einsum('ij,...j->...i', transformation_matrix, img)
    return res


def _lms2xyz(img: np.ndarray):
    transformation_matrix = np.array(
        [
            [ 1.2269, -0.5578,  0.2814],
            [-0.0406,  1.1123, -0.0716],
            [-0.0764, -0.4214,  1.5870],
        ]
    )

    res = np.einsum('ij,...j->...i', transformation_matrix, img)
    return res


def _lms2oklab(img: np.ndarray) -> np.ndarray:
    transformation_matrix = np.array(
        [
            [0.2105,  0.7936, -0.0041],
            [1.9780, -2.4286,  0.4506],
            [0.0259,  0.7828, -0.8087],
        ]
    )

    res = np.einsum('ij,...j->...i', transformation_matrix, np.cbrt(img))
    return res


def rgb2oklab(img: np.ndarray) -> np.ndarray:
    return _lms2oklab(rgb2lms(img))
