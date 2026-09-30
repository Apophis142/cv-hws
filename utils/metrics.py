import typing as ty

import numba as nb
import numpy as np


@nb.njit(cache=True, inline="always")
def euclid2(x: np.ndarray, y: ty.Optional[np.ndarray] = None) -> np.ndarray:
    if y is None:
        y = x
    return np.sum((x[:, None, ...] - y[None, :, ...]) ** 2, axis=-1)


@nb.njit(cache=True, inline="always")
def euclid(x: np.ndarray, y: ty.Optional[np.ndarray] = None) -> np.ndarray:
    if y is None:
        y = x
    return np.sqrt(euclid2(x, y))


@nb.njit(cache=True, inline="always")
def cosine(x: np.ndarray, y: ty.Optional[np.ndarray] = None) -> np.ndarray:
    len_x = np.sqrt(np.sum(x**2, axis=-1))
    len_x[len_x == 0] = 1

    if y is None:
        y = x
        len_y = len_x
    else:
        len_y = np.sqrt(np.sum(y ** 2, axis=-1))
        len_y[len_y == 0] = 1
    return 1 - np.sum(x[:, None, ...] * y[None, :, ...], axis=-1) / (len_x[..., None] * len_y[None, ...])


@nb.njit(cache=True, inline="always")
def _max_last_axis(a):
    s = a.shape
    lead = 1
    for d in s[:-1]:
        lead *= d
    n = s[-1]
    flat = a.reshape(lead, n)
    out = np.empty(lead, dtype=a.dtype)
    for i in range(lead):
        m = flat[i, 0]
        for k in range(1, n):
            v = flat[i, k]
            if v > m:
                m = v
        out[i] = m
    return out.reshape(s[:-1])


@nb.njit(cache=True, inline="always")
def chebyshev(x: np.ndarray, y: ty.Optional[np.ndarray] = None) -> np.ndarray:
    if y is None:
        y = x
    diff = np.abs(x[:, None, ...] - y[None, :, ...])
    return _max_last_axis(diff)


@nb.njit(cache=True, inline="always")
def manhattan(x: np.ndarray, y: ty.Optional[np.ndarray] = None) -> np.ndarray:
    if y is None:
        y = x
    return np.abs(x[:, None, ...] - y[None, :, ...]).sum(axis=-1)


if __name__ == "__main__":
    x = np.random.randn(5, 3)
    y = np.random.randn(4, 3)
    ref = np.abs(x[:, None, :] - y[None, :, :]).max(axis=-1)
    got = chebyshev.py_func(x, y)
    assert np.allclose(ref, got)
