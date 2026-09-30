import typing as ty

import numpy as np

import utils


def build_signature_arrays(
    signatures: ty.Dict[str, ty.List[ty.Dict[str, ty.Any]]],
    colors: ty.Iterable[str],
) -> ty.Dict[str, np.ndarray]:
    result: ty.Dict[str, np.ndarray] = {}
    for color in colors:
        sigs = signatures.get(color)
        if not sigs:
            continue
        result[color] = np.asarray(
            [[s["L"], s["a"], s["b"]] for s in sigs],
            dtype=np.float64,
        )
    return result


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Парные расстояния CIEDE2000 между двумя наборами Lab-цветов.

    Parameters
    ----------
    lab1 : (N, 3) float
    lab2 : (M, 3) float

    Returns
    -------
    (N, M) float
    """
    L1 = lab1[:, 0][:, None]
    a1 = lab1[:, 1][:, None]
    b1 = lab1[:, 2][:, None]
    L2 = lab2[:, 0][None, :]
    a2 = lab2[:, 1][None, :]
    b2 = lab2[:, 2][None, :]

    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    C_bar = 0.5 * (C1 + C2)

    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where((C1p * C2p) == 0.0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) * 0.5)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    h_sum = h1p + h2p
    h_abs_diff = np.abs(h1p - h2p)

    hp_bar = np.where(
        C1p * C2p == 0.0,
        h_sum,
        np.where(
            h_abs_diff <= 180.0,
            h_sum * 0.5,
            np.where(
                h_sum < 360.0,
                (h_sum + 360.0) * 0.5,
                (h_sum - 360.0) * 0.5,
            ),
        ),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )

    dTheta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar ** 7
    RC = 2.0 * np.sqrt(Cp_bar7 / (Cp_bar7 + 25.0 ** 7))
    SL = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_bar
    SH = 1.0 + 0.015 * Cp_bar * T
    RT = -np.sin(np.radians(2.0 * dTheta)) * RC

    return np.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )


def pairwise_color_distance(
    pixels: np.ndarray,
    signatures: np.ndarray,
    metric: ty.Literal["euclid", "ciede2000"] = "ciede2000",
) -> np.ndarray:
    if metric == "ciede2000":
        return ciede2000(pixels, signatures)
    if metric == "euclid":
        return utils.metrics.euclid(pixels.astype(np.float64), signatures.astype(np.float64))
    raise ValueError(f"Unknown metric: {metric}")
