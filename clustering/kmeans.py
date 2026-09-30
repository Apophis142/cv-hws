import typing as ty

import numpy as np

import utils.metrics


def kmeans_clustering(
        array: np.array,
        k: int,
        max_iters: int = 1000,
        tol: float = 1e-4,
        metric: ty.Literal['euclid', 'euclid2', 'manhattan', 'chebyshev', 'cosine'] = "euclid",
        random_state: int = 42,
):
    assert isinstance(k, int) and 0 < k <= array.shape[0]

    match metric:
        case "euclid" | "euclid2":
            metric_function = utils.metrics.euclid2
        case "manhattan":
            metric_function = utils.metrics.manhattan
        case "chebyshev":
            metric_function = utils.metrics.chebyshev
        case "cosine":
            metric_function = utils.metrics.cosine
        case _:
            raise ValueError

    n = array.shape[0]

    cluster_labels = np.arange(k)
    np.random.seed(random_state)
    random_idx = np.random.choice(n, k, replace=False)
    centroids = array[random_idx]

    labels = np.zeros(n)

    for _ in range(max_iters):
        distances = metric_function(centroids, array)
        labels = np.argmin(distances, axis=0)

        indicator = labels[:, None].__eq__(cluster_labels)
        sums = indicator.sum(axis=0)
        mask = sums.__ne__(0)
        new_centroids = np.empty_like(centroids)
        new_centroids[mask] = (indicator.T @ array)[mask, ...] / sums[mask, None]
        if (empty_mask := ~mask).any():
            new_centroids[empty_mask] = array[np.random.choice(n, empty_mask.sum(), replace=False)]

        if ((centroids - (centroids := new_centroids)) ** 2).sum() < tol:
            distances = metric_function(centroids, array)
            labels = np.argmin(distances, axis=0)
            break

    return labels
