import cv2
import numpy as np

from .config import StitchingConfig


def extract_features(images: list[np.ndarray], config: StitchingConfig) -> list:
    orb = cv2.ORB_create(nfeatures=config.n_features)
    return [
        orb.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        for image in images
    ]


def estimate_pairwise_homographies(features: list, config: StitchingConfig) -> list[np.ndarray]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    homographies = []
    for i in range(len(features) - 1):
        previous_points, previous_descriptors = features[i]
        next_points, next_descriptors = features[i + 1]
        pair = f"Кадры №{i + 1} и №{i + 2}"
        if previous_descriptors is None or next_descriptors is None:
            raise ValueError(f"{pair}: ORB не обнаружил достаточно признаков")

        matches = matcher.knnMatch(next_descriptors, previous_descriptors, k=2)
        good = [
            candidates[0] for candidates in matches
            if len(candidates) == 2
            and candidates[0].distance < config.ratio_threshold * candidates[1].distance
        ]
        if len(good) < config.min_matches:
            raise ValueError(f"{pair}: недостаточно совпадений ({len(good)})")

        source = np.float32([next_points[m.queryIdx].pt for m in good])
        target = np.float32([previous_points[m.trainIdx].pt for m in good])
        homography, inlier_mask = cv2.findHomography(
            source, target, cv2.RANSAC, config.ransac_threshold,
        )
        if homography is None or inlier_mask is None or not np.isfinite(homography).all():
            raise ValueError(f"{pair}: не удалось оценить гомографию")
        inliers = int(inlier_mask.sum())
        if inliers < config.min_inliers or inliers / len(good) < config.min_inlier_ratio:
            raise ValueError(f"{pair}: недостаточно согласованных точек ({inliers}/{len(good)})")
        homographies.append(homography)
    return homographies


def compose_transforms(pairwise_homographies: list[np.ndarray]) -> list[np.ndarray]:
    to_first = [np.eye(3)]
    for homography in pairwise_homographies:
        to_first.append(to_first[-1] @ homography)
    try:
        views = [np.linalg.inv(matrix) for matrix in to_first]
        views = [matrix / np.cbrt(np.linalg.det(matrix)) for matrix in views]
        first_to_reference = np.mean(views, axis=0)
        transforms = [first_to_reference @ matrix for matrix in to_first]
        if any(not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3
               for matrix in transforms):
            raise ValueError("Цепочка гомографий вырождена")
    except np.linalg.LinAlgError as exc:
        raise ValueError("Цепочка гомографий вырождена") from exc
    return transforms


def render_panorama(
    images: list[np.ndarray], transforms: list[np.ndarray], config: StitchingConfig,
) -> np.ndarray:
    projected_corners = []
    for image, transform in zip(images, transforms):
        height, width = image.shape[:2]
        corners = np.float64([[0, 0], [width, 0], [width, height], [0, height]])
        homogeneous = np.column_stack((corners, np.ones(4))) @ transform.T
        denominator = homogeneous[:, 2]
        if not np.isfinite(homogeneous).all() or not (
            np.all(denominator > 1e-8) or np.all(denominator < -1e-8)
        ):
            raise ValueError("Кадр пересекает горизонт общей проекции")
        projected_corners.append(homogeneous[:, :2] / denominator[:, None])

    corners = np.concatenate(projected_corners)
    lower = np.floor(corners.min(axis=0))
    upper = np.ceil(corners.max(axis=0))
    width, height = upper - lower
    if width < 1 or height < 1 or width >= 32767 or height >= 32767 \
            or width * height > config.max_canvas_pixels:
        raise ValueError("Слишком большой холст: проверьте перекрытие и перспективу кадров")
    size = (int(width), int(height))
    translation = np.array([[1, 0, -lower[0]], [0, 1, -lower[1]], [0, 0, 1]])

    pixel_sum = np.zeros((size[1], size[0], 3), dtype=np.float32)
    weight_sum = np.zeros((size[1], size[0]), dtype=np.float32)
    for image, transform in zip(images, transforms):
        matrix = translation @ transform
        pixel_sum += cv2.warpPerspective(image.astype(np.float32), matrix, size)
        weight_sum += cv2.warpPerspective(
            np.ones(image.shape[:2], dtype=np.float32), matrix, size,
        )

    covered = weight_sum > 1e-6
    np.divide(pixel_sum, weight_sum[..., None], out=pixel_sum, where=covered[..., None])
    panorama = np.clip(np.rint(pixel_sum), 0, 255).astype(np.uint8)
    rows = np.flatnonzero(covered.any(axis=1))
    columns = np.flatnonzero(covered.any(axis=0))
    if not len(rows) or not len(columns):
        raise ValueError("Панорама не содержит видимых пикселей")
    return panorama[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]


def stitch_images(
    images: list[np.ndarray], config: StitchingConfig | None = None,
) -> np.ndarray:
    if len(images) < 2:
        raise ValueError("Для панорамы нужны хотя бы два изображения")
    for i, image in enumerate(images):
        if image is None or image.ndim != 3 or image.shape[2] != 3 \
                or image.dtype != np.uint8 or image.size == 0:
            raise ValueError(f"Кадр №{i + 1}: ожидается непустое BGR uint8 изображение")
    config = config or StitchingConfig()
    features = extract_features(images, config)
    pairwise = estimate_pairwise_homographies(features, config)
    transforms = compose_transforms(pairwise)
    return render_panorama(images, transforms, config)
