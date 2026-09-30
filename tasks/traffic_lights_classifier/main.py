import argparse
import os
import typing as ty

import cv2
import numpy as np

import utils
from clustering.kmeans import kmeans_clustering as kmeans
from . import ColorClassifierConfig, classify_pixels, vote_cluster_colors, build_signature_arrays, config_from_json
from . import decide_image_class, ImageDecision, UNDEFINED_CLASS


def classify_clusters(
        lab_pixels: np.ndarray,
        labels: np.ndarray,
        config: ty.Optional[ColorClassifierConfig] = None,
) -> ty.Dict[int, ty.Dict[str, ty.Any]]:
    if config is None:
        config = ColorClassifierConfig()

    cluster_labels_flat = labels.reshape(-1)

    n_clusters = cluster_labels_flat.max() + 1
    signature_arrays = build_signature_arrays(config.signatures, config.target_colors)
    pixel_labels, pixel_confidence = classify_pixels(lab_pixels, config, signature_arrays)
    return vote_cluster_colors(
        pixel_labels,
        pixel_confidence,
        cluster_labels_flat,
        n_clusters,
        config.target_colors,
        tau=config.tau,
        margin=config.margin,
        min_valid_ratio=config.min_valid_ratio,
        total_pixels=lab_pixels.shape[0],
        min_fill_ratio=config.min_fill_ratio,
        max_spatial_spread=config.max_spatial_spread,
    )


def clusterize_colors(img: np.array, n_clusters: int = 5) -> ty.Tuple:
    lab = utils.color_transformation.rgb2lab(utils.color_transformation.bgr2rgb(img))
    pixels = lab.reshape(-1, 3)
    labels = kmeans(
        pixels[..., 1:],
        k=n_clusters,
        max_iters=1000,
        metric="euclid",
        random_state=29092026,
    )

    return labels, pixels


def classify_image_with_retry(
    img_bgr: np.ndarray,
    config: ty.Optional[ColorClassifierConfig] = None,
) -> ty.Tuple[ImageDecision, ty.Dict, np.ndarray]:
    if config is None:
        config = ColorClassifierConfig()

    n_clusters = config.initial_clusters

    while True:
        labels, lab_pixels = clusterize_colors(img_bgr, n_clusters=n_clusters)
        labels_2d = labels.reshape(img_bgr.shape[:-1])

        cluster_results = classify_clusters(lab_pixels, labels_2d, config)
        decision = decide_image_class(cluster_results, config)

        if not decision.is_retry:
            return decision, cluster_results, labels_2d

        if n_clusters >= config.max_clusters:
            return (
                ImageDecision(
                    UNDEFINED_CLASS, "max_clusters_reached",
                    details={"n_clusters": n_clusters},
                ),
                cluster_results,
                labels_2d,
            )

        n_clusters = min(n_clusters + config.cluster_step, config.max_clusters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/traffic_lights/images/", help="Directory with images to classify")
    parser.add_argument("--output", default="tasks/traffic_lights_classifier/answers/", help="Directory to write answers files")
    parser.add_argument("--config", default=None, help="Path to .json file with signatures for colors and thresholds")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show details on decisions")

    args = parser.parse_args()

    directory = args.input
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Не директория: {directory}")
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    os.makedirs(args.output, exist_ok=True)

    config = config_from_json(args.config) if args.config else ColorClassifierConfig()
    config.initial_clusters = args.clusters

    with (open(args.output + "red.txt", "w", encoding="utf-8") as red,
          open(args.output + "yellow.txt", "w", encoding="utf-8") as yellow,
          open(args.output + "green.txt", "w", encoding="utf-8") as green):
        files = {"red": red, "yellow": yellow, "green": green}

        for filename in sorted(os.listdir(directory)):
            if os.path.splitext(filename)[1].lower() not in valid_ext:
                continue

            path = os.path.join(directory, filename)
            img = cv2.imread(path)

            if img is None:
                print(f"[WARN] Не удалось прочитать: {path}")
                continue

            decision, cluster_results, labels_2d = classify_image_with_retry(img, config)
            if args.verbose:
                print(f"\n{filename}: class={decision.class_color}  reason={decision.reason}")
                for cid, info in cluster_results.items():
                    spread = info["compactness"]["spatial_spread"] if info.get("compactness") else None
                    print(
                        f"  cluster {cid:>2}: color={info['color']:<10} "
                        f"ratio={info['ratio']:.2f} "
                        f"spread={spread:.3f} area={info['area_ratio']:.3f}"
                    )
            print(filename, file=files.get(decision.class_color, yellow))


if __name__ == "__main__":
    main()
