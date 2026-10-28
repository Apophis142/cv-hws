from dataclasses import dataclass


@dataclass(frozen=True)
class StitchingConfig:
    n_features: int = 4000
    ratio_threshold: float = 0.75
    ransac_threshold: float = 5.0
    min_matches: int = 10
    min_inliers: int = 8
    min_inlier_ratio: float = 0.3
    max_canvas_pixels: int = 40_000_000
