from dataclasses import dataclass, field
import json
import typing as ty


DEFAULT_SIGNATURES: ty.Dict[str, ty.List[ty.Dict[str, ty.Any]]] = {
    "red": [
        {"name": "maroon",      "L": 28, "a": 45, "b": 30},
        {"name": "brick",       "L": 38, "a": 50, "b": 35},
        {"name": "pure_red",    "L": 53, "a": 80, "b": 67},
        {"name": "scarlet",     "L": 50, "a": 65, "b": 50},
        {"name": "light_red",   "L": 68, "a": 45, "b": 30},
        {"name": "pinkish_red", "L": 75, "a": 35, "b": 15},
    ],
    "orange": [
        {"name": "dark_orange",   "L": 55, "a": 30, "b": 50},
        {"name": "burnt_orange",  "L": 50, "a": 35, "b": 55},
        {"name": "pure_orange",   "L": 65, "a": 40, "b": 65},
        {"name": "bright_orange", "L": 70, "a": 30, "b": 75},
        {"name": "light_orange",  "L": 80, "a": 20, "b": 65},
    ],
    "yellow": [
        {"name": "amber",         "L": 75, "a":  5, "b": 75},
        {"name": "golden_yellow", "L": 85, "a": -5, "b": 85},
        {"name": "pure_yellow",   "L": 95, "a": -10, "b": 90},
        {"name": "pale_yellow",   "L": 95, "a": -5, "b": 40},
    ],
    "green": [
        {"name": "dark_green",   "L": 35, "a": -40, "b": 35},
        {"name": "forest_green", "L": 45, "a": -45, "b": 40},
        {"name": "grass_green",  "L": 70, "a": -55, "b": 60},
        {"name": "pure_green",   "L": 85, "a": -80, "b": 80},
        {"name": "lime",         "L": 85, "a": -50, "b": 70},
        {"name": "light_green",  "L": 85, "a": -35, "b": 45},
        {"name": "mint",         "L": 88, "a": -35, "b": 15},
    ],
    "blue": [
        {"name": "dark_blue", "L": 30, "a": 20,  "b": -45},
        {"name": "pure_blue", "L": 32, "a": 79,  "b": -108},
        {"name": "sky_blue",  "L": 70, "a": -10, "b": -40},
        {"name": "cyan",      "L": 80, "a": -30, "b": -15},
    ],
    "achromatic": [
        {"name": "black",      "L": 10, "a": 0, "b": 0},
        {"name": "dark_gray",  "L": 30, "a": 0, "b": 0},
        {"name": "mid_gray",   "L": 50, "a": 0, "b": 0},
        {"name": "light_gray", "L": 70, "a": 0, "b": 0},
        {"name": "white",      "L": 95, "a": 0, "b": 0},
    ],
}

DEFAULT_HUE_BOUNDARIES: ty.Dict[str, float] = {
    "red_orange": 45.0,
    "orange_yellow": 72.0,
    "yellow_green": 105.0,
}

DEFAULT_BLUE_HUE_RANGE: ty.Tuple[float, float] = (180.0, 320.0)
DEFAULT_ACHROMATIC_CHROMA_THRESHOLD: float = 15.0
DEFAULT_DISPUTED_HUE_MARGIN: float = 8.0

BACKGROUND_COLORS: ty.Tuple[str, ...] = ("blue", "achromatic")
TARGET_COLORS: ty.Tuple[str, ...] = ("red", "orange", "yellow", "green")


@dataclass
class ColorClassifierConfig:
    signatures: ty.Dict[str, ty.List[ty.Dict[str, ty.Any]]] = field(
        default_factory=lambda: DEFAULT_SIGNATURES
    )
    hue_boundaries: ty.Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_HUE_BOUNDARIES)
    )
    blue_hue_range: ty.Tuple[float, float] = DEFAULT_BLUE_HUE_RANGE
    achromatic_chroma_threshold: float = DEFAULT_ACHROMATIC_CHROMA_THRESHOLD
    disputed_hue_margin: float = DEFAULT_DISPUTED_HUE_MARGIN

    background_colors: ty.Tuple[str, ...] = BACKGROUND_COLORS
    target_colors: ty.Tuple[str, ...] = TARGET_COLORS

    fine_metric: ty.Literal["euclid", "ciede2000"] = "ciede2000"
    use_hue_prefilter: bool = True

    tau: float = 0.5
    margin: float = 0.1
    min_valid_ratio: float = 0.0
    min_cluster_area_ratio: float = 0.0
    min_fill_ratio: float = 0.0
    max_spatial_spread: float = 500.0

    initial_clusters: int = 10
    max_clusters: int = 25
    cluster_step: int = 5

    final_class_colors = {"red", "yellow", "green"}


def config_from_json(path: str) -> ColorClassifierConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ColorClassifierConfig(
        signatures=data.get("color_signatures", DEFAULT_SIGNATURES),
        hue_boundaries=data.get("hue_boundaries_deg", dict(DEFAULT_HUE_BOUNDARIES)),
        achromatic_chroma_threshold=float(
            data.get("achromatic_chroma_threshold", DEFAULT_ACHROMATIC_CHROMA_THRESHOLD)
        ),
        blue_hue_range=tuple(data.get("blue_hue_range", DEFAULT_BLUE_HUE_RANGE)),
        disputed_hue_margin=float(
            data.get("disputed_hue_margin", DEFAULT_DISPUTED_HUE_MARGIN)
        ),
        fine_metric=data.get("fine_metric", "ciede2000"),
        use_hue_prefilter=bool(data.get("use_hue_prefilter", True)),
        tau=float(data.get("tau", 0.5)),
        margin=float(data.get("margin", 0.1)),
        min_valid_ratio=float(data.get("min_valid_ratio", 0.0)),
    )
