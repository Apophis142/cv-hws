from dataclasses import dataclass, field


import typing as ty
from dataclasses import dataclass, field, replace


@dataclass
class HoughCircleConfig:
    threshold1: float = 0.08
    threshold2: float = 0.017
    r_min: ty.Union[int, float] = 15
    r_max: ty.Union[int, float] = 100
    r_delta: ty.Union[int, float] = 2
    ab_delta: ty.Union[int, float] = 2
    threshold: float = 0.3
    min_dist: ty.Union[int, float] = 10
    max_circles: int = 20


def _resolve_px(value: ty.Union[int, float], scale: int, minimum: int = 1) -> int:
    """
    int   -> абсолютное значение в пикселях.
    float в (0, 1) -> доля от scale (обычно min(H, W)).
    """
    if isinstance(value, float) and 0.0 < value < 1.0:
        return max(minimum, int(round(value * scale)))
    return max(minimum, int(value))


def resolve_circles_config(
    cfg: HoughCircleConfig,
    shape: ty.Tuple[int, ...],
) -> HoughCircleConfig:
    """Приводит параметры, заданные долями, к пикселям для конкретного изображения."""
    scale = min(shape[0], shape[1])
    return replace(
        cfg,
        r_min=_resolve_px(cfg.r_min, scale, minimum=1),
        r_max=_resolve_px(cfg.r_max, scale, minimum=2),
        r_delta=_resolve_px(cfg.r_delta, scale, minimum=1),
        ab_delta=_resolve_px(cfg.ab_delta, scale, minimum=1),
        min_dist=_resolve_px(cfg.min_dist, scale, minimum=0),
    )

@dataclass
class PipelineConfig:
    input_path: str
    output_dir: str
    report_format: str = "txt"
    circle: HoughCircleConfig = field(default_factory=HoughCircleConfig)
