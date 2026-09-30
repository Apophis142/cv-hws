from dataclasses import dataclass, field
import typing as ty

from .config import ColorClassifierConfig


RETRY_CLASS = "__retry__"
UNDEFINED_CLASS = "undefined"
_UNDEFINED_COLOR = "undefined"

_ORANGE_TO_YELLOW: ty.Dict[str, str] = {"orange": "yellow"}


def _to_final_class(color: str) -> str:
    return _ORANGE_TO_YELLOW.get(color, color)


@dataclass
class ImageDecision:
    """Результат определения класса изображения."""

    class_color: str
    reason: str
    chosen_cluster: ty.Optional[int] = None
    candidates: ty.List[int] = field(default_factory=list)
    details: ty.Dict[str, ty.Any] = field(default_factory=dict)

    @property
    def is_retry(self) -> bool:
        return self.class_color == RETRY_CLASS

    @property
    def is_resolved(self) -> bool:
        return self.class_color in ("red", "yellow", "green")

    def __str__(self) -> str:
        return (
            f"ImageDecision(class={self.class_color!r}, reason={self.reason!r}, "
            f"cluster={self.chosen_cluster}, candidates={self.candidates}, "
            f"details={self.details})"
        )


def _filter_undefined(
    cluster_results: ty.Dict[int, ty.Dict[str, ty.Any]],
) -> ty.Dict[int, ty.Dict[str, ty.Any]]:
    return {
        cid: info
        for cid, info in cluster_results.items()
        if info["color"] != _UNDEFINED_COLOR
    }


def _step1_single_color(
    D: ty.Dict[int, ty.Dict[str, ty.Any]],
    config: ColorClassifierConfig,
) -> ty.Optional[str]:
    """Все кластеры одного цвета, и он входит в final_class_colors."""
    if not D:
        return None
    colors = {info["color"] for info in D.values()}
    if len(colors) != 1:
        return None
    c = colors.pop()
    return c if c in config.final_class_colors else None


def _step2_orange_yellow(
    D: ty.Dict[int, ty.Dict[str, ty.Any]],
) -> ty.Optional[str]:
    """Все кластеры в {orange, yellow} (в т.ч. только orange) → yellow."""
    if not D:
        return None
    if all(info["color"] in ("orange", "yellow") for info in D.values()):
        return "yellow"
    return None


def _step3_ratio_dominant(
    D: ty.Dict[int, ty.Dict[str, ty.Any]],
) -> ty.Optional[ty.Tuple[str, int, float]]:
    if not D:
        return None

    sorted_items = sorted(D.items(), key=lambda kv: kv[1]["ratio"], reverse=True)
    top_id, top = sorted_items[0]
    top_color = top["color"]
    top_ratio = float(top["ratio"])

    if top_ratio < 0.9:
        return None

    other_ratios = [
        float(info["ratio"])
        for _, info in sorted_items
        if info["color"] != top_color
    ]
    if not other_ratios:
        return None

    second_ratio = max(other_ratios)

    if (top_ratio - second_ratio) >= 0.15:
        return top_color, top_id, top_ratio - second_ratio

    return None


def _step35_filter(
    D: ty.Dict[int, ty.Dict[str, ty.Any]],
) -> ty.Tuple[ty.Dict[int, ty.Dict[str, ty.Any]], bool]:
    """Вернуть (D_after, did_filter). did_filter=False, если триггера не было."""
    if not any(float(info["ratio"]) >= 0.9 for info in D.values()):
        return D, False
    D_new = {
        cid: info
        for cid, info in D.items()
        if float(info["ratio"]) > 0.6
    }
    return D_new, True


def _step4_spread_dominant(
    D: ty.Dict[int, ty.Dict[str, ty.Any]],
) -> ty.Optional[ty.Tuple[str, int, float, float]]:
    """
    Кластер со spatial_spread минимум в 2 раза меньше, чем у всех
    кластеров другого цвета.

    Возвращает (цвет, cluster_id, top_spread, others_min) или None.
    """
    if len(D) < 2:
        return None

    def spread(info: ty.Dict[str, ty.Any]) -> float:
        c = info.get("compactness")
        if c is None:
            raise ValueError(
                "vote_cluster_colors должен заполнять compactness для каждого кластера"
            )
        return float(c["spatial_spread"])

    sorted_items = sorted(D.items(), key=lambda kv: spread(kv[1]))
    top_id, top = sorted_items[0]
    top_color = top["color"]
    top_spread = spread(top)

    other_spreads = [
        spread(info)
        for _, info in sorted_items
        if info["color"] != top_color
    ]
    if not other_spreads:
        return None

    others_min = min(other_spreads)
    if top_spread * 2.0 <= others_min:
        return top_color, top_id, top_spread, others_min
    return None


def _step5_area_filter(
    D: ty.Dict[int, ty.Dict[str, ty.Any]],
    lo: float = 0.02,
    hi: float = 0.08,
) -> ty.Tuple[ty.Dict[int, ty.Dict[str, ty.Any]], bool]:
    """
    Оставить только кластеры с area_ratio ∈ (lo, hi).

    did_filter=True только если D изменился.
    """
    D_new = {
        cid: info
        for cid, info in D.items()
        if lo < float(info.get("area_ratio", 0.0)) < hi
    }
    return D_new, len(D_new) < len(D)


def decide_image_class(
    cluster_results: ty.Dict[int, ty.Dict[str, ty.Any]],
    config: ty.Optional[ColorClassifierConfig] = None,
) -> ImageDecision:
    """Прогнать кластеры через дерево решений."""
    if config is None:
        config = ColorClassifierConfig()

    D = _filter_undefined(cluster_results)
    if not D:
        return ImageDecision(RETRY_CLASS, "step0_empty_D")

    filtered_35 = False
    filtered_5 = False

    while True:
        c = _step1_single_color(D, config)
        if c is not None:
            return ImageDecision(
                c, "step1_single_color",
                candidates=sorted(D.keys()),
            )

        c = _step2_orange_yellow(D)
        if c is not None:
            return ImageDecision(
                c, "step2_orange_yellow",
                candidates=sorted(D.keys()),
            )

        r3 = _step3_ratio_dominant(D)
        if r3 is not None:
            raw_color, cid, gap = r3
            return ImageDecision(
                _to_final_class(raw_color), "step3_ratio_dominant",
                chosen_cluster=cid,
                candidates=sorted(D.keys()),
                details={"gap": gap, "raw_color": raw_color},
            )

        if not filtered_35:
            D_new, did_filter = _step35_filter(D)
            if did_filter:
                D = D_new
                filtered_35 = True
                continue

        r4 = _step4_spread_dominant(D)
        if r4 is not None:
            raw_color, cid, top_spread, others_min = r4
            return ImageDecision(
                _to_final_class(raw_color), "step4_spread_dominant",
                chosen_cluster=cid,
                candidates=sorted(D.keys()),
                details={
                    "top_spread": top_spread,
                    "others_min": others_min,
                    "raw_color": raw_color,
                },
            )

        if not filtered_5:
            D_new, did_filter = _step5_area_filter(D)
            filtered_5 = True
            if did_filter:
                D = D_new
                continue

        break

    return ImageDecision(
        RETRY_CLASS, "step6_auto_yellow",
        candidates=sorted(D.keys()),
    )
