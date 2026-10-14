from dataclasses import asdict
from pathlib import Path

from utils.hough import hough_circles

from .config import PipelineConfig, resolve_circles_config
from .io_utils import iter_images, load_image, save_image
from .report import ReportEntry, write_report
from .visualize import circle_mask, draw_circles, overlay_mask


def process_file(path: Path, config: PipelineConfig):
    img = load_image(path)
    circle_cfg = resolve_circles_config(config.circle, img.shape)
    circles, edges = hough_circles(img, return_edges=True, **asdict(circle_cfg))
    return ReportEntry(name=path.name, circles=circles), edges


def run_pipeline(config: PipelineConfig) -> Path:
    output_dir = Path(config.output_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    edges_dir = output_dir / "edges"

    entries = []

    for path in iter_images(config.input_path):
        entry, edges = process_file(path, config)
        entries.append(entry)

        img = load_image(path)
        mask = circle_mask(img.shape, entry.circles, thickness=1)
        overlay = overlay_mask(img, mask)
        overlay = draw_circles(overlay, entry.circles)

        save_image(mask_dir / f"{path.stem}_mask.png", mask)
        save_image(overlay_dir / f"{path.stem}_overlay.png", overlay)
        save_image(edges_dir / f"{path.stem}_canny.png", edges)

        print(f"[{path.name}] circles: {entry.n_circles}")

    report_path = write_report(entries, output_dir, config.report_format)
    print(f"Report written to: {report_path}")
    return report_path
