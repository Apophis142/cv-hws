import argparse
import sys
from pathlib import Path

from .config import HoughCircleConfig, PipelineConfig
from .pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hough_circles",
        description="Hough transform pipeline for circle detection",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Image file or directory with images",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Directory for reports, masks and overlays",
    )
    parser.add_argument(
        "--report-format",
        choices=("txt", "md"),
        default="txt",
    )

    g = parser.add_argument_group("Canny")
    g.add_argument("--canny-low", type=float, default=0.08)
    g.add_argument("--canny-high", type=float, default=0.17)

    g = parser.add_argument_group("Hough circles")
    g.add_argument("--r-min", type=float, default=0.03,
                   help="Min radius: pixels (int) or fraction of min(H,W) (float in (0,1))")
    g.add_argument("--r-max", type=float, default=0.15,
                   help="Max radius: pixels or fraction of min(H,W)")
    g.add_argument("--r-delta", type=float, default=0.005,
                   help="Radius step: pixels or fraction of min(H,W)")
    g.add_argument("--ab-delta", type=float, default=0.005,
                   help="Center bin size: pixels or fraction of min(H,W); larger = faster")
    g.add_argument("--threshold", type=float, default=0.4,
                   help="Minimum visible-arc fraction (0..1). Scale-invariant to radius.")
    g.add_argument("--min-dist", type=float, default=0.02,
                   help="Min center distance: pixels or fraction of min(H,W)")
    g.add_argument("--max-circles", type=int, default=20)
    
    return parser


def _parse_config() -> PipelineConfig:
    args = _build_parser().parse_args()

    if not args.input_path.exists():
        print(f"error: input path does not exist: {args.input_path}", file=sys.stderr)
        sys.exit(2)
    if not (0 < args.canny_low < args.canny_high <= 1):
        print("error: require 0 < canny_low < canny_high <= 1", file=sys.stderr)
        sys.exit(2)
    if not (0 < args.r_min <= args.r_max):
        print("error: require 0 < r_min <= r_max", file=sys.stderr)
        sys.exit(2)
    if args.r_delta <= 0 or args.ab_delta <= 0:
        print("error: r_delta and ab_delta must be > 0", file=sys.stderr)
        sys.exit(2)
    if not (0.0 < args.threshold <= 1.0):
        print("error: require 0 < threshold <= 1", file=sys.stderr)
        sys.exit(2)

    
    circle_cfg = HoughCircleConfig(
        threshold1=args.canny_low,
        threshold2=args.canny_high,
        r_min=args.r_min,
        r_max=args.r_max,
        r_delta=args.r_delta,
        ab_delta=args.ab_delta,
        threshold=args.threshold,
        min_dist=args.min_dist,
        max_circles=args.max_circles,
    )

    return PipelineConfig(
        input_path=str(args.input_path),
        output_dir=str(args.output_dir),
        report_format=args.report_format,
        circle=circle_cfg,
    )


def main() -> None:
    config = _parse_config()
    run_pipeline(config)


if __name__ == "__main__":
    main()
