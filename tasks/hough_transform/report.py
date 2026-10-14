import typing as ty
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ReportEntry:
    name: str
    circles: np.ndarray

    @property
    def n_circles(self) -> int:
        return int(self.circles.shape[0])


def write_report(
    entries: ty.List[ReportEntry],
    output_dir: ty.Union[str, Path],
    fmt: str = "txt",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "md":
        path = output_dir / "report.md"
        content = _format_md(entries)
    elif fmt == "txt":
        path = output_dir / "report.txt"
        content = _format_txt(entries)
    else:
        raise ValueError(f"Unknown report format: {fmt}")

    path.write_text(content, encoding="utf-8")
    return path


def _format_md(entries: ty.List[ReportEntry]) -> str:
    total = sum(e.n_circles for e in entries)
    lines = [
        "# Hough Circles Report",
        "",
        f"- Images processed: **{len(entries)}**",
        f"- Circles detected: **{total}**",
        "",
        "| File | Circles |",
        "|------|--------:|",
    ]
    for e in entries:
        lines.append(f"| {e.name} | {e.n_circles} |")
    lines.append("")
    return "\n".join(lines)


def _format_txt(entries: ty.List[ReportEntry]) -> str:
    total = sum(e.n_circles for e in entries)
    lines = [
        "Hough Circles Report",
        "=" * 20,
        f"Images processed: {len(entries)}",
        f"Circles detected: {total}",
        "",
        "Details:",
    ]
    for e in entries:
        lines.append(f"  {e.name}: {e.n_circles}")
    return "\n".join(lines) + "\n"
