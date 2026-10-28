import argparse
import re
from pathlib import Path

import cv2

from .stitching import stitch_images

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Имя изображения должно заканчиваться числом: {path.name}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch ordered building photographs into a panorama")
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Directory with images ordered by the trailing number in each filename")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("tasks/panorama_stitching/output/panorama.png"),
                        help="Output image path (default: %(default)s)")
    args = parser.parse_args()

    try:
        if not args.input.is_dir():
            raise ValueError(f"Не директория: {args.input}")
        paths = sorted(
            (path for path in args.input.iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
             and path.resolve() != args.output.resolve()),
            key=frame_number,
        )
        if len(paths) < 2:
            raise ValueError("Во входной папке должны находиться хотя бы два изображения")
        images = []
        for i, path in enumerate(paths, start=1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Не удалось прочитать изображение: {path}")
            print(f"Кадр №{i}: {path.name}", flush=True)
            images.append(image)
        panorama = stitch_images(images)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), panorama):
            raise OSError(f"Не удалось сохранить панораму: {args.output}")
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")
    print(f"Панорама сохранена: {args.output}")


if __name__ == "__main__":
    main()
