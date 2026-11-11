import argparse
import subprocess
import zipfile
from pathlib import Path

from .data import file_hash, save_json


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def export_source(output: Path) -> dict:
    output = Path(output)
    if output.suffix.lower() != ".zip":
        raise ValueError("Source archive must have a .zip extension")
    files = []
    for path in sorted(PACKAGE.iterdir()):
        if not path.is_file():
            continue
        if path.suffix in {".py", ".md", ".ipynb"} or path.name in {
            "requirements.txt",
            ".gitignore",
        }:
            files.append(path)
    hashes = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in files}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "status",
                "--porcelain",
                "--",
                "tasks/transfer_learning",
            ],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit, status = None, None
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    metadata = {
        "git_commit": commit,
        "git_status_task": status,
        "files_sha256": hashes,
        "archive_sha256": file_hash(output),
    }
    save_json(output.with_suffix(".json"), metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export transfer learning sources")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_source(args.output)
    print(f"Source archive: {args.output}")


if __name__ == "__main__":
    main()
