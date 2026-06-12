"""Export batch result images as WebP files for cloud upload.

Usage:
    1. Paste the finished batch folder into TARGET_BATCH_DIR below.
    2. Run this file from the IDE or with:
       uv run python scripts/export_batch_images_to_webp.py

The default output is:
    cloud_upload/<project-name>/
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


# Paste the batch folder you want to export here.
TARGET_BATCH_DIR = Path("data/projects/jojo-style-0611/batch-t2i/batch-005")

# Output folder at the StyleClaw project root.
OUTPUT_ROOT = Path("cloud_upload")

WEBP_QUALITY = 50
OVERWRITE = True

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the StyleClaw project root."""
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def project_name_from_batch_dir(batch_dir: Path) -> str:
    """Infer project name from data/projects/<project-name>/... paths."""
    parts = batch_dir.parts
    try:
        projects_index = parts.index("projects")
    except ValueError:
        return batch_dir.parent.name

    if projects_index + 1 >= len(parts):
        return batch_dir.parent.name
    return parts[projects_index + 1]


def source_prefix_from_batch_dir(batch_dir: Path) -> str:
    """Build a stable prefix such as batch-t2i__batch-004."""
    parts = batch_dir.parts
    try:
        projects_index = parts.index("projects")
    except ValueError:
        return safe_filename_segment(batch_dir.name)

    batch_parts = parts[projects_index + 2 :]
    if not batch_parts:
        return safe_filename_segment(batch_dir.name)
    return "__".join(safe_filename_segment(part) for part in batch_parts)


def safe_filename_segment(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    value = value.strip(".-_")
    return value or "image"


def iter_image_files(batch_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in batch_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(batch_dir).parts)
    )


def output_name_for(source: Path, batch_dir: Path) -> str:
    relative = source.relative_to(batch_dir).with_suffix("")
    parts = [safe_filename_segment(part) for part in relative.parts]
    return "__".join(part for part in parts if part) + ".webp"


def save_webp(source: Path, destination: Path, quality: int) -> None:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        image.load()

        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        elif image.mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")

        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="WEBP", quality=quality, method=6)


def export_images(
    batch_dir: Path,
    output_root: Path,
    quality: int,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int, list[str]]:
    batch_dir = resolve_project_path(batch_dir)
    output_root = resolve_project_path(output_root)

    if not batch_dir.is_dir():
        raise FileNotFoundError(f"Batch folder not found: {batch_dir}")

    project_name = project_name_from_batch_dir(batch_dir)
    source_prefix = source_prefix_from_batch_dir(batch_dir)
    output_dir = output_root / project_name / source_prefix
    images = iter_image_files(batch_dir)

    converted = 0
    skipped = 0
    errors: list[str] = []

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for source in images:
        destination = output_dir / output_name_for(source, batch_dir)
        if destination.exists() and not overwrite:
            skipped += 1
            continue

        try:
            if not dry_run:
                save_webp(source, destination, quality)
            converted += 1
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            errors.append(f"{source}: {exc}")

    action = "Would export" if dry_run else "Exported"
    print(f"{action} {converted} image(s) to: {output_dir}")
    if skipped:
        print(f"Skipped {skipped} existing file(s).")
    if errors:
        print(f"Failed {len(errors)} file(s):")
        for message in errors:
            print(f"  - {message}")

    return converted, skipped, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all images in a batch folder to WebP for cloud upload."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=TARGET_BATCH_DIR,
        help="Batch folder to export. Defaults to TARGET_BATCH_DIR in this file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root output folder. Defaults to cloud_upload at the project root.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=WEBP_QUALITY,
        help="WebP quality, 1-100. Defaults to 50.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip files that already exist instead of overwriting them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be exported without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.quality <= 100:
        raise ValueError("--quality must be between 1 and 100")

    export_images(
        batch_dir=args.source,
        output_root=args.output_root,
        quality=args.quality,
        overwrite=not args.no_overwrite and OVERWRITE,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
