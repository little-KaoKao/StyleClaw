from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from styleclaw.core.config import IMAGE_ENCODE_CONCURRENCY

MAX_REF_IMAGE_BYTES = 50 * 1024 * 1024

MAX_LONG_EDGE = 1024

# Bound the number of Pillow-decoding worker threads so building image
# blocks for an evaluate call (often 20+ images) doesn't spawn one thread
# per image.
_ENCODE_SEMAPHORE: asyncio.Semaphore | None = None


def _encode_semaphore() -> asyncio.Semaphore:
    # Lazy init because event-loop-aware primitives can't be constructed at
    # import time (no running loop yet).
    global _ENCODE_SEMAPHORE
    if _ENCODE_SEMAPHORE is None:
        _ENCODE_SEMAPHORE = asyncio.Semaphore(IMAGE_ENCODE_CONCURRENCY)
    return _ENCODE_SEMAPHORE


def verify_ref_image(path: Path | str, max_bytes: int = MAX_REF_IMAGE_BYTES) -> None:
    """Validate that a user-supplied reference image is safe to copy into the
    project directory: it must exist, be under `max_bytes`, and be decodable
    by Pillow. Raises ValueError with a human-readable message on failure.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"Image not found: {p}")
    size = p.stat().st_size
    if size > max_bytes:
        mb = size / (1024 * 1024)
        limit_mb = max_bytes / (1024 * 1024)
        raise ValueError(
            f"Image too large: {p.name} is {mb:.1f} MB (limit: {limit_mb:.0f} MB)"
        )
    try:
        # Image.verify() only checks chunk CRCs and headers — it does NOT
        # decompress the pixel data. Use load() so a PNG with corrupt zlib
        # inside IDAT (valid CRC, garbage payload) is rejected at init time
        # instead of crashing later inside an async resize.
        with Image.open(p) as img:
            img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"Not a valid image: {p.name} ({exc})") from exc


def _needs_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "PA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )


def _output_format(img: Image.Image) -> str:
    # Always use WebP for better compression
    return "WEBP"


def resize_for_llm(image_path: Path | str) -> tuple[bytes, str]:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    try:
        w, h = img.size
        long_edge = max(w, h)

        if long_edge > MAX_LONG_EDGE:
            scale = MAX_LONG_EDGE / long_edge
            new_w = int(w * scale)
            new_h = int(h * scale)
            # BICUBIC is roughly 2× faster than LANCZOS; the difference is
            # invisible after the LLM's vision encoder downsamples again.
            resized = img.resize((new_w, new_h), Image.BICUBIC)
            img.close()
            img = resized

        fmt = _output_format(img)
        if fmt == "WEBP":
            # Convert to RGB for WebP (no alpha support in quality mode)
            if img.mode not in ("RGB", "L"):
                converted = img.convert("RGB")
                img.close()
                img = converted
        elif fmt == "JPEG" and img.mode not in ("RGB", "L"):
            converted = img.convert("RGB")
            img.close()
            img = converted
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=85)
    finally:
        img.close()
    if fmt == "WEBP":
        media_type = "image/webp"
    elif fmt == "PNG":
        media_type = "image/png"
    else:
        media_type = "image/jpeg"
    return buf.getvalue(), media_type



def encode_image_for_llm(image_path: Path | str) -> tuple[str, str]:
    data, media_type = resize_for_llm(image_path)
    return base64.b64encode(data).decode("utf-8"), media_type


def build_image_block(image_path: Path | str) -> dict:
    b64_data, media_type = encode_image_for_llm(image_path)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": b64_data},
    }


async def build_image_block_async(image_path: Path | str) -> dict:
    """Async variant that offloads Pillow decode/resize/encode to a worker
    thread, so the event loop stays responsive while processing many images.
    """
    async with _encode_semaphore():
        return await asyncio.to_thread(build_image_block, image_path)


async def build_image_blocks_async(image_paths: list[Path | str]) -> list[dict]:
    """Build image blocks for several paths concurrently (one thread each,
    capped by IMAGE_ENCODE_CONCURRENCY)."""
    return list(await asyncio.gather(*(build_image_block_async(p) for p in image_paths)))
