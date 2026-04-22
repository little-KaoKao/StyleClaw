from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

MAX_LONG_EDGE = 1024


def resize_for_llm(image_path: Path | str) -> bytes:
    img = Image.open(image_path)
    w, h = img.size
    long_edge = max(w, h)

    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    fmt = "PNG" if img.mode == "RGBA" else "JPEG"
    img.save(buf, format=fmt, quality=85)
    return buf.getvalue()


def image_to_base64(image_path: Path | str) -> str:
    data = resize_for_llm(image_path)
    return base64.b64encode(data).decode("utf-8")


def media_type_for(image_path: Path | str) -> str:
    suffix = Path(image_path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
