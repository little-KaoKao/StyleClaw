from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

MAX_LONG_EDGE = 1024


def _output_format(img: Image.Image) -> str:
    return "PNG" if img.mode == "RGBA" else "JPEG"


def resize_for_llm(image_path: Path | str) -> tuple[bytes, str]:
    img = Image.open(image_path)
    w, h = img.size
    long_edge = max(w, h)

    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    fmt = _output_format(img)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    media_type = "image/png" if fmt == "PNG" else "image/jpeg"
    return buf.getvalue(), media_type


def image_to_base64(image_path: Path | str) -> str:
    data, _ = resize_for_llm(image_path)
    return base64.b64encode(data).decode("utf-8")


def media_type_for(image_path: Path | str) -> str:
    img = Image.open(image_path)
    fmt = _output_format(img)
    return "image/png" if fmt == "PNG" else "image/jpeg"


def encode_image_for_llm(image_path: Path | str) -> tuple[str, str]:
    data, media_type = resize_for_llm(image_path)
    return base64.b64encode(data).decode("utf-8"), media_type
