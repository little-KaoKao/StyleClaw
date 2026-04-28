from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

MAX_LONG_EDGE = 1024


def _needs_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "PA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )


def _output_format(img: Image.Image) -> str:
    return "PNG" if _needs_alpha(img) else "JPEG"


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
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            img.close()
            img = resized

        fmt = _output_format(img)
        if fmt == "JPEG" and img.mode not in ("RGB", "L"):
            converted = img.convert("RGB")
            img.close()
            img = converted
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=85)
    finally:
        img.close()
    media_type = "image/png" if fmt == "PNG" else "image/jpeg"
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
