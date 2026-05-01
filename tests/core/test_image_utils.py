from __future__ import annotations

import base64
from pathlib import Path

import pytest
from PIL import Image

from styleclaw.core.image_utils import (
    build_image_block,
    encode_image_for_llm,
    resize_for_llm,
    verify_ref_image,
)


@pytest.fixture
def rgb_image(tmp_path: Path) -> Path:
    img = Image.new("RGB", (2048, 1024), color=(255, 0, 0))
    p = tmp_path / "rgb.jpg"
    img.save(p, "JPEG")
    return p


@pytest.fixture
def rgba_image(tmp_path: Path) -> Path:
    img = Image.new("RGBA", (512, 2048), color=(0, 0, 255, 128))
    p = tmp_path / "rgba.png"
    img.save(p, "PNG")
    return p


@pytest.fixture
def small_image(tmp_path: Path) -> Path:
    img = Image.new("RGB", (100, 50), color=(0, 255, 0))
    p = tmp_path / "small.jpg"
    img.save(p, "JPEG")
    return p


class TestResizeForLlm:
    def test_landscape_resized(self, rgb_image: Path) -> None:
        data, media_type = resize_for_llm(rgb_image)
        img = Image.open(__import__("io").BytesIO(data))
        assert img.width == 1024
        assert img.height == 512

    def test_portrait_resized(self, rgba_image: Path) -> None:
        data, media_type = resize_for_llm(rgba_image)
        img = Image.open(__import__("io").BytesIO(data))
        assert img.height == 1024
        assert img.width == 256

    def test_small_image_not_upscaled(self, small_image: Path) -> None:
        data, _ = resize_for_llm(small_image)
        img = Image.open(__import__("io").BytesIO(data))
        assert img.width == 100
        assert img.height == 50

    def test_rgba_returns_png_media_type(self, rgba_image: Path) -> None:
        _, media_type = resize_for_llm(rgba_image)
        assert media_type == "image/png"

    def test_rgb_returns_jpeg_media_type(self, rgb_image: Path) -> None:
        _, media_type = resize_for_llm(rgb_image)
        assert media_type == "image/jpeg"


class TestEncodeImageForLlm:
    def test_rgb_returns_jpeg(self, rgb_image: Path) -> None:
        b64_str, media_type = encode_image_for_llm(rgb_image)
        assert media_type == "image/jpeg"
        decoded = base64.b64decode(b64_str)
        assert decoded[:2] == b"\xff\xd8"

    def test_rgba_returns_png(self, rgba_image: Path) -> None:
        b64_str, media_type = encode_image_for_llm(rgba_image)
        assert media_type == "image/png"
        decoded = base64.b64decode(b64_str)
        assert decoded[:4] == b"\x89PNG"

    def test_returns_valid_base64(self, small_image: Path) -> None:
        b64_str, _ = encode_image_for_llm(small_image)
        decoded = base64.b64decode(b64_str)
        assert len(decoded) > 0


class TestBuildImageBlock:
    def test_returns_image_block_dict(self, small_image: Path) -> None:
        block = build_image_block(small_image)
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/jpeg"
        decoded = base64.b64decode(block["source"]["data"])
        assert len(decoded) > 0

    def test_rgba_returns_png_block(self, rgba_image: Path) -> None:
        block = build_image_block(rgba_image)
        assert block["source"]["media_type"] == "image/png"


class TestVerifyRefImage:
    def test_accepts_valid_png(self, small_image: Path) -> None:
        verify_ref_image(small_image)  # does not raise

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            verify_ref_image(tmp_path / "nope.png")

    def test_rejects_non_image(self, tmp_path: Path) -> None:
        p = tmp_path / "secrets.txt"
        p.write_bytes(b"KEY=deadbeef\n" * 100)
        with pytest.raises(ValueError, match="Not a valid image"):
            verify_ref_image(p)

    def test_rejects_oversize(self, tmp_path: Path) -> None:
        p = tmp_path / "big.png"
        Image.new("RGB", (8, 8)).save(p, "PNG")
        with pytest.raises(ValueError, match="too large"):
            verify_ref_image(p, max_bytes=10)

    def test_rejects_file_with_image_extension_but_garbage_bytes(self, tmp_path: Path) -> None:
        p = tmp_path / "fake.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 100)
        with pytest.raises(ValueError, match="Not a valid image"):
            verify_ref_image(p)
