from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from styleclaw.storage.image_store import download_image


class TestDownloadImage:
    @respx.mock
    async def test_downloads_to_dest(self, tmp_path: Path) -> None:
        dest = tmp_path / "images" / "output.png"
        respx.get("https://cdn.example.com/image.png").respond(
            content=b"fake image bytes",
            headers={"content-type": "image/png"},
        )
        result = await download_image("https://cdn.example.com/image.png", dest)
        assert result == dest
        assert dest.read_bytes() == b"fake image bytes"

    @respx.mock
    async def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        dest = tmp_path / "a" / "b" / "c" / "img.png"
        respx.get("https://cdn.example.com/image.png").respond(
            content=b"data", headers={"content-type": "image/png"},
        )
        await download_image("https://cdn.example.com/image.png", dest)
        assert dest.exists()

    @respx.mock
    async def test_raises_on_http_error(self, tmp_path: Path) -> None:
        dest = tmp_path / "img.png"
        respx.get("https://cdn.example.com/image.png").respond(status_code=404)
        with pytest.raises(RuntimeError, match="failed after"):
            await download_image("https://cdn.example.com/image.png", dest)

    @respx.mock
    async def test_uses_provided_client(self, tmp_path: Path) -> None:
        dest = tmp_path / "img.png"
        respx.get("https://cdn.example.com/image.png").respond(
            content=b"data", headers={"content-type": "image/png"},
        )
        async with httpx.AsyncClient() as client:
            result = await download_image("https://cdn.example.com/image.png", dest, client=client)
        assert result == dest
        assert dest.read_bytes() == b"data"

    @respx.mock
    async def test_detects_jpeg_content_type(self, tmp_path: Path) -> None:
        dest = tmp_path / "output.png"
        respx.get("https://cdn.example.com/image").respond(
            content=b"jpeg data",
            headers={"content-type": "image/jpeg"},
        )
        result = await download_image("https://cdn.example.com/image", dest)
        assert result.suffix == ".jpg"
        assert result.read_bytes() == b"jpeg data"

    @respx.mock
    async def test_retries_on_transport_error(self, tmp_path: Path) -> None:
        dest = tmp_path / "output.png"
        route = respx.get("https://cdn.example.com/image.png")
        route.side_effect = [
            httpx.ConnectError("connection refused"),
            httpx.Response(200, content=b"data", headers={"content-type": "image/png"}),
        ]
        result = await download_image("https://cdn.example.com/image.png", dest)
        assert result.read_bytes() == b"data"
