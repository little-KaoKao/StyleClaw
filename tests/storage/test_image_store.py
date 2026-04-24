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
            content=b"fake image bytes"
        )
        result = await download_image("https://cdn.example.com/image.png", dest)
        assert result == dest
        assert dest.read_bytes() == b"fake image bytes"

    @respx.mock
    async def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        dest = tmp_path / "a" / "b" / "c" / "img.png"
        respx.get("https://cdn.example.com/image.png").respond(content=b"data")
        await download_image("https://cdn.example.com/image.png", dest)
        assert dest.exists()

    @respx.mock
    async def test_raises_on_http_error(self, tmp_path: Path) -> None:
        dest = tmp_path / "img.png"
        respx.get("https://cdn.example.com/image.png").respond(status_code=404)
        with pytest.raises(httpx.HTTPStatusError):
            await download_image("https://cdn.example.com/image.png", dest)

    @respx.mock
    async def test_uses_provided_client(self, tmp_path: Path) -> None:
        dest = tmp_path / "img.png"
        respx.get("https://cdn.example.com/image.png").respond(content=b"data")
        async with httpx.AsyncClient() as client:
            result = await download_image("https://cdn.example.com/image.png", dest, client=client)
        assert result == dest
        assert dest.read_bytes() == b"data"
