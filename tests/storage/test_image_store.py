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

    async def test_rejects_non_http_url(self, tmp_path: Path) -> None:
        dest = tmp_path / "x.png"
        with pytest.raises(RuntimeError, match="non-HTTP"):
            await download_image("file:///etc/passwd", dest)
        with pytest.raises(RuntimeError, match="non-HTTP"):
            await download_image("ftp://example.com/x", dest)

    @respx.mock
    async def test_aborts_when_body_exceeds_size_cap(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Small cap so a tiny body trips it.
        monkeypatch.setattr(
            "styleclaw.storage.image_store.MAX_DOWNLOAD_BYTES_PER_FILE", 100,
        )
        dest = tmp_path / "huge.png"
        big_body = b"x" * 5000  # well over the 100-byte cap
        respx.get("https://cdn.example.com/huge.png").respond(
            content=big_body, headers={"content-type": "image/png"},
        )
        with pytest.raises(RuntimeError, match="size cap"):
            await download_image("https://cdn.example.com/huge.png", dest)
        # Temp file must have been cleaned up.
        assert not list(tmp_path.glob("*.tmp"))
        assert not dest.exists()

    async def test_rejects_loopback_hostname(self, tmp_path: Path) -> None:
        dest = tmp_path / "x.png"
        with pytest.raises(RuntimeError, match="loopback"):
            await download_image("http://localhost/x.png", dest)

    async def test_rejects_private_ip(self, tmp_path: Path, monkeypatch) -> None:
        # Pretend example.invalid resolves to a private RFC1918 address.
        def fake_getaddrinfo(host, *_a, **_kw):
            return [(0, 0, 0, "", ("192.168.1.1", 0))]
        monkeypatch.setattr(
            "styleclaw.storage.image_store.socket.getaddrinfo", fake_getaddrinfo,
        )
        dest = tmp_path / "x.png"
        with pytest.raises(RuntimeError, match="disallowed"):
            await download_image("https://example.invalid/x.png", dest)

    @respx.mock
    async def test_refuses_to_follow_redirect(self, tmp_path: Path) -> None:
        dest = tmp_path / "x.png"
        respx.get("https://cdn.example.com/image.png").respond(
            status_code=302,
            headers={"location": "http://127.0.0.1/loot"},
        )
        with pytest.raises(RuntimeError, match="redirect"):
            await download_image("https://cdn.example.com/image.png", dest)
