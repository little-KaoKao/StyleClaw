from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx

from styleclaw.core.config import MAX_DOWNLOAD_BYTES_PER_FILE

logger = logging.getLogger(__name__)

DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_DELAY = 2
DOWNLOAD_CHUNK_SIZE = 64 * 1024

FAILED_DOWNLOADS_FILE = "failed_downloads.json"

_CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

OUTPUT_IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def _ext_from_response(resp: httpx.Response, default: str = ".png") -> str:
    ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    return _CONTENT_TYPE_TO_EXT.get(ct, default)


def list_output_images(dir_path: Path, prefix: str = "output-") -> list[Path]:
    """List generated output images in a directory, supporting all extensions
    produced by `download_image` (png/jpg/jpeg/webp/gif).

    Sorted by filename for stable ordering across formats.
    """
    if not dir_path.exists():
        return []
    images: list[Path] = []
    for ext in OUTPUT_IMAGE_EXTENSIONS:
        images.extend(dir_path.glob(f"{prefix}*{ext}"))
    return sorted(images, key=lambda p: p.name)


def count_failed_downloads(dir_path: Path) -> int:
    """Count entries in a `failed_downloads.json` sidecar written by
    scripts/poll.py. Returns 0 when the file is missing or unparseable —
    this is informational only and shouldn't block report rendering."""
    sidecar = dir_path / FAILED_DOWNLOADS_FILE
    if not sidecar.exists():
        return 0
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        urls = data.get("failed_urls", [])
        return len(urls) if isinstance(urls, list) else 0
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", sidecar, exc)
        return 0


async def download_image(
    url: str,
    dest: Path,
    client: httpx.AsyncClient | None = None,
) -> Path:
    if not url.startswith(("http://", "https://")):
        raise RuntimeError(f"Refusing to download non-HTTP URL: {url[:80]}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None

    @asynccontextmanager
    async def _acquire_client() -> AsyncIterator[httpx.AsyncClient]:
        if client is not None:
            yield client
        else:
            async with httpx.AsyncClient(timeout=60) as new_client:
                yield new_client

    for attempt in range(DOWNLOAD_RETRIES):
        try:
            async with _acquire_client() as c:
                async with c.stream("GET", url) as resp:
                    resp.raise_for_status()
                    ext = _ext_from_response(resp, dest.suffix or ".png")
                    actual_dest = dest.with_suffix(ext)
                    tmp = actual_dest.with_suffix(actual_dest.suffix + ".tmp")
                    total = 0
                    try:
                        with open(tmp, "wb") as fh:
                            async for chunk in resp.aiter_bytes(DOWNLOAD_CHUNK_SIZE):
                                total += len(chunk)
                                if total > MAX_DOWNLOAD_BYTES_PER_FILE:
                                    fh.close()
                                    tmp.unlink(missing_ok=True)
                                    raise RuntimeError(
                                        f"Download exceeded "
                                        f"{MAX_DOWNLOAD_BYTES_PER_FILE} bytes "
                                        f"for {url[:80]} (size cap)"
                                    )
                                fh.write(chunk)
                    except BaseException:
                        tmp.unlink(missing_ok=True)
                        raise
            tmp.replace(actual_dest)
            return actual_dest
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < DOWNLOAD_RETRIES - 1:
                logger.warning(
                    "Download failed (attempt %d/%d) for %s: %s. Retrying...",
                    attempt + 1, DOWNLOAD_RETRIES, url[:80], exc,
                )
                await asyncio.sleep(DOWNLOAD_RETRY_DELAY * (attempt + 1))

    logger.error("Download failed after %d retries for %s", DOWNLOAD_RETRIES, url[:80])
    raise RuntimeError(
        f"Image download failed after {DOWNLOAD_RETRIES} retries: {url[:80]}"
    ) from last_exc
