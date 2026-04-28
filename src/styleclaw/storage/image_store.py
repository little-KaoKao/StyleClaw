from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_DELAY = 2

_CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def _ext_from_response(resp: httpx.Response, default: str = ".png") -> str:
    ct = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    return _CONTENT_TYPE_TO_EXT.get(ct, default)


async def download_image(
    url: str,
    dest: Path,
    client: httpx.AsyncClient | None = None,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None

    for attempt in range(DOWNLOAD_RETRIES):
        try:
            if client is not None:
                resp = await client.get(url)
            else:
                async with httpx.AsyncClient(timeout=60) as c:
                    resp = await c.get(url)
            resp.raise_for_status()

            ext = _ext_from_response(resp, dest.suffix or ".png")
            actual_dest = dest.with_suffix(ext)
            tmp = actual_dest.with_suffix(actual_dest.suffix + ".tmp")
            tmp.write_bytes(resp.content)
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
