from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlsplit

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


def _is_disallowed_host(hostname: str) -> tuple[bool, str]:
    """Return ``(True, reason)`` if ``hostname`` resolves to an address we
    refuse to fetch from — loopback, link-local, private RFC1918, multicast,
    or unspecified. Used as SSRF defense for arbitrary URLs the LLM/cloud
    APIs hand us.

    Resolution failures themselves are NOT a block — let httpx surface a
    normal connection error instead of a confusing "disallowed host" message.
    """
    if not hostname:
        return True, "empty hostname"
    lowered = hostname.lower()
    if lowered in {"localhost", "ip6-localhost", "ip6-loopback"}:
        return True, f"loopback hostname: {hostname}"

    try:
        results = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False, ""

    for _family, _type, _proto, _canon, sockaddr in results:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_loopback or ip.is_link_local or ip.is_private
            or ip.is_multicast or ip.is_unspecified or ip.is_reserved
        ):
            return True, f"{hostname} resolves to {ip_str} ({ip.__class__.__name__.lower()} disallowed)"
    return False, ""


# Process-wide cache of SSRF host-check results. A typical batch poll hits the
# same RunningHub CDN hostname for every download, so re-resolving via
# ``socket.getaddrinfo`` per URL wastes a thread-pool slot (which the PIL
# encoder also shares) for up to the OS DNS timeout. The cache is unbounded by
# design — host counts in practice are small (CDN domains), and entries are
# tiny. DNS rebinding after the first check is an inherent SSRF-guard
# limitation, not a regression from caching.
_HOST_CHECK_CACHE: dict[str, tuple[bool, str]] = {}


async def _is_disallowed_host_cached(hostname: str) -> tuple[bool, str]:
    cached = _HOST_CHECK_CACHE.get(hostname)
    if cached is not None:
        return cached
    result = await asyncio.to_thread(_is_disallowed_host, hostname)
    _HOST_CHECK_CACHE[hostname] = result
    return result


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

    parsed = urlsplit(url)
    blocked, reason = await _is_disallowed_host_cached(parsed.hostname or "")
    if blocked:
        raise RuntimeError(f"Refusing to download from {url[:80]}: {reason}")

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
                async with c.stream("GET", url, follow_redirects=False) as resp:
                    if 300 <= resp.status_code < 400:
                        location = resp.headers.get("location", "?")
                        raise RuntimeError(
                            f"Refusing to follow redirect ({resp.status_code}) "
                            f"from {url[:80]} -> {location[:80]}"
                        )
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
