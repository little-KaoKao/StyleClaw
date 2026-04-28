from __future__ import annotations

from pathlib import Path

import httpx


async def download_image(
    url: str,
    dest: Path,
    client: httpx.AsyncClient | None = None,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if client is not None:
        resp = await client.get(url)
        resp.raise_for_status()
        tmp.write_bytes(resp.content)
    else:
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.get(url)
            resp.raise_for_status()
            tmp.write_bytes(resp.content)
    tmp.replace(dest)
    return dest
