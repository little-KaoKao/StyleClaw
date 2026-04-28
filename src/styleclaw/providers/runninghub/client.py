from __future__ import annotations

import asyncio
import logging
from typing import Any, Self

import httpx

from styleclaw.core.config import CONCURRENCY_LIMIT

logger = logging.getLogger(__name__)

BASE_URL = "https://www.runninghub.cn"
MAX_RETRIES = 3


class RunningHubClient:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def post(self, path: str, json_data: dict[str, Any]) -> dict[str, Any]:
        async with self._semaphore:
            return await self._post_with_retry(path, json_data)

    async def upload(self, path: str, file_path: str) -> dict[str, Any]:
        async with self._semaphore:
            return await self._upload_with_retry(path, file_path)

    async def _post_with_retry(
        self, path: str, json_data: dict[str, Any]
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = await self._client.post(path, json=json_data)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning(
                        "Request to %s failed (attempt %d/%d): %s. Retrying in %ds.",
                        path, attempt + 1, MAX_RETRIES, exc, wait,
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(
            f"Request to {path} failed after {MAX_RETRIES} retries"
        ) from last_exc

    async def _upload_with_retry(
        self, path: str, file_path: str
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                with open(file_path, "rb") as f:
                    resp = await self._client.post(
                        path,
                        files={"file": f},
                    )
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning(
                        "Upload to %s failed (attempt %d/%d): %s. Retrying in %ds.",
                        path, attempt + 1, MAX_RETRIES, exc, wait,
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(
            f"Upload to {path} failed after {MAX_RETRIES} retries"
        ) from last_exc

    async def close(self) -> None:
        await self._client.aclose()
