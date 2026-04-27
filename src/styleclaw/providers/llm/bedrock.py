from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Self

import httpx

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class BedrockProvider:
    def __init__(
        self,
        region: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._model_id = model_id or os.getenv(
            "CLAUDE_MODEL", "anthropic.claude-sonnet-4-20250514"
        )
        bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
        if not bearer_token:
            raise ValueError(
                "AWS_BEARER_TOKEN_BEDROCK is not set. "
                "Please set it in your .env file or environment."
            )
        base_url = f"https://bedrock-runtime.{self._region}.amazonaws.com"
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def invoke(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        url = f"/model/{self._model_id}/invoke"
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = await self._http.post(url, content=json.dumps(body))
                if resp.status_code >= 500:
                    resp.raise_for_status()
                resp.raise_for_status()
                result = resp.json()
                text_blocks = [
                    block["text"]
                    for block in result.get("content", [])
                    if block.get("type") == "text"
                ]
                return "\n".join(text_blocks)
            except httpx.TransportError as exc:
                last_exc = exc
                wait = 2**attempt
                logger.warning(
                    "Bedrock request failed (attempt %d/%d): %s. Retrying in %ds.",
                    attempt + 1, MAX_RETRIES, exc, wait,
                )
                await asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_exc = exc
                wait = 2**attempt
                logger.warning(
                    "Bedrock request failed (attempt %d/%d): %s. Retrying in %ds.",
                    attempt + 1, MAX_RETRIES, exc, wait,
                )
                await asyncio.sleep(wait)
        raise RuntimeError(
            f"Bedrock invoke failed after {MAX_RETRIES} retries"
        ) from last_exc
