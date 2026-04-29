from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Self

import httpx

from styleclaw.providers.llm.base import LLMResponse

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class BedrockProvider:
    def __init__(
        self,
        region: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._region = region or os.getenv("AWS_REGION", "")
        if not self._region:
            self._region = "us-east-1"
            logger.warning("AWS_REGION not set, defaulting to 'us-east-1'")
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
        result = await self._post(body)
        text_blocks = [
            b["text"] for b in result.get("content", [])
            if b.get("type") == "text"
        ]
        if not text_blocks:
            raise ValueError("Bedrock returned no text content in response")
        return "\n".join(text_blocks)

    async def invoke_with_thinking(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        thinking_budget: int = 5000,
    ) -> LLMResponse:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            # Extended thinking requires temperature == 1.0.
            "temperature": 1.0,
            "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
        }
        result = await self._post(body)
        blocks = result.get("content", [])
        text_parts = [b["text"] for b in blocks if b.get("type") == "text"]
        thinking_parts = [
            b.get("thinking", "") for b in blocks if b.get("type") == "thinking"
        ]
        if not text_parts:
            raise ValueError("Bedrock returned no text content in response")
        return LLMResponse(
            text="\n".join(text_parts),
            thinking="\n\n".join(t for t in thinking_parts if t),
        )

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"/model/{self._model_id}/invoke"
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = await self._http.post(url, content=json.dumps(body))
                resp.raise_for_status()
                return resp.json()
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
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
                if attempt < MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning(
                        "Bedrock request failed (attempt %d/%d): %s. Retrying in %ds.",
                        attempt + 1, MAX_RETRIES, exc, wait,
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(
            f"Bedrock invoke failed after {MAX_RETRIES} retries"
        ) from last_exc
