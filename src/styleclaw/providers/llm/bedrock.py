from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Self

import httpx
from pydantic import SecretStr

from styleclaw.core.config import (
    LLM_CONCURRENCY_LIMIT,
    LLM_CONNECT_TIMEOUT,
    LLM_READ_TIMEOUT,
    LLM_WRITE_TIMEOUT,
)
from styleclaw.providers.llm._retry import llm_retry_loop
from styleclaw.providers.llm.base import LLMResponse

logger = logging.getLogger(__name__)


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
        self._model_id = model_id or os.getenv("LLM_MODEL") or os.getenv(
            "CLAUDE_MODEL", "anthropic.claude-sonnet-4-20250514"
        )
        raw_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
        if not raw_token:
            raise ValueError(
                "AWS_BEARER_TOKEN_BEDROCK is not set. "
                "Please set it in your .env file or environment."
            )
        self._token = SecretStr(raw_token)
        base_url = f"https://bedrock-runtime.{self._region}.amazonaws.com"
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {self._token.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                LLM_READ_TIMEOUT, connect=LLM_CONNECT_TIMEOUT, write=LLM_WRITE_TIMEOUT,
            ),
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=50,
                keepalive_expiry=30.0,
            ),
        )
        self._semaphore = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)

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
        # Anthropic requires thinking.budget_tokens < max_tokens. When the
        # caller's max_tokens leaves no room for a non-trivial response after
        # the thinking budget, lift it so the API doesn't reject the request.
        effective_max_tokens = max(max_tokens, thinking_budget + 1024)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system,
            "messages": messages,
            "max_tokens": effective_max_tokens,
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

        async def _attempt() -> dict[str, Any]:
            async with self._semaphore:
                resp = await self._http.post(url, content=json.dumps(body))
            resp.raise_for_status()
            return resp.json()

        return await llm_retry_loop("Bedrock invoke", _attempt)
