from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Self

import httpx
from pydantic import SecretStr

from styleclaw.core.config import LLM_CONCURRENCY_LIMIT
from styleclaw.providers.llm.base import LLMResponse

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

DEFAULT_BASE_URL = "https://llm.runninghub.cn/v1"
DEFAULT_MODEL = "rh-llm-a/rh-c-o-47"


def _message_text_and_thinking(msg: dict[str, Any]) -> tuple[str, str]:
    text = (msg.get("content") or "") if isinstance(msg.get("content"), str) else ""
    raw_thinking = msg.get("reasoning_content")
    if raw_thinking is None:
        raw_thinking = msg.get("reasoning")
    thinking = raw_thinking if isinstance(raw_thinking, str) else ""
    return text, thinking


class RunningHubLLMProvider:
    """RunningHub OpenAI-compatible LLM API (https://llm.runninghub.cn/v1)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("RUNNINGHUB_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._model_id = model_id or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        _api_key = api_key or os.getenv("RUNNINGHUB_API_KEY", "")
        if not _api_key:
            raise ValueError("RUNNINGHUB_API_KEY is not set (required for RunningHub LLM).")
        self._api_key = SecretStr(_api_key)
        self._reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else (os.getenv("RUNNINGHUB_LLM_REASONING_EFFORT") or "high").strip().lower()
        )
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(300.0, connect=30.0, write=60.0),
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
        oai_messages = [{"role": "system", "content": system}, *messages]
        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        result = await self._post(body)
        msg = result["choices"][0]["message"]
        text, _ = _message_text_and_thinking(msg)
        if not text:
            raise ValueError(f"RunningHub LLM returned empty content: {result!r}")
        return text

    async def invoke_with_thinking(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        thinking_budget: int = 5000,
    ) -> LLMResponse:
        _ = thinking_budget  # protocol parity; RunningHub uses reasoning_effort instead
        oai_messages = [{"role": "system", "content": system}, *messages]
        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": 1,
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        if self._reasoning_effort and self._reasoning_effort not in ("off", "false", "0", "no"):
            body["reasoning_effort"] = self._reasoning_effort
        result = await self._post(body)
        msg = result["choices"][0]["message"]
        text, thinking = _message_text_and_thinking(msg)
        if not text:
            raise ValueError(f"RunningHub LLM returned empty content: {result!r}")
        return LLMResponse(text=text, thinking=thinking)

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                async with self._semaphore:
                    async with self._http.stream(
                        "POST",
                        "/chat/completions",
                        content=json.dumps(body),
                    ) as resp:
                        resp.raise_for_status()
                        content_type = resp.headers.get("content-type", "")
                        if "text/event-stream" in content_type:
                            chunks: list[str] = []
                            async for line in resp.aiter_lines():
                                if not line.startswith("data: "):
                                    continue
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        chunks.append(delta)
                                except (KeyError, IndexError, json.JSONDecodeError):
                                    continue
                            return {"choices": [{"message": {"content": "".join(chunks)}}]}
                        await resp.aread()
                        return resp.json()
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning(
                        "RunningHub LLM request failed (attempt %d/%d): %s: %s. Retrying in %ds.",
                        attempt + 1,
                        MAX_RETRIES,
                        type(exc).__name__,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning(
                        "RunningHub LLM request failed (attempt %d/%d): %s: %s. Retrying in %ds.",
                        attempt + 1,
                        MAX_RETRIES,
                        type(exc).__name__,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(f"RunningHub LLM invoke failed after {MAX_RETRIES} retries") from last_exc
