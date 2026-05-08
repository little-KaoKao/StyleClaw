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


class OpenAICompatProvider:
    """OpenAI-compatible LLM provider (e.g. gptproto.com)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._base_url = base_url or os.getenv("OPENAI_COMPAT_BASE_URL", "")
        if not self._base_url:
            raise ValueError("OPENAI_COMPAT_BASE_URL is not set.")
        self._model_id = model_id or os.getenv("LLM_MODEL", "gemini-2.5-pro-preview-05-06")
        _api_key = api_key or os.getenv("OPENAI_COMPAT_API_KEY", "")
        if not _api_key:
            raise ValueError("OPENAI_COMPAT_API_KEY is not set.")
        self._api_key = SecretStr(_api_key)
        self._http = httpx.AsyncClient(
            base_url=self._base_url.rstrip("/"),
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
        stream: bool = False,
    ) -> str:
        oai_messages = [{"role": "system", "content": system}, *messages]
        body = {
            "model": self._model_id,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stream:
            body["stream"] = True
            return await self._post_stream(body)
        else:
            result = await self._post(body)
            try:
                return result["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:
                raise ValueError(f"Unexpected response format: {result}") from exc

    async def invoke_with_thinking(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        thinking_budget: int = 5000,
    ) -> LLMResponse:
        # OpenAI-compat providers don't expose thinking blocks; fall back to plain invoke.
        text = await self.invoke(system, messages, max_tokens=max_tokens, temperature=1.0, stream=True)
        return LLMResponse(text=text)

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                async with self._semaphore:
                    # Use stream() even for non-streaming to handle SSE responses
                    chunks: list[str] = []
                    async with self._http.stream("POST", "/chat/completions", content=json.dumps(body)) as resp:
                        resp.raise_for_status()
                        # Check if response is SSE (server-sent events)
                        content_type = resp.headers.get("content-type", "")
                        if "text/event-stream" in content_type:
                            # Server returned SSE, parse it
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
                            # Return as if it were a normal response
                            return {
                                "choices": [{
                                    "message": {"content": "".join(chunks)}
                                }]
                            }
                        else:
                            # Normal JSON response - must read body first
                            await resp.aread()
                            return resp.json()
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning("Request failed (attempt %d/%d): %s: %s. Retrying in %ds.",
                                   attempt + 1, MAX_RETRIES, type(exc).__name__, exc, wait)
                    await asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning("Request failed (attempt %d/%d): %s: %s. Retrying in %ds.",
                                   attempt + 1, MAX_RETRIES, type(exc).__name__, exc, wait)
                    await asyncio.sleep(wait)
        raise RuntimeError(f"LLM invoke failed after {MAX_RETRIES} retries") from last_exc


    async def _post_stream(self, body: dict[str, Any]) -> str:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                async with self._semaphore:
                    chunks: list[str] = []
                    async with self._http.stream("POST", "/chat/completions", content=json.dumps(body)) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                                if delta:
                                    print(delta, end="", flush=True)
                                    chunks.append(delta)
                            except (KeyError, IndexError, json.JSONDecodeError):
                                continue
                    print()  # newline after stream ends
                    return "".join(chunks)
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning("Request failed (attempt %d/%d): %s: %s. Retrying in %ds.",
                                   attempt + 1, MAX_RETRIES, type(exc).__name__, exc, wait)
                    await asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning("Request failed (attempt %d/%d): %s: %s. Retrying in %ds.",
                                   attempt + 1, MAX_RETRIES, type(exc).__name__, exc, wait)
                    await asyncio.sleep(wait)
        raise RuntimeError(f"LLM invoke failed after {MAX_RETRIES} retries") from last_exc
