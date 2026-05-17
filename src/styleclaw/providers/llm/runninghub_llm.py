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
    STREAM_DISPLAY,
)
from styleclaw.providers.llm._retry import llm_retry_loop
from styleclaw.providers.llm.base import LLMResponse

logger = logging.getLogger(__name__)

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
            timeout=httpx.Timeout(
                LLM_READ_TIMEOUT, connect=LLM_CONNECT_TIMEOUT, write=LLM_WRITE_TIMEOUT,
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
        oai_messages = [{"role": "system", "content": system}, *messages]
        body: dict[str, Any] = {
            "model": self._model_id,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        result = await self._post(body)
        msg = result["choices"][0]["message"]
        text, thinking = _message_text_and_thinking(msg)
        if not text and thinking:
            logger.warning(
                "RunningHub LLM returned empty content but non-empty reasoning; "
                "falling back to reasoning_content as response text."
            )
            text = thinking
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
            "stream": True,
        }
        if self._reasoning_effort and self._reasoning_effort not in ("off", "false", "0", "no"):
            body["reasoning_effort"] = self._reasoning_effort
        result = await self._post(body)
        msg = result["choices"][0]["message"]
        text, thinking = _message_text_and_thinking(msg)
        if not text and thinking:
            # Some thinking models emit the JSON answer inside reasoning_content
            # while leaving content empty. Fall back to the thinking text so that
            # downstream JSON parsers can still extract the structured output.
            logger.warning(
                "RunningHub LLM returned empty content but non-empty reasoning; "
                "falling back to reasoning_content as response text."
            )
            text = thinking
            thinking = ""
        if not text:
            raise ValueError(f"RunningHub LLM returned empty content: {result!r}")
        return LLMResponse(text=text, thinking=thinking)

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        async def _attempt() -> dict[str, Any]:
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
                        reasoning_chunks: list[str] = []
                        # active_stream: None | "think" | "answer" — tracks which
                        # marker was printed last so we only emit a new prefix on
                        # a transition (e.g. think → answer).
                        active_stream: str | None = None
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                delta_obj = json.loads(data)["choices"][0]["delta"]
                            except (KeyError, IndexError, json.JSONDecodeError):
                                continue
                            reasoning = delta_obj.get("reasoning_content", "")
                            if reasoning:
                                if STREAM_DISPLAY:
                                    if active_stream != "think":
                                        print("\n  💭 " if active_stream else "  💭 ", end="", flush=True)
                                        active_stream = "think"
                                    print(reasoning, end="", flush=True)
                                reasoning_chunks.append(reasoning)
                            delta = delta_obj.get("content", "")
                            if delta:
                                if STREAM_DISPLAY:
                                    if active_stream != "answer":
                                        print("\n  ↓ " if active_stream else "  ↓ ", end="", flush=True)
                                        active_stream = "answer"
                                    print(delta, end="", flush=True)
                                chunks.append(delta)
                        if STREAM_DISPLAY and active_stream is not None:
                            print()
                        return {
                            "choices": [{
                                "message": {
                                    "content": "".join(chunks),
                                    "reasoning_content": "".join(reasoning_chunks),
                                }
                            }]
                        }
                    await resp.aread()
                    return resp.json()

        return await llm_retry_loop("RunningHub LLM invoke", _attempt)
