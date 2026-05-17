"""Shared retry helper for LLM providers.

The three LLM providers (OpenAI-compat, RunningHub-LLM, Bedrock) each had a
near-identical retry loop around their ``_post`` body — same 3 attempts,
same fixed ``2**attempt`` backoff, same predicate (5xx and 429 retry, other
4xx fail fast), same redact-then-log warning. This module collapses that
into one place and adds ±20% jitter on the backoff so concurrent LLM calls
that all 429 on the same beat don't retry on the same beat too (consistent
with the jitter we added to RunningHub client retries in b5f4e72).
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, TypeVar

import httpx

from styleclaw.core.redact import redact_exc

logger = logging.getLogger(__name__)

LLM_MAX_RETRIES = 3

T = TypeVar("T")


async def llm_retry_loop(
    label: str,
    operation: Callable[[], Awaitable[T]],
    max_retries: int = LLM_MAX_RETRIES,
) -> T:
    """Retry ``operation`` on ``httpx.TransportError`` and on
    ``httpx.HTTPStatusError`` with 5xx / 429.

    Backoff is ``(2 ** attempt) * jitter`` where jitter is sampled from
    ``[0.8, 1.2]`` per attempt. Fails fast on other 4xx (auth, malformed
    request, etc.) — those won't get better by waiting.

    Wraps the final failure in ``RuntimeError(f"{label} failed after ...")``
    chained via ``__cause__`` so callers and ``redact_exc`` can still walk
    back to the underlying httpx error.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await operation()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status < 500 and status != 429:
                raise
            last_exc = exc
        except httpx.TransportError as exc:
            last_exc = exc
        if attempt < max_retries - 1:
            wait = (2 ** attempt) * random.uniform(0.8, 1.2)
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1fs.",
                label, attempt + 1, max_retries, redact_exc(last_exc), wait,
            )
            await asyncio.sleep(wait)
    raise RuntimeError(f"{label} failed after {max_retries} retries") from last_exc
