"""Shared retry helper for LLM providers.

Wraps the OpenAI-compatible provider's ``_post`` body in a retry loop with
*per-error-type backoff*:

- 5xx and transport errors: 3 attempts, ``2**attempt`` seconds backoff with
  ±20% jitter (~1s, ~2s). These tend to clear on the second-scale.
- 429 Too Many Requests: up to 4 attempts. If the server sends a
  ``Retry-After`` header (delta-seconds or HTTP-date), honor it (capped at
  ``RETRY_AFTER_MAX`` to avoid pathological "come back in an hour" responses
  blocking the orchestrator). Otherwise fall back to a slow schedule
  (~10s, ~20s, ~40s) because rate-limit windows are minute-scale and a 1s/2s
  backoff is essentially "fail fast" for RPM/TPM caps.

Fails fast on other 4xx (auth, malformed request, etc.) — those won't get
better by waiting. Jitter prevents concurrent LLM calls that all 429 on the
same beat from retrying on the same beat too (consistent with the jitter we
added to RunningHub client retries in b5f4e72).
"""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, TypeVar

import httpx

from styleclaw.core.redact import redact_exc

logger = logging.getLogger(__name__)

LLM_MAX_RETRIES = 3
LLM_MAX_RETRIES_429 = 4
BACKOFF_BASE_5XX = 1.0
BACKOFF_BASE_429 = 10.0
RETRY_AFTER_MAX = 120.0

T = TypeVar("T")


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header into seconds.

    RFC 7231 allows either delta-seconds (non-negative integer) or an
    HTTP-date. Returns ``None`` for missing, malformed, or negative values
    so the caller can fall back to its own backoff schedule.
    """
    if not value:
        return None
    text = value.strip()
    try:
        seconds = float(text)
        return seconds if seconds >= 0 else None
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = (dt - datetime.now(timezone.utc)).total_seconds()
    return delta if delta >= 0 else None


def _wait_for_429(response: httpx.Response, attempt: int) -> float:
    """Pick a wait duration for a 429 response.

    Honors ``Retry-After`` when present and reasonable; otherwise uses a
    slow exponential backoff that has a chance of clearing a 60s RPM
    window before the retry budget is exhausted.
    """
    retry_after = _parse_retry_after(response.headers.get("retry-after"))
    if retry_after is not None and retry_after <= RETRY_AFTER_MAX:
        return retry_after * random.uniform(0.95, 1.05)
    return BACKOFF_BASE_429 * (2 ** attempt) * random.uniform(0.8, 1.2)


def _wait_for_5xx(attempt: int) -> float:
    return BACKOFF_BASE_5XX * (2 ** attempt) * random.uniform(0.8, 1.2)


async def llm_retry_loop(
    label: str,
    operation: Callable[[], Awaitable[T]],
    max_retries: int = LLM_MAX_RETRIES,
) -> T:
    """Retry ``operation`` on transient failures with per-error backoff.

    Retried errors: ``httpx.TransportError`` and ``httpx.HTTPStatusError``
    with status 5xx or 429. Other 4xx errors raise immediately.

    The retry budget is dynamic: if any attempt has hit a 429, the loop
    promotes its effective max to ``LLM_MAX_RETRIES_429`` so rate-limit
    windows have a realistic chance of clearing. Wraps the final failure
    in ``RuntimeError(f"{label} failed after ...")`` chained via
    ``__cause__`` so callers and ``redact_exc`` can still walk back to the
    underlying httpx error.
    """
    last_exc: Exception | None = None
    seen_429 = False
    attempt = 0
    while True:
        wait: float
        try:
            return await operation()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status < 500 and status != 429:
                raise
            last_exc = exc
            if status == 429:
                seen_429 = True
                wait = _wait_for_429(exc.response, attempt)
            else:
                wait = _wait_for_5xx(attempt)
        except httpx.TransportError as exc:
            last_exc = exc
            wait = _wait_for_5xx(attempt)

        attempt += 1
        effective_max = max(max_retries, LLM_MAX_RETRIES_429) if seen_429 else max_retries
        if attempt >= effective_max:
            break
        logger.warning(
            "%s failed (attempt %d/%d): %s. Retrying in %.1fs.",
            label, attempt, effective_max, redact_exc(last_exc), wait,
        )
        await asyncio.sleep(wait)
    raise RuntimeError(f"{label} failed after {attempt} retries") from last_exc
