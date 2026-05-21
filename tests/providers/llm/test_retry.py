from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from styleclaw.providers.llm._retry import (
    BACKOFF_BASE_429,
    BACKOFF_BASE_5XX,
    LLM_MAX_RETRIES_429,
    RETRY_AFTER_MAX,
    _parse_retry_after,
    llm_retry_loop,
)


def _make_429(headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://example.invalid/x")
    resp = httpx.Response(429, request=req, headers=headers or {})
    return httpx.HTTPStatusError("rate limit", request=req, response=resp)


class TestLlmRetryLoop:
    async def test_returns_result_on_first_success(self) -> None:
        attempts = 0

        async def op() -> str:
            nonlocal attempts
            attempts += 1
            return "ok"

        result = await llm_retry_loop("test op", op)
        assert result == "ok"
        assert attempts == 1

    async def test_retries_on_5xx_then_succeeds(self, monkeypatch) -> None:
        # Compress jitter for speed; otherwise the test would take ~3s waiting.
        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep",
            AsyncMock(return_value=None),
        )
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                req = httpx.Request("POST", "https://example.invalid/x")
                resp = httpx.Response(500, request=req)
                raise httpx.HTTPStatusError("boom", request=req, response=resp)
            return "recovered"

        result = await llm_retry_loop("test op", op)
        assert result == "recovered"
        assert calls == 3

    async def test_retries_on_429(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep",
            AsyncMock(return_value=None),
        )
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                req = httpx.Request("POST", "https://example.invalid/x")
                resp = httpx.Response(429, request=req)
                raise httpx.HTTPStatusError("rate limit", request=req, response=resp)
            return "ok"

        await llm_retry_loop("test op", op)
        assert calls == 2

    async def test_fails_fast_on_4xx_other_than_429(self) -> None:
        async def op() -> str:
            req = httpx.Request("POST", "https://example.invalid/x")
            resp = httpx.Response(400, request=req)
            raise httpx.HTTPStatusError("bad input", request=req, response=resp)

        # Should raise immediately, not retry, not wrap in RuntimeError.
        with pytest.raises(httpx.HTTPStatusError):
            await llm_retry_loop("test op", op)

    async def test_retries_on_transport_error(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep",
            AsyncMock(return_value=None),
        )
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise httpx.ConnectError("conn refused")
            return "ok"

        await llm_retry_loop("test op", op)
        assert calls == 2

    async def test_raises_runtime_error_after_max_retries(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        async def op() -> str:
            raise httpx.ConnectError("permanent network failure")

        with pytest.raises(RuntimeError, match="failed after 3 retries"):
            await llm_retry_loop("my op", op)

    async def test_chains_cause_through_to_runtime_error(self, monkeypatch) -> None:
        # The final RuntimeError must chain via __cause__ so redact_exc and
        # debuggers can still walk back to the underlying httpx error.
        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        async def op() -> str:
            raise httpx.ConnectError("orig error msg")

        try:
            await llm_retry_loop("my op", op)
        except RuntimeError as exc:
            cause = exc.__cause__
            assert isinstance(cause, httpx.ConnectError)
            assert "orig error msg" in str(cause)

    async def test_uses_jitter_in_backoff(self, monkeypatch) -> None:
        # Capture sleep durations and confirm each lands within the ±20%
        # envelope around the bare 2**attempt baseline.
        durations: list[float] = []

        async def _fake_sleep(d: float) -> None:
            durations.append(d)

        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep", _fake_sleep,
        )

        async def op() -> str:
            raise httpx.ConnectError("nope")

        with pytest.raises(RuntimeError):
            await llm_retry_loop("op", op)

        # 2 retries → 2 sleeps; attempt 0 base=1, attempt 1 base=2.
        assert len(durations) == 2
        assert 0.8 <= durations[0] <= 1.2
        assert 1.6 <= durations[1] <= 2.4

    async def test_429_uses_slow_backoff_without_retry_after(
        self, monkeypatch,
    ) -> None:
        # When the server doesn't provide Retry-After, 429 must fall back to
        # the slow exponential schedule (~10s, ~20s, ~40s ± jitter) — the
        # whole point of the per-error split.
        durations: list[float] = []

        async def _fake_sleep(d: float) -> None:
            durations.append(d)

        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep", _fake_sleep,
        )

        async def op() -> str:
            raise _make_429()

        with pytest.raises(RuntimeError, match="failed after 4 retries"):
            await llm_retry_loop("op", op)

        # 4 attempts (LLM_MAX_RETRIES_429) → 3 sleeps; base 10 * 2**attempt.
        assert len(durations) == LLM_MAX_RETRIES_429 - 1
        for i, d in enumerate(durations):
            base = BACKOFF_BASE_429 * (2 ** i)
            assert 0.8 * base <= d <= 1.2 * base, (i, d, base)

    async def test_429_honors_retry_after_seconds(self, monkeypatch) -> None:
        durations: list[float] = []

        async def _fake_sleep(d: float) -> None:
            durations.append(d)

        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep", _fake_sleep,
        )
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _make_429({"retry-after": "7"})
            return "ok"

        result = await llm_retry_loop("op", op)
        assert result == "ok"
        assert len(durations) == 1
        # ±5% jitter around the header value.
        assert 7 * 0.95 <= durations[0] <= 7 * 1.05

    async def test_429_honors_retry_after_http_date(self, monkeypatch) -> None:
        durations: list[float] = []

        async def _fake_sleep(d: float) -> None:
            durations.append(d)

        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep", _fake_sleep,
        )

        future = datetime.now(timezone.utc) + timedelta(seconds=12)
        http_date = format_datetime(future, usegmt=True)
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _make_429({"retry-after": http_date})
            return "ok"

        await llm_retry_loop("op", op)
        # Allow generous bounds: clock skew during the test + jitter envelope.
        # ~12s ± 5% jitter, but clamp to a window that survives 1-2s of slop.
        assert len(durations) == 1
        assert 9 <= durations[0] <= 14

    async def test_429_falls_back_when_retry_after_exceeds_cap(
        self, monkeypatch,
    ) -> None:
        # If a server says "come back in an hour", we ignore it and use the
        # slow backoff instead — better to fail fast than block the
        # orchestrator for an hour on a flaky upstream.
        durations: list[float] = []

        async def _fake_sleep(d: float) -> None:
            durations.append(d)

        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep", _fake_sleep,
        )
        absurd = str(int(RETRY_AFTER_MAX) + 1000)
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _make_429({"retry-after": absurd})
            return "ok"

        await llm_retry_loop("op", op)
        assert len(durations) == 1
        # Slow backoff for attempt 0 → ~10s ± 20%.
        assert 0.8 * BACKOFF_BASE_429 <= durations[0] <= 1.2 * BACKOFF_BASE_429

    async def test_429_ignores_malformed_retry_after(self, monkeypatch) -> None:
        durations: list[float] = []

        async def _fake_sleep(d: float) -> None:
            durations.append(d)

        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep", _fake_sleep,
        )
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _make_429({"retry-after": "soonish"})
            return "ok"

        await llm_retry_loop("op", op)
        assert len(durations) == 1
        assert 0.8 * BACKOFF_BASE_429 <= durations[0] <= 1.2 * BACKOFF_BASE_429

    async def test_429_gets_more_attempts_than_5xx(self, monkeypatch) -> None:
        # The whole reason for the 429-specific budget: it must out-retry
        # the default 5xx budget when 429s start showing up.
        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep",
            AsyncMock(return_value=None),
        )
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise _make_429()

        with pytest.raises(RuntimeError, match=f"failed after {LLM_MAX_RETRIES_429} retries"):
            await llm_retry_loop("op", op)
        assert calls == LLM_MAX_RETRIES_429

    async def test_5xx_then_429_still_promotes_budget(self, monkeypatch) -> None:
        # Mixed-error path: a 5xx burns one attempt, then a 429 lifts the
        # cap. We should still get the full 429 budget in total.
        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep",
            AsyncMock(return_value=None),
        )
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            req = httpx.Request("POST", "https://example.invalid/x")
            if calls == 1:
                resp = httpx.Response(500, request=req)
                raise httpx.HTTPStatusError("boom", request=req, response=resp)
            raise _make_429()

        with pytest.raises(RuntimeError):
            await llm_retry_loop("op", op)
        assert calls == LLM_MAX_RETRIES_429

    async def test_5xx_uses_fast_backoff_unchanged(self, monkeypatch) -> None:
        # Regression guard: the per-error split shouldn't slow down the
        # 5xx/transport path, only speed up the 429 path's tolerance.
        durations: list[float] = []

        async def _fake_sleep(d: float) -> None:
            durations.append(d)

        monkeypatch.setattr(
            "styleclaw.providers.llm._retry.asyncio.sleep", _fake_sleep,
        )

        async def op() -> str:
            req = httpx.Request("POST", "https://example.invalid/x")
            resp = httpx.Response(503, request=req)
            raise httpx.HTTPStatusError("nope", request=req, response=resp)

        with pytest.raises(RuntimeError, match="failed after 3 retries"):
            await llm_retry_loop("op", op)

        assert len(durations) == 2
        for i, d in enumerate(durations):
            base = BACKOFF_BASE_5XX * (2 ** i)
            assert 0.8 * base <= d <= 1.2 * base, (i, d, base)


class TestParseRetryAfter:
    def test_returns_none_for_empty(self) -> None:
        assert _parse_retry_after(None) is None
        assert _parse_retry_after("") is None
        assert _parse_retry_after("   ") is None

    def test_parses_integer_seconds(self) -> None:
        assert _parse_retry_after("5") == 5.0
        assert _parse_retry_after("  30 ") == 30.0
        assert _parse_retry_after("0") == 0.0

    def test_parses_float_seconds(self) -> None:
        assert _parse_retry_after("2.5") == 2.5

    def test_rejects_negative(self) -> None:
        assert _parse_retry_after("-1") is None

    def test_parses_http_date(self) -> None:
        future = datetime.now(timezone.utc) + timedelta(seconds=10)
        http_date = format_datetime(future, usegmt=True)
        parsed = _parse_retry_after(http_date)
        assert parsed is not None
        assert 8 <= parsed <= 11

    def test_past_http_date_returns_none(self) -> None:
        past = datetime.now(timezone.utc) - timedelta(seconds=60)
        http_date = format_datetime(past, usegmt=True)
        assert _parse_retry_after(http_date) is None

    def test_garbage_returns_none(self) -> None:
        assert _parse_retry_after("soonish") is None
        assert _parse_retry_after("not a date") is None
