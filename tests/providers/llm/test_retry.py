from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from styleclaw.providers.llm._retry import llm_retry_loop


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
