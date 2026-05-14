from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from styleclaw.providers.llm.openai_compat import OpenAICompatProvider


@pytest.fixture
def provider(monkeypatch) -> OpenAICompatProvider:
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://oai.test/v1")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
    return OpenAICompatProvider(
        base_url="https://oai.test/v1", api_key="k", model_id="test",
    )


class TestOpenAICompatRetry:
    @respx.mock
    async def test_retries_on_429_rate_limit(
        self, provider: OpenAICompatProvider,
    ) -> None:
        route = respx.post("https://oai.test/v1/chat/completions")
        ok_body = {"choices": [{"message": {"content": "ok"}}]}
        route.side_effect = [
            httpx.Response(429, text="rate limited"),
            httpx.Response(429, text="rate limited"),
            httpx.Response(200, json=ok_body),
        ]
        with patch("styleclaw.providers.llm.openai_compat.asyncio.sleep", new_callable=AsyncMock):
            text = await provider.invoke("s", [{"role": "user", "content": "x"}])
        assert text == "ok"
        assert route.call_count == 3
        await provider.close()

    @respx.mock
    async def test_no_retry_on_400(
        self, provider: OpenAICompatProvider,
    ) -> None:
        route = respx.post("https://oai.test/v1/chat/completions").respond(
            status_code=400, text="bad request"
        )
        with pytest.raises(httpx.HTTPStatusError):
            await provider.invoke("s", [])
        assert route.call_count == 1
        await provider.close()
