from __future__ import annotations

import json

import httpx
import pytest
import respx

from styleclaw.providers.llm.bedrock import BedrockProvider


@pytest.fixture
def provider(monkeypatch) -> BedrockProvider:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    return BedrockProvider(region="us-east-1", model_id="test-model")


class TestBedrockProvider:
    @respx.mock
    async def test_invoke_returns_text(self, provider: BedrockProvider) -> None:
        route = respx.post(
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(
            json={
                "content": [{"type": "text", "text": "hello world"}],
            }
        )

        result = await provider.invoke(
            system="test system",
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == "hello world"
        assert route.called

    @respx.mock
    async def test_invoke_joins_multiple_text_blocks(self, provider: BedrockProvider) -> None:
        respx.post(
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(
            json={
                "content": [
                    {"type": "text", "text": "line1"},
                    {"type": "text", "text": "line2"},
                ],
            }
        )
        result = await provider.invoke(system="s", messages=[])
        assert result == "line1\nline2"

    @respx.mock
    async def test_invoke_skips_non_text_blocks(self, provider: BedrockProvider) -> None:
        respx.post(
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(
            json={
                "content": [
                    {"type": "image", "data": "..."},
                    {"type": "text", "text": "only text"},
                ],
            }
        )
        result = await provider.invoke(system="s", messages=[])
        assert result == "only text"

    @respx.mock
    async def test_invoke_raises_on_5xx_after_retries(self, provider: BedrockProvider) -> None:
        respx.post(
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(status_code=500, text="Internal Server Error")
        with pytest.raises(RuntimeError, match="failed after 3 retries"):
            await provider.invoke(system="s", messages=[])

    @respx.mock
    async def test_invoke_raises_on_empty_content(self, provider: BedrockProvider) -> None:
        respx.post(
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(json={"content": []})
        with pytest.raises(ValueError, match="no text content"):
            await provider.invoke(system="s", messages=[])

    async def test_close(self, provider: BedrockProvider) -> None:
        await provider.close()

    def test_raises_without_token(self, monkeypatch) -> None:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "")
        with pytest.raises(ValueError, match="AWS_BEARER_TOKEN_BEDROCK"):
            BedrockProvider()

    def test_default_env_values(self, monkeypatch) -> None:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "valid-token")
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("CLAUDE_MODEL", raising=False)
        p = BedrockProvider()
        assert p._region == "us-east-1"
        assert "claude-sonnet" in p._model_id


class TestBedrockAsyncContextManager:
    async def test_aenter_returns_self(self, provider: BedrockProvider) -> None:
        async with provider as ctx:
            assert ctx is provider

    async def test_aexit_closes_http_client(self, provider: BedrockProvider) -> None:
        async with provider:
            pass
        assert provider._http.is_closed

    @respx.mock
    async def test_usable_inside_context(self, monkeypatch) -> None:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
        respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(json={"content": [{"type": "text", "text": "hello"}]})

        async with BedrockProvider(region="us-east-1", model_id="test-model") as llm:
            result = await llm.invoke(system="s", messages=[])
            assert result == "hello"


class TestBedrockRetry:
    @respx.mock
    async def test_retries_on_5xx_then_succeeds(self, provider: BedrockProvider) -> None:
        route = respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        )
        route.side_effect = [
            httpx.Response(500, text="error"),
            httpx.Response(502, text="bad gateway"),
            httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]}),
        ]
        result = await provider.invoke(system="s", messages=[])
        assert result == "ok"
        assert route.call_count == 3

    @respx.mock
    async def test_raises_after_max_retries(self, provider: BedrockProvider) -> None:
        respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(status_code=500, text="error")
        with pytest.raises(RuntimeError, match="failed after 3 retries"):
            await provider.invoke(system="s", messages=[])

    @respx.mock
    async def test_no_retry_on_4xx(self, provider: BedrockProvider) -> None:
        route = respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        )
        route.respond(status_code=400, text="bad request")
        with pytest.raises(httpx.HTTPStatusError):
            await provider.invoke(system="s", messages=[])
        assert route.call_count == 1

    @respx.mock
    async def test_retries_on_transport_error(self, provider: BedrockProvider) -> None:
        route = respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        )
        route.side_effect = [
            httpx.ConnectError("connection refused"),
            httpx.Response(200, json={"content": [{"type": "text", "text": "recovered"}]}),
        ]
        result = await provider.invoke(system="s", messages=[])
        assert result == "recovered"
