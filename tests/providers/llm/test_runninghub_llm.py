from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from styleclaw.providers.llm.runninghub_llm import RunningHubLLMProvider


@pytest.fixture
def provider(monkeypatch) -> RunningHubLLMProvider:
    monkeypatch.setenv("RUNNINGHUB_API_KEY", "test-key")
    return RunningHubLLMProvider(
        base_url="https://llm.test/v1",
        api_key="test-key",
        model_id="rh-test",
        reasoning_effort="high",
    )


class TestRunningHubLLMProviderInit:
    def test_requires_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("RUNNINGHUB_API_KEY", raising=False)
        with pytest.raises(ValueError, match="RUNNINGHUB_API_KEY"):
            RunningHubLLMProvider(base_url="https://llm.test/v1", api_key="")


class TestRunningHubLLMInvoke:
    @respx.mock
    async def test_invoke_returns_text(self, provider: RunningHubLLMProvider) -> None:
        respx.post("https://llm.test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"ok": true}'}}],
                },
            )
        )
        text = await provider.invoke("sys", [{"role": "user", "content": "x"}])
        assert text == '{"ok": true}'
        await provider.close()

    @respx.mock
    async def test_invoke_raises_on_empty_content(self, provider: RunningHubLLMProvider) -> None:
        respx.post("https://llm.test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": ""}}]},
            )
        )
        with pytest.raises(ValueError, match="empty content"):
            await provider.invoke("s", [])
        await provider.close()

    @respx.mock
    async def test_invoke_with_thinking_passes_reasoning_effort(
        self, provider: RunningHubLLMProvider,
    ) -> None:
        captured: dict = {}

        def _side_effect(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "answer",
                                "reasoning_content": "step 1",
                            },
                        },
                    ],
                },
            )

        respx.post("https://llm.test/v1/chat/completions").mock(side_effect=_side_effect)
        r = await provider.invoke_with_thinking("s", [{"role": "user", "content": "q"}])
        assert r.text == "answer"
        assert r.thinking == "step 1"
        body = json.loads(captured["body"])
        assert body["reasoning_effort"] == "high"
        assert body["temperature"] == 1
        await provider.close()

    @respx.mock
    async def test_invoke_with_thinking_omits_effort_when_off(self, monkeypatch) -> None:
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        p = RunningHubLLMProvider(
            base_url="https://llm.test/v1",
            api_key="k",
            model_id="m",
            reasoning_effort="off",
        )
        captured: dict = {}

        def _side_effect(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content.decode()
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        respx.post("https://llm.test/v1/chat/completions").mock(side_effect=_side_effect)
        await p.invoke_with_thinking("s", [])
        assert "reasoning_effort" not in json.loads(captured["body"])
        await p.close()


class TestRunningHubLLMRetry:
    @respx.mock
    async def test_retries_on_5xx_then_succeeds(self, provider: RunningHubLLMProvider) -> None:
        route = respx.post("https://llm.test/v1/chat/completions")
        n = {"i": 0}

        def side_effect(_request: httpx.Request) -> httpx.Response:
            n["i"] += 1
            if n["i"] == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        route.mock(side_effect=side_effect)
        with patch("styleclaw.providers.llm.runninghub_llm.asyncio.sleep", new_callable=AsyncMock):
            text = await provider.invoke("s", [])
        assert text == "ok"
        assert route.call_count == 2
        await provider.close()

    @respx.mock
    async def test_no_retry_on_4xx(self, provider: RunningHubLLMProvider) -> None:
        respx.post("https://llm.test/v1/chat/completions").mock(return_value=httpx.Response(400))
        with pytest.raises(httpx.HTTPStatusError):
            await provider.invoke("s", [])
        await provider.close()


class TestRunningHubLLMAsyncContextManager:
    async def test_aenter_returns_self(self, provider: RunningHubLLMProvider) -> None:
        async with provider as ctx:
            assert ctx is provider

    async def test_aexit_closes_http(self, provider: RunningHubLLMProvider) -> None:
        async with provider:
            pass
        assert provider._http.is_closed
