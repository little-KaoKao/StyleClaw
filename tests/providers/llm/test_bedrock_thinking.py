from __future__ import annotations

import json

import httpx
import pytest
import respx

from styleclaw.providers.llm.base import LLMResponse
from styleclaw.providers.llm.bedrock import BedrockProvider


@pytest.fixture
def provider(monkeypatch) -> BedrockProvider:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    return BedrockProvider(region="us-east-1", model_id="test-model")


class TestBedrockInvokeWithThinking:
    @respx.mock
    async def test_returns_both_text_and_thinking(self, provider: BedrockProvider) -> None:
        route = respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(
            json={
                "content": [
                    {"type": "thinking", "thinking": "Let me reason about this..."},
                    {"type": "text", "text": "final answer"},
                ],
            }
        )

        result = await provider.invoke_with_thinking(
            system="s",
            messages=[{"role": "user", "content": "q"}],
            thinking_budget=5000,
        )
        assert isinstance(result, LLMResponse)
        assert result.text == "final answer"
        assert result.thinking == "Let me reason about this..."
        assert route.called

    @respx.mock
    async def test_request_body_enables_thinking(self, provider: BedrockProvider) -> None:
        captured = {}

        def _handler(request):
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "ok"}]},
            )

        respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).mock(side_effect=_handler)

        await provider.invoke_with_thinking(
            system="s", messages=[], thinking_budget=3000,
        )
        assert captured["body"]["thinking"] == {"type": "enabled", "budget_tokens": 3000}
        # Extended thinking requires temperature == 1.0
        assert captured["body"]["temperature"] == 1.0

    @respx.mock
    async def test_thinking_empty_when_no_thinking_block(
        self, provider: BedrockProvider,
    ) -> None:
        respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(json={"content": [{"type": "text", "text": "answer"}]})

        result = await provider.invoke_with_thinking(
            system="s", messages=[], thinking_budget=5000,
        )
        assert result.text == "answer"
        assert result.thinking == ""

    @respx.mock
    async def test_joins_multiple_thinking_blocks(
        self, provider: BedrockProvider,
    ) -> None:
        respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).respond(
            json={
                "content": [
                    {"type": "thinking", "thinking": "step 1"},
                    {"type": "thinking", "thinking": "step 2"},
                    {"type": "text", "text": "done"},
                ],
            }
        )
        result = await provider.invoke_with_thinking(
            system="s", messages=[], thinking_budget=5000,
        )
        assert result.thinking == "step 1\n\nstep 2"
        assert result.text == "done"

    @respx.mock
    async def test_invoke_without_thinking_unchanged(
        self, provider: BedrockProvider,
    ) -> None:
        """Existing invoke() must not regress."""
        captured = {}

        def _handler(request):
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(
                200, json={"content": [{"type": "text", "text": "ok"}]},
            )

        respx.post(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/invoke"
        ).mock(side_effect=_handler)

        result = await provider.invoke(system="s", messages=[], temperature=0.3)
        assert result == "ok"
        assert "thinking" not in captured["body"]
        assert captured["body"]["temperature"] == 0.3
