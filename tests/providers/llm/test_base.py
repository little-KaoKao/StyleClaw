from __future__ import annotations

from unittest.mock import AsyncMock

from styleclaw.providers.llm.base import LLMProvider, LLMResponse


class TestLLMProvider:
    def test_protocol_check_with_mock(self) -> None:
        mock = AsyncMock()
        mock.invoke = AsyncMock(return_value="result")
        assert isinstance(mock, LLMProvider)


class TestLLMResponse:
    def test_has_text_and_thinking_fields(self) -> None:
        r = LLMResponse(text="hi", thinking="because")
        assert r.text == "hi"
        assert r.thinking == "because"

    def test_thinking_defaults_to_empty(self) -> None:
        r = LLMResponse(text="hi")
        assert r.thinking == ""


class TestLLMProviderProtocol:
    def test_protocol_requires_invoke(self) -> None:
        class Fake:
            async def invoke(self, system, messages, max_tokens=4096, temperature=0.3):
                return "ok"

            async def invoke_with_thinking(
                self, system, messages, max_tokens=4096, thinking_budget=5000
            ):
                return LLMResponse(text="ok")

        assert isinstance(Fake(), LLMProvider)

    def test_protocol_requires_invoke_with_thinking(self) -> None:
        class Fake:
            async def invoke(self, system, messages, max_tokens=4096, temperature=0.3):
                return "ok"

            async def invoke_with_thinking(
                self, system, messages, max_tokens=4096, thinking_budget=5000
            ):
                return LLMResponse(text="ok", thinking="why")

        assert isinstance(Fake(), LLMProvider)
