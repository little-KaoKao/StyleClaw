from __future__ import annotations

from unittest.mock import AsyncMock

from styleclaw.providers.llm.base import LLMProvider


class TestLLMProvider:
    def test_protocol_check_with_mock(self) -> None:
        mock = AsyncMock()
        mock.invoke = AsyncMock(return_value="result")
        assert isinstance(mock, LLMProvider)
