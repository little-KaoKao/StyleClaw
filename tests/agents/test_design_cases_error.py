from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from styleclaw.agents.design_cases import design_cases


class TestDesignCasesErrorRecovery:
    async def test_recovers_from_truncated_json(self) -> None:
        raw = (
            '{"cases": [{"id": "am-01", "category": "adult_male", "description": "warrior"}'
            ', {"id"]}'
        )

        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = raw

        config = await design_cases(mock_llm, "anime", "bold style", 1)
        assert len(config.cases) == 1
        assert config.cases[0].id == "am-01"

    async def test_raises_on_no_closing_brace(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = '{"cases": [{"id": "am-01"'

        with pytest.raises(json.JSONDecodeError):
            await design_cases(mock_llm, "anime", "bold style", 1)

    async def test_raises_on_no_closing_bracket(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = '{"cases": [{"id": "am-01"}'

        with pytest.raises(json.JSONDecodeError):
            await design_cases(mock_llm, "anime", "bold style", 1)

    async def test_raises_on_completely_invalid_json(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "not json at all, no braces"

        with pytest.raises(json.JSONDecodeError):
            await design_cases(mock_llm, "anime", "bold style", 1)

    async def test_raises_on_empty_cases(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = json.dumps({"cases": []})

        with pytest.raises(ValueError, match="zero cases"):
            await design_cases(mock_llm, "anime", "bold style", 1)

    async def test_valid_response_parses_normally(self) -> None:
        valid = json.dumps({
            "cases": [
                {"id": "am-01", "category": "adult_male", "description": "warrior", "aspect_ratio": "9:16"},
            ]
        })

        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = valid

        config = await design_cases(mock_llm, "anime", "bold style", 1)
        assert len(config.cases) == 1
        assert config.trigger_phrase == "bold style"
        assert config.batch == 1
