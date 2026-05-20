from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from styleclaw.agents.design_cases import design_cases


class TestDesignCasesErrorRecovery:
    @pytest.fixture(autouse=True)
    def _force_single_shard(self, monkeypatch):
        monkeypatch.setattr(
            "styleclaw.agents.design_cases.DESIGN_CASES_SHARDS", 1
        )

    async def test_recovers_from_truncated_json(self) -> None:
        raw = (
            '{"cases": [{"id": "am-01", "category": "adult_male", "description": "warrior"}'
            ', {"id"]}'
        )

        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = raw

        with pytest.raises(ValueError, match="returned 1 cases.*expected 100"):
            await design_cases(mock_llm, "anime", "bold style", 1)

    async def test_raises_on_no_closing_brace(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = '{"cases": [{"id": "am-01"'

        with pytest.raises(json.JSONDecodeError):
            await design_cases(mock_llm, "anime", "bold style", 1)

    async def test_raises_on_no_closing_bracket(self) -> None:
        """Input has a `}` but no `]`. The recovery helper closes the array,
        but the recovered case is missing required fields, so Pydantic
        validation fails — which is the correct surface for "garbage in,
        clear failure out"."""
        from pydantic import ValidationError
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = '{"cases": [{"id": "am-01"}'

        with pytest.raises((json.JSONDecodeError, ValidationError, ValueError)):
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

    async def test_valid_short_response_rejected_by_total_check(self) -> None:
        valid = json.dumps({
            "cases": [
                {"id": "am-01", "category": "adult_male", "description": "warrior", "aspect_ratio": "9:16"},
            ]
        })

        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = valid

        with pytest.raises(ValueError, match="returned 1 cases.*expected 100"):
            await design_cases(mock_llm, "anime", "bold style", 1)
