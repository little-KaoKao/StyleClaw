from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from styleclaw.agents.design_cases import _format_skeleton, design_cases
from styleclaw.core.case_generator import generate_case_skeleton


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    cases = [
        {"id": f"am-{i:03d}", "category": "adult_male", "description": f"Male char {i}", "aspect_ratio": "9:16"}
        for i in range(1, 11)
    ]
    llm.invoke.return_value = json.dumps({"cases": cases})
    return llm


class TestDesignCases:
    async def test_returns_batch_config(self, mock_llm) -> None:
        result = await design_cases(mock_llm, "anime", "bold style", batch_num=1)
        assert result.batch == 1
        assert result.trigger_phrase == "bold style"
        assert len(result.cases) == 10

    async def test_ip_info_in_system_prompt(self, mock_llm) -> None:
        await design_cases(mock_llm, "Spider-Verse", "trigger", batch_num=1)
        call_args = mock_llm.invoke.call_args
        assert "Spider-Verse" in call_args.kwargs["system"]


class TestFormatSkeleton:
    def test_formats_categories(self) -> None:
        skeleton = generate_case_skeleton()
        text = _format_skeleton(skeleton)
        assert "adult_male" in text
        assert "adult_female" in text
        assert "creature" in text
        assert "(fill in description)" in text
