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

    async def test_no_feedback_section_when_empty(self, mock_llm) -> None:
        await design_cases(mock_llm, "anime", "x", batch_num=1)
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        # Placeholder must not leak through and there must be no feedback heading
        assert "{feedback_section}" not in sys_prompt
        assert "User feedback on previous batch" not in sys_prompt

    async def test_feedback_appended_to_prompt(self, mock_llm) -> None:
        feedback = "上一批室内场景太少，多加一些咖啡馆和书房"
        await design_cases(mock_llm, "anime", "x", batch_num=2, feedback=feedback)
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        assert "User feedback on previous batch" in sys_prompt
        assert feedback in sys_prompt

    async def test_whitespace_only_feedback_treated_as_empty(self, mock_llm) -> None:
        await design_cases(mock_llm, "anime", "x", batch_num=2, feedback="   \n  ")
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        assert "User feedback on previous batch" not in sys_prompt


class TestFormatSkeleton:
    def test_formats_categories(self) -> None:
        skeleton = generate_case_skeleton()
        text = _format_skeleton(skeleton)
        assert "adult_male" in text
        assert "adult_female" in text
        assert "creature" in text
        assert "(fill in description)" in text
