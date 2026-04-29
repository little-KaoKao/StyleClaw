from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from styleclaw.agents.analyze_style import (
    analyze_style,
    analyze_style_with_thinking,
)
from styleclaw.core.models import StyleAnalysis
from styleclaw.providers.llm.base import LLMResponse


@pytest.fixture
def ref_image(tmp_path: Path) -> Path:
    from PIL import Image
    p = tmp_path / "ref.png"
    Image.new("RGB", (32, 32), "red").save(p)
    return p


class TestAnalyzeStyleWithThinking:
    async def test_returns_analysis_and_thinking(self, ref_image, tmp_path):
        fake_llm = AsyncMock()
        fake_llm.invoke_with_thinking = AsyncMock(
            return_value=LLMResponse(
                text='{"trigger_phrase": "bold ink style"}',
                thinking="I observed heavy linework and high contrast.",
            )
        )

        analysis, thinking = await analyze_style_with_thinking(
            fake_llm, [ref_image], ip_info="test ip", thinking_budget=3000,
        )
        assert isinstance(analysis, StyleAnalysis)
        assert analysis.trigger_phrase == "bold ink style"
        assert thinking == "I observed heavy linework and high contrast."

    async def test_plain_analyze_style_still_returns_just_analysis(
        self, ref_image,
    ):
        fake_llm = AsyncMock()
        fake_llm.invoke = AsyncMock(
            return_value='{"trigger_phrase": "plain"}'
        )
        analysis = await analyze_style(fake_llm, [ref_image], ip_info="test")
        assert isinstance(analysis, StyleAnalysis)
        assert analysis.trigger_phrase == "plain"


from styleclaw.agents.refine_prompt import (
    refine_prompt,
    refine_prompt_with_thinking,
)
from styleclaw.core.models import PromptConfig


class TestRefinePromptWithThinking:
    async def test_returns_config_and_thinking(self, ref_image):
        fake_llm = AsyncMock()
        fake_llm.invoke_with_thinking = AsyncMock(
            return_value=LLMResponse(
                text='{"trigger_phrase": "new trigger"}',
                thinking="Color scored low so I added 'vivid palette'.",
            )
        )
        config, thinking = await refine_prompt_with_thinking(
            fake_llm, [ref_image],
            current_trigger="old trigger", round_num=2,
            ip_info="ip", evaluations=[], human_direction="",
            thinking_budget=3000,
        )
        assert isinstance(config, PromptConfig)
        assert config.trigger_phrase == "new trigger"
        assert config.round == 2
        assert thinking.startswith("Color scored low")
