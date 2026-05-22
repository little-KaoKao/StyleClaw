from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from styleclaw.agents.analyze_style import analyze_style


@pytest.fixture
def ref_images(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(2):
        p = tmp_path / f"ref-{i}.png"
        Image.new("RGB", (100, 100), color=(i * 50, 0, 0)).save(p)
        paths.append(p)
    return paths


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.invoke.return_value = json.dumps({
        "visual_style": "anime illustration",
        "color_science": "warm tones",
        "lighting_quality": "dramatic",
        "material_texture": "smooth",
        "post_processing": "minimal presence",
        "spatial_perspective": "centered",
        "dynamic_state": "energetic",
        "trigger_phrase": "bold colorful anime style",
        "trigger_variants": ["variant1"],
    })
    return llm


class TestAnalyzeStyle:
    async def test_returns_style_analysis(self, mock_llm, ref_images) -> None:
        result = await analyze_style(mock_llm, ref_images, "anime IP")
        assert result.trigger_phrase == "bold colorful anime style"
        assert result.color_science == "warm tones"

    async def test_sends_images_to_llm(self, mock_llm, ref_images) -> None:
        await analyze_style(mock_llm, ref_images, "anime IP")
        call_args = mock_llm.invoke.call_args
        messages = call_args.kwargs["messages"]
        content = messages[0]["content"]
        image_blocks = [c for c in content if c["type"] == "image"]
        assert len(image_blocks) == 2

    async def test_handles_markdown_fenced_response(self, ref_images) -> None:
        llm = AsyncMock()
        llm.invoke.return_value = '```json\n' + json.dumps({
            "trigger_phrase": "test trigger",
        }) + '\n```'
        result = await analyze_style(llm, ref_images, "test")
        assert result.trigger_phrase == "test trigger"

    async def test_ip_info_in_system_prompt(self, mock_llm, ref_images) -> None:
        await analyze_style(mock_llm, ref_images, "Spider-Verse style")
        call_args = mock_llm.invoke.call_args
        assert "Spider-Verse style" in call_args.kwargs["system"]

    async def test_test_subjects_parsed_when_provided(self, ref_images) -> None:
        llm = AsyncMock()
        llm.invoke.return_value = json.dumps({
            "trigger_phrase": "test trigger",
            "test_subjects": {
                "male": "a young man in a red and blue patterned bodysuit",
                "female": "a young woman with long dark hair in a tactical outfit",
            },
        })
        result = await analyze_style(llm, ref_images, "test")
        assert result.test_subjects == {
            "male": "a young man in a red and blue patterned bodysuit",
            "female": "a young woman with long dark hair in a tactical outfit",
        }

    async def test_test_subjects_defaults_to_empty(self, mock_llm, ref_images) -> None:
        # mock_llm response has no test_subjects key → must default to {}
        result = await analyze_style(mock_llm, ref_images, "anime IP")
        assert result.test_subjects == {}

    async def test_partial_test_subjects_preserved(self, ref_images) -> None:
        llm = AsyncMock()
        llm.invoke.return_value = json.dumps({
            "trigger_phrase": "test trigger",
            "test_subjects": {"male": "M-only"},
        })
        result = await analyze_style(llm, ref_images, "test")
        assert result.test_subjects == {"male": "M-only"}

    async def test_analyze_prompt_mentions_test_subjects(self, mock_llm, ref_images) -> None:
        await analyze_style(mock_llm, ref_images, "anime IP")
        call_args = mock_llm.invoke.call_args
        system_prompt = call_args.kwargs["system"]
        assert "test_subjects" in system_prompt
        # IP-stripping guidance must be in the prompt
        assert "Strip IP-identifying" in system_prompt or "IP-identifying" in system_prompt
