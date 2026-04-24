from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from styleclaw.agents.refine_prompt import _build_history_text, refine_prompt
from styleclaw.core.models import DimensionScores, RoundEvaluation, RoundScore


@pytest.fixture
def ref_images(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(2):
        p = tmp_path / f"ref-{i}.png"
        Image.new("RGB", (100, 100)).save(p)
        paths.append(p)
    return paths


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.invoke.return_value = json.dumps({
        "trigger_phrase": "refined bold anime style with high contrast",
        "model_params": {},
        "adjustment_note": "increased contrast, warmer tones",
    })
    return llm


@pytest.fixture
def sample_evaluations() -> list[RoundEvaluation]:
    scores = DimensionScores(
        color_palette=7.0, line_style=6.5, lighting=7.5, texture=7.0, overall_mood=8.0,
    )
    return [
        RoundEvaluation(
            round=1,
            evaluations=[RoundScore(model="mj-v7", scores=scores, total=7.2)],
            next_direction="increase contrast",
        ),
    ]


class TestRefinePrompt:
    async def test_returns_prompt_config(self, mock_llm, ref_images, sample_evaluations) -> None:
        result = await refine_prompt(
            mock_llm, ref_images, "current trigger", 2, "anime IP", sample_evaluations,
        )
        assert "refined bold" in result.trigger_phrase
        assert result.round == 2
        assert result.derived_from == "round-001"

    async def test_first_round_derived_from_initial(self, mock_llm, ref_images) -> None:
        result = await refine_prompt(
            mock_llm, ref_images, "initial", 1, "anime", [],
        )
        assert result.derived_from == "initial-analysis"

    async def test_human_direction_passed(self, mock_llm, ref_images) -> None:
        await refine_prompt(
            mock_llm, ref_images, "trigger", 1, "ip", [], human_direction="warmer colors",
        )
        call_args = mock_llm.invoke.call_args
        assert "warmer colors" in call_args.kwargs["system"]


class TestBuildHistoryText:
    def test_empty_evaluations(self) -> None:
        assert _build_history_text([]) == "(no previous evaluations)"

    def test_formats_scores(self, sample_evaluations) -> None:
        text = _build_history_text(sample_evaluations)
        assert "Round 1" in text
        assert "mj-v7" in text
        assert "color=7.0" in text

    def test_limits_to_last_3(self) -> None:
        evals = [
            RoundEvaluation(round=i, evaluations=[], next_direction=f"dir-{i}")
            for i in range(1, 6)
        ]
        text = _build_history_text(evals)
        assert "Round 3" in text
        assert "Round 4" in text
        assert "Round 5" in text
        assert "Round 1" not in text
