from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from styleclaw.agents.evaluate_result import evaluate_round


@pytest.fixture
def ref_images(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(2):
        p = tmp_path / f"ref-{i}.png"
        Image.new("RGB", (100, 100)).save(p)
        paths.append(p)
    return paths


@pytest.fixture
def model_images(tmp_path: Path) -> dict[str, list[Path]]:
    result = {}
    for model in ("mj-v7",):
        paths = []
        for i in range(2):
            p = tmp_path / f"{model}-{i}.png"
            Image.new("RGB", (100, 100)).save(p)
            paths.append(p)
        result[model] = paths
    return result


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.invoke.return_value = json.dumps({
        "round": 1,
        "evaluations": [{
            "model": "mj-v7",
            "scores": {
                "color_palette": 8.0,
                "line_style": 7.5,
                "lighting": 7.0,
                "texture": 7.5,
                "overall_mood": 8.0,
            },
            "total": 7.6,
            "analysis": "good match",
        }],
        "recommendation": "approve",
        "next_direction": "increase contrast",
    })
    return llm


class TestEvaluateRound:
    async def test_returns_round_evaluation(self, mock_llm, ref_images, model_images) -> None:
        result = await evaluate_round(mock_llm, ref_images, model_images, round_num=1)
        assert result.recommendation == "approve"
        assert len(result.evaluations) == 1
        assert result.evaluations[0].model == "mj-v7"

    async def test_includes_round_in_system_prompt(self, mock_llm, ref_images, model_images) -> None:
        await evaluate_round(mock_llm, ref_images, model_images, round_num=3)
        call_args = mock_llm.invoke.call_args
        assert "3" in call_args.kwargs["system"]
