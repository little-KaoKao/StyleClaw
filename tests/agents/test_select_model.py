from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from styleclaw.agents.select_model import evaluate_models


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
    for model in ("mj-v7", "niji7"):
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
        "evaluations": [
            {"model": "mj-v7", "scores": {"color": 8.0}, "total": 8.0, "analysis": "good", "suggestions": "none"},
            {"model": "niji7", "scores": {"color": 7.0}, "total": 7.0, "analysis": "ok", "suggestions": "try more"},
        ],
        "recommendation": "mj-v7",
        "next_direction": "increase saturation",
    })
    return llm


class TestEvaluateModels:
    async def test_returns_evaluation(self, mock_llm, ref_images, model_images) -> None:
        result = await evaluate_models(mock_llm, ref_images, model_images)
        assert result.recommendation == "mj-v7"
        assert len(result.evaluations) == 2

    async def test_sends_ref_and_model_images(self, mock_llm, ref_images, model_images) -> None:
        await evaluate_models(mock_llm, ref_images, model_images)
        call_args = mock_llm.invoke.call_args
        messages = call_args.kwargs["messages"]
        content = messages[0]["content"]
        image_blocks = [c for c in content if c["type"] == "image"]
        assert len(image_blocks) == 6  # 2 ref + 2*2 model
