from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from styleclaw.agents.select_model_panel import select_models_with_panel
from styleclaw.core.models import ModelEvaluation, PanelResult


def _stub(model_id: str, recommendation: str, score: float):
    prov = AsyncMock()
    prov._model_id = model_id
    eval_json = json.dumps({
        "evaluations": [],
        "recommendation": recommendation,
        "recommended_variant": "prompt-only",
        "next_direction": "",
    })
    score_json = json.dumps({"score": score, "rationale": "ok"})
    prov.invoke.side_effect = [eval_json] + [score_json] * 10
    return prov


@pytest.mark.asyncio
async def test_select_models_with_panel(tmp_path, monkeypatch):
    # Disable the LLM image cache: this test runs 3 concurrent proposals that
    # all encode the same images, and the cache writer (single .tmp filename
    # per cache key) races on Windows file replace.
    monkeypatch.setenv("STYLECLAW_LLM_IMAGE_CACHE", "0")

    refs = [tmp_path / "ref1.png"]
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()

    img = Image.new("RGB", (64, 64), color="red")
    img.save(refs[0])
    img.save(img_dir / "out1.png")

    model_images = {"mj-v7/prompt-only": [img_dir / "out1.png"]}
    llms = [
        _stub("a", "mj-v7", 9.0),
        _stub("b", "niji7", 7.0),
        _stub("c", "mj-v7", 8.5),
    ]
    labels = ["A", "B", "C"]

    evaluation, panel = await select_models_with_panel(
        llms, labels, refs, model_images,
    )

    assert isinstance(evaluation, ModelEvaluation)
    assert isinstance(panel, PanelResult)
    assert panel.winner_model_id in {"a", "b", "c"}
    # Winner's recommendation should be reflected in the merged ModelEvaluation.
    winner_payload = next(p.payload for p in panel.proposals if p.model_id == panel.winner_model_id)
    assert evaluation.recommendation == winner_payload["recommendation"]
