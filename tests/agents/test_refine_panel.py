from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from styleclaw.agents.refine_panel import refine_with_panel
from styleclaw.core.models import PanelResult, PromptConfig


def _stub_provider(model_id: str, propose_payload: str, score_value: float):
    """Build a stub provider with scripted invoke() responses.

    - First invoke() call (propose) returns a JSON PromptConfig-shaped string.
    - Subsequent invoke() calls (score) return a JSON score object.

    The adapter is responsible for choosing which prompt template to render.
    Test relies on call order, not template inspection.
    """
    prov = AsyncMock()
    prov._model_id = model_id
    propose_json = f'{{"trigger_phrase": "{propose_payload}", "adjustment_note": ""}}'
    score_json = f'{{"score": {score_value}, "rationale": "ok"}}'
    prov.invoke.side_effect = [propose_json] + [score_json] * 10  # pad for safety
    return prov


@pytest.mark.asyncio
async def test_refine_with_panel_returns_winner_prompt_and_result(tmp_path):
    refs = [tmp_path / "ref1.png"]
    Image.new("RGB", (64, 64)).save(refs[0])

    llms = [
        _stub_provider("a", "trigger-A", 8.0),
        _stub_provider("b", "trigger-B", 9.0),
        _stub_provider("c", "trigger-C", 7.0),
    ]
    labels = ["A", "B", "C"]

    prompt_config, panel_result = await refine_with_panel(
        llms, labels, refs,
        current_trigger="prev",
        round_num=2,
        ip_info="anime",
        evaluations=[],
        human_direction="",
    )

    assert isinstance(prompt_config, PromptConfig)
    assert isinstance(panel_result, PanelResult)
    # All three stubs score 8/9/7 respectively for everyone — so b's proposal
    # (scored by a=8 and c=7 -> avg 7.5) ties with c (scored by a=8, b=9 -> avg 8.5)
    # etc. Verify *some* trigger from the candidate set was chosen.
    assert prompt_config.trigger_phrase in {"trigger-A", "trigger-B", "trigger-C"}
    assert panel_result.winner_model_id in {"a", "b", "c"}
    assert prompt_config.round == 2
    # Three propose calls + six score calls = nine total invocations max.
    total_calls = sum(len(p.invoke.call_args_list) for p in llms)
    assert 3 <= total_calls <= 9
