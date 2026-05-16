"""Tests that the HTML reports surface panel.json sidecar data when present,
and silently omit the block when the sidecar is absent."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from styleclaw.core.models import (
    DimensionScores,
    ModelEvaluation,
    ModelScore,
    PanelProposal,
    PanelResult,
    Phase,
    ProjectConfig,
    ProjectState,
    PromptConfig,
    RoundEvaluation,
    RoundScore,
    StyleAnalysis,
)
from styleclaw.scripts.report import (
    generate_model_select_report,
    generate_style_refine_report,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


def _create_project(phase: Phase = Phase.STYLE_REFINE, current_round: int = 1) -> str:
    name = "panel-test-proj"
    config = ProjectConfig(
        name=name,
        ip_info="anime style",
        ref_images=["refs/ref-001.png"],
    )
    root = project_store.create_project(config)
    Image.new("RGB", (100, 100), color=(0, 128, 255)).save(root / "refs" / "ref-001.png")
    state = ProjectState(
        phase=phase,
        current_round=current_round,
        current_batch=1,
    )
    project_store.save_state(name, state)
    return name


def _sample_panel_result(label: str = "Opus") -> PanelResult:
    return PanelResult(
        proposals=[PanelProposal(model_id="a", label=label, payload={"trigger_phrase": "t"})],
        scores=[],
        winner_model_id="a",
        averages={"a": 8.0},
    )


def _seed_style_refine(name: str) -> None:
    project_store.save_prompt_config(name, 1, PromptConfig(round=1, trigger_phrase="test trigger"))
    project_store.save_round_evaluation(
        name,
        1,
        RoundEvaluation(
            round=1,
            evaluations=[
                RoundScore(
                    model="mj-v7",
                    scores=DimensionScores(
                        color_palette=7.0,
                        line_style=7.0,
                        lighting=7.0,
                        texture=7.0,
                        overall_mood=7.0,
                    ),
                    total=7.0,
                    analysis="ok",
                )
            ],
        ),
    )


def _seed_model_select(name: str) -> None:
    project_store.save_analysis(name, StyleAnalysis(trigger_phrase="bold anime"))
    project_store.save_evaluation(
        name,
        ModelEvaluation(
            evaluations=[
                ModelScore(
                    model="mj-v7",
                    scores=DimensionScores(
                        color_palette=8.0,
                        line_style=7.5,
                        lighting=7.0,
                        texture=7.5,
                        overall_mood=8.0,
                    ),
                    total=7.8,
                    analysis="great",
                    suggestions="none",
                ),
            ],
            recommendation="mj-v7",
        ),
    )


class TestStyleRefineReportPanel:
    def test_includes_panel_block_when_sidecar_present(self) -> None:
        name = _create_project(phase=Phase.STYLE_REFINE, current_round=1)
        _seed_style_refine(name)
        project_store.save_round_panel_result(name, 1, _sample_panel_result("Opus"))

        path = generate_style_refine_report(name, round_num=1)
        html = Path(path).read_text(encoding="utf-8")
        assert "Panel review" in html
        assert "Opus" in html

    def test_omits_panel_block_when_no_sidecar(self) -> None:
        name = _create_project(phase=Phase.STYLE_REFINE, current_round=1)
        _seed_style_refine(name)

        path = generate_style_refine_report(name, round_num=1)
        html = Path(path).read_text(encoding="utf-8")
        assert "Panel review" not in html


class TestModelSelectReportPanel:
    def test_includes_panel_block_when_sidecar_present(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT, current_round=1)
        _seed_model_select(name)
        project_store.save_model_select_panel_result(name, _sample_panel_result("Haiku"))

        path = generate_model_select_report(name)
        html = Path(path).read_text(encoding="utf-8")
        assert "Panel review" in html
        assert "Haiku" in html

    def test_omits_panel_block_when_no_sidecar(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT, current_round=1)
        _seed_model_select(name)

        path = generate_model_select_report(name)
        html = Path(path).read_text(encoding="utf-8")
        assert "Panel review" not in html
