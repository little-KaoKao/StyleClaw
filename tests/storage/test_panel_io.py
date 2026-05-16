from __future__ import annotations

import pytest

from styleclaw.core.models import (
    PanelProposal,
    PanelResult,
    PanelScore,
    ProjectConfig,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


def _make_project(name: str = "p") -> str:
    project_store.create_project(ProjectConfig(name=name))
    return name


def _sample_result() -> PanelResult:
    return PanelResult(
        proposals=[PanelProposal(model_id="a", payload={"x": 1})],
        scores=[PanelScore(evaluator_model_id="a", target_model_id="b", score=7.0)],
        winner_model_id="a",
        averages={"a": 7.0},
    )


class TestPanelStorage:
    def test_save_and_load_round_panel(self):
        name = _make_project()
        result = _sample_result()
        project_store.save_round_panel_result(name, round_num=1, result=result, pass_num=1)
        loaded = project_store.load_round_panel_result(name, round_num=1, pass_num=1)
        assert loaded == result

    def test_save_and_load_model_select_panel(self):
        name = _make_project()
        result = _sample_result()
        project_store.save_model_select_panel_result(name, result=result, pass_num=1)
        loaded = project_store.load_model_select_panel_result(name, pass_num=1)
        assert loaded == result

    def test_load_returns_none_when_missing(self):
        name = _make_project()
        assert project_store.load_round_panel_result(name, round_num=1, pass_num=1) is None
        assert project_store.load_model_select_panel_result(name, pass_num=1) is None
