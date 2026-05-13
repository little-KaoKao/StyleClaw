from __future__ import annotations

import pytest

from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.orchestrator.suggestions import suggest_next_steps
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


def _setup(name: str, phase: Phase, **state_kwargs) -> None:
    project_store.create_project(ProjectConfig(name=name))
    project_store.save_state(name, ProjectState(phase=phase, **state_kwargs))


def _joined(suggestions: list[str]) -> str:
    return "\n".join(suggestions)


class TestSuggestNextSteps:
    def test_returns_non_empty_list_of_strings(self):
        _setup("p", Phase.INIT)
        result = suggest_next_steps("p")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(s, str) and s for s in result)

    def test_init_phase_suggests_analyze(self):
        _setup("p", Phase.INIT)
        result = suggest_next_steps("p")
        assert 1 <= len(result) <= 5
        assert "分析" in _joined(result) or "analyze" in _joined(result).lower()

    def test_model_select_phase_covers_keywords(self):
        _setup("p", Phase.MODEL_SELECT)
        result = suggest_next_steps("p")
        assert 3 <= len(result) <= 5
        joined = _joined(result)
        assert "sref" in joined or "重测" in joined or "选" in joined or "进入精炼" in joined

    def test_model_select_includes_pass_or_retest(self):
        _setup("p", Phase.MODEL_SELECT)
        result = suggest_next_steps("p")
        joined = _joined(result)
        assert "pass" in joined.lower() or "重测" in joined

    def test_style_refine_phase_keywords(self):
        _setup("p", Phase.STYLE_REFINE, current_round=2)
        result = suggest_next_steps("p")
        assert 3 <= len(result) <= 5
        joined = _joined(result)
        assert ("精炼" in joined) or ("approve" in joined.lower()) or ("回退" in joined)

    def test_style_refine_all_three_keywords_present(self):
        _setup("p", Phase.STYLE_REFINE, current_round=3)
        result = suggest_next_steps("p")
        joined = _joined(result)
        assert "精炼" in joined
        assert "approve" in joined.lower()
        assert "回退" in joined

    def test_batch_t2i_phase_keywords(self):
        _setup("p", Phase.BATCH_T2I)
        result = suggest_next_steps("p")
        assert 3 <= len(result) <= 5
        joined = _joined(result)
        assert "用例" in joined or "批量" in joined or "报告" in joined or "图生图" in joined

    def test_batch_t2i_covers_design_submit_report(self):
        _setup("p", Phase.BATCH_T2I)
        result = suggest_next_steps("p")
        joined = _joined(result)
        assert "用例" in joined
        assert "批量" in joined
        assert "报告" in joined

    def test_batch_i2i_phase_keywords(self):
        _setup("p", Phase.BATCH_I2I)
        result = suggest_next_steps("p")
        assert 3 <= len(result) <= 5
        joined = _joined(result)
        assert "i2i" in joined.lower() or "图生图" in joined or "完成" in joined or "报告" in joined

    def test_batch_i2i_includes_approve_complete(self):
        _setup("p", Phase.BATCH_I2I)
        result = suggest_next_steps("p")
        joined = _joined(result)
        assert "完成" in joined or "结束" in joined

    def test_completed_phase_short_list(self):
        _setup("p", Phase.COMPLETED)
        result = suggest_next_steps("p")
        assert 1 <= len(result) <= 5

    def test_each_suggestion_uses_styleclaw_run_with_project(self):
        _setup("alpha-beta", Phase.MODEL_SELECT)
        result = suggest_next_steps("alpha-beta")
        for line in result:
            assert line.startswith("styleclaw run ")
            assert "alpha-beta" in line
            assert "<project>" not in line

    def test_no_placeholder_left_for_any_phase(self):
        project_store.create_project(ProjectConfig(name="proj-x"))
        for ph in (
            Phase.INIT,
            Phase.MODEL_SELECT,
            Phase.STYLE_REFINE,
            Phase.BATCH_T2I,
            Phase.BATCH_I2I,
            Phase.COMPLETED,
        ):
            project_store.save_state("proj-x", ProjectState(phase=ph))
            result = suggest_next_steps("proj-x")
            for line in result:
                assert "<project>" not in line
                assert "proj-x" in line
