from __future__ import annotations

import pytest

from styleclaw.core.models import (
    ModelEvaluation,
    ModelScore,
    ProjectConfig,
    StyleAnalysis,
    TaskRecord,
    TaskStatus,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def proj():
    cfg = ProjectConfig(name="p", ip_info="x", ref_images=[])
    project_store.create_project(cfg)
    return "p"


class TestModelSelectDir:
    def test_pass_dir_created(self, proj):
        d = project_store.model_select_dir(proj, pass_num=1)
        assert d.exists()
        assert d.name == "pass-001"
        assert d.parent.name == "model-select"

    def test_pass_dir_separates_passes(self, proj):
        d1 = project_store.model_select_dir(proj, pass_num=1)
        d2 = project_store.model_select_dir(proj, pass_num=2)
        assert d1 != d2
        assert d2.name == "pass-002"


class TestPassScopedAnalysis:
    def test_save_and_load_analysis_uses_pass(self, proj):
        analysis = StyleAnalysis(trigger_phrase="bold ink")
        project_store.save_analysis(proj, analysis, pass_num=1)
        loaded = project_store.load_analysis(proj, pass_num=1)
        assert loaded.trigger_phrase == "bold ink"

    def test_load_analysis_pass_2_falls_back_to_pass_1(self, proj):
        """When a project enters a new pass, the analysis from pass 1 is the
        starting point until retest-models seeds pass-N (F2)."""
        project_store.save_analysis(
            proj, StyleAnalysis(trigger_phrase="from pass 1"), pass_num=1,
        )
        loaded = project_store.load_analysis(proj, pass_num=2)
        assert loaded.trigger_phrase == "from pass 1"


class TestPassScopedTaskRecords:
    def test_save_and_load_task_record_uses_pass(self, proj):
        record = TaskRecord(task_id="t1", model_id="mj-v7", status=TaskStatus.SUCCESS)
        project_store.save_task_record(
            proj, "mj-v7", record, variant="prompt-only", pass_num=2,
        )
        loaded = project_store.load_task_record(
            proj, "mj-v7", variant="prompt-only", pass_num=2,
        )
        assert loaded.task_id == "t1"

    def test_load_all_pass_records_are_isolated(self, proj):
        r1 = TaskRecord(task_id="t-pass1", model_id="mj-v7", status=TaskStatus.SUCCESS)
        r2 = TaskRecord(task_id="t-pass2", model_id="mj-v7", status=TaskStatus.SUCCESS)
        project_store.save_task_record(proj, "mj-v7", r1, variant="prompt-only", pass_num=1)
        project_store.save_task_record(proj, "mj-v7", r2, variant="prompt-only", pass_num=2)

        pass1 = project_store.load_all_task_records(proj, pass_num=1)
        pass2 = project_store.load_all_task_records(proj, pass_num=2)
        assert pass1["mj-v7/prompt-only"].task_id == "t-pass1"
        assert pass2["mj-v7/prompt-only"].task_id == "t-pass2"


class TestPassScopedEvaluation:
    def test_save_and_load_evaluation_uses_pass(self, proj):
        ev = ModelEvaluation(
            evaluations=[ModelScore(model="mj-v7", total=8.5)],
            recommendation="mj-v7",
        )
        project_store.save_evaluation(proj, ev, pass_num=1)
        loaded = project_store.load_evaluation(proj, pass_num=1)
        assert loaded.recommendation == "mj-v7"

    def test_evaluations_isolated_between_passes(self, proj):
        ev1 = ModelEvaluation(recommendation="mj-v7")
        ev2 = ModelEvaluation(recommendation="niji7")
        project_store.save_evaluation(proj, ev1, pass_num=1)
        project_store.save_evaluation(proj, ev2, pass_num=2)
        assert project_store.load_evaluation(proj, pass_num=1).recommendation == "mj-v7"
        assert project_store.load_evaluation(proj, pass_num=2).recommendation == "niji7"
