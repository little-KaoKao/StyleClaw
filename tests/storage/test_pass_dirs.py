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

    def test_legacy_analysis_readable_at_pass_1(self, proj):
        """A pre-existing `model-select/initial-analysis.json` is readable as pass 1."""
        legacy_dir = project_store.project_dir(proj) / "model-select"
        legacy_dir.mkdir(exist_ok=True)
        legacy_path = legacy_dir / "initial-analysis.json"
        legacy_path.write_text('{"trigger_phrase": "legacy"}', encoding="utf-8")

        loaded = project_store.load_analysis(proj, pass_num=1)
        assert loaded.trigger_phrase == "legacy"

    def test_pass_specific_shadows_legacy(self, proj):
        """If pass-001 has its own analysis, prefer it over legacy."""
        legacy_dir = project_store.project_dir(proj) / "model-select"
        legacy_dir.mkdir(exist_ok=True)
        (legacy_dir / "initial-analysis.json").write_text(
            '{"trigger_phrase": "legacy"}', encoding="utf-8",
        )
        new = StyleAnalysis(trigger_phrase="pass1 override")
        project_store.save_analysis(proj, new, pass_num=1)

        loaded = project_store.load_analysis(proj, pass_num=1)
        assert loaded.trigger_phrase == "pass1 override"


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

    def test_legacy_flat_records_readable_at_pass_1(self, proj):
        legacy = (
            project_store.project_dir(proj)
            / "model-select" / "results" / "mj-v7" / "prompt-only"
        )
        legacy.mkdir(parents=True)
        (legacy / "task.json").write_text(
            '{"task_id": "legacy-t", "model_id": "mj-v7", "status": "SUCCESS"}',
            encoding="utf-8",
        )
        records = project_store.load_all_task_records(proj, pass_num=1)
        assert records["mj-v7/prompt-only"].task_id == "legacy-t"


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
