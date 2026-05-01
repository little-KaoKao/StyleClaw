from __future__ import annotations

from pathlib import Path

import pytest

from styleclaw.core.models import ProjectConfig
from styleclaw.scripts.migrate import migrate_project
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def proj() -> str:
    project_store.create_project(ProjectConfig(name="p", ip_info="x"))
    return "p"


def _legacy_ms(proj: str) -> Path:
    return project_store.project_dir(proj) / "model-select"


def _legacy_sr(proj: str) -> Path:
    return project_store.project_dir(proj) / "style-refine"


class TestMigrateModelSelect:
    def test_moves_loose_files_into_pass_001(self, proj):
        ms = _legacy_ms(proj)
        (ms / "initial-analysis.json").write_text('{"trigger_phrase": "t"}', encoding="utf-8")
        (ms / "evaluation.json").write_text('{"recommendation": "mj-v7"}', encoding="utf-8")

        result = migrate_project(proj)
        assert result.model_select_migrated

        pass1 = ms / "pass-001"
        assert (pass1 / "initial-analysis.json").exists()
        assert (pass1 / "evaluation.json").exists()
        assert not (ms / "initial-analysis.json").exists()
        assert not (ms / "evaluation.json").exists()

    def test_moves_results_subtree(self, proj):
        ms = _legacy_ms(proj)
        legacy_results = ms / "results" / "mj-v7" / "prompt-only"
        legacy_results.mkdir(parents=True)
        (legacy_results / "task.json").write_text(
            '{"task_id": "t1", "model_id": "mj-v7", "status": "SUCCESS"}',
            encoding="utf-8",
        )

        result = migrate_project(proj)
        assert result.model_select_migrated

        new = ms / "pass-001" / "results" / "mj-v7" / "prompt-only" / "task.json"
        assert new.exists()
        assert not (ms / "results").exists()

    def test_idempotent(self, proj):
        ms = _legacy_ms(proj)
        (ms / "initial-analysis.json").write_text('{"trigger_phrase": "t"}', encoding="utf-8")

        first = migrate_project(proj)
        assert first.model_select_migrated

        second = migrate_project(proj)
        assert not second.model_select_migrated
        assert not second.anything_migrated

    def test_refuses_if_pass_001_target_exists(self, proj):
        ms = _legacy_ms(proj)
        (ms / "initial-analysis.json").write_text('{"trigger_phrase": "legacy"}', encoding="utf-8")
        pass1 = ms / "pass-001"
        pass1.mkdir()
        (pass1 / "initial-analysis.json").write_text(
            '{"trigger_phrase": "already here"}', encoding="utf-8",
        )

        with pytest.raises(FileExistsError):
            migrate_project(proj)


class TestMigrateStyleRefine:
    def test_moves_round_dirs_into_pass_001(self, proj):
        sr = _legacy_sr(proj)
        r1 = sr / "round-001"
        r2 = sr / "round-002"
        r1.mkdir(parents=True)
        r2.mkdir(parents=True)
        (r1 / "prompt.json").write_text("{}", encoding="utf-8")

        result = migrate_project(proj)
        assert result.style_refine_rounds_migrated == [1, 2]

        assert (sr / "pass-001" / "round-001" / "prompt.json").exists()
        assert (sr / "pass-001" / "round-002").exists()
        assert not r1.exists()
        assert not r2.exists()

    def test_noop_when_no_legacy_rounds(self, proj):
        result = migrate_project(proj)
        assert result.style_refine_rounds_migrated == []


class TestMigrateProjectNotFound:
    def test_raises_for_missing_project(self):
        with pytest.raises(FileNotFoundError):
            migrate_project("does-not-exist")


class TestMigrateCombined:
    def test_migrates_both_layouts_in_one_call(self, proj):
        ms = _legacy_ms(proj)
        (ms / "initial-analysis.json").write_text('{"trigger_phrase": "t"}', encoding="utf-8")
        sr = _legacy_sr(proj)
        (sr / "round-001").mkdir(parents=True)

        result = migrate_project(proj)
        assert result.model_select_migrated
        assert result.style_refine_rounds_migrated == [1]

        assert (ms / "pass-001" / "initial-analysis.json").exists()
        assert (sr / "pass-001" / "round-001").exists()
