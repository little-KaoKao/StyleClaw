from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from styleclaw.cli import app
from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store

runner = CliRunner()


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")
    monkeypatch.setenv("RUNNINGHUB_API_KEY", "test-key")


def _make_project(name: str, *, phase: Phase, last_updated: str) -> None:
    config = ProjectConfig(name=name, ip_info="ip", ref_images=["refs/x.png"])
    project_store.create_project(config)
    project_store.save_state(name, ProjectState(phase=phase, last_updated=last_updated))


def _iso_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


class TestArchiveCommand:
    def test_archive_moves_project_to_archive_dir(self) -> None:
        _make_project("alpha", phase=Phase.MODEL_SELECT, last_updated=_iso_days_ago(1))

        src = project_store.project_dir("alpha")
        assert src.exists()

        result = runner.invoke(app, ["archive", "alpha"])
        assert result.exit_code == 0, result.output

        assert not src.exists()
        archive_root = project_store.DATA_ROOT / ".archive"
        assert archive_root.exists()
        archived = list(archive_root.iterdir())
        assert len(archived) == 1
        moved = archived[0]
        assert moved.name.endswith("-alpha")
        # Timestamp prefix YYYYMMDD-HHMMSS (15 chars + dash)
        assert len(moved.name.split("-alpha")[0]) == len("20260101-120000")
        assert (moved / "config.json").exists()
        assert (moved / "state.json").exists()

    def test_archive_missing_project_exits_1(self) -> None:
        result = runner.invoke(app, ["archive", "ghost"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "no such" in result.output.lower()

    def test_archived_project_no_longer_listed(self) -> None:
        _make_project("alpha", phase=Phase.MODEL_SELECT, last_updated=_iso_days_ago(1))
        _make_project("beta", phase=Phase.INIT, last_updated=_iso_days_ago(1))

        runner.invoke(app, ["archive", "alpha"])

        remaining = project_store.list_projects()
        assert "alpha" not in remaining
        assert "beta" in remaining


class TestCleanCommand:
    def test_dry_run_lists_stalled_projects(self) -> None:
        _make_project("stale-1", phase=Phase.MODEL_SELECT, last_updated=_iso_days_ago(30))
        _make_project("stale-2", phase=Phase.BATCH_T2I, last_updated=_iso_days_ago(10))
        _make_project("fresh", phase=Phase.MODEL_SELECT, last_updated=_iso_days_ago(2))
        _make_project("done", phase=Phase.COMPLETED, last_updated=_iso_days_ago(60))

        result = runner.invoke(app, ["clean", "--stalled", "--days", "7"])
        assert result.exit_code == 0, result.output

        assert "stale-1" in result.output
        assert "stale-2" in result.output
        assert "fresh" not in result.output
        assert "done" not in result.output
        # Dry run: source dirs are intact
        assert project_store.project_dir("stale-1").exists()
        assert project_store.project_dir("stale-2").exists()

    def test_default_threshold_is_7_days(self) -> None:
        _make_project("stale", phase=Phase.MODEL_SELECT, last_updated=_iso_days_ago(8))
        _make_project("fresh", phase=Phase.MODEL_SELECT, last_updated=_iso_days_ago(6))

        result = runner.invoke(app, ["clean", "--stalled"])
        assert result.exit_code == 0
        assert "stale" in result.output
        # "fresh" is a substring of nothing else in output we care about
        assert "fresh" not in result.output

    def test_yes_flag_archives_stalled_projects(self) -> None:
        _make_project("stale", phase=Phase.MODEL_SELECT, last_updated=_iso_days_ago(30))
        _make_project("fresh", phase=Phase.INIT, last_updated=_iso_days_ago(1))

        result = runner.invoke(app, ["clean", "--stalled", "--yes"])
        assert result.exit_code == 0, result.output

        assert not project_store.project_dir("stale").exists()
        assert project_store.project_dir("fresh").exists()

        archive_root = project_store.DATA_ROOT / ".archive"
        archived = list(archive_root.iterdir())
        assert len(archived) == 1
        assert archived[0].name.endswith("-stale")

    def test_no_stalled_projects_reports_clean(self) -> None:
        _make_project("fresh", phase=Phase.INIT, last_updated=_iso_days_ago(1))

        result = runner.invoke(app, ["clean", "--stalled"])
        assert result.exit_code == 0
        assert "fresh" not in result.output
        assert "stalled" in result.output.lower() or "no" in result.output.lower()
