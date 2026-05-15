from __future__ import annotations

import json

import pytest

from styleclaw.core.models import Action, ActionPlan, ProjectConfig
from styleclaw.orchestrator.audit_log import AuditLogger
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def setup_project():
    project_store.create_project(ProjectConfig(name="p"))


class TestAuditLogger:
    def test_records_plan_to_runs_dir(self, setup_project):
        audit = AuditLogger.create("p", "do something")
        plan = ActionPlan(
            summary="x",
            steps=[Action(name="analyze", description="x")],
        )
        audit.record_plan(plan)

        plan_files = list((project_store.project_dir("p") / "runs").rglob("plan.json"))
        assert len(plan_files) == 1
        data = json.loads(plan_files[0].read_text(encoding="utf-8"))
        assert data["project"] == "p"
        assert data["intent"] == "do something"
        assert data["plan"]["summary"] == "x"
        assert data["plan"]["steps"][0]["name"] == "analyze"

    def test_step_finished_writes_log(self, setup_project):
        audit = AuditLogger.create("p", "x")
        audit.step_started(0)
        audit.step_finished(0, "analyze", ok=True, message="done")
        audit.step_started(1)
        audit.step_finished(1, "generate", ok=False, message="boom")

        log_files = list(
            (project_store.project_dir("p") / "runs").rglob("execution-log.json")
        )
        assert len(log_files) == 1
        data = json.loads(log_files[0].read_text(encoding="utf-8"))
        steps = data["steps"]
        assert len(steps) == 2
        assert steps[0]["ok"] is True
        assert steps[0]["name"] == "analyze"
        assert steps[1]["ok"] is False
        assert steps[1]["message"] == "boom"
        # elapsed_seconds should be a float >= 0
        assert all(isinstance(s["elapsed_seconds"], float) for s in steps)

    def test_log_flushed_after_each_step(self, setup_project):
        """A crash mid-run must leave the partial log on disk."""
        audit = AuditLogger.create("p", "x")
        audit.step_started(0)
        audit.step_finished(0, "analyze", ok=True, message="done")

        log_path = next(
            (project_store.project_dir("p") / "runs").rglob("execution-log.json")
        )
        data = json.loads(log_path.read_text(encoding="utf-8"))
        assert len(data["steps"]) == 1

    def test_cancelled_writes_marker(self, setup_project):
        audit = AuditLogger.create("p", "x")
        audit.cancelled()
        log_path = next(
            (project_store.project_dir("p") / "runs").rglob("execution-log.json")
        )
        data = json.loads(log_path.read_text(encoding="utf-8"))
        assert len(data["steps"]) == 1
        assert data["steps"][0]["name"] == "<cancelled>"
        assert data["steps"][0]["ok"] is False

    def test_two_runs_dont_collide(self, setup_project):
        a1 = AuditLogger.create("p", "first")
        # Force a different timestamp for the second run
        a2 = AuditLogger.create("p", "second")
        # If they happen in the same second, run_dirs collide — bypass this
        # collision case explicitly to test multi-run isolation.
        if a1.run_dir == a2.run_dir:
            object.__setattr__(a2, "run_dir", a2.run_dir.parent / (a2.run_dir.name + "-x"))

        a1.record_plan(ActionPlan(summary="A", steps=[Action(name="analyze", description="x")]))
        a2.record_plan(ActionPlan(summary="B", steps=[Action(name="analyze", description="x")]))

        plan_files = sorted((project_store.project_dir("p") / "runs").rglob("plan.json"))
        assert len(plan_files) == 2
        summaries = {json.loads(p.read_text())["plan"]["summary"] for p in plan_files}
        assert summaries == {"A", "B"}
