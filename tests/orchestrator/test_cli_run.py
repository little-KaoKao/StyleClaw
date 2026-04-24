from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from styleclaw.cli import app
from styleclaw.core.models import Action, ActionPlan, Phase, ProjectConfig, ProjectState
from styleclaw.orchestrator.actions import StepResult
from styleclaw.storage import project_store

runner = CliRunner()


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")
    monkeypatch.setenv("RUNNINGHUB_API_KEY", "test-key")


@pytest.fixture
def setup_project():
    config = ProjectConfig(name="test-proj", ip_info="anime")
    root = project_store.create_project(config)
    (root / "refs" / "ref-001.png").write_bytes(b"fake image")
    state = ProjectState(phase=Phase.INIT)
    project_store.save_state("test-proj", state)
    return root


def _make_plan(summary: str = "Test plan") -> ActionPlan:
    return ActionPlan(
        summary=summary,
        steps=[Action(name="analyze", description="分析参考图片")],
    )


class TestRunCommand:
    def test_no_project_found(self) -> None:
        result = runner.invoke(app, ["run", "do something"])
        assert result.exit_code == 1
        assert "No projects found" in result.output

    def test_multiple_projects_no_flag(self) -> None:
        project_store.create_project(ProjectConfig(name="proj-a"))
        project_store.create_project(ProjectConfig(name="proj-b"))
        result = runner.invoke(app, ["run", "do something"])
        assert result.exit_code == 1
        assert "Multiple projects" in result.output

    @patch("styleclaw.providers.llm.bedrock.BedrockProvider")
    @patch("styleclaw.orchestrator.planner.plan", new_callable=AsyncMock)
    def test_auto_selects_single_project(self, mock_plan, mock_llm_cls, setup_project) -> None:
        mock_plan.return_value = _make_plan("Test plan")
        result = runner.invoke(app, ["run", "analyze style"], input="n\n")
        assert result.exit_code == 0
        assert "Test plan" in result.output

    @patch("styleclaw.providers.llm.bedrock.BedrockProvider")
    @patch("styleclaw.orchestrator.planner.plan", new_callable=AsyncMock)
    def test_cancel(self, mock_plan, mock_llm_cls, setup_project) -> None:
        mock_plan.return_value = _make_plan("Plan")
        result = runner.invoke(app, ["run", "analyze", "-p", "test-proj"], input="n\n")
        assert "Cancelled" in result.output

    @patch("styleclaw.providers.llm.bedrock.BedrockProvider")
    @patch("styleclaw.orchestrator.planner.plan", new_callable=AsyncMock)
    def test_explicit_project(self, mock_plan, mock_llm_cls, setup_project) -> None:
        mock_plan.return_value = _make_plan("Analyze")
        result = runner.invoke(app, ["run", "analyze", "-p", "test-proj"], input="n\n")
        assert "Analyze" in result.output
