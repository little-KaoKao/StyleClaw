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
        assert "No projects yet" in result.output

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

    @patch("styleclaw.core.llm_routing.RoleRouter.from_env")
    @patch("styleclaw.orchestrator.planner.plan", new_callable=AsyncMock)
    def test_uses_configured_llm_provider(self, mock_plan, mock_from_env, setup_project) -> None:
        fake_llm = MagicMock()
        fake_router = MagicMock()
        fake_router.get.return_value = fake_llm
        fake_router.close = AsyncMock()
        mock_from_env.return_value = fake_router
        mock_plan.return_value = _make_plan("Configured provider")

        result = runner.invoke(app, ["run", "analyze", "-p", "test-proj"], input="n\n")

        assert result.exit_code == 0
        mock_from_env.assert_called_once()
        # planner gets the planner-role provider.
        assert mock_plan.call_args.args[0] is fake_llm
        fake_router.close.assert_awaited_once()

    @patch("styleclaw.cli._build_context")
    @patch("styleclaw.core.llm_routing.RoleRouter.from_env")
    @patch("styleclaw.orchestrator.planner.plan", new_callable=AsyncMock)
    def test_dry_run_skips_execution(
        self, mock_plan, mock_from_env, mock_build_context, setup_project,
    ) -> None:
        fake_router = MagicMock()
        fake_router.get.return_value = MagicMock()
        fake_router.close = AsyncMock()
        mock_from_env.return_value = fake_router
        mock_plan.return_value = _make_plan("Dry-run plan")

        result = runner.invoke(
            app, ["run", "analyze", "-p", "test-proj", "--dry-run"],
        )

        assert result.exit_code == 0, result.output
        assert "Dry-run plan" in result.output
        assert "(dry-run) 未执行" in result.output
        # Trailing "Done." (from successful execution) must not appear.
        assert "Done." not in result.output
        # Router was built+closed exactly once for planning.
        mock_from_env.assert_called_once()
        mock_plan.assert_awaited_once()
        fake_router.close.assert_awaited_once()
        # Execution context must never be constructed in dry-run.
        mock_build_context.assert_not_called()

    @patch("styleclaw.cli._build_context")
    @patch("styleclaw.core.llm_routing.RoleRouter.from_env")
    @patch("styleclaw.orchestrator.planner.plan", new_callable=AsyncMock)
    def test_dry_run_overrides_yes(
        self, mock_plan, mock_from_env, mock_build_context, setup_project,
    ) -> None:
        fake_router = MagicMock()
        fake_router.get.return_value = MagicMock()
        fake_router.close = AsyncMock()
        mock_from_env.return_value = fake_router
        mock_plan.return_value = _make_plan("Dry beats yes")

        result = runner.invoke(
            app, ["run", "analyze", "-p", "test-proj", "--yes", "--dry-run"],
        )

        assert result.exit_code == 0, result.output
        assert "(dry-run) 未执行" in result.output
        mock_build_context.assert_not_called()
