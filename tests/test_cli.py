from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from styleclaw.cli import app
from styleclaw.core.models import (
    HistoryEntry,
    Phase,
    ProjectConfig,
    ProjectState,
    PromptConfig,
    StyleAnalysis,
)
from styleclaw.orchestrator.actions import StepResult
from styleclaw.storage import project_store

runner = CliRunner()


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")
    monkeypatch.setenv("RUNNINGHUB_API_KEY", "test-key")


@pytest.fixture
def setup_project():
    config = ProjectConfig(name="test-proj", ip_info="anime", ref_images=["refs/ref-001.png"])
    root = project_store.create_project(config)
    (root / "refs" / "ref-001.png").write_bytes(b"fake image")
    return root


def _set_state(phase: Phase, **kwargs) -> None:
    state = ProjectState(phase=phase, **kwargs)
    project_store.save_state("test-proj", state)


class TestStatusCommand:
    def test_list_all_projects(self, setup_project) -> None:
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "test-proj" in result.output

    def test_list_no_projects(self) -> None:
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "No projects found" in result.output

    def test_show_project_detail(self, setup_project) -> None:
        result = runner.invoke(app, ["status", "test-proj"])
        assert result.exit_code == 0
        assert "test-proj" in result.output
        assert "INIT" in result.output
        assert "anime" in result.output


class TestInitCommand:
    def test_init_missing_ref_file(self, tmp_path) -> None:
        result = runner.invoke(app, ["init", "new-proj", "--ref", str(tmp_path / "missing.png")])
        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("styleclaw.scripts.init_project.init_project")
    @patch("styleclaw.providers.runninghub.client.RunningHubClient")
    def test_init_success(self, mock_client_cls, mock_init, tmp_path) -> None:
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"fake")
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client
        mock_init.return_value = tmp_path / "projects" / "new-proj"
        result = runner.invoke(
            app,
            ["init", "new-proj", "--ref", str(ref), "--info", "anime style"],
        )
        assert result.exit_code == 0
        assert "initialized" in result.output


class TestAnalyzeCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)
        result = runner.invoke(app, ["analyze", "test-proj"])
        assert result.exit_code == 1
        assert "INIT" in result.output

    @patch("styleclaw.cli._run_action")
    def test_success(self, mock_run, setup_project) -> None:
        _set_state(Phase.INIT)
        mock_run.return_value = StepResult(ok=True, message="Trigger: bold anime")
        result = runner.invoke(app, ["analyze", "test-proj"])
        assert result.exit_code == 0
        assert "Analysis complete" in result.output
        mock_run.assert_called_once_with(
            "test-proj", "analyze", show_thinking=True, thinking_budget=5000,
        )


class TestGenerateCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["generate", "test-proj"])
        assert result.exit_code == 1
        assert "Cannot generate" in result.output

    @patch("styleclaw.cli._run_action")
    def test_model_select_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)
        mock_run.return_value = StepResult(ok=True, message="Submitted 5 model tasks")
        result = runner.invoke(app, ["generate", "test-proj"])
        assert result.exit_code == 0
        assert "Submitted 5 model tasks" in result.output

    @patch("styleclaw.cli._run_action")
    def test_style_refine_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=1)
        mock_run.return_value = StepResult(ok=True, message="Submitted 2 refine tasks")
        result = runner.invoke(app, ["generate", "test-proj"])
        assert result.exit_code == 0

    def test_style_refine_no_round(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=0)
        result = runner.invoke(app, ["generate", "test-proj"])
        assert result.exit_code == 1
        assert "Run 'refine' first" in result.output

    @patch("styleclaw.cli._run_action")
    def test_models_filter_passed_through(self, mock_run, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)
        mock_run.return_value = StepResult(
            ok=True, message="Submitted 8 model tasks (pass 1) [filtered: mj-v7, niji7]",
        )
        result = runner.invoke(
            app, ["generate", "test-proj", "--models", "mj-v7,niji7"],
        )
        assert result.exit_code == 0
        # _run_action receives the filter in args
        action_args = mock_run.call_args.args[2]
        assert action_args.get("models") == "mj-v7,niji7"

    def test_models_filter_rejected_outside_model_select(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=1)
        result = runner.invoke(
            app, ["generate", "test-proj", "--models", "mj-v7"],
        )
        assert result.exit_code == 1
        assert "MODEL_SELECT" in result.output

    def test_models_dry_run_unknown_model(self, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)
        result = runner.invoke(
            app, ["generate", "test-proj", "--models", "ghost-model", "--dry-run"],
        )
        assert result.exit_code == 1
        assert "Unknown model" in result.output


class TestPollCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 1
        assert "Nothing to poll" in result.output

    @patch("styleclaw.cli._run_action")
    def test_model_select_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)
        mock_run.return_value = StepResult(ok=True, message="5/5 completed")
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 0
        assert "5/5 completed" in result.output

    @patch("styleclaw.cli._run_action")
    def test_style_refine_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=1)
        mock_run.return_value = StepResult(ok=True, message="2/2 completed")
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 0

    @patch("styleclaw.cli._run_action")
    def test_batch_t2i_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.BATCH_T2I, current_batch=1)
        mock_run.return_value = StepResult(ok=True, message="100/100 completed")
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 0
        assert "100/100 completed" in result.output

    @patch("styleclaw.cli._run_action")
    def test_batch_i2i_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.BATCH_I2I, current_batch=1)
        mock_run.return_value = StepResult(ok=True, message="50/50 completed")
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 0

    @patch("styleclaw.cli._run_action")
    def test_poll_timeout(self, mock_run, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)
        mock_run.return_value = StepResult(ok=False, message="Poll timed out after 60 cycles")
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 1
        assert "timed out" in result.output


class TestEvaluateCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["evaluate", "test-proj"])
        assert result.exit_code == 1
        assert "Cannot evaluate" in result.output

    @patch("styleclaw.cli._run_action")
    def test_success(self, mock_run, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)
        mock_run.return_value = StepResult(ok=True, message="Recommendation: mj-v7")
        result = runner.invoke(app, ["evaluate", "test-proj"])
        assert result.exit_code == 0
        assert "Recommendation" in result.output


class TestSelectModelCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["select-model", "test-proj", "--models", "mj-v7"])
        assert result.exit_code == 1

    def test_unknown_model(self, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)
        result = runner.invoke(app, ["select-model", "test-proj", "--models", "unknown"])
        assert result.exit_code == 1
        assert "Unknown model" in result.output

    def test_advances_from_model_select(self, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)
        result = runner.invoke(app, ["select-model", "test-proj", "--models", "mj-v7"])
        assert result.exit_code == 0
        assert "STYLE_REFINE" in result.output
        state = project_store.load_state("test-proj")
        assert state.phase == Phase.STYLE_REFINE
        assert state.selected_models == ["mj-v7"]

    def test_updates_in_style_refine(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, selected_models=["mj-v7"])
        result = runner.invoke(app, ["select-model", "test-proj", "--models", "niji7,mj-v7"])
        assert result.exit_code == 0
        assert "Updated models" in result.output
        state = project_store.load_state("test-proj")
        assert state.selected_models == ["niji7", "mj-v7"]


class TestRefineCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["refine", "test-proj"])
        assert result.exit_code == 1

    def test_max_rounds_exceeded(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=5)
        result = runner.invoke(app, ["refine", "test-proj"])
        assert result.exit_code == 1
        assert "max auto rounds" in result.output

    @patch("styleclaw.cli._run_action")
    def test_success(self, mock_run, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=0)
        mock_run.return_value = StepResult(ok=True, message="Round 1: bold anime style")
        result = runner.invoke(app, ["refine", "test-proj"])
        assert result.exit_code == 0
        assert "Round 1" in result.output


class TestApproveCommand:
    def test_approve_batch_t2i_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["approve", "test-proj", "--yes"])
        assert result.exit_code == 1
        assert "STYLE_REFINE" in result.output

    def test_approve_batch_t2i(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=1)
        analysis = StyleAnalysis(trigger_phrase="bold anime")
        project_store.save_analysis("test-proj", analysis)
        prompt = PromptConfig(round=1, trigger_phrase="refined trigger")
        project_store.save_prompt_config("test-proj", 1, prompt)

        result = runner.invoke(app, ["approve", "test-proj", "--yes"])
        assert result.exit_code == 0
        assert "BATCH_T2I" in result.output

    def test_approve_completed_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE)
        result = runner.invoke(app, ["approve", "test-proj", "--phase", "completed", "--yes"])
        assert result.exit_code == 1
        assert "BATCH_I2I" in result.output

    def test_approve_completed(self, setup_project) -> None:
        _set_state(Phase.BATCH_I2I, selected_models=["mj-v7"])
        result = runner.invoke(app, ["approve", "test-proj", "--phase", "completed", "--yes"])
        assert result.exit_code == 0
        assert "COMPLETED" in result.output

    def test_approve_unknown_phase(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE)
        result = runner.invoke(app, ["approve", "test-proj", "--phase", "unknown", "--yes"])
        assert result.exit_code == 1
        assert "Unknown target phase" in result.output

    def test_approve_cancel(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=1)
        analysis = StyleAnalysis(trigger_phrase="bold anime")
        project_store.save_analysis("test-proj", analysis)
        prompt = PromptConfig(round=1, trigger_phrase="trigger")
        project_store.save_prompt_config("test-proj", 1, prompt)

        result = runner.invoke(app, ["approve", "test-proj"], input="n\n")
        assert "Cancelled" in result.output


class TestAdjustCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["adjust", "test-proj", "--direction", "warmer"])
        assert result.exit_code == 1


class TestRollbackCommand:
    def test_rollback_to_style_refine(self, setup_project) -> None:
        history = [
            HistoryEntry(phase=Phase.INIT, completed_at="2024-01-01T00:00:00+00:00"),
            HistoryEntry(phase=Phase.MODEL_SELECT, completed_at="2024-01-01T00:00:01+00:00"),
            HistoryEntry(phase=Phase.STYLE_REFINE, completed_at="2024-01-01T00:00:02+00:00"),
        ]
        _set_state(Phase.BATCH_T2I, history=history, current_round=2)
        rd = project_store.round_dir("test-proj", 2, pass_num=1)
        rd.mkdir(parents=True, exist_ok=True)
        result = runner.invoke(app, ["rollback", "test-proj", "--to", "STYLE_REFINE", "--round", "2"])
        assert result.exit_code == 0
        state = project_store.load_state("test-proj")
        assert state.phase == Phase.STYLE_REFINE
        assert state.current_round == 2


class TestDesignCasesCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["design-cases", "test-proj"])
        assert result.exit_code == 1

    @patch("styleclaw.cli._run_action")
    def test_success(self, mock_run, setup_project) -> None:
        _set_state(Phase.BATCH_T2I)
        mock_run.return_value = StepResult(ok=True, message="Designed 100 cases")
        result = runner.invoke(app, ["design-cases", "test-proj"])
        assert result.exit_code == 0
        assert "100 cases" in result.output


class TestBatchSubmitCommand:
    def test_t2i_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["batch-submit", "test-proj"])
        assert result.exit_code == 1

    def test_i2i_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["batch-submit", "test-proj", "--i2i"])
        assert result.exit_code == 1

    def test_t2i_no_model(self, setup_project) -> None:
        _set_state(Phase.BATCH_T2I, selected_models=[])
        result = runner.invoke(app, ["batch-submit", "test-proj"])
        assert result.exit_code == 1
        assert "No model selected" in result.output

    def test_i2i_no_model(self, setup_project) -> None:
        _set_state(Phase.BATCH_I2I, selected_models=[])
        result = runner.invoke(app, ["batch-submit", "test-proj", "--i2i"])
        assert result.exit_code == 1
        assert "No model selected" in result.output

    @patch("styleclaw.cli._run_action")
    def test_t2i_success(self, mock_run, setup_project) -> None:
        _set_state(Phase.BATCH_T2I, current_batch=1, selected_models=["mj-v7"])
        mock_run.return_value = StepResult(ok=True, message="Submitted 100 t2i tasks")
        result = runner.invoke(app, ["batch-submit", "test-proj"])
        assert result.exit_code == 0
        assert "t2i tasks" in result.output

    @patch("styleclaw.cli._run_action")
    def test_i2i_success(self, mock_run, setup_project) -> None:
        _set_state(Phase.BATCH_I2I, current_batch=1, current_round=1, selected_models=["mj-v7"])
        mock_run.return_value = StepResult(ok=True, message="Submitted 50 i2i tasks")
        result = runner.invoke(app, ["batch-submit", "test-proj", "--i2i"])
        assert result.exit_code == 0
        assert "i2i tasks" in result.output


class TestReportCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.COMPLETED)
        result = runner.invoke(app, ["report", "test-proj"])
        assert result.exit_code == 1
        assert "No report available" in result.output

    @patch("styleclaw.scripts.report.generate_model_select_report")
    def test_model_select_passes_current_pass_num(self, mock_report, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT, current_model_select_pass=3)
        mock_report.return_value = Path("/fake/p3.html")
        result = runner.invoke(app, ["report", "test-proj"])
        assert result.exit_code == 0
        assert mock_report.call_args.kwargs.get("pass_num") == 3

    @patch("styleclaw.scripts.report.generate_style_refine_report")
    def test_style_refine_passes_current_pass_num(self, mock_report, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=2, current_model_select_pass=2)
        mock_report.return_value = Path("/fake/r2.html")
        result = runner.invoke(app, ["report", "test-proj"])
        assert result.exit_code == 0
        assert mock_report.call_args.kwargs.get("pass_num") == 2
        # round_num is the second positional arg (after name)
        assert mock_report.call_args.args[1] == 2


class TestGetCurrentTrigger:
    def test_from_round(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=1)
        prompt = PromptConfig(round=1, trigger_phrase="from round")
        project_store.save_prompt_config("test-proj", 1, prompt)

        from styleclaw.cli import _get_current_trigger
        assert _get_current_trigger("test-proj", project_store.load_state("test-proj")) == "from round"

    def test_from_analysis(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=0)
        analysis = StyleAnalysis(trigger_phrase="from analysis")
        project_store.save_analysis("test-proj", analysis)

        from styleclaw.cli import _get_current_trigger
        assert _get_current_trigger("test-proj", project_store.load_state("test-proj")) == "from analysis"

    def test_no_analysis(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=0)
        from styleclaw.cli import _get_current_trigger
        result = _get_current_trigger("test-proj", project_store.load_state("test-proj"))
        assert "not found" in result


class TestBuildContext:
    def test_api_key_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("RUNNINGHUB_API_KEY", raising=False)
        import typer
        from styleclaw.cli import _get_api_key
        with pytest.raises(typer.Exit):
            _get_api_key()


class TestShowThinkingFlag:
    def test_analyze_forwards_show_thinking(self, setup_project) -> None:
        captured_ctx = {}

        async def fake_action(ctx, args):
            captured_ctx["show_thinking"] = ctx.show_thinking
            captured_ctx["thinking_budget"] = ctx.thinking_budget
            from styleclaw.core.state_machine import advance
            state = project_store.load_state(ctx.project)
            project_store.save_state(ctx.project, advance(state, Phase.MODEL_SELECT))
            return StepResult(ok=True, message="ok")

        with patch.dict(
            "styleclaw.orchestrator.actions.ACTION_REGISTRY",
            {"analyze": MagicMock(
                fn=fake_action, needs_client=False, needs_llm=True,
                requires_confirmation=False,
            )},
        ):
            with patch("styleclaw.providers.llm.bedrock.BedrockProvider") as FakeLLM:
                FakeLLM.return_value = MagicMock()
                FakeLLM.return_value.close = AsyncMock()
                result = runner.invoke(
                    app,
                    ["analyze", "test-proj", "--show-thinking", "--thinking-budget", "4000"],
                )

        assert result.exit_code == 0, result.output
        assert captured_ctx["show_thinking"] is True
        assert captured_ctx["thinking_budget"] == 4000


class TestRetestModelsCommand:
    def test_retest_models_advances_to_model_select(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=1, current_model_select_pass=1)
        result = runner.invoke(app, ["retest-models", "test-proj"])
        assert result.exit_code == 0
        new_state = project_store.load_state("test-proj")
        assert new_state.phase == Phase.MODEL_SELECT
        assert new_state.current_model_select_pass == 2

    def test_retest_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["retest-models", "test-proj"])
        assert result.exit_code == 1
        assert "STYLE_REFINE or BATCH_T2I" in result.output


class TestBackToT2iCommand:
    def test_back_to_t2i_from_batch_i2i(self, setup_project) -> None:
        _set_state(Phase.BATCH_I2I, current_batch=1)
        result = runner.invoke(app, ["back-to-t2i", "test-proj"])
        assert result.exit_code == 0
        assert project_store.load_state("test-proj").phase == Phase.BATCH_T2I

    def test_back_to_t2i_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE)
        result = runner.invoke(app, ["back-to-t2i", "test-proj"])
        assert result.exit_code == 1
        assert "BATCH_I2I" in result.output


class TestStatusShowsPass:
    def test_status_includes_pass_number(self, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT, current_model_select_pass=2)
        result = runner.invoke(app, ["status", "test-proj"])
        assert result.exit_code == 0
        assert "Pass:" in result.output
        assert "2" in result.output


class TestStatusShowsTaskCounts:
    """``status`` should answer "did the batch finish?" without making the
    user open task.json by hand."""

    def test_task_counts_for_batch_t2i(self, setup_project) -> None:
        from styleclaw.core.models import TaskRecord

        _set_state(Phase.BATCH_T2I, current_batch=1)
        # 3 SUCCESS, 1 FAILED, 1 still QUEUED.
        for cid, status in [
            ("c-001", "SUCCESS"), ("c-002", "SUCCESS"), ("c-003", "SUCCESS"),
            ("c-004", "FAILED"), ("c-005", "QUEUED"),
        ]:
            project_store.save_batch_task_record(
                "test-proj", 1, cid,
                TaskRecord(task_id=f"t-{cid}", model_id="mj-v7", status=status),
            )

        result = runner.invoke(app, ["status", "test-proj"])
        assert result.exit_code == 0
        assert "Tasks:" in result.output
        assert "3 ok" in result.output
        assert "1 failed" in result.output
        assert "1 pending" in result.output
        assert "5 total" in result.output

    def test_task_counts_omitted_when_no_records(self, setup_project) -> None:
        # MODEL_SELECT phase with no submitted tasks yet — the line should
        # be suppressed entirely, not show "0 ok, 0 failed, 0 pending".
        _set_state(Phase.MODEL_SELECT)
        result = runner.invoke(app, ["status", "test-proj"])
        assert result.exit_code == 0
        assert "Tasks:" not in result.output

    def test_task_counts_omitted_for_init_phase(self, setup_project) -> None:
        # INIT has no task-level state by construction.
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["status", "test-proj"])
        assert result.exit_code == 0
        assert "Tasks:" not in result.output


class TestInterruptHint:
    """Ctrl-C during a long-running action shouldn't leave the user wondering
    whether anything stuck — print a resume hint pointing at the right
    follow-up command."""

    @patch("styleclaw.cli.asyncio.run")
    def test_poll_prints_resume_hint_on_interrupt(self, mock_run, setup_project):
        _set_state(Phase.MODEL_SELECT)
        mock_run.side_effect = KeyboardInterrupt()
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 130
        assert "中断已捕获" in result.output
        assert "styleclaw poll test-proj" in result.output

    @patch("styleclaw.cli.asyncio.run")
    def test_batch_submit_resume_hint_mentions_checkpoint(
        self, mock_run, setup_project,
    ):
        _set_state(Phase.BATCH_T2I, current_batch=1, selected_models=["mj-v7"])
        mock_run.side_effect = KeyboardInterrupt()
        result = runner.invoke(app, ["batch-submit", "test-proj"])
        assert result.exit_code == 130
        assert "checkpoint" in result.output

    @patch("styleclaw.cli.asyncio.run")
    def test_generate_resume_hint_mentions_success_skip(
        self, mock_run, setup_project,
    ):
        _set_state(Phase.MODEL_SELECT)
        mock_run.side_effect = KeyboardInterrupt()
        result = runner.invoke(app, ["generate", "test-proj"])
        assert result.exit_code == 130
        assert "SUCCESS" in result.output

    def test_print_interrupt_hint_handles_no_project(self):
        # The `run` command can be invoked with no project (init flow); the
        # hint must still be safe to print.
        from styleclaw.cli import _print_interrupt_hint
        # Should not raise.
        _print_interrupt_hint("run", None)
