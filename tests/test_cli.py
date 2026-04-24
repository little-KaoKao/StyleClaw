from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from typer.testing import CliRunner

from styleclaw.cli import app
from styleclaw.core.models import (
    BatchCase,
    BatchConfig,
    DimensionScores,
    ModelEvaluation,
    ModelScore,
    Phase,
    ProjectConfig,
    ProjectState,
    PromptConfig,
    RoundEvaluation,
    RoundScore,
    StyleAnalysis,
    TaskRecord,
    UploadRecord,
)
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

    @patch("styleclaw.cli._run_with_client")
    def test_init_success(self, mock_run, tmp_path) -> None:
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"fake")
        mock_run.return_value = tmp_path / "projects" / "new-proj"
        result = runner.invoke(app, ["init", "new-proj", "--ref", str(ref)])
        assert result.exit_code == 0


class TestAnalyzeCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)
        result = runner.invoke(app, ["analyze", "test-proj"])
        assert result.exit_code == 1
        assert "INIT" in result.output


class TestGenerateCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["generate", "test-proj"])
        assert result.exit_code == 1
        assert "Cannot generate" in result.output

    @patch("styleclaw.cli._run_with_client")
    def test_model_select_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)

        analysis = StyleAnalysis(trigger_phrase="bold anime")
        project_store.save_analysis("test-proj", analysis)
        uploads = [UploadRecord(local_path="r.png", url="https://cdn.example.com/r.png", file_name="r.png")]
        project_store.save_uploads("test-proj", uploads)

        mock_run.return_value = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="QUEUED"),
        }
        result = runner.invoke(app, ["generate", "test-proj"])
        assert result.exit_code == 0
        assert "mj-v7" in result.output

    @patch("styleclaw.cli._run_with_client")
    def test_style_refine_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=1)

        prompt = PromptConfig(round=1, trigger_phrase="bold anime")
        project_store.save_prompt_config("test-proj", 1, prompt)
        uploads = [UploadRecord(local_path="r.png", url="https://cdn.example.com/r.png", file_name="r.png")]
        project_store.save_uploads("test-proj", uploads)

        mock_run.return_value = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="QUEUED"),
        }
        result = runner.invoke(app, ["generate", "test-proj"])
        assert result.exit_code == 0

    def test_style_refine_no_round(self, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=0)
        result = runner.invoke(app, ["generate", "test-proj"])
        assert result.exit_code == 1
        assert "Run 'refine' first" in result.output


class TestPollCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 1
        assert "Nothing to poll" in result.output

    @patch("styleclaw.cli._run_with_client")
    def test_model_select_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.MODEL_SELECT)
        mock_run.return_value = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS", results=[{"url": "http://img.png"}]),
        }
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 0
        assert "mj-v7" in result.output

    @patch("styleclaw.cli._run_with_client")
    def test_style_refine_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.STYLE_REFINE, current_round=1)
        mock_run.return_value = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS", results=[]),
        }
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 0

    @patch("styleclaw.cli._run_with_client")
    def test_batch_t2i_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.BATCH_T2I, current_batch=1)
        mock_run.return_value = {
            "am-001": TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS", results=[]),
        }
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 0
        assert "1/1 completed" in result.output

    @patch("styleclaw.cli._run_with_client")
    def test_batch_i2i_phase(self, mock_run, setup_project) -> None:
        _set_state(Phase.BATCH_I2I, current_batch=1)
        mock_run.return_value = {
            "i2i-001": TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS", results=[]),
        }
        result = runner.invoke(app, ["poll", "test-proj"])
        assert result.exit_code == 0


class TestEvaluateCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.INIT)
        result = runner.invoke(app, ["evaluate", "test-proj"])
        assert result.exit_code == 1
        assert "Cannot evaluate" in result.output


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
        _set_state(Phase.BATCH_T2I)
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

    @patch("styleclaw.cli._run_with_client")
    def test_t2i_success(self, mock_run, setup_project) -> None:
        _set_state(Phase.BATCH_T2I, current_batch=1, selected_models=["mj-v7"])
        uploads = [UploadRecord(local_path="r.png", url="https://cdn.example.com/r.png", file_name="r.png")]
        project_store.save_uploads("test-proj", uploads)
        mock_run.return_value = {"am-001": TaskRecord(task_id="t1", model_id="mj-v7", status="QUEUED")}
        result = runner.invoke(app, ["batch-submit", "test-proj"])
        assert result.exit_code == 0
        assert "1 t2i tasks" in result.output

    @patch("styleclaw.cli._run_with_client")
    def test_i2i_success(self, mock_run, setup_project) -> None:
        _set_state(Phase.BATCH_I2I, current_batch=1, current_round=1, selected_models=["mj-v7"])
        prompt = PromptConfig(round=1, trigger_phrase="bold anime")
        project_store.save_prompt_config("test-proj", 1, prompt)
        mock_run.return_value = {"i2i-001": TaskRecord(task_id="t1", model_id="mj-v7", status="QUEUED")}
        result = runner.invoke(app, ["batch-submit", "test-proj", "--i2i"])
        assert result.exit_code == 0
        assert "1 i2i tasks" in result.output


class TestReportCommand:
    def test_wrong_phase(self, setup_project) -> None:
        _set_state(Phase.COMPLETED)
        result = runner.invoke(app, ["report", "test-proj"])
        assert result.exit_code == 1
        assert "No report available" in result.output


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
        assert _get_current_trigger("test-proj", project_store.load_state("test-proj")) == "(unknown)"


class TestRunWithClient:
    def test_api_key_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("RUNNINGHUB_API_KEY", raising=False)
        import typer
        from styleclaw.cli import _get_api_key
        with pytest.raises(typer.Exit):
            _get_api_key()
