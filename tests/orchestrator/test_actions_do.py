from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    TaskStatus,
    UploadRecord,
)
from styleclaw.orchestrator.actions import (
    ExecutionContext,
    StepResult,
    do_add_refs,
    do_analyze,
    do_approve,
    do_batch_submit,
    do_design_cases,
    do_evaluate,
    do_generate,
    do_init,
    do_poll,
    do_refine,
    do_report,
    do_select_model,
    do_set_pass,
    do_set_sref,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


def _create_project(
    phase: Phase = Phase.INIT,
    selected_models: list[str] | None = None,
    current_round: int = 0,
    current_batch: int = 0,
    ref_images: list[str] | None = None,
) -> str:
    name = "test-proj"
    config = ProjectConfig(
        name=name,
        ip_info="anime style",
        ref_images=ref_images or ["ref1.png"],
    )
    project_store.create_project(config)
    state = ProjectState(
        phase=phase,
        selected_models=selected_models or [],
        current_round=current_round,
        current_batch=current_batch,
    )
    project_store.save_state(name, state)
    root = project_store.project_dir(name)
    for img in config.ref_images:
        (root / img).write_bytes(b"fake-image")
    return name


def _ctx(
    name: str = "test-proj",
    client: object | None = None,
    llm: object | None = None,
    model_id: str = "test-model",
) -> ExecutionContext:
    from tests.orchestrator._routing_helpers import MockRouter
    router = MockRouter(llm, model_id) if llm is not None else None
    return ExecutionContext(
        project=name,
        client=client,
        llm_router=router,
        poll_interval=0.0,
    )


class TestDoAnalyze:
    async def test_analyzes_and_advances_to_model_select(self) -> None:
        name = _create_project(phase=Phase.INIT)
        analysis = StyleAnalysis(trigger_phrase="bold anime lineart")
        mock_llm = AsyncMock()

        with patch(
            "styleclaw.agents.analyze_style.analyze_style",
            new_callable=AsyncMock,
            return_value=analysis,
        ):
            result = await do_analyze(_ctx(name, llm=mock_llm, model_id="gemini-2.5-pro"), {})

        assert result.ok is True
        assert "bold anime lineart" in result.message
        saved = project_store.load_analysis(name)
        assert saved.trigger_phrase == "bold anime lineart"
        # Router records which model produced this artifact.
        assert saved.model_id == "gemini-2.5-pro"
        state = project_store.load_state(name)
        assert state.phase == Phase.MODEL_SELECT


class TestDoGenerate:
    async def test_model_select_phase(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        analysis = StyleAnalysis(trigger_phrase="bold anime")
        project_store.save_analysis(name, analysis)
        project_store.save_uploads(name, [
            UploadRecord(local_path="ref1.png", url="http://img/1", file_name="ref1.png"),
        ])

        mock_client = AsyncMock()
        records = {"mj-v7": TaskRecord(task_id="t1", model_id="mj-v7")}

        with patch(
            "styleclaw.scripts.generate.generate_model_select",
            new_callable=AsyncMock,
            return_value=records,
        ):
            result = await do_generate(_ctx(name, client=mock_client), {})

        assert result.ok is True
        assert "1" in result.message

    async def test_style_refine_phase(self) -> None:
        name = _create_project(
            phase=Phase.STYLE_REFINE, selected_models=["mj-v7"], current_round=1,
        )
        project_store.save_prompt_config(
            name, 1, PromptConfig(round=1, trigger_phrase="refined style"),
        )
        project_store.save_uploads(name, [
            UploadRecord(local_path="ref1.png", url="http://img/1", file_name="ref1.png"),
        ])

        mock_client = AsyncMock()
        records = {"mj-v7": TaskRecord(task_id="t2", model_id="mj-v7")}

        with patch(
            "styleclaw.scripts.generate.generate_style_refine",
            new_callable=AsyncMock,
            return_value=records,
        ):
            result = await do_generate(_ctx(name, client=mock_client), {})

        assert result.ok is True
        assert "1" in result.message

    async def test_wrong_phase_returns_error(self) -> None:
        name = _create_project(phase=Phase.BATCH_T2I)
        result = await do_generate(_ctx(name), {})
        assert result.ok is False
        assert "Cannot generate" in result.message


class TestDoPoll:
    async def test_all_completed_returns_immediately(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        completed_records = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS"),
        }

        with patch(
            "styleclaw.scripts.poll.poll_model_select",
            new_callable=AsyncMock,
            return_value=completed_records,
        ):
            result = await do_poll(_ctx(name), {})

        assert result.ok is True
        assert "1/1" in result.message

    async def test_style_refine_poll(self) -> None:
        name = _create_project(
            phase=Phase.STYLE_REFINE, selected_models=["mj-v7"], current_round=1,
        )
        completed_records = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS"),
        }

        with patch(
            "styleclaw.scripts.poll.poll_style_refine",
            new_callable=AsyncMock,
            return_value=completed_records,
        ):
            result = await do_poll(_ctx(name), {})

        assert result.ok is True

    async def test_batch_t2i_poll(self) -> None:
        name = _create_project(phase=Phase.BATCH_T2I, current_batch=1)
        completed_records = {
            "case-001": TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS"),
        }

        with patch(
            "styleclaw.scripts.poll.poll_batch",
            new_callable=AsyncMock,
            return_value=completed_records,
        ):
            result = await do_poll(_ctx(name), {})

        assert result.ok is True

    async def test_batch_i2i_poll(self) -> None:
        name = _create_project(phase=Phase.BATCH_I2I, current_batch=1)
        completed_records = {
            "i2i-001": TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS"),
        }

        with patch(
            "styleclaw.scripts.poll.poll_batch",
            new_callable=AsyncMock,
            return_value=completed_records,
        ):
            result = await do_poll(_ctx(name), {})

        assert result.ok is True

    async def test_wrong_phase_returns_error(self) -> None:
        name = _create_project(phase=Phase.INIT)
        result = await do_poll(_ctx(name), {})
        assert result.ok is False
        assert "Nothing to poll" in result.message

    async def test_retries_until_all_complete(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        pending = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="RUNNING"),
        }
        completed = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS"),
        }
        call_count = 0

        async def _mock_poll(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return pending if call_count == 1 else completed

        with patch(
            "styleclaw.scripts.poll.poll_model_select",
            side_effect=_mock_poll,
        ):
            result = await do_poll(_ctx(name), {})

        assert result.ok is True
        assert call_count == 2

    async def test_tqdm_bar_created_in_tty(self, monkeypatch) -> None:
        # In a TTY, do_poll should create a tqdm bar instead of emitting the
        # "Waiting..." log line. We mock sys.stdout.isatty so the test
        # exercises the TTY path even though pytest's stdout isn't a TTY.
        name = _create_project(phase=Phase.MODEL_SELECT)
        pending = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="RUNNING"),
        }
        completed = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS"),
        }
        call_count = 0

        async def _mock_poll(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return pending if call_count == 1 else completed

        # Pretend stdout is a TTY for the duration of the call.
        import sys
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

        created: list = []
        from tqdm import tqdm as _real_tqdm

        class _SpyTqdm(_real_tqdm):
            def __init__(self, *a, **kw):
                created.append((a, kw))
                super().__init__(*a, **kw)

        monkeypatch.setattr("styleclaw.orchestrator.actions.tqdm", _SpyTqdm, raising=False)

        with patch(
            "styleclaw.scripts.poll.poll_model_select", side_effect=_mock_poll,
        ):
            result = await do_poll(_ctx(name), {})

        assert result.ok is True
        # Exactly one bar created across however many cycles ran.
        assert len(created) == 1
        # And it was given the right unit / total.
        _, kw = created[0]
        assert kw.get("unit") == "task"
        assert kw.get("total") == 1

    async def test_non_tty_keeps_waiting_log_line(self, monkeypatch, caplog) -> None:
        # In CI / log capture there is no TTY; the log line must keep firing
        # so the operator still gets per-cycle progress in the logs.
        name = _create_project(phase=Phase.MODEL_SELECT)
        pending = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="RUNNING"),
        }
        completed = {
            "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS"),
        }
        call_count = 0

        async def _mock_poll(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return pending if call_count == 1 else completed

        import sys
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)

        import logging
        caplog.set_level(logging.INFO, logger="styleclaw.orchestrator.actions")

        with patch(
            "styleclaw.scripts.poll.poll_model_select", side_effect=_mock_poll,
        ):
            result = await do_poll(_ctx(name), {})

        assert result.ok is True
        assert any("Waiting..." in rec.message for rec in caplog.records)


class TestDoEvaluate:
    async def test_model_select_no_images(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)

        mock_llm = AsyncMock()
        result = await do_evaluate(_ctx(name, llm=mock_llm), {})
        assert result.ok is False
        assert "No generated images" in result.message

    async def test_model_select_with_images(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        results_dir = project_store.model_results_dir(name, "mj-v7")
        (results_dir / "output-001.png").write_bytes(b"fake-img")
        project_store.save_task_record(
            name, "mj-v7", TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS"),
        )

        evaluation = ModelEvaluation(
            recommendation="mj-v7",
            evaluations=[ModelScore(model="mj-v7", total=8.0)],
        )
        mock_llm = AsyncMock()

        with (
            patch(
                "styleclaw.agents.select_model.evaluate_models",
                new_callable=AsyncMock,
                return_value=evaluation,
            ),
            patch(
                "styleclaw.scripts.report.generate_model_select_report",
            ),
        ):
            result = await do_evaluate(_ctx(name, llm=mock_llm), {})

        assert result.ok is True
        assert "mj-v7" in result.message
        assert result.data["recommendation"] == "mj-v7"

    async def test_style_refine_no_images(self) -> None:
        name = _create_project(
            phase=Phase.STYLE_REFINE, selected_models=["mj-v7"], current_round=1,
        )

        mock_llm = AsyncMock()
        result = await do_evaluate(_ctx(name, llm=mock_llm), {})
        assert result.ok is False
        assert "No generated images" in result.message

    async def test_style_refine_with_images(self) -> None:
        name = _create_project(
            phase=Phase.STYLE_REFINE, selected_models=["mj-v7"], current_round=1,
        )
        results_dir = project_store.round_results_dir(name, 1, "mj-v7")
        (results_dir / "output-001.png").write_bytes(b"fake-img")
        project_store.save_round_task_record(
            name, 1, "mj-v7",
            TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS"),
        )

        high_scores = DimensionScores(
            color_palette=8.0, line_style=8.0, lighting=8.0, texture=8.0, overall_mood=8.0,
        )
        evaluation = RoundEvaluation(
            round=1,
            evaluations=[RoundScore(model="mj-v7", total=8.0, scores=high_scores)],
        )
        mock_llm = AsyncMock()

        with (
            patch(
                "styleclaw.agents.evaluate_result.evaluate_round",
                new_callable=AsyncMock,
                return_value=evaluation,
            ),
            patch(
                "styleclaw.scripts.report.generate_style_refine_report",
            ),
        ):
            result = await do_evaluate(_ctx(name, llm=mock_llm), {})

        assert result.ok is True
        assert "PASS" in result.message
        assert result.data["passed"] is True

    async def test_wrong_phase_returns_error(self) -> None:
        name = _create_project(phase=Phase.INIT)
        result = await do_evaluate(_ctx(name), {})
        assert result.ok is False
        assert "Cannot evaluate" in result.message


class TestDoSelectModel:
    async def test_no_models_specified(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        result = await do_select_model(_ctx(name), {"models": ""})
        assert result.ok is False
        assert "No models" in result.message

    async def test_unknown_model(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        result = await do_select_model(_ctx(name), {"models": "unknown-model"})
        assert result.ok is False
        assert "Unknown model" in result.message

    async def test_model_select_advances_to_style_refine(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        result = await do_select_model(_ctx(name), {"models": "mj-v7"})
        assert result.ok is True
        assert "STYLE_REFINE" in result.message
        state = project_store.load_state(name)
        assert state.phase == Phase.STYLE_REFINE
        assert state.selected_models == ["mj-v7"]

    async def test_style_refine_updates_models(self) -> None:
        name = _create_project(
            phase=Phase.STYLE_REFINE, selected_models=["mj-v7"],
        )
        result = await do_select_model(_ctx(name), {"models": "niji7,mj-v7"})
        assert result.ok is True
        assert "Updated" in result.message
        state = project_store.load_state(name)
        assert state.selected_models == ["niji7", "mj-v7"]

    async def test_wrong_phase_returns_error(self) -> None:
        name = _create_project(phase=Phase.BATCH_T2I)
        result = await do_select_model(_ctx(name), {"models": "mj-v7"})
        assert result.ok is False
        assert "Cannot select model" in result.message


class TestDoRefine:
    async def test_first_round_uses_analysis_trigger(self) -> None:
        name = _create_project(
            phase=Phase.STYLE_REFINE, selected_models=["mj-v7"], current_round=0,
        )
        project_store.save_analysis(
            name, StyleAnalysis(trigger_phrase="initial trigger"),
        )

        new_prompt = PromptConfig(round=1, trigger_phrase="refined trigger")
        mock_llm = AsyncMock()

        with patch(
            "styleclaw.agents.refine_prompt.refine_prompt",
            new_callable=AsyncMock,
            return_value=new_prompt,
        ):
            result = await do_refine(_ctx(name, llm=mock_llm), {})

        assert result.ok is True
        assert "refined trigger" in result.message
        state = project_store.load_state(name)
        assert state.current_round == 1

    async def test_subsequent_round_uses_previous_prompt(self) -> None:
        name = _create_project(
            phase=Phase.STYLE_REFINE, selected_models=["mj-v7"], current_round=1,
        )
        project_store.save_prompt_config(
            name, 1, PromptConfig(round=1, trigger_phrase="round 1 trigger"),
        )

        new_prompt = PromptConfig(round=2, trigger_phrase="round 2 trigger")
        mock_llm = AsyncMock()

        with patch(
            "styleclaw.agents.refine_prompt.refine_prompt",
            new_callable=AsyncMock,
            return_value=new_prompt,
        ) as mock_fn:
            result = await do_refine(_ctx(name, llm=mock_llm), {"direction": "more contrast"})

        assert result.ok is True
        assert "round 2 trigger" in result.message
        mock_fn.assert_called_once()
        call_args = mock_fn.call_args
        assert call_args[0][2] == "round 1 trigger"
        assert call_args[0][5] == []
        assert call_args[0][6] == "more contrast"

    async def test_max_rounds_exceeded(self) -> None:
        name = _create_project(
            phase=Phase.STYLE_REFINE, selected_models=["mj-v7"], current_round=5,
        )
        result = await do_refine(_ctx(name), {})
        assert result.ok is False
        assert "Max rounds" in result.message

    async def test_loads_previous_evaluations(self) -> None:
        name = _create_project(
            phase=Phase.STYLE_REFINE, selected_models=["mj-v7"], current_round=2,
        )
        project_store.save_prompt_config(
            name, 2, PromptConfig(round=2, trigger_phrase="round 2 trigger"),
        )
        eval1 = RoundEvaluation(round=1, evaluations=[])
        project_store.save_round_evaluation(name, 1, eval1)
        eval2 = RoundEvaluation(round=2, evaluations=[])
        project_store.save_round_evaluation(name, 2, eval2)

        new_prompt = PromptConfig(round=3, trigger_phrase="round 3 trigger")
        mock_llm = AsyncMock()

        with patch(
            "styleclaw.agents.refine_prompt.refine_prompt",
            new_callable=AsyncMock,
            return_value=new_prompt,
        ) as mock_fn:
            result = await do_refine(_ctx(name, llm=mock_llm), {})

        assert result.ok is True
        evaluations_arg = mock_fn.call_args[0][5]
        assert len(evaluations_arg) == 2


class TestDoApprove:
    async def test_approve_batch_t2i_from_style_refine(self) -> None:
        name = _create_project(phase=Phase.STYLE_REFINE)
        result = await do_approve(_ctx(name), {"target": "batch-t2i"})
        assert result.ok is True
        state = project_store.load_state(name)
        assert state.phase == Phase.BATCH_T2I

    async def test_approve_completed_from_batch_i2i(self) -> None:
        name = _create_project(phase=Phase.BATCH_I2I)
        result = await do_approve(_ctx(name), {"target": "completed"})
        assert result.ok is True
        state = project_store.load_state(name)
        assert state.phase == Phase.COMPLETED

    async def test_approve_completed_wrong_phase(self) -> None:
        name = _create_project(phase=Phase.STYLE_REFINE)
        result = await do_approve(_ctx(name), {"target": "completed"})
        assert result.ok is False
        assert "BATCH_I2I" in result.message

    async def test_approve_batch_t2i_wrong_phase(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        result = await do_approve(_ctx(name), {"target": "batch-t2i"})
        assert result.ok is False
        assert "STYLE_REFINE" in result.message

    async def test_approve_default_target_is_batch_t2i(self) -> None:
        name = _create_project(phase=Phase.STYLE_REFINE)
        result = await do_approve(_ctx(name), {})
        assert result.ok is True
        state = project_store.load_state(name)
        assert state.phase == Phase.BATCH_T2I


class TestDoDesignCases:
    async def test_designs_and_saves_cases(self) -> None:
        name = _create_project(
            phase=Phase.BATCH_T2I, selected_models=["mj-v7"], current_round=1,
        )
        project_store.save_prompt_config(
            name, 1, PromptConfig(round=1, trigger_phrase="anime trigger"),
        )

        cases = [
            BatchCase(id="am-01", category="adult_male", description="warrior"),
            BatchCase(id="am-02", category="adult_male", description="scholar"),
        ]
        batch_config = BatchConfig(batch=1, trigger_phrase="anime trigger", cases=cases)
        mock_llm = AsyncMock()

        with patch(
            "styleclaw.agents.design_cases.design_cases",
            new_callable=AsyncMock,
            return_value=batch_config,
        ):
            result = await do_design_cases(_ctx(name, llm=mock_llm), {})

        assert result.ok is True
        assert "2" in result.message
        state = project_store.load_state(name)
        assert state.current_batch == 1

    async def test_wrong_phase_rejected(self) -> None:
        name = _create_project(phase=Phase.STYLE_REFINE, current_round=1)
        result = await do_design_cases(_ctx(name, llm=AsyncMock()), {})
        assert result.ok is False
        assert "BATCH_T2I" in result.message

    async def test_zero_round_rejected(self) -> None:
        """In BATCH_T2I with current_round=0, there's no prompt.json to load —
        action must fail with a clean message instead of FileNotFoundError."""
        name = _create_project(phase=Phase.BATCH_T2I, current_round=0)
        result = await do_design_cases(_ctx(name, llm=AsyncMock()), {})
        assert result.ok is False
        assert "current_round=0" in result.message


class TestDoBatchSubmit:
    async def test_no_model_selected(self) -> None:
        name = _create_project(phase=Phase.BATCH_T2I, current_batch=1)
        result = await do_batch_submit(_ctx(name), {})
        assert result.ok is False
        assert "No model" in result.message

    async def test_t2i_submit(self) -> None:
        name = _create_project(
            phase=Phase.BATCH_T2I,
            selected_models=["mj-v7"],
            current_batch=1,
        )
        project_store.save_uploads(name, [
            UploadRecord(local_path="ref1.png", url="http://img/1", file_name="ref1.png"),
        ])

        records = {"case-001": TaskRecord(task_id="t1", model_id="mj-v7")}
        mock_client = AsyncMock()

        with patch(
            "styleclaw.scripts.batch_submit.batch_submit_t2i",
            new_callable=AsyncMock,
            return_value=records,
        ):
            result = await do_batch_submit(_ctx(name, client=mock_client), {})

        assert result.ok is True
        assert "1" in result.message
        assert "t2i" in result.message

    async def test_i2i_submit(self) -> None:
        name = _create_project(
            phase=Phase.BATCH_I2I,
            selected_models=["mj-v7"],
            current_round=1,
            current_batch=1,
        )
        project_store.save_prompt_config(
            name, 1, PromptConfig(round=1, trigger_phrase="trigger"),
        )

        records = {"i2i-001": TaskRecord(task_id="t1", model_id="mj-v7")}
        mock_client = AsyncMock()

        with patch(
            "styleclaw.scripts.batch_submit.batch_submit_i2i",
            new_callable=AsyncMock,
            return_value=records,
        ):
            result = await do_batch_submit(_ctx(name, client=mock_client), {})

        assert result.ok is True
        assert "1" in result.message
        assert "i2i" in result.message

    async def test_wrong_phase_returns_error(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT, selected_models=["mj-v7"])
        result = await do_batch_submit(_ctx(name), {"model": "mj-v7"})
        assert result.ok is False
        assert "Cannot batch-submit" in result.message

    async def test_uses_explicit_model_arg(self) -> None:
        name = _create_project(
            phase=Phase.BATCH_T2I,
            selected_models=["mj-v7"],
            current_batch=1,
        )
        project_store.save_uploads(name, [
            UploadRecord(local_path="ref1.png", url="http://img/1", file_name="ref1.png"),
        ])

        records = {"case-001": TaskRecord(task_id="t1", model_id="niji7")}
        mock_client = AsyncMock()

        with patch(
            "styleclaw.scripts.batch_submit.batch_submit_t2i",
            new_callable=AsyncMock,
            return_value=records,
        ) as mock_fn:
            result = await do_batch_submit(
                _ctx(name, client=mock_client), {"model": "niji7"},
            )

        assert result.ok is True


class TestDoReport:
    async def test_batch_t2i_report(self) -> None:
        name = _create_project(phase=Phase.BATCH_T2I, current_batch=1)

        with patch(
            "styleclaw.scripts.report.generate_batch_t2i_report",
            return_value=Path("/fake/report.html"),
        ):
            result = await do_report(_ctx(name), {})

        assert result.ok is True
        assert "report" in result.message.lower()

    async def test_batch_i2i_report(self) -> None:
        name = _create_project(phase=Phase.BATCH_I2I, current_batch=1)

        with patch(
            "styleclaw.scripts.report.generate_batch_i2i_report",
            return_value=Path("/fake/report.html"),
        ):
            result = await do_report(_ctx(name), {})

        assert result.ok is True
        assert "report" in result.message.lower()

    async def test_wrong_phase_returns_error(self) -> None:
        name = _create_project(phase=Phase.STYLE_REFINE)
        result = await do_report(_ctx(name), {})
        assert result.ok is False
        assert "No report" in result.message


class TestDoInit:
    async def test_creates_project_from_ref_dir(self, tmp_path) -> None:
        ref_dir = tmp_path / "input-refs"
        ref_dir.mkdir()
        (ref_dir / "a.png").write_bytes(b"fake-png-1")
        (ref_dir / "b.jpg").write_bytes(b"fake-jpg-2")

        fake_root = tmp_path / "fake-project-root"
        fake_init = AsyncMock(return_value=fake_root)

        with patch("styleclaw.scripts.init_project.init_project", fake_init):
            result = await do_init(
                _ctx("new-proj", client=MagicMock()),
                {
                    "ref_dir": str(ref_dir),
                    "ip_info": "anime style",
                    "description": "from test",
                    "force": False,
                },
            )

        assert result.ok is True, result.message
        assert "new-proj" in result.message
        assert "2 ref images" in result.message
        fake_init.assert_awaited_once()
        call_args = fake_init.await_args
        assert call_args.args[0] == "new-proj"
        # Image paths discovered from ref_dir, sorted
        passed_refs = call_args.args[1]
        assert [p.name for p in passed_refs] == ["a.png", "b.jpg"]
        assert call_args.args[2] == "anime style"
        assert call_args.args[3] == "from test"
        assert call_args.kwargs["force"] is False

    async def test_missing_ref_dir_returns_error(self) -> None:
        result = await do_init(
            _ctx("new-proj", client=MagicMock()),
            {"ref_dir": "", "ip_info": "x"},
        )
        assert result.ok is False
        assert "ref_dir" in result.message

    async def test_nonexistent_ref_dir_returns_error(self, tmp_path) -> None:
        result = await do_init(
            _ctx("new-proj", client=MagicMock()),
            {"ref_dir": str(tmp_path / "does-not-exist"), "ip_info": "x"},
        )
        assert result.ok is False
        assert "not a directory" in result.message

    async def test_empty_ref_dir_returns_error(self, tmp_path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = await do_init(
            _ctx("new-proj", client=MagicMock()),
            {"ref_dir": str(empty_dir), "ip_info": "x"},
        )
        assert result.ok is False
        assert "No images" in result.message


class TestDoSetSref:
    async def test_updates_sref_index(self) -> None:
        name = _create_project(
            phase=Phase.MODEL_SELECT,
            ref_images=["refs/ref-001.png", "refs/ref-002.png", "refs/ref-003.png"],
        )
        result = await do_set_sref(_ctx(name), {"index": 2})
        assert result.ok is True
        assert "ref-003" in result.message
        assert project_store.load_config(name).sref_index == 2

    async def test_index_out_of_range(self) -> None:
        name = _create_project(
            phase=Phase.MODEL_SELECT,
            ref_images=["refs/ref-001.png"],
        )
        result = await do_set_sref(_ctx(name), {"index": 5})
        assert result.ok is False
        assert "out of range" in result.message

    async def test_missing_index(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        result = await do_set_sref(_ctx(name), {})
        assert result.ok is False
        assert "args.index" in result.message

    async def test_non_integer_index(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        result = await do_set_sref(_ctx(name), {"index": "foo"})
        assert result.ok is False
        assert "integer" in result.message


class TestDoSetPass:
    async def test_updates_pass_number(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        result = await do_set_pass(_ctx(name), {"pass_num": 3})
        assert result.ok is True
        assert "3" in result.message
        assert project_store.load_state(name).current_model_select_pass == 3

    async def test_zero_pass_rejected(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        result = await do_set_pass(_ctx(name), {"pass_num": 0})
        assert result.ok is False
        assert ">= 1" in result.message

    async def test_missing_pass_num(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        result = await do_set_pass(_ctx(name), {})
        assert result.ok is False
        assert "args.pass_num" in result.message


class TestDoAddRefs:
    async def test_advances_from_t2i_and_uploads(self, tmp_path) -> None:
        name = _create_project(phase=Phase.BATCH_T2I, current_batch=1)

        i2i_dir = tmp_path / "i2i-srcs"
        i2i_dir.mkdir()
        (i2i_dir / "src1.png").write_bytes(b"x")
        (i2i_dir / "src2.jpg").write_bytes(b"y")

        with patch(
            "styleclaw.providers.runninghub.upload.upload_file",
            new_callable=AsyncMock,
            side_effect=[
                UploadRecord(local_path="x", url="http://u/1", file_name="src1.png"),
                UploadRecord(local_path="y", url="http://u/2", file_name="src2.jpg"),
            ],
        ), patch(
            "styleclaw.core.image_utils.verify_ref_image",
            return_value=None,
        ):
            result = await do_add_refs(
                _ctx(name, client=MagicMock()),
                {"image_dir": str(i2i_dir)},
            )

        assert result.ok is True, result.message
        assert "2 ref images" in result.message
        state = project_store.load_state(name)
        assert state.phase == Phase.BATCH_I2I
        uploads = project_store.load_i2i_uploads(name, 1)
        assert len(uploads) == 2

    async def test_missing_image_dir(self) -> None:
        name = _create_project(phase=Phase.BATCH_T2I)
        result = await do_add_refs(_ctx(name, client=MagicMock()), {})
        assert result.ok is False
        assert "image_dir" in result.message

    async def test_wrong_phase(self, tmp_path) -> None:
        name = _create_project(phase=Phase.STYLE_REFINE, current_round=1)
        d = tmp_path / "i"
        d.mkdir()
        (d / "a.png").write_bytes(b"x")
        result = await do_add_refs(
            _ctx(name, client=MagicMock()),
            {"image_dir": str(d)},
        )
        assert result.ok is False
        assert "BATCH_T2I or BATCH_I2I" in result.message

    async def test_empty_dir(self, tmp_path) -> None:
        name = _create_project(phase=Phase.BATCH_T2I)
        d = tmp_path / "empty"
        d.mkdir()
        result = await do_add_refs(
            _ctx(name, client=MagicMock()),
            {"image_dir": str(d)},
        )
        assert result.ok is False
        assert "No images" in result.message


class TestDoGenerateModelsFilter:
    async def test_passes_models_string_arg(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        analysis = StyleAnalysis(trigger_phrase="bold anime")
        project_store.save_analysis(name, analysis)
        project_store.save_uploads(name, [
            UploadRecord(local_path="ref1.png", url="http://img/1", file_name="ref1.png"),
        ])

        captured: dict = {}

        async def _fake_generate(name, client, trigger, *, sref_url, models, pass_num, force):
            captured["models"] = models
            return {"mj-v7/prompt-only-male": TaskRecord(task_id="t1", model_id="mj-v7")}

        with patch(
            "styleclaw.scripts.generate.generate_model_select",
            new=_fake_generate,
        ):
            result = await do_generate(
                _ctx(name, client=AsyncMock()),
                {"models": "mj-v7,niji7"},
            )

        assert result.ok is True
        assert "filtered: mj-v7, niji7" in result.message
        assert captured["models"] == ["mj-v7", "niji7"]

    async def test_unknown_model_rejected(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        analysis = StyleAnalysis(trigger_phrase="x")
        project_store.save_analysis(name, analysis)
        project_store.save_uploads(name, [
            UploadRecord(local_path="ref1.png", url="http://img/1", file_name="ref1.png"),
        ])

        result = await do_generate(
            _ctx(name, client=AsyncMock()),
            {"models": "mj-v7,not-a-model"},
        )
        assert result.ok is False
        assert "not-a-model" in result.message
        assert "Unknown model" in result.message

    async def test_no_models_arg_means_all(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        analysis = StyleAnalysis(trigger_phrase="x")
        project_store.save_analysis(name, analysis)
        project_store.save_uploads(name, [
            UploadRecord(local_path="ref1.png", url="http://img/1", file_name="ref1.png"),
        ])

        captured: dict = {}

        async def _fake_generate(name, client, trigger, *, sref_url, models, pass_num, force):
            captured["models"] = models
            return {}

        with patch(
            "styleclaw.scripts.generate.generate_model_select",
            new=_fake_generate,
        ):
            result = await do_generate(_ctx(name, client=AsyncMock()), {})

        assert result.ok is True
        assert captured["models"] is None
        assert "filtered" not in result.message


class TestDoRefinePanel:
    """do_refine should branch on STYLECLAW_PANEL_REFINE."""

    @pytest.fixture(autouse=True)
    def _reset_config_after_test(self, monkeypatch):
        # Tests in this class flip STYLECLAW_PANEL_REFINE on and importlib.reload
        # config_mod so do_refine sees the new state. monkeypatch reverts the env
        # at teardown but does NOT re-reload the module, leaving PANEL_REFINE_ENABLED
        # stuck True for downstream tests. Force a clean reload after each test.
        yield
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODELS", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_LABELS", raising=False)
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)

    @pytest.mark.asyncio
    async def test_panel_off_routes_to_single_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        name = _create_project(phase=Phase.STYLE_REFINE, selected_models=["mj-v7"])
        # Seed analysis so the round-1 path can read a current_trigger.
        project_store.save_analysis(name, StyleAnalysis(trigger_phrase="seed"))

        with patch(
            "styleclaw.agents.refine_prompt.refine_prompt",
            new=AsyncMock(return_value=PromptConfig(round=1, trigger_phrase="single-model-win")),
        ) as single, patch(
            "styleclaw.agents.refine_panel.refine_with_panel",
            new=AsyncMock(),
        ) as panel:
            result = await do_refine(_ctx(name, llm=AsyncMock()), {})

        assert result.ok
        single.assert_awaited_once()
        panel.assert_not_awaited()
        # No panel.json sidecar.
        round_d = project_store.round_dir(name, 1)
        assert not (round_d / "panel.json").exists()

    @pytest.mark.asyncio
    async def test_panel_on_routes_through_panel_and_writes_sidecar(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        import importlib, styleclaw.core.config as config_mod
        importlib.reload(config_mod)

        from styleclaw.core.models import PanelProposal, PanelResult

        name = _create_project(phase=Phase.STYLE_REFINE, selected_models=["mj-v7"])
        project_store.save_analysis(name, StyleAnalysis(trigger_phrase="seed"))

        panel_result = PanelResult(
            proposals=[PanelProposal(model_id="m1", payload={"trigger_phrase": "panel-win"})],
            scores=[],
            winner_model_id="m1",
            averages={"m1": 9.0},
        )
        panel_prompt = PromptConfig(round=1, trigger_phrase="panel-win", derived_from="initial-analysis")

        with patch(
            "styleclaw.providers.llm.panel_factory.build_panel_providers",
            return_value=[(AsyncMock(_model_id=f"m{i}"), f"L{i}") for i in (1, 2, 3)],
        ), patch(
            "styleclaw.providers.llm.panel_factory.close_panel_providers",
            new=AsyncMock(),
        ), patch(
            "styleclaw.agents.refine_panel.refine_with_panel",
            new=AsyncMock(return_value=(panel_prompt, panel_result)),
        ) as panel_call, patch(
            "styleclaw.agents.refine_prompt.refine_prompt",
            new=AsyncMock(),
        ) as single_call:
            result = await do_refine(_ctx(name, llm=AsyncMock()), {})

        assert result.ok
        panel_call.assert_awaited_once()
        single_call.assert_not_awaited()

        # Main artifact contains winner's trigger phrase (unchanged downstream contract).
        loaded = project_store.load_prompt_config(name, round_num=1)
        assert loaded.trigger_phrase == "panel-win"

        # Sidecar exists.
        loaded_panel = project_store.load_round_panel_result(name, round_num=1)
        assert loaded_panel is not None
        assert loaded_panel.winner_model_id == "m1"

    @pytest.mark.asyncio
    async def test_panel_failure_returns_step_failure(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        import importlib, styleclaw.core.config as config_mod
        importlib.reload(config_mod)

        name = _create_project(phase=Phase.STYLE_REFINE, selected_models=["mj-v7"])
        project_store.save_analysis(name, StyleAnalysis(trigger_phrase="seed"))

        with patch(
            "styleclaw.providers.llm.panel_factory.build_panel_providers",
            return_value=[(AsyncMock(_model_id=f"m{i}"), f"L{i}") for i in (1, 2, 3)],
        ), patch(
            "styleclaw.providers.llm.panel_factory.close_panel_providers",
            new=AsyncMock(),
        ), patch(
            "styleclaw.agents.refine_panel.refine_with_panel",
            new=AsyncMock(side_effect=RuntimeError("Refine panel produced no winner")),
        ):
            result = await do_refine(_ctx(name, llm=AsyncMock()), {})

        assert result.ok is False
        assert "panel" in result.message.lower()

    @pytest.mark.asyncio
    async def test_degraded_panel_refuses_to_persist_prompt_or_advance(
        self, monkeypatch,
    ):
        # A degraded panel result must NOT save prompt.json or advance the
        # state — otherwise a half-validated trigger taints downstream rounds.
        # panel.json IS saved (forensic record).
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.delenv("STYLECLAW_ALLOW_DEGRADED_PANEL", raising=False)
        import importlib, styleclaw.core.config as config_mod
        importlib.reload(config_mod)

        from styleclaw.core.models import PanelProposal, PanelResult

        name = _create_project(phase=Phase.STYLE_REFINE, selected_models=["mj-v7"])
        project_store.save_analysis(name, StyleAnalysis(trigger_phrase="seed"))

        degraded = PanelResult(
            proposals=[PanelProposal(model_id="m1", payload={"trigger_phrase": "shaky"})],
            scores=[], winner_model_id="m1", averages={"m1": 7.0},
            degraded=True, error_log=["propose[m2]: TimeoutError: ..."],
        )
        panel_prompt = PromptConfig(round=1, trigger_phrase="shaky", derived_from="initial-analysis")

        with patch(
            "styleclaw.providers.llm.panel_factory.build_panel_providers",
            return_value=[(AsyncMock(_model_id=f"m{i}"), f"L{i}") for i in (1, 2, 3)],
        ), patch(
            "styleclaw.providers.llm.panel_factory.close_panel_providers",
            new=AsyncMock(),
        ), patch(
            "styleclaw.agents.refine_panel.refine_with_panel",
            new=AsyncMock(return_value=(panel_prompt, degraded)),
        ):
            result = await do_refine(_ctx(name, llm=AsyncMock()), {})

        assert result.ok is False
        assert "degraded" in result.message.lower()
        assert "STYLECLAW_ALLOW_DEGRADED_PANEL" in result.message

        # panel.json IS saved (forensic), prompt.json is NOT, state didn't advance.
        loaded_panel = project_store.load_round_panel_result(name, round_num=1)
        assert loaded_panel is not None
        assert loaded_panel.degraded is True
        with pytest.raises(FileNotFoundError):
            project_store.load_prompt_config(name, round_num=1)
        state = project_store.load_state(name)
        assert state.current_round == 0  # unchanged

    @pytest.mark.asyncio
    async def test_allow_degraded_env_overrides_refusal(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.setenv("STYLECLAW_ALLOW_DEGRADED_PANEL", "1")
        import importlib, styleclaw.core.config as config_mod
        importlib.reload(config_mod)

        from styleclaw.core.models import PanelProposal, PanelResult

        name = _create_project(phase=Phase.STYLE_REFINE, selected_models=["mj-v7"])
        project_store.save_analysis(name, StyleAnalysis(trigger_phrase="seed"))

        degraded = PanelResult(
            proposals=[PanelProposal(model_id="m1", payload={"trigger_phrase": "shaky"})],
            scores=[], winner_model_id="m1", averages={"m1": 7.0},
            degraded=True, error_log=["propose[m2]: TimeoutError: ..."],
        )
        panel_prompt = PromptConfig(round=1, trigger_phrase="shaky", derived_from="initial-analysis")

        with patch(
            "styleclaw.providers.llm.panel_factory.build_panel_providers",
            return_value=[(AsyncMock(_model_id=f"m{i}"), f"L{i}") for i in (1, 2, 3)],
        ), patch(
            "styleclaw.providers.llm.panel_factory.close_panel_providers",
            new=AsyncMock(),
        ), patch(
            "styleclaw.agents.refine_panel.refine_with_panel",
            new=AsyncMock(return_value=(panel_prompt, degraded)),
        ):
            result = await do_refine(_ctx(name, llm=AsyncMock()), {})

        assert result.ok is True
        # Both panel.json and prompt.json persisted; state advanced.
        assert project_store.load_round_panel_result(name, round_num=1) is not None
        assert project_store.load_prompt_config(name, round_num=1).trigger_phrase == "shaky"
        assert project_store.load_state(name).current_round == 1


class TestArgsBoundChecks:
    async def test_poll_max_cycles_negative_rejected(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        result = await do_poll(_ctx(name), {"max_cycles": 0})
        assert result.ok is False
        assert ">= 1" in result.message

    async def test_poll_max_cycles_non_integer_rejected(self) -> None:
        name = _create_project(phase=Phase.MODEL_SELECT)
        result = await do_poll(_ctx(name), {"max_cycles": "lots"})
        assert result.ok is False
        assert "integer" in result.message

    async def test_poll_max_cycles_clamped_to_config(self, monkeypatch) -> None:
        # Tiny clamp so the test isn't slow.
        import styleclaw.orchestrator.actions as actions_mod
        monkeypatch.setattr(actions_mod, "MAX_POLL_CYCLES", 2)

        name = _create_project(phase=Phase.MODEL_SELECT)
        # All tasks already SUCCESS so we exit fast — the clamp itself is the
        # behavior under test.
        project_store.save_task_record(
            name, "mj-v7",
            TaskRecord(task_id="t1", model_id="mj-v7", status=TaskStatus.SUCCESS),
        )

        with patch(
            "styleclaw.scripts.poll.poll_model_select",
            new_callable=AsyncMock,
            return_value={
                "mj-v7": TaskRecord(task_id="t1", model_id="mj-v7", status=TaskStatus.SUCCESS),
            },
        ):
            result = await do_poll(_ctx(name), {"max_cycles": 99})
        # Even with max_cycles=99 the action shouldn't crash; clamp produces
        # a successful exit.
        assert result.ok is True

    async def test_batch_submit_unknown_model_rejected(self) -> None:
        name = _create_project(
            phase=Phase.BATCH_T2I, selected_models=["mj-v7"], current_batch=1,
        )
        result = await do_batch_submit(
            _ctx(name, client=AsyncMock()),
            {"model": "not-a-model"},
        )
        assert result.ok is False
        assert "Unknown model" in result.message


class TestDoEvaluatePanel:
    """do_evaluate (MODEL_SELECT) should branch on STYLECLAW_PANEL_MODEL_SELECT."""

    @pytest.fixture(autouse=True)
    def _reset_config_after_test(self, monkeypatch):
        # Same rationale as TestDoRefinePanel._reset_config_after_test.
        yield
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODELS", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_LABELS", raising=False)
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)

    @pytest.mark.asyncio
    async def test_panel_off_routes_to_single_model(self, monkeypatch):
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        name = _create_project(phase=Phase.MODEL_SELECT)
        # Seed at least one model output so the path doesn't short-circuit.
        results_dir = project_store.model_results_dir(name, "mj-v7", variant="prompt-only")
        import PIL.Image
        PIL.Image.new("RGB", (64, 64), color="red").save(str(results_dir / "output-001.png"))
        project_store.save_task_record(
            name, "mj-v7",
            TaskRecord(task_id="t", model_id="mj-v7", status="SUCCESS"),
            variant="prompt-only",
        )

        single_eval = ModelEvaluation(recommendation="mj-v7", recommended_variant="prompt-only")
        with patch(
            "styleclaw.agents.select_model.evaluate_models",
            new=AsyncMock(return_value=single_eval),
        ) as single, patch(
            "styleclaw.agents.select_model_panel.select_models_with_panel",
            new=AsyncMock(),
        ) as panel, patch(
            "styleclaw.scripts.report.generate_model_select_report",
            return_value=Path("/tmp/x.html"),
        ):
            result = await do_evaluate(_ctx(name, llm=AsyncMock()), {})

        assert result.ok
        single.assert_awaited_once()
        panel.assert_not_awaited()
        assert project_store.load_model_select_panel_result(name) is None

    @pytest.mark.asyncio
    async def test_panel_on_routes_through_panel_and_writes_sidecar(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_MODEL_SELECT", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        import importlib, styleclaw.core.config as config_mod
        importlib.reload(config_mod)

        from styleclaw.core.models import PanelProposal, PanelResult

        name = _create_project(phase=Phase.MODEL_SELECT)
        results_dir = project_store.model_results_dir(name, "mj-v7", variant="prompt-only")
        import PIL.Image
        PIL.Image.new("RGB", (64, 64), color="red").save(str(results_dir / "output-001.png"))
        project_store.save_task_record(
            name, "mj-v7",
            TaskRecord(task_id="t", model_id="mj-v7", status="SUCCESS"),
            variant="prompt-only",
        )

        panel_eval = ModelEvaluation(recommendation="mj-v7", recommended_variant="prompt-sref")
        panel_result = PanelResult(
            proposals=[PanelProposal(model_id="m1", payload=panel_eval.model_dump())],
            scores=[],
            winner_model_id="m1",
            averages={"m1": 9.0},
        )

        with patch(
            "styleclaw.providers.llm.panel_factory.build_panel_providers",
            return_value=[(AsyncMock(_model_id=f"m{i}"), f"L{i}") for i in (1, 2, 3)],
        ), patch(
            "styleclaw.providers.llm.panel_factory.close_panel_providers",
            new=AsyncMock(),
        ), patch(
            "styleclaw.agents.select_model_panel.select_models_with_panel",
            new=AsyncMock(return_value=(panel_eval, panel_result)),
        ), patch(
            "styleclaw.scripts.report.generate_model_select_report",
            return_value=Path("/tmp/x.html"),
        ):
            result = await do_evaluate(_ctx(name, llm=AsyncMock()), {})

        assert result.ok
        loaded = project_store.load_evaluation(name)
        assert loaded.recommendation == "mj-v7"
        assert loaded.recommended_variant == "prompt-sref"
        loaded_panel = project_store.load_model_select_panel_result(name)
        assert loaded_panel is not None
        assert loaded_panel.winner_model_id == "m1"
        # Restore config module so subsequent tests see clean env state.
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        importlib.reload(config_mod)

    @pytest.mark.asyncio
    async def test_degraded_panel_refuses_to_save_evaluation(self, monkeypatch):
        # A degraded MODEL_SELECT panel must not save evaluation.json — the
        # downstream `select-model` would otherwise lock in a blind winner.
        monkeypatch.setenv("STYLECLAW_PANEL_MODEL_SELECT", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.delenv("STYLECLAW_ALLOW_DEGRADED_PANEL", raising=False)
        import importlib, styleclaw.core.config as config_mod
        importlib.reload(config_mod)

        from styleclaw.core.models import PanelProposal, PanelResult

        name = _create_project(phase=Phase.MODEL_SELECT)
        results_dir = project_store.model_results_dir(name, "mj-v7", variant="prompt-only")
        import PIL.Image
        PIL.Image.new("RGB", (64, 64), color="red").save(str(results_dir / "output-001.png"))
        project_store.save_task_record(
            name, "mj-v7",
            TaskRecord(task_id="t", model_id="mj-v7", status="SUCCESS"),
            variant="prompt-only",
        )

        panel_eval = ModelEvaluation(recommendation="mj-v7", recommended_variant="prompt-only")
        degraded = PanelResult(
            proposals=[PanelProposal(model_id="m1", payload=panel_eval.model_dump())],
            scores=[], winner_model_id="m1", averages={"m1": 6.0},
            degraded=True, error_log=["propose[m3]: TimeoutError"],
        )

        with patch(
            "styleclaw.providers.llm.panel_factory.build_panel_providers",
            return_value=[(AsyncMock(_model_id=f"m{i}"), f"L{i}") for i in (1, 2, 3)],
        ), patch(
            "styleclaw.providers.llm.panel_factory.close_panel_providers",
            new=AsyncMock(),
        ), patch(
            "styleclaw.agents.select_model_panel.select_models_with_panel",
            new=AsyncMock(return_value=(panel_eval, degraded)),
        ), patch(
            "styleclaw.scripts.report.generate_model_select_report",
            return_value=Path("/tmp/x.html"),
        ) as report_call:
            result = await do_evaluate(_ctx(name, llm=AsyncMock()), {})

        assert result.ok is False
        assert "degraded" in result.message.lower()
        # panel.json saved (forensic), evaluation.json NOT, report NOT generated.
        assert project_store.load_model_select_panel_result(name) is not None
        with pytest.raises(FileNotFoundError):
            project_store.load_evaluation(name)
        report_call.assert_not_called()
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        importlib.reload(config_mod)
