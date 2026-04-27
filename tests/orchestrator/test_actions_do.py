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
    UploadRecord,
)
from styleclaw.orchestrator.actions import (
    ExecutionContext,
    StepResult,
    do_analyze,
    do_approve,
    do_batch_submit,
    do_design_cases,
    do_evaluate,
    do_generate,
    do_poll,
    do_refine,
    do_report,
    do_select_model,
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
) -> ExecutionContext:
    return ExecutionContext(
        project=name,
        client=client,
        llm=llm,
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
            result = await do_analyze(_ctx(name, llm=mock_llm), {})

        assert result.ok is True
        assert "bold anime lineart" in result.message
        saved = project_store.load_analysis(name)
        assert saved.trigger_phrase == "bold anime lineart"
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
