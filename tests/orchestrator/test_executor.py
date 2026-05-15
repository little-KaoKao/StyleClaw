from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from styleclaw.core.models import (
    Action,
    ActionPlan,
    DimensionScores,
    LoopConfig,
    Phase,
    ProjectConfig,
    ProjectState,
    PromptConfig,
    RoundEvaluation,
    RoundScore,
    StyleAnalysis,
)
from styleclaw.orchestrator.actions import ActionDef, ExecutionContext, StepResult
from styleclaw.orchestrator.executor import _should_continue_loop, display_plan, execute
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def setup_project():
    config = ProjectConfig(name="test-proj", ip_info="anime")
    root = project_store.create_project(config)
    (root / "refs" / "ref-001.png").write_bytes(b"fake image")
    return root


@pytest.fixture
def ctx():
    return ExecutionContext(project="test-proj")


class TestShouldContinueLoop:
    def test_no_round(self, setup_project, ctx) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.STYLE_REFINE, current_round=0))
        assert _should_continue_loop(ctx) is False

    def test_no_evaluation(self, setup_project, ctx) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.STYLE_REFINE, current_round=1))
        assert _should_continue_loop(ctx) is False

    def test_passing_scores(self, setup_project, ctx) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.STYLE_REFINE, current_round=1))
        scores = DimensionScores(color_palette=8.0, line_style=8.0, lighting=8.0, texture=8.0, overall_mood=8.0)
        evaluation = RoundEvaluation(
            round=1,
            evaluations=[RoundScore(model="mj-v7", scores=scores, total=8.0)],
        )
        project_store.save_round_evaluation("test-proj", 1, evaluation)
        assert _should_continue_loop(ctx) is False

    def test_failing_scores(self, setup_project, ctx) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.STYLE_REFINE, current_round=1))
        scores = DimensionScores(color_palette=5.0, line_style=6.0, lighting=6.0, texture=6.0, overall_mood=6.0)
        evaluation = RoundEvaluation(
            round=1,
            evaluations=[RoundScore(model="mj-v7", scores=scores, total=5.8)],
        )
        project_store.save_round_evaluation("test-proj", 1, evaluation)
        assert _should_continue_loop(ctx) is True

    def test_needs_human_prints_diagnosis(self, setup_project, ctx, capsys) -> None:
        project_store.save_state(
            "test-proj",
            ProjectState(phase=Phase.STYLE_REFINE, current_round=2),
        )
        weak = DimensionScores(color_palette=4.5, line_style=7.0, lighting=7.0, texture=7.0, overall_mood=7.0)
        strong = DimensionScores(color_palette=8.0, line_style=8.0, lighting=8.0, texture=8.0, overall_mood=8.0)
        evaluation = RoundEvaluation(
            round=2,
            evaluations=[
                RoundScore(model="mj-v7", scores=weak, total=6.7),
                RoundScore(model="niji7", scores=strong, total=8.0),
            ],
        )
        project_store.save_round_evaluation("test-proj", 2, evaluation)

        assert _should_continue_loop(ctx) is False

        err = capsys.readouterr().err
        assert "needs_human" in err
        assert "色彩调性得分 4.5 最弱" in err
        assert 'styleclaw run "提高色彩饱和度、减弱色差光" -p test-proj' in err
        assert "style-refine/pass-001/round-002/report.html" in err


class TestDisplayPlan:
    def test_display_without_loop(self, setup_project, capsys) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.INIT))
        plan = ActionPlan(
            summary="Analyze style",
            steps=[Action(name="analyze", description="分析参考图片")],
        )
        display_plan(plan, "test-proj")
        captured = capsys.readouterr()
        assert "Analyze style" in captured.out
        assert "analyze" in captured.out

    def test_display_with_loop(self, setup_project, capsys) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.STYLE_REFINE))
        plan = ActionPlan(
            summary="Refine loop",
            steps=[
                Action(name="refine", description="精炼"),
                Action(name="generate", description="生成"),
                Action(name="poll", description="等待"),
                Action(name="evaluate", description="评估"),
            ],
            loop=LoopConfig(start_step=0, end_step=3, max_iterations=3),
        )
        display_plan(plan, "test-proj")
        captured = capsys.readouterr()
        assert "Loop" in captured.out
        assert "max 3" in captured.out

    def test_display_with_stop_summary(self, setup_project, capsys) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.INIT))
        plan = ActionPlan(
            summary="Analyze",
            steps=[Action(name="analyze", description="分析")],
            stop_summary="分析完后停在 MODEL_SELECT，告诉我选哪个模型",
        )
        display_plan(plan, "test-proj")
        captured = capsys.readouterr()
        assert "执行完后" in captured.out
        assert "告诉我选哪个模型" in captured.out

    def test_display_no_stop_summary_omits_block(self, setup_project, capsys) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.INIT))
        plan = ActionPlan(
            summary="x",
            steps=[Action(name="analyze", description="x")],
        )
        display_plan(plan, "test-proj")
        captured = capsys.readouterr()
        assert "执行完后" not in captured.out


class TestExecute:
    async def test_single_step(self, setup_project, ctx) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.INIT))

        mock_result = StepResult(ok=True, message="done")

        plan = ActionPlan(
            summary="Test",
            steps=[Action(name="analyze", description="分析")],
        )

        with patch.dict(
            "styleclaw.orchestrator.executor.ACTION_REGISTRY",
            {"analyze": ActionDef(fn=AsyncMock(return_value=mock_result))},
        ):
            results = await execute(plan, ctx)

        assert len(results) == 1
        assert results[0].ok is True

    async def test_failure_stops_early(self, setup_project, ctx) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.INIT))

        fail_result = StepResult(ok=False, message="error")
        plan = ActionPlan(
            summary="Fail test",
            steps=[
                Action(name="analyze", description="分析"),
                Action(name="generate", description="生成"),
            ],
        )

        with patch.dict(
            "styleclaw.orchestrator.executor.ACTION_REGISTRY",
            {"analyze": ActionDef(fn=AsyncMock(return_value=fail_result))},
        ):
            results = await execute(plan, ctx)

        assert len(results) == 1
        assert results[0].ok is False

    async def test_unknown_action_stops_early(self, setup_project, ctx) -> None:
        plan = ActionPlan(
            summary="Unknown",
            steps=[Action(name="nonexistent", description="不存在")],
        )

        results = await execute(plan, ctx)
        assert len(results) == 1
        assert results[0].ok is False
        assert "Unknown action" in results[0].message

    async def test_callbacks_called(self, setup_project, ctx) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.INIT))

        mock_result = StepResult(ok=True, message="done")
        plan = ActionPlan(
            summary="Test",
            steps=[Action(name="analyze", description="分析")],
        )

        starts = []
        dones = []

        with patch.dict(
            "styleclaw.orchestrator.executor.ACTION_REGISTRY",
            {"analyze": ActionDef(fn=AsyncMock(return_value=mock_result))},
        ):
            await execute(
                plan, ctx,
                on_step_start=lambda i, n, d: starts.append((i, n)),
                on_step_done=lambda i, n, r: dones.append((i, n, r.ok)),
            )

        assert starts == [(0, "analyze")]
        assert dones == [(0, "analyze", True)]

    async def test_loop_execution(self, setup_project, ctx) -> None:
        project_store.save_state("test-proj", ProjectState(phase=Phase.STYLE_REFINE, current_round=1))

        call_count = 0

        async def mock_fn(c, args):
            nonlocal call_count
            call_count += 1
            return StepResult(ok=True, message=f"step {call_count}")

        mock_action = ActionDef(fn=mock_fn)

        plan = ActionPlan(
            summary="Loop test",
            steps=[
                Action(name="refine", description="精炼"),
                Action(name="evaluate", description="评估"),
            ],
            loop=LoopConfig(start_step=0, end_step=1, max_iterations=2),
        )

        with patch.dict(
            "styleclaw.orchestrator.executor.ACTION_REGISTRY",
            {"refine": mock_action, "evaluate": mock_action},
        ), patch(
            "styleclaw.orchestrator.executor._should_continue_loop",
            side_effect=[True, False],
        ):
            results = await execute(plan, ctx)

        assert call_count == 4


class TestArgsValidation:
    async def test_unknown_arg_key_rejected(self, setup_project, ctx):
        from styleclaw.orchestrator.actions import ACTION_REGISTRY

        plan = ActionPlan(
            summary="x",
            steps=[Action(name="generate", description="x", args={"banana": 1})],
        )
        # Real action registry; the unknown key should never reach action.fn.
        mock_fn = AsyncMock(return_value=StepResult(ok=True, message="should not run"))
        mock_action = type(ACTION_REGISTRY["generate"])(
            fn=mock_fn,
            needs_client=False,
            needs_llm=False,
        )
        with patch.dict(
            "styleclaw.orchestrator.executor.ACTION_REGISTRY",
            {"generate": mock_action},
        ):
            results = await execute(plan, ctx)

        assert len(results) == 1
        assert results[0].ok is False
        assert "unknown args" in results[0].message
        assert "banana" in results[0].message
        mock_fn.assert_not_called()

    async def test_known_arg_passes_through(self, setup_project, ctx):
        from styleclaw.orchestrator.actions import ACTION_REGISTRY

        plan = ActionPlan(
            summary="x",
            steps=[Action(name="generate", description="x", args={"force": True})],
        )
        captured = {}

        async def _capture(c, args):
            captured.update(args)
            return StepResult(ok=True, message="captured")

        mock_action = type(ACTION_REGISTRY["generate"])(
            fn=_capture, needs_client=False, needs_llm=False,
        )
        with patch.dict(
            "styleclaw.orchestrator.executor.ACTION_REGISTRY",
            {"generate": mock_action},
        ):
            results = await execute(plan, ctx)

        assert results[0].ok is True
        assert captured == {"force": True}
