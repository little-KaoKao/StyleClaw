from __future__ import annotations

import pytest

from styleclaw.core.models import (
    Action,
    ActionPlan,
    BatchCase,
    BatchConfig,
    LoopConfig,
    Phase,
    ProjectConfig,
    ProjectState,
)
from styleclaw.orchestrator.cost_estimate import (
    PlanEstimate,
    StepEstimate,
    estimate_plan,
    format_plan_estimate,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


def _setup(name: str, phase: Phase, **state_kwargs) -> None:
    project_store.create_project(ProjectConfig(name=name))
    project_store.save_state(name, ProjectState(phase=phase, **state_kwargs))


class TestEstimateGenerate:
    def test_model_select_all_models(self):
        _setup("p", Phase.MODEL_SELECT)
        plan = ActionPlan(
            summary="x",
            steps=[Action(name="generate", description="x")],
        )
        est = estimate_plan(plan, "p")
        # 5 models × 2 variants × 2 genders = 20 tasks
        assert est.total_tasks == 20
        # mj-v7 (4) + niji7 (4) + nb2 (1) + seedream (1) + gpt-image-2 (1) = 11 per gender×variant cell
        # 11 * 4 = 44 expected images
        assert est.total_images == 44

    def test_model_select_filtered(self):
        _setup("p", Phase.MODEL_SELECT)
        plan = ActionPlan(
            summary="x",
            steps=[Action(name="generate", description="x", args={"models": "mj-v7,niji7"})],
        )
        est = estimate_plan(plan, "p")
        assert est.total_tasks == 8  # 2 models × 4 cells
        assert est.total_images == 32  # both 4-image models

    def test_style_refine_uses_selected_models(self):
        _setup("p", Phase.STYLE_REFINE, selected_models=["mj-v7", "nb2"], current_round=1)
        plan = ActionPlan(
            summary="x",
            steps=[Action(name="generate", description="x")],
        )
        est = estimate_plan(plan, "p")
        assert est.total_tasks == 2
        # mj-v7=4, nb2 in refine mode=3
        assert est.total_images == 7


class TestEstimateBatchSubmit:
    def test_pending_cases_counted(self, tmp_path):
        _setup(
            "p", Phase.BATCH_T2I,
            selected_models=["mj-v7"], current_round=1, current_batch=1,
        )
        cases = [
            BatchCase(id=f"c{i:03d}", category="adult_male", description="x", status="pending")
            for i in range(7)
        ]
        cases.append(
            BatchCase(id="c-done", category="adult_male", description="x", status="success")
        )
        cfg = BatchConfig(batch=1, trigger_phrase="t", cases=cases)
        project_store.save_batch_config("p", 1, cfg)

        plan = ActionPlan(
            summary="x",
            steps=[Action(name="batch-submit", description="x")],
        )
        est = estimate_plan(plan, "p")
        assert est.total_tasks == 7
        assert est.total_images == 28  # mj-v7 returns 4

    def test_no_cases_yet(self):
        _setup("p", Phase.BATCH_T2I, current_batch=0)
        plan = ActionPlan(
            summary="x", steps=[Action(name="batch-submit", description="x")],
        )
        est = estimate_plan(plan, "p")
        assert est.total_tasks == 0
        assert "design-cases" in est.steps[0].note


class TestLoopMultiplier:
    def test_total_multiplied_by_loop(self):
        _setup("p", Phase.STYLE_REFINE, selected_models=["mj-v7"], current_round=1)
        plan = ActionPlan(
            summary="x",
            steps=[
                Action(name="refine", description="x"),
                Action(name="generate", description="x"),
                Action(name="poll", description="x"),
                Action(name="evaluate", description="x"),
            ],
            loop=LoopConfig(start_step=0, end_step=3, max_iterations=5),
        )
        est = estimate_plan(plan, "p")
        # 1 task per iter × 5 iters
        assert est.total_tasks == 5
        assert est.total_images == 20  # 4 × 5


class TestNoCost:
    def test_llm_only_steps_have_zero_tasks(self):
        _setup("p", Phase.INIT)
        plan = ActionPlan(
            summary="x",
            steps=[Action(name="analyze", description="x")],
        )
        est = estimate_plan(plan, "p")
        assert est.total_tasks == 0
        assert "LLM" in est.steps[0].note

    def test_no_project_returns_empty_estimate(self):
        plan = ActionPlan(
            summary="init",
            steps=[Action(name="init", description="x")],
        )
        est = estimate_plan(plan, "ghost")
        assert est.steps == []


class TestFormatting:
    def test_format_includes_total_line(self):
        est = PlanEstimate(
            steps=[StepEstimate("generate", tasks=20, images=44, note="5 模型")],
            loop_iterations_max=1,
        )
        lines = format_plan_estimate(est)
        assert any("20 任务" in line for line in lines)
        assert any("44 张图" in line for line in lines)
        assert any("合计" in line for line in lines)

    def test_format_loop_suffix(self):
        est = PlanEstimate(
            steps=[StepEstimate("generate", tasks=1, images=4)],
            loop_iterations_max=5,
        )
        lines = format_plan_estimate(est)
        joined = "\n".join(lines)
        assert "最多 5 轮" in joined
        assert "5 任务" in joined  # 1 × 5

    def test_format_empty_estimate_returns_empty(self):
        assert format_plan_estimate(PlanEstimate(steps=[])) == []
