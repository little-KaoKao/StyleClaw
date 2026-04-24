from __future__ import annotations

from styleclaw.core.models import Action, ActionPlan, LoopConfig


class TestActionPlanModels:
    def test_action_basic(self) -> None:
        action = Action(name="analyze", description="Analyze style")
        assert action.name == "analyze"
        assert action.args == {}

    def test_action_with_args(self) -> None:
        action = Action(name="select-model", description="Select", args={"models": "mj-v7"})
        assert action.args["models"] == "mj-v7"

    def test_loop_config(self) -> None:
        loop = LoopConfig(start_step=0, end_step=3, max_iterations=5)
        assert loop.start_step == 0
        assert loop.max_iterations == 5

    def test_action_plan_without_loop(self) -> None:
        plan = ActionPlan(
            summary="Test plan",
            steps=[Action(name="analyze", description="Analyze")],
        )
        assert plan.loop is None
        assert len(plan.steps) == 1

    def test_action_plan_with_loop(self) -> None:
        plan = ActionPlan(
            summary="Refine loop",
            steps=[
                Action(name="refine", description="Refine trigger"),
                Action(name="generate", description="Generate"),
                Action(name="poll", description="Wait"),
                Action(name="evaluate", description="Score"),
            ],
            loop=LoopConfig(start_step=0, end_step=3),
        )
        assert plan.loop is not None
        assert plan.loop.end_step == 3

    def test_action_plan_from_json(self) -> None:
        data = {
            "summary": "Run analysis",
            "steps": [
                {"name": "analyze", "description": "Analyze images", "args": {}},
            ],
            "loop": None,
        }
        plan = ActionPlan.model_validate(data)
        assert plan.steps[0].name == "analyze"

    def test_action_plan_immutability(self) -> None:
        plan = ActionPlan(
            summary="Test",
            steps=[Action(name="analyze", description="A")],
        )
        new_plan = plan.model_copy(update={"summary": "Updated"})
        assert plan.summary == "Test"
        assert new_plan.summary == "Updated"
