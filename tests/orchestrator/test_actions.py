from __future__ import annotations

from styleclaw.core.models import Phase
from styleclaw.orchestrator.actions import ACTION_REGISTRY, PHASE_ACTIONS


class TestActionRegistry:
    def test_all_actions_registered(self) -> None:
        expected = {
            "analyze", "generate", "poll", "evaluate", "select-model",
            "refine", "approve", "design-cases", "batch-submit", "report",
        }
        assert set(ACTION_REGISTRY.keys()) == expected

    def test_all_phase_actions_exist_in_registry(self) -> None:
        for phase, actions in PHASE_ACTIONS.items():
            for action_name in actions:
                assert action_name in ACTION_REGISTRY, f"{action_name} not in registry (phase={phase})"

    def test_init_phase_actions(self) -> None:
        assert PHASE_ACTIONS[Phase.INIT] == ["analyze"]

    def test_model_select_phase_actions(self) -> None:
        actions = PHASE_ACTIONS[Phase.MODEL_SELECT]
        assert "generate" in actions
        assert "poll" in actions
        assert "evaluate" in actions
        assert "select-model" in actions

    def test_style_refine_phase_actions(self) -> None:
        actions = PHASE_ACTIONS[Phase.STYLE_REFINE]
        assert "refine" in actions
        assert "approve" in actions

    def test_batch_t2i_phase_actions(self) -> None:
        actions = PHASE_ACTIONS[Phase.BATCH_T2I]
        assert "design-cases" in actions
        assert "batch-submit" in actions
        assert "report" in actions

    def test_completed_phase_no_actions(self) -> None:
        assert PHASE_ACTIONS[Phase.COMPLETED] == []

    def test_action_client_requirements(self) -> None:
        assert ACTION_REGISTRY["generate"].needs_client is True
        assert ACTION_REGISTRY["poll"].needs_client is True
        assert ACTION_REGISTRY["batch-submit"].needs_client is True
        assert ACTION_REGISTRY["analyze"].needs_client is False
        assert ACTION_REGISTRY["evaluate"].needs_client is False

    def test_action_llm_requirements(self) -> None:
        assert ACTION_REGISTRY["analyze"].needs_llm is True
        assert ACTION_REGISTRY["evaluate"].needs_llm is True
        assert ACTION_REGISTRY["refine"].needs_llm is True
        assert ACTION_REGISTRY["design-cases"].needs_llm is True
        assert ACTION_REGISTRY["generate"].needs_llm is False
        assert ACTION_REGISTRY["approve"].needs_llm is False


class TestExecutionContextThinking:
    def test_thinking_defaults(self):
        from styleclaw.orchestrator.actions import ExecutionContext
        ctx = ExecutionContext(project="p")
        assert ctx.show_thinking is False
        assert ctx.thinking_budget == 5000

    def test_thinking_can_be_set(self):
        from styleclaw.orchestrator.actions import ExecutionContext
        ctx = ExecutionContext(project="p", show_thinking=True, thinking_budget=8000)
        assert ctx.show_thinking is True
        assert ctx.thinking_budget == 8000
