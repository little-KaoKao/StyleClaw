import pytest
from pydantic import ValidationError

from styleclaw.core.models import (
    Phase,
    PanelProposal,
    PanelResult,
    PanelScore,
    ProjectConfig,
    ProjectState,
)


def test_phase_enum_values():
    assert Phase.INIT == "INIT"
    assert Phase.MODEL_SELECT == "MODEL_SELECT"
    assert Phase.COMPLETED == "COMPLETED"


def test_project_config_defaults():
    config = ProjectConfig(name="test")
    assert config.name == "test"
    assert config.description == ""
    assert config.ref_images == []
    assert config.created_at  # should be auto-populated


def test_project_state_defaults():
    state = ProjectState()
    assert state.phase == Phase.INIT
    assert state.selected_models == []
    assert state.current_round == 0
    assert state.history == []


def test_with_phase_returns_new_state():
    state = ProjectState()
    new_state = state.with_phase(Phase.MODEL_SELECT)

    assert new_state is not state
    assert new_state.phase == Phase.MODEL_SELECT
    assert state.phase == Phase.INIT  # original unchanged
    assert len(new_state.history) == 1
    assert new_state.history[0].phase == Phase.INIT


def test_with_selected_models_returns_new_state():
    state = ProjectState(phase=Phase.MODEL_SELECT)
    new_state = state.with_selected_models(["mj-v7", "niji7"])

    assert new_state is not state
    assert new_state.selected_models == ["mj-v7", "niji7"]
    assert state.selected_models == []  # original unchanged


def test_with_round_returns_new_state():
    state = ProjectState()
    new_state = state.with_round(3)

    assert new_state is not state
    assert new_state.current_round == 3
    assert state.current_round == 0


class TestPanelModels:
    def test_proposal_defaults(self):
        p = PanelProposal(model_id="m1", payload={"trigger_phrase": "foo"})
        assert p.label == ""
        assert p.thinking == ""
        assert p.payload == {"trigger_phrase": "foo"}

    def test_proposal_is_frozen(self):
        p = PanelProposal(model_id="m1", payload={})
        with pytest.raises(ValidationError):
            p.model_id = "m2"

    def test_score_required_fields(self):
        s = PanelScore(evaluator_model_id="e", target_model_id="t", score=8.5)
        assert s.rationale == ""

    def test_result_defaults(self):
        r = PanelResult(
            proposals=[],
            scores=[],
            winner_model_id="",
            averages={},
        )
        assert r.degraded is False
        assert r.error_log == []

    def test_result_holds_full_panel(self):
        proposals = [
            PanelProposal(model_id="a", payload={"x": 1}),
            PanelProposal(model_id="b", payload={"x": 2}),
            PanelProposal(model_id="c", payload={"x": 3}),
        ]
        scores = [
            PanelScore(evaluator_model_id="a", target_model_id="b", score=7.0),
            PanelScore(evaluator_model_id="a", target_model_id="c", score=6.0),
            PanelScore(evaluator_model_id="b", target_model_id="a", score=8.0),
            PanelScore(evaluator_model_id="b", target_model_id="c", score=7.5),
            PanelScore(evaluator_model_id="c", target_model_id="a", score=9.0),
            PanelScore(evaluator_model_id="c", target_model_id="b", score=8.5),
        ]
        r = PanelResult(
            proposals=proposals,
            scores=scores,
            winner_model_id="a",
            averages={"a": 8.5, "b": 7.75, "c": 6.75},
        )
        assert r.winner_model_id == "a"
        assert len(r.scores) == 6


from styleclaw.core.models import (
    BatchConfig,
    ModelEvaluation,
    PromptConfig,
    RoundEvaluation,
    StyleAnalysis,
)


class TestModelIdField:
    """All 5 LLM-derived artifact models carry an optional model_id field.

    Default "" lets old on-disk JSON (without the field) round-trip cleanly,
    and lets action code fill it post-parse via model_copy(update=...).
    """

    def test_style_analysis_defaults_empty(self):
        assert StyleAnalysis().model_id == ""

    def test_model_evaluation_defaults_empty(self):
        assert ModelEvaluation().model_id == ""

    def test_round_evaluation_defaults_empty(self):
        assert RoundEvaluation().model_id == ""

    def test_prompt_config_defaults_empty(self):
        assert PromptConfig().model_id == ""

    def test_batch_config_defaults_empty(self):
        assert BatchConfig().model_id == ""

    def test_old_json_without_field_loads(self):
        # Simulate a legacy file written before the field was added.
        legacy_json = '{"trigger_phrase": "x"}'
        analysis = StyleAnalysis.model_validate_json(legacy_json)
        assert analysis.trigger_phrase == "x"
        assert analysis.model_id == ""

    def test_model_copy_update_sets_field(self):
        analysis = StyleAnalysis(trigger_phrase="t")
        updated = analysis.model_copy(update={"model_id": "claude-sonnet-4-6"})
        assert updated.model_id == "claude-sonnet-4-6"
        # Original frozen instance unchanged.
        assert analysis.model_id == ""

    def test_round_evaluation_with_round_still_works(self):
        # RoundEvaluation has helper methods — make sure model_id doesn't
        # interfere with them.
        ev = RoundEvaluation(round=1, model_id="m").model_copy(
            update={"evaluations": []}
        )
        assert ev.model_id == "m"
        assert ev.should_approve() is False
