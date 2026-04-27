import pytest

from styleclaw.core.models import Phase, ProjectState
from styleclaw.core.state_machine import (
    advance,
    can_advance,
    can_rollback,
    rollback,
)


class TestCanAdvance:
    def test_init_to_model_select(self):
        assert can_advance(Phase.INIT, Phase.MODEL_SELECT) is True

    def test_init_to_style_refine_blocked(self):
        assert can_advance(Phase.INIT, Phase.STYLE_REFINE) is False

    def test_model_select_to_style_refine(self):
        assert can_advance(Phase.MODEL_SELECT, Phase.STYLE_REFINE) is True

    def test_style_refine_self_loop(self):
        assert can_advance(Phase.STYLE_REFINE, Phase.STYLE_REFINE) is True

    def test_style_refine_to_batch_t2i(self):
        assert can_advance(Phase.STYLE_REFINE, Phase.BATCH_T2I) is True

    def test_batch_t2i_to_style_refine(self):
        assert can_advance(Phase.BATCH_T2I, Phase.STYLE_REFINE) is True

    def test_batch_t2i_to_completed(self):
        assert can_advance(Phase.BATCH_T2I, Phase.COMPLETED) is True

    def test_batch_t2i_to_completed_advance(self):
        state = ProjectState(phase=Phase.BATCH_T2I)
        new_state = advance(state, Phase.COMPLETED)
        assert new_state.phase == Phase.COMPLETED

    def test_batch_i2i_to_completed(self):
        assert can_advance(Phase.BATCH_I2I, Phase.COMPLETED) is True


class TestAdvance:
    def test_advance_succeeds(self):
        state = ProjectState(phase=Phase.INIT)
        new_state = advance(state, Phase.MODEL_SELECT)
        assert new_state.phase == Phase.MODEL_SELECT
        assert len(new_state.history) == 1

    def test_advance_invalid_raises(self):
        state = ProjectState(phase=Phase.INIT)
        with pytest.raises(ValueError, match="Cannot transition"):
            advance(state, Phase.COMPLETED)

    def test_advance_preserves_other_fields(self):
        state = ProjectState(
            phase=Phase.MODEL_SELECT,
            selected_models=["mj-v7"],
            current_round=2,
        )
        new_state = advance(state, Phase.STYLE_REFINE)
        assert new_state.selected_models == ["mj-v7"]
        assert new_state.current_round == 2


class TestRollback:
    def test_can_rollback_to_earlier_phase(self):
        assert can_rollback(Phase.STYLE_REFINE, Phase.INIT) is True

    def test_cannot_rollback_from_init(self):
        assert can_rollback(Phase.INIT, Phase.INIT) is False

    def test_cannot_rollback_to_later_phase(self):
        assert can_rollback(Phase.INIT, Phase.MODEL_SELECT) is False

    def test_rollback_succeeds(self):
        state = ProjectState(phase=Phase.STYLE_REFINE)
        new_state = rollback(state, Phase.INIT)
        assert new_state.phase == Phase.INIT
        assert new_state.history[-1].metadata["rollback_from"] == "STYLE_REFINE"

    def test_rollback_invalid_raises(self):
        state = ProjectState(phase=Phase.INIT)
        with pytest.raises(ValueError, match="Cannot rollback"):
            rollback(state, Phase.INIT)
