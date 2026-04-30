import pytest

from styleclaw.core.models import Phase, ProjectState
from styleclaw.core.state_machine import advance, can_advance


class TestBidirectionalTransitions:
    def test_style_refine_to_model_select_allowed(self):
        assert can_advance(Phase.STYLE_REFINE, Phase.MODEL_SELECT) is True

    def test_batch_t2i_to_model_select_allowed(self):
        assert can_advance(Phase.BATCH_T2I, Phase.MODEL_SELECT) is True

    def test_batch_i2i_to_batch_t2i_allowed(self):
        assert can_advance(Phase.BATCH_I2I, Phase.BATCH_T2I) is True

    def test_advance_style_refine_to_model_select(self):
        state = ProjectState(phase=Phase.STYLE_REFINE, current_round=2)
        new_state = advance(state, Phase.MODEL_SELECT)
        assert new_state.phase == Phase.MODEL_SELECT
        assert new_state.current_round == 2

    def test_advance_batch_t2i_to_model_select(self):
        state = ProjectState(phase=Phase.BATCH_T2I, current_batch=1)
        new_state = advance(state, Phase.MODEL_SELECT)
        assert new_state.phase == Phase.MODEL_SELECT
        assert new_state.current_batch == 1

    def test_advance_batch_i2i_to_batch_t2i(self):
        state = ProjectState(phase=Phase.BATCH_I2I, current_batch=1)
        new_state = advance(state, Phase.BATCH_T2I)
        assert new_state.phase == Phase.BATCH_T2I

    def test_existing_transitions_not_broken(self):
        assert can_advance(Phase.INIT, Phase.MODEL_SELECT) is True
        assert can_advance(Phase.MODEL_SELECT, Phase.STYLE_REFINE) is True
        assert can_advance(Phase.STYLE_REFINE, Phase.BATCH_T2I) is True
        assert can_advance(Phase.STYLE_REFINE, Phase.STYLE_REFINE) is True
        assert can_advance(Phase.BATCH_T2I, Phase.BATCH_I2I) is True
        assert can_advance(Phase.BATCH_I2I, Phase.COMPLETED) is True

    def test_invalid_transitions_still_blocked(self):
        assert can_advance(Phase.INIT, Phase.BATCH_T2I) is False
        assert can_advance(Phase.INIT, Phase.COMPLETED) is False
        assert can_advance(Phase.COMPLETED, Phase.INIT) is False
