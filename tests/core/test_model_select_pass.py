from styleclaw.core.models import Phase, ProjectState


class TestModelSelectPass:
    def test_default_is_zero(self):
        state = ProjectState()
        assert state.current_model_select_pass == 0

    def test_with_pass_returns_new_instance(self):
        state = ProjectState()
        new_state = state.with_model_select_pass(1)
        assert new_state.current_model_select_pass == 1
        assert state.current_model_select_pass == 0

    def test_with_pass_updates_timestamp(self):
        state = ProjectState()
        original_ts = state.last_updated
        new_state = state.with_model_select_pass(2)
        assert new_state.last_updated != original_ts

    def test_with_pass_preserves_other_fields(self):
        state = ProjectState(
            phase=Phase.MODEL_SELECT,
            selected_models=["mj-v7"],
            current_round=3,
            current_batch=1,
        )
        new_state = state.with_model_select_pass(2)
        assert new_state.phase == Phase.MODEL_SELECT
        assert new_state.selected_models == ["mj-v7"]
        assert new_state.current_round == 3
        assert new_state.current_batch == 1
        assert new_state.current_model_select_pass == 2

    def test_pass_persists_through_with_phase(self):
        state = ProjectState(current_model_select_pass=2)
        new_state = state.with_phase(Phase.MODEL_SELECT)
        assert new_state.current_model_select_pass == 2
