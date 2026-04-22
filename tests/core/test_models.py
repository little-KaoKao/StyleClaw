from styleclaw.core.models import (
    Phase,
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
