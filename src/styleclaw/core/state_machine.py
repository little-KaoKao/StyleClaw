from __future__ import annotations

from styleclaw.core.models import Phase, ProjectState

TRANSITIONS: dict[Phase, list[Phase]] = {
    Phase.INIT: [Phase.MODEL_SELECT],
    Phase.MODEL_SELECT: [Phase.STYLE_REFINE],
    Phase.STYLE_REFINE: [Phase.BATCH_T2I, Phase.STYLE_REFINE, Phase.MODEL_SELECT],
    Phase.BATCH_T2I: [Phase.BATCH_I2I, Phase.STYLE_REFINE, Phase.MODEL_SELECT],
    Phase.BATCH_I2I: [Phase.STYLE_REFINE, Phase.BATCH_T2I, Phase.COMPLETED],
}

ALL_PHASES: list[Phase] = [
    Phase.INIT,
    Phase.MODEL_SELECT,
    Phase.STYLE_REFINE,
    Phase.BATCH_T2I,
    Phase.BATCH_I2I,
    Phase.COMPLETED,
]


def can_advance(current: Phase, target: Phase) -> bool:
    return target in TRANSITIONS.get(current, [])


def advance(state: ProjectState, target: Phase) -> ProjectState:
    if not can_advance(state.phase, target):
        raise ValueError(
            f"Cannot transition from {state.phase} to {target}. "
            f"Allowed: {TRANSITIONS.get(state.phase, [])}"
        )
    return state.with_phase(target)


def can_rollback(state: ProjectState, target: Phase) -> bool:
    if state.phase == Phase.INIT:
        return False
    current_idx = ALL_PHASES.index(state.phase)
    target_idx = ALL_PHASES.index(target)
    if target_idx >= current_idx:
        return False
    visited = {entry.phase for entry in state.history}
    visited.add(state.phase)
    return target in visited


def rollback(state: ProjectState, target: Phase) -> ProjectState:
    if not can_rollback(state, target):
        raise ValueError(
            f"Cannot rollback from {state.phase} to {target}. "
            f"Target must be an earlier, previously visited phase."
        )
    return state.with_phase(target, metadata={"rollback_from": str(state.phase)})
