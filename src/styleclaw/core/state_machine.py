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

# Phase-specific suggestions to surface when a transition is blocked.
_PHASE_HINTS: dict[Phase, str] = {
    Phase.INIT: "Run 'styleclaw analyze <project>' to advance to MODEL_SELECT.",
    Phase.MODEL_SELECT: (
        "Run 'styleclaw generate', 'poll', then 'evaluate', "
        "and finally 'select-model --models ...' to advance to STYLE_REFINE."
    ),
    Phase.STYLE_REFINE: (
        "Run 'styleclaw refine' / 'generate' / 'poll' / 'evaluate' rounds, "
        "then 'styleclaw approve' to advance to BATCH_T2I."
    ),
    Phase.BATCH_T2I: (
        "Run 'styleclaw design-cases', 'batch-submit', 'poll', 'report'. "
        "Use 'add-refs' to advance to BATCH_I2I."
    ),
    Phase.BATCH_I2I: (
        "Run 'styleclaw batch-submit --i2i', 'poll', 'report'. "
        "Use 'styleclaw approve --phase completed' when done."
    ),
    Phase.COMPLETED: "Project already completed.",
}


def _hint_for(phase: Phase) -> str:
    return _PHASE_HINTS.get(phase, "")


def can_advance(current: Phase, target: Phase) -> bool:
    return target in TRANSITIONS.get(current, [])


def advance(state: ProjectState, target: Phase) -> ProjectState:
    if not can_advance(state.phase, target):
        allowed = TRANSITIONS.get(state.phase, [])
        msg = (
            f"Cannot transition from {state.phase} to {target}. "
            f"Allowed: {allowed}."
        )
        hint = _hint_for(state.phase)
        if hint:
            msg += f"\nNext step: {hint}"
        raise ValueError(msg)
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
        visited = sorted({entry.phase for entry in state.history} | {state.phase}, key=ALL_PHASES.index)
        raise ValueError(
            f"Cannot rollback from {state.phase} to {target}. "
            f"Target must be an earlier, previously visited phase. "
            f"Visited phases: {[p.value for p in visited]}"
        )
    return state.with_phase(target, metadata={"rollback_from": str(state.phase)})
