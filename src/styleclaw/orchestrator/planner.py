from __future__ import annotations

import logging
from pathlib import Path
from string import Template

from styleclaw.core.models import ActionPlan, Phase
from styleclaw.core.text_utils import parse_llm_response
from styleclaw.orchestrator.actions import ACTION_REGISTRY, PHASE_ACTIONS
from styleclaw.providers.llm.base import LLMProvider
from styleclaw.storage import project_store

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "plan.md"

# Phases from which the planner is allowed to see the next phase's actions in
# the same plan. The transition into the next phase for each of these is
# either automatic (INIT → MODEL_SELECT via `analyze`) or already lives behind
# its own user-confirmation gate inside the current phase's action list
# (`approve` in STYLE_REFINE / BATCH_T2I, `back-to-t2i` / `approve` in
# BATCH_I2I). MODEL_SELECT is deliberately excluded: its `select-model` gate
# must not be bypassed by an LLM-emitted plan.
CROSS_PHASE_PLANNABLE_FROM = frozenset({
    Phase.INIT, Phase.STYLE_REFINE, Phase.BATCH_T2I, Phase.BATCH_I2I,
})

# Actions that advance the project across a phase boundary, change pass
# scope, or otherwise require explicit user intent. They are only made
# plannable when the project is already in the phase that owns them — never
# via cross-phase extension. (`retest-models` opens a new model-select pass
# and should never be inserted just because the user said "analyze".)
GATED_CROSS_PHASE_ACTIONS: frozenset[str] = frozenset({"select-model", "approve", "retest-models"})


def _build_actions_text(actions: list[str]) -> str:
    return "\n".join(f"- `{a}`" for a in actions)


def _unknown_actions(plan: ActionPlan, available: list[str]) -> list[str]:
    """Return step names that either don't exist in ACTION_REGISTRY or aren't
    allowed in the current phase. Preserves duplicates to give the LLM exact
    feedback."""
    allowed = set(available)
    return [
        s.name for s in plan.steps
        if s.name not in ACTION_REGISTRY or s.name not in allowed
    ]


async def plan(llm: LLMProvider, project: str, intent: str) -> ActionPlan:
    state = project_store.load_state(project)
    config = project_store.load_config(project)

    available = PHASE_ACTIONS.get(state.phase, [])

    if state.phase in CROSS_PHASE_PLANNABLE_FROM:
        from styleclaw.core.state_machine import TRANSITIONS
        next_phases_actions: list[str] = []
        for next_phase in TRANSITIONS.get(state.phase, []):
            next_phases_actions.extend(PHASE_ACTIONS.get(next_phase, []))
        extended = [a for a in next_phases_actions if a not in GATED_CROSS_PHASE_ACTIONS]
        available = list(dict.fromkeys(available + extended))

    template = Template(PROMPT_PATH.read_text(encoding="utf-8"))
    system_prompt = template.safe_substitute(
        project_name=project,
        phase=state.phase.value,
        current_round=state.current_round,
        current_batch=state.current_batch,
        selected_models=", ".join(state.selected_models) or "(none)",
        ip_info=config.ip_info or "(none)",
        available_actions=_build_actions_text(available),
        intent=intent,
    )

    messages: list[dict] = [{"role": "user", "content": intent}]
    raw = await llm.invoke(
        system=system_prompt,
        messages=messages,
        max_tokens=2048,
        temperature=0.3,
    )
    first_plan = parse_llm_response(raw, ActionPlan, "action plan")

    bad = _unknown_actions(first_plan, available)
    if not bad:
        return first_plan

    logger.warning(
        "Planner produced unknown/disallowed actions %s for phase %s; retrying once.",
        bad, state.phase.value,
    )
    retry_messages = messages + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                f"The plan contains action names that are not available in "
                f"phase {state.phase.value}: {bad}. "
                f"Choose only from this exact list: {available}. "
                f"Return a corrected ActionPlan JSON."
            ),
        },
    ]
    retry_raw = await llm.invoke(
        system=system_prompt,
        messages=retry_messages,
        max_tokens=2048,
        temperature=0.3,
    )
    retried = parse_llm_response(retry_raw, ActionPlan, "action plan (retry)")
    still_bad = _unknown_actions(retried, available)
    if still_bad:
        raise ValueError(
            f"Planner still produced unknown/disallowed actions after retry: {still_bad}. "
            f"Allowed in phase {state.phase.value}: {available}."
        )
    return retried
