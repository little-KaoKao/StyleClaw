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
GATED_CROSS_PHASE_ACTIONS: frozenset[str] = frozenset({"select-model", "approve", "retest-models", "add-refs"})


def _build_actions_text(actions: list[str]) -> str:
    return "\n".join(f"- `{a}`" for a in actions)


def _self_check_phase_actions() -> None:
    """Module-load-time sanity check on PHASE_ACTIONS / ACTION_REGISTRY /
    GATED_CROSS_PHASE_ACTIONS:

    1. Every action named in PHASE_ACTIONS exists in ACTION_REGISTRY.
    2. Every gated action exists in ACTION_REGISTRY.
    3. CROSS_PHASE_PLANNABLE_FROM only names real Phase values.

    Run once at import. Misconfiguration here is a developer error (typo on
    a new action), not a user-facing one, so we raise immediately."""
    for phase, actions in PHASE_ACTIONS.items():
        missing = [a for a in actions if a not in ACTION_REGISTRY]
        if missing:
            raise RuntimeError(
                f"PHASE_ACTIONS[{phase.value}] references unknown actions: {missing}"
            )
    gated_missing = [a for a in GATED_CROSS_PHASE_ACTIONS if a not in ACTION_REGISTRY]
    if gated_missing:
        raise RuntimeError(
            f"GATED_CROSS_PHASE_ACTIONS references unknown actions: {gated_missing}"
        )


_self_check_phase_actions()


def _unknown_actions(plan: ActionPlan, available: list[str]) -> list[str]:
    """Return step names that either don't exist in ACTION_REGISTRY or aren't
    allowed in the current phase. Preserves duplicates to give the LLM exact
    feedback."""
    allowed = set(available)
    return [
        s.name for s in plan.steps
        if s.name not in ACTION_REGISTRY or s.name not in allowed
    ]


def _sanitize_for_tag(text: str) -> str:
    """Neutralize closing-tag markers so user input can't escape the
    ``<user_intent>``/``<user_ip_info>`` containers in the system prompt.

    We only need to defang the close tags — the planner's system prompt
    explicitly tells the model to ignore instructions inside these tags, so a
    raw ``<`` is fine as long as the matching close tag is broken.
    """
    return (
        text.replace("</user_intent>", "&lt;/user_intent&gt;")
            .replace("</user_ip_info>", "&lt;/user_ip_info&gt;")
    )


async def plan(llm: LLMProvider, project: str, intent: str) -> ActionPlan:
    try:
        state = project_store.load_state(project)
        config = project_store.load_config(project)
    except FileNotFoundError:
        # No project on disk yet — only `init` is plannable, and the
        # confirmation callback will collect ref_dir / ip_info from the user.
        return await _plan_init_only(llm, project, intent)

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
        ip_info=_sanitize_for_tag(config.ip_info) if config.ip_info else "(none)",
        available_actions=_build_actions_text(available),
        intent=_sanitize_for_tag(intent),
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


async def _plan_init_only(llm: LLMProvider, project: str, intent: str) -> ActionPlan:
    """Build a plan for a project that doesn't exist yet.

    The plan is always a single ``init`` step; ref_dir and ip_info come from
    the confirmation callback in the CLI, not from the LLM. We still pass the
    intent through ``args`` so the confirmation prompt can show it as a hint.
    """
    from styleclaw.core.models import ActionPlan, Action

    return ActionPlan(
        summary=f"创建新项目 '{project}'",
        steps=[
            Action(
                name="init",
                description=f"根据用户意图创建项目 '{project}'：{intent}",
                args={"ref_dir": "", "ip_info": "", "description": "", "force": False},
            ),
        ],
        loop=None,
        stop_summary="项目创建完成，phase 进入 INIT，下一步可以让我分析风格。",
    )
