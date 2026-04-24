from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template

from styleclaw.core.models import ActionPlan
from styleclaw.core.text_utils import clean_json
from styleclaw.orchestrator.actions import PHASE_ACTIONS
from styleclaw.providers.llm.base import LLMProvider
from styleclaw.storage import project_store

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "plan.md"


def _build_actions_text(actions: list[str]) -> str:
    return "\n".join(f"- `{a}`" for a in actions)


async def plan(llm: LLMProvider, project: str, intent: str) -> ActionPlan:
    state = project_store.load_state(project)
    config = project_store.load_config(project)

    available = PHASE_ACTIONS.get(state.phase, [])

    if state.phase.value in ("INIT", "MODEL_SELECT"):
        next_phases_actions: list[str] = []
        from styleclaw.core.state_machine import TRANSITIONS
        for next_phase in TRANSITIONS.get(state.phase, []):
            next_phases_actions.extend(PHASE_ACTIONS.get(next_phase, []))
        available = list(dict.fromkeys(available + next_phases_actions))

    template = Template(PROMPT_PATH.read_text())
    system_prompt = template.substitute(
        project_name=project,
        phase=state.phase.value,
        current_round=state.current_round,
        current_batch=state.current_batch,
        selected_models=", ".join(state.selected_models) or "(none)",
        ip_info=config.ip_info or "(none)",
        available_actions=_build_actions_text(available),
        intent=intent,
    )

    raw = await llm.invoke(
        system=system_prompt,
        messages=[{"role": "user", "content": intent}],
        max_tokens=2048,
        temperature=0.3,
    )

    cleaned = clean_json(raw)
    data = json.loads(cleaned)
    return ActionPlan.model_validate(data)
