from __future__ import annotations

import json
import logging
from pathlib import Path

from styleclaw.core.image_utils import build_image_block
from styleclaw.core.models import PromptConfig, RoundEvaluation
from styleclaw.core.text_utils import clean_json, parse_llm_response, sanitize_braces
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "refine.md"
)

MAX_HISTORY_ROUNDS = 3


async def refine_prompt(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    current_trigger: str,
    round_num: int,
    ip_info: str,
    evaluations: list[RoundEvaluation],
    human_direction: str = "",
) -> PromptConfig:
    history_text = _build_history_text(evaluations)

    system_prompt = (
        PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("{trigger_phrase}", current_trigger)
        .replace("{round_num}", str(round_num))
        .replace("{ip_info}", sanitize_braces(ip_info))
        .replace("{history_scores}", history_text)
        .replace("{human_direction}", sanitize_braces(human_direction) if human_direction else "(none)")
    )

    content: list[dict] = [build_image_block(p) for p in ref_image_paths]
    content.append({
        "type": "text",
        "text": "Refine the trigger phrase based on the evaluation history and reference images.",
    })

    messages = [{"role": "user", "content": content}]
    raw = await llm.invoke(system=system_prompt, messages=messages)

    cleaned = clean_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON for prompt refinement: {exc}") from exc

    data["round"] = round_num
    data.setdefault("derived_from", f"round-{round_num - 1:03d}" if round_num > 1 else "initial-analysis")
    config = parse_llm_response(
        json.dumps(data), PromptConfig, "prompt refinement",
    )
    logger.info("Refined trigger (round %d): %s", round_num, config.trigger_phrase[:80])
    return config


def _build_history_text(evaluations: list[RoundEvaluation]) -> str:
    if not evaluations:
        return "(no previous evaluations)"

    recent = evaluations[-MAX_HISTORY_ROUNDS:]
    lines: list[str] = []
    for ev in recent:
        lines.append(f"### Round {ev.round}")
        for score in ev.evaluations:
            s = score.scores
            lines.append(
                f"- {score.model}: color={s.color_palette} line={s.line_style} "
                f"light={s.lighting} texture={s.texture} mood={s.overall_mood} "
                f"total={score.total:.1f}"
            )
        if ev.next_direction:
            lines.append(f"  Direction: {ev.next_direction}")
    return "\n".join(lines)
