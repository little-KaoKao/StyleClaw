from __future__ import annotations

import json
import logging
from pathlib import Path

from styleclaw.agents.refine_prompt import refine_prompt
from styleclaw.core.image_utils import build_image_blocks_async
from styleclaw.core.models import PanelResult, PromptConfig, RoundEvaluation
from styleclaw.core.panel import run_panel
from styleclaw.core.text_utils import clean_json, sanitize_braces
from styleclaw.providers.llm.base import LLMProvider
from styleclaw.storage.project_store import round_label

logger = logging.getLogger(__name__)

SCORE_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "providers" / "llm" / "prompts" / "score_refine_proposal.md"
)


def _build_history_text(evaluations: list[RoundEvaluation]) -> str:
    # Mirror refine_prompt._build_history_text but keep it local to avoid a
    # private cross-module import. Cap at 3 most recent rounds.
    if not evaluations:
        return "(no previous evaluations)"
    recent = evaluations[-3:]
    lines: list[str] = []
    for ev in recent:
        lines.append(f"### Round {ev.round}")
        for s in ev.evaluations:
            d = s.scores
            lines.append(
                f"- {s.model}: style={d.visual_style} color={d.color_science} "
                f"light={d.lighting_quality} texture={d.material_texture} "
                f"post={d.post_processing} space={d.spatial_perspective} "
                f"motion={d.dynamic_state} total={s.total:.1f}"
            )
        if ev.next_direction:
            lines.append(f"  Direction: {ev.next_direction}")
    return "\n".join(lines)


async def refine_with_panel(
    llms: list[LLMProvider],
    labels: list[str],
    ref_image_paths: list[Path],
    current_trigger: str,
    round_num: int,
    ip_info: str,
    evaluations: list[RoundEvaluation],
    human_direction: str = "",
) -> tuple[PromptConfig, PanelResult]:
    """Run three propose + cross-score rounds and return the winner's prompt.

    Returns (winner_prompt_config, panel_result). When the panel degrades to
    no winner, the function raises RuntimeError — callers should surface this
    as a StepResult(ok=False).
    """

    async def propose(llm: LLMProvider) -> dict:
        config = await refine_prompt(
            llm, ref_image_paths, current_trigger, round_num,
            ip_info, evaluations, human_direction,
        )
        return config.model_dump()

    score_template = SCORE_PROMPT_PATH.read_text(encoding="utf-8")
    history_text = _build_history_text(evaluations)

    # Build the image blocks once: every (evaluator, proposal) pair sees the
    # same refs, so encoding in the closure would re-do the work N times in
    # parallel inside the panel TaskGroup.
    blocks = await build_image_blocks_async(list(ref_image_paths))

    async def score(evaluator: LLMProvider, payload: dict) -> tuple[float, str]:
        rendered = (
            score_template
            .replace("{ip_info}", sanitize_braces(ip_info))
            .replace("{round_num}", str(round_num))
            .replace("{history_scores}", history_text)
            .replace("{candidate_trigger}", sanitize_braces(payload.get("trigger_phrase", "")))
            .replace("{candidate_note}", sanitize_braces(payload.get("adjustment_note", "")))
        )
        messages = [{
            "role": "user",
            "content": [
                *blocks,
                {"type": "text", "text": "Grade the candidate trigger phrase against the references."},
            ],
        }]
        raw = await evaluator.invoke(system=rendered, messages=messages)
        data = json.loads(clean_json(raw))
        return float(data["score"]), str(data.get("rationale", ""))

    panel = await run_panel(llms, labels, propose, score)

    if not panel.winner_model_id:
        raise RuntimeError(
            "Refine panel produced no winner: "
            + ("; ".join(panel.error_log) or "all proposals/scores failed")
        )

    winner = next(p for p in panel.proposals if p.model_id == panel.winner_model_id)
    prompt_config = PromptConfig.model_validate({
        **winner.payload,
        "round": round_num,
        "derived_from": (
            round_label(round_num - 1) if round_num > 1 else "initial-analysis"
        ),
    })
    logger.info(
        "Panel-refined trigger (round %d, winner=%s): %s",
        round_num, panel.winner_model_id, prompt_config.trigger_phrase[:80],
    )
    return prompt_config, panel
