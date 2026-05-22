from __future__ import annotations

import json
import logging
from pathlib import Path

from styleclaw.agents.analyze_style import analyze_style
from styleclaw.core.image_utils import build_image_blocks_async
from styleclaw.core.models import PanelResult, StyleAnalysis
from styleclaw.core.panel import run_panel
from styleclaw.core.text_utils import clean_json, sanitize_braces
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

SCORE_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "providers" / "llm" / "prompts" / "score_analyze_proposal.md"
)


async def analyze_style_with_panel(
    llms: list[LLMProvider],
    labels: list[str],
    ref_image_paths: list[Path],
    ip_info: str,
) -> tuple[StyleAnalysis, PanelResult]:
    """Run three propose + cross-score rounds and return the winner's analysis.

    Returns (winner_analysis, panel_result). When the panel degrades to
    no winner, the function raises RuntimeError — callers should surface this
    as a StepResult(ok=False).
    """

    async def propose(llm: LLMProvider) -> dict:
        analysis = await analyze_style(llm, ref_image_paths, ip_info)
        return analysis.model_dump()

    score_template = SCORE_PROMPT_PATH.read_text(encoding="utf-8")

    # Build the image blocks once: every (evaluator, proposal) pair sees the
    # same refs, so encoding in the closure would re-do the work N times in
    # parallel inside the panel TaskGroup.
    blocks = await build_image_blocks_async(list(ref_image_paths))

    async def score(evaluator: LLMProvider, payload: dict) -> tuple[float, str]:
        rendered = (
            score_template
            .replace("{ip_info}", sanitize_braces(ip_info))
            .replace(
                "{candidate_analysis}",
                sanitize_braces(json.dumps(payload, ensure_ascii=False, indent=2)),
            )
        )
        messages = [{
            "role": "user",
            "content": [
                *blocks,
                {"type": "text", "text": "Grade the candidate analysis against the references."},
            ],
        }]
        raw = await evaluator.invoke(system=rendered, messages=messages)
        data = json.loads(clean_json(raw))
        return float(data["score"]), str(data.get("rationale", ""))

    panel = await run_panel(llms, labels, propose, score)

    if not panel.winner_model_id:
        raise RuntimeError(
            "Analyze panel produced no winner: "
            + ("; ".join(panel.error_log) or "all proposals/scores failed")
        )

    winner = next(p for p in panel.proposals if p.model_id == panel.winner_model_id)
    analysis = StyleAnalysis.model_validate(winner.payload)
    logger.info(
        "Panel style analysis complete (winner=%s). Trigger: %s",
        panel.winner_model_id, analysis.trigger_phrase[:80],
    )
    return analysis, panel
