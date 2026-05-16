from __future__ import annotations

import json
import logging
from pathlib import Path

from styleclaw.agents.select_model import evaluate_models
from styleclaw.core.image_utils import build_image_blocks_async
from styleclaw.core.models import ModelEvaluation, PanelResult
from styleclaw.core.panel import run_panel
from styleclaw.core.text_utils import clean_json, sanitize_braces
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

SCORE_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "providers" / "llm" / "prompts" / "score_model_select_proposal.md"
)


async def select_models_with_panel(
    llms: list[LLMProvider],
    labels: list[str],
    ref_image_paths: list[Path],
    model_images: dict[str, list[Path]],
) -> tuple[ModelEvaluation, PanelResult]:
    """Run three propose + cross-score rounds and return the winner's evaluation."""

    async def propose(llm: LLMProvider) -> dict:
        evaluation = await evaluate_models(llm, ref_image_paths, model_images)
        return evaluation.model_dump()

    score_template = SCORE_PROMPT_PATH.read_text(encoding="utf-8")

    async def score(evaluator: LLMProvider, payload: dict) -> tuple[float, str]:
        rendered = score_template.replace(
            "{candidate_evaluation}",
            sanitize_braces(json.dumps(payload, ensure_ascii=False, indent=2)),
        )
        # Show the same images the proposer saw so the scorer can verify.
        all_paths: list[Path] = list(ref_image_paths)
        for imgs in model_images.values():
            all_paths.extend(imgs)
        blocks = await build_image_blocks_async(all_paths)
        messages = [{
            "role": "user",
            "content": [
                *blocks,
                {"type": "text", "text": "Grade the candidate evaluation against the references and generations."},
            ],
        }]
        raw = await evaluator.invoke(system=rendered, messages=messages, max_tokens=4096)
        data = json.loads(clean_json(raw))
        return float(data["score"]), str(data.get("rationale", ""))

    panel = await run_panel(llms, labels, propose, score)

    if not panel.winner_model_id:
        raise RuntimeError(
            "Model-select panel produced no winner: "
            + ("; ".join(panel.error_log) or "all proposals/scores failed")
        )

    winner = next(p for p in panel.proposals if p.model_id == panel.winner_model_id)
    evaluation = ModelEvaluation.model_validate(winner.payload)
    logger.info(
        "Panel model-select complete (winner=%s, recommendation=%s)",
        panel.winner_model_id, evaluation.recommendation,
    )
    return evaluation, panel
