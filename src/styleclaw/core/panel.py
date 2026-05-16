"""Domain-agnostic three-model panel coordinator.

Runs three phases:
  1. Propose  — each LLM in ``llms`` calls the caller-supplied ``propose``
     coroutine concurrently; failures are captured, not raised.
  2. Cross-evaluate — every (evaluator, proposal) pair where evaluator != author
     calls the caller-supplied ``score`` coroutine concurrently; failures are
     also captured via ``_safe_score`` so the TaskGroup never sees an exception.
  3. Aggregate — compute per-proposal mean scores, apply a stable tie-break
     (earliest position in ``llms``), and return a ``PanelResult``.

The ``degraded`` flag is set whenever any error was logged OR fewer proposals
arrived than the number of ``llms`` supplied.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from styleclaw.core.models import PanelProposal, PanelResult, PanelScore
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

ProposeFn = Callable[[LLMProvider], Awaitable[dict[str, Any]]]
ScoreFn = Callable[[LLMProvider, dict[str, Any]], Awaitable[tuple[float, str]]]


async def run_panel(
    llms: list[LLMProvider],
    labels: list[str],
    propose: ProposeFn,
    score: ScoreFn,
    min_proposals: int = 2,
    min_scores_per_proposal: int = 1,
) -> PanelResult:
    """Run a three-model panel: propose → cross-score → aggregate.

    ``llms`` and ``labels`` are positional pairs. ``propose`` is called once per
    provider; ``score`` is called for every (evaluator, proposal) pair where
    evaluator != proposal author. Only LLMs whose proposals succeeded participate
    as evaluators in Phase 2; failed proposers do not score others. Failures in
    either phase are captured in ``error_log`` rather than raised.

    Args:
        llms: List of LLMProvider instances; one per panel participant.
        labels: Human-readable labels paired by index with ``llms``.
        propose: Async callable ``(llm) -> dict`` — returns the proposal payload.
        score: Async callable ``(evaluator, payload) -> (float, str)`` — returns
            (score_value, rationale).
        min_proposals: Minimum number of successful proposals to proceed; if
            fewer arrive, returns a degraded ``PanelResult`` with no winner.
        min_scores_per_proposal: Minimum number of valid cross-scores a proposal
            must receive to be included in the averages.

    Returns:
        ``PanelResult`` with proposals, scores, averages, winner, and error log.

    Raises:
        ValueError: If ``llms`` and ``labels`` have different lengths.
    """
    if len(llms) != len(labels):
        raise ValueError(
            f"llms and labels must be the same length (got {len(llms)} vs {len(labels)})"
        )

    error_log: list[str] = []

    # Phase 1: propose — gather with return_exceptions so one failure doesn't
    # kill the other proposals.
    propose_results = await asyncio.gather(
        *(propose(llm) for llm in llms),
        return_exceptions=True,
    )
    proposals: list[PanelProposal] = []
    for llm, label, outcome in zip(llms, labels, propose_results):
        mid = _model_id_of(llm)
        if isinstance(outcome, BaseException):
            error_log.append(f"propose[{mid}]: {type(outcome).__name__}: {outcome}")
            continue
        if not isinstance(outcome, dict):
            error_log.append(f"propose[{mid}]: expected dict, got {type(outcome).__name__}")
            continue
        proposals.append(PanelProposal(model_id=mid, label=label, payload=outcome))

    if len(proposals) < min_proposals:
        return PanelResult(
            proposals=proposals,
            scores=[],
            winner_model_id="",
            averages={},
            degraded=True,
            error_log=error_log,
        )

    # Phase 2: cross-evaluation (no self-scoring).
    # Only LLMs that successfully produced a proposal participate as evaluators.
    # _safe_score wraps the score call and returns exceptions instead of
    # raising them — that keeps all sibling tasks alive inside the TaskGroup.
    proposal_ids = {p.model_id for p in proposals}
    evaluator_llms = [llm for llm in llms if _model_id_of(llm) in proposal_ids]

    score_tasks: list[tuple[str, str, asyncio.Task]] = []
    async with asyncio.TaskGroup() as tg:
        for evaluator in evaluator_llms:
            ev_id = _model_id_of(evaluator)
            for proposal in proposals:
                if proposal.model_id == ev_id:
                    continue
                fut = tg.create_task(_safe_score(score, evaluator, proposal))
                score_tasks.append((ev_id, proposal.model_id, fut))

    scores: list[PanelScore] = []
    for ev_id, tgt_id, fut in score_tasks:
        outcome = fut.result()
        if isinstance(outcome, BaseException):
            error_log.append(f"score[{ev_id}->{tgt_id}]: {type(outcome).__name__}: {outcome}")
            continue
        score_val, rationale = outcome
        scores.append(PanelScore(
            evaluator_model_id=ev_id,
            target_model_id=tgt_id,
            score=float(score_val),
            rationale=rationale,
        ))

    # Phase 3: aggregate per-proposal mean scores.
    averages: dict[str, float] = {}
    for proposal in proposals:
        received = [s.score for s in scores if s.target_model_id == proposal.model_id]
        if len(received) < min_scores_per_proposal:
            error_log.append(
                f"proposal {proposal.model_id}: insufficient scores "
                f"({len(received)} < {min_scores_per_proposal})"
            )
            continue
        averages[proposal.model_id] = sum(received) / len(received)

    if not averages:
        return PanelResult(
            proposals=proposals,
            scores=scores,
            winner_model_id="",
            averages={},
            degraded=True,
            error_log=error_log,
        )

    # Stable tie-break: highest average wins; ties go to the participant that
    # appears earliest in the ``llms`` list.
    ordering = {_model_id_of(llm): idx for idx, llm in enumerate(llms)}
    winner = min(
        averages.keys(),
        key=lambda mid: (-averages[mid], ordering.get(mid, 1_000_000)),
    )

    degraded = bool(error_log) or len(proposals) < len(llms)
    return PanelResult(
        proposals=proposals,
        scores=scores,
        winner_model_id=winner,
        averages=averages,
        degraded=degraded,
        error_log=error_log,
    )


async def _safe_score(
    score: ScoreFn,
    evaluator: LLMProvider,
    proposal: PanelProposal,
) -> tuple[float, str] | BaseException:
    """Wrap a score call so exceptions are returned rather than raised.

    Returning the exception instead of raising it means the enclosing
    ``asyncio.TaskGroup`` never sees an exception, so all sibling scoring
    tasks continue to completion regardless of individual failures.
    """
    try:
        return await score(evaluator, proposal.payload)
    except BaseException as exc:  # noqa: BLE001  — intentionally broad; we record and continue
        return exc


def _model_id_of(llm: LLMProvider) -> str:
    """Best-effort identifier for an LLMProvider.

    Real providers expose ``_model_id`` (see ``OpenAICompatProvider``); test
    doubles set the same attribute to keep things uniform.
    """
    return getattr(llm, "_model_id", repr(llm))
