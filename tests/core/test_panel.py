from __future__ import annotations

import pytest

from styleclaw.core.panel import run_panel
from styleclaw.core.models import PanelResult


def _llm(model_id: str):
    """Lightweight stand-in for an LLMProvider: only needs an identifier.

    `run_panel` doesn't call any LLMProvider methods directly — it passes
    the provider into the caller-supplied propose/score functions. So a bare
    object with the right attribute is enough here.
    """
    obj = object.__new__(type("_LLM", (), {}))
    obj._model_id = model_id
    return obj


@pytest.mark.asyncio
class TestRunPanel:
    async def test_full_success_picks_highest_average(self):
        llms = [_llm("a"), _llm("b"), _llm("c")]
        labels = ["A", "B", "C"]

        async def propose(llm):
            return {"trigger": llm._model_id}

        # Hand-crafted scoring: a's proposal averages 8.5, b's 7.75, c's 6.75.
        score_table = {
            ("a", "b"): 7.0, ("a", "c"): 6.0,
            ("b", "a"): 8.0, ("b", "c"): 7.5,
            ("c", "a"): 9.0, ("c", "b"): 8.5,
        }

        async def score(evaluator, payload):
            target = payload["trigger"]
            return score_table[(evaluator._model_id, target)], f"r-{evaluator._model_id}-{target}"

        result: PanelResult = await run_panel(llms, labels, propose, score)

        assert result.winner_model_id == "a"
        assert len(result.proposals) == 3
        assert len(result.scores) == 6
        assert result.averages == pytest.approx({"a": 8.5, "b": 7.75, "c": 6.75})
        assert result.degraded is False
        assert result.error_log == []

    async def test_one_proposal_raises_continues_with_survivors(self):
        llms = [_llm("a"), _llm("b"), _llm("c")]
        labels = ["A", "B", "C"]

        async def propose(llm):
            if llm._model_id == "b":
                raise RuntimeError("boom-b")
            return {"trigger": llm._model_id}

        async def score(evaluator, payload):
            return 7.0, "ok"

        result = await run_panel(llms, labels, propose, score)

        assert len(result.proposals) == 2
        # Only 2 surviving proposals; the surviving evaluator (a, c) scores 1 other each, so 2 scoring calls total.
        assert len(result.scores) == 2
        assert any("boom-b" in m for m in result.error_log)
        assert result.degraded is True
        assert result.winner_model_id in {"a", "c"}

    async def test_below_min_proposals_returns_degraded_no_winner(self):
        llms = [_llm("a"), _llm("b"), _llm("c")]
        labels = ["A", "B", "C"]

        async def propose(llm):
            if llm._model_id != "a":
                raise RuntimeError(f"down-{llm._model_id}")
            return {"trigger": "a"}

        async def score(evaluator, payload):
            return 9.0, ""

        result = await run_panel(llms, labels, propose, score)
        assert result.winner_model_id == ""
        assert result.degraded is True
        assert result.averages == {}
        assert len(result.proposals) == 1

    async def test_score_exception_logged_proposal_still_aggregated(self):
        llms = [_llm("a"), _llm("b"), _llm("c")]
        labels = ["A", "B", "C"]

        async def propose(llm):
            return {"trigger": llm._model_id}

        async def score(evaluator, payload):
            if evaluator._model_id == "c" and payload["trigger"] == "a":
                raise RuntimeError("score-fail")
            return 7.0, ""

        result = await run_panel(llms, labels, propose, score)
        # 6 scheduled - 1 failure = 5 scores; a's proposal received 1 valid score (from b only).
        assert len(result.scores) == 5
        assert "a" in result.averages
        assert any("score-fail" in m for m in result.error_log)
        assert result.degraded is True

    async def test_all_scores_for_one_proposal_missing_drops_it(self):
        llms = [_llm("a"), _llm("b"), _llm("c")]
        labels = ["A", "B", "C"]

        async def propose(llm):
            return {"trigger": llm._model_id}

        async def score(evaluator, payload):
            if payload["trigger"] == "a":
                raise RuntimeError("no-score-for-a")
            return 7.0, ""

        result = await run_panel(llms, labels, propose, score)
        assert "a" not in result.averages
        assert result.winner_model_id != "a"
        assert any("a" in m and "insufficient" in m for m in result.error_log)
        assert result.degraded is True

    async def test_tie_break_uses_position_in_llms(self):
        llms = [_llm("a"), _llm("b"), _llm("c")]
        labels = ["A", "B", "C"]

        async def propose(llm):
            return {"trigger": llm._model_id}

        async def score(evaluator, payload):
            # Every proposal averages exactly 7.0.
            return 7.0, ""

        result = await run_panel(llms, labels, propose, score)
        assert result.averages["a"] == result.averages["b"] == result.averages["c"] == 7.0
        # Tie-break: earliest position wins.
        assert result.winner_model_id == "a"
        assert result.degraded is False

    async def test_llms_labels_length_mismatch_raises(self):
        llms = [_llm("a"), _llm("b")]
        labels = ["A", "B", "C"]  # mismatch

        async def propose(llm):
            return {}

        async def score(evaluator, payload):
            return 7.0, ""

        with pytest.raises(ValueError, match="same length"):
            await run_panel(llms, labels, propose, score)
