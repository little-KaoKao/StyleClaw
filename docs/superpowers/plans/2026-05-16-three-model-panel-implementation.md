# Three-Model Review Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a three-model panel (propose → cross-score → pick winner) into `STYLE_REFINE.refine` and `MODEL_SELECT.evaluate`, gated by independent env switches that default OFF.

**Architecture:** A domain-agnostic `core/panel.py::run_panel` coordinator runs Phase 1 (3 concurrent proposals) → Phase 2 (≤6 concurrent cross-evaluations, no self-scoring) → Phase 3 (mean aggregation + stable tie-break). Two domain adapters (`agents/refine_panel.py`, `agents/select_model_panel.py`) supply `propose` / `score` callables that wrap the existing single-model agents and read new scoring prompts. A factory (`providers/llm/panel_factory.py`) instantiates three `OpenAICompatProvider` instances differing only by `model_id`. The winner's payload lands in the existing artifact files (`prompt.json` / `evaluation.json`) — downstream code is untouched — and the full `PanelResult` is persisted to a sibling `panel.json`.

**Tech Stack:** Python 3.11+, Pydantic v2 (`_FrozenModel`), `asyncio.gather(return_exceptions=True)`, httpx (existing provider transport), Jinja2 (existing report templates), pytest with `AsyncMock` for LLM stubs.

**Spec reference:** [docs/superpowers/specs/2026-05-14-three-model-panel-design.md](../specs/2026-05-14-three-model-panel-design.md)

---

## File Structure

### Files to create

| Path | Responsibility |
|---|---|
| `src/styleclaw/core/panel.py` | Domain-agnostic `run_panel()` coordinator + helpers |
| `src/styleclaw/agents/refine_panel.py` | Domain adapter: refine `propose` + `score` + writer |
| `src/styleclaw/agents/select_model_panel.py` | Domain adapter: model-select `propose` + `score` + writer |
| `src/styleclaw/providers/llm/panel_factory.py` | `build_panel_providers()` reads env, returns `[(provider, label), ...]` |
| `src/styleclaw/providers/llm/prompts/score_refine_proposal.md` | Scoring rubric for candidate trigger phrase |
| `src/styleclaw/providers/llm/prompts/score_model_select_proposal.md` | Scoring rubric for candidate model recommendation |
| `tests/core/test_panel.py` | Unit tests for `run_panel()` |
| `tests/agents/test_refine_panel.py` | Adapter tests for refine path |
| `tests/agents/test_select_model_panel.py` | Adapter tests for model-select path |
| `tests/providers/llm/test_panel_factory.py` | Factory + env validation tests |

### Files to modify

| Path | Change |
|---|---|
| `src/styleclaw/core/models.py` | Add `PanelProposal`, `PanelScore`, `PanelResult` |
| `src/styleclaw/core/config.py` | Add `PANEL_REFINE_ENABLED`, `PANEL_MODEL_SELECT_ENABLED`, `PANEL_MODELS`, `PANEL_LABELS` + startup validation |
| `src/styleclaw/storage/project_store.py` | Add `save_panel_result` / `load_panel_result` for refine round + model-select pass |
| `src/styleclaw/orchestrator/actions.py` | Branch in `do_refine` / `do_evaluate` on panel toggles |
| `src/styleclaw/reports/templates/style_refine.html` | Optional `{% if panel %}` review block |
| `src/styleclaw/reports/templates/model_select.html` | Optional `{% if panel %}` review block |
| `src/styleclaw/scripts/report.py` | Load `panel.json` (if present) and pass into report context |
| `tests/orchestrator/test_actions_do.py` | Add panel-on / panel-off coverage for `do_refine` and `do_evaluate` |
| `tests/core/test_config.py` | Add panel env validation tests |
| `CLAUDE.md` | Document the new env vars under "Environment Variables" |

---

## Task 1: Add Panel data models

**Files:**
- Modify: `src/styleclaw/core/models.py:1-9` (imports) and end of file (new classes)
- Test: `tests/core/test_models.py` (extend)

- [ ] **Step 1: Write failing tests for the three new models**

Append to `tests/core/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from styleclaw.core.models import PanelProposal, PanelResult, PanelScore


class TestPanelModels:
    def test_proposal_defaults(self):
        p = PanelProposal(model_id="m1", payload={"trigger_phrase": "foo"})
        assert p.label == ""
        assert p.thinking == ""
        assert p.payload == {"trigger_phrase": "foo"}

    def test_proposal_is_frozen(self):
        p = PanelProposal(model_id="m1", payload={})
        with pytest.raises(ValidationError):
            p.model_id = "m2"

    def test_score_required_fields(self):
        s = PanelScore(evaluator_model_id="e", target_model_id="t", score=8.5)
        assert s.rationale == ""

    def test_result_defaults(self):
        r = PanelResult(
            proposals=[],
            scores=[],
            winner_model_id="",
            averages={},
        )
        assert r.degraded is False
        assert r.error_log == []

    def test_result_holds_full_panel(self):
        proposals = [
            PanelProposal(model_id="a", payload={"x": 1}),
            PanelProposal(model_id="b", payload={"x": 2}),
            PanelProposal(model_id="c", payload={"x": 3}),
        ]
        scores = [
            PanelScore(evaluator_model_id="a", target_model_id="b", score=7.0),
            PanelScore(evaluator_model_id="a", target_model_id="c", score=6.0),
            PanelScore(evaluator_model_id="b", target_model_id="a", score=8.0),
            PanelScore(evaluator_model_id="b", target_model_id="c", score=7.5),
            PanelScore(evaluator_model_id="c", target_model_id="a", score=9.0),
            PanelScore(evaluator_model_id="c", target_model_id="b", score=8.5),
        ]
        r = PanelResult(
            proposals=proposals,
            scores=scores,
            winner_model_id="a",
            averages={"a": 8.5, "b": 7.75, "c": 6.75},
        )
        assert r.winner_model_id == "a"
        assert len(r.scores) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/core/test_models.py::TestPanelModels -v`
Expected: FAIL with `ImportError: cannot import name 'PanelProposal'`.

- [ ] **Step 3: Implement the models**

Append to `src/styleclaw/core/models.py` (after `ActionPlan`):

```python
class PanelProposal(_FrozenModel):
    """One participant's submitted artifact in a panel round."""
    model_id: str
    label: str = ""
    payload: dict[str, Any]
    thinking: str = ""


class PanelScore(_FrozenModel):
    """One evaluator's score for one proposal (never the evaluator's own)."""
    evaluator_model_id: str
    target_model_id: str
    score: float
    rationale: str = ""


class PanelResult(_FrozenModel):
    """Aggregate outcome of a three-model panel round."""
    proposals: list[PanelProposal] = Field(default_factory=list)
    scores: list[PanelScore] = Field(default_factory=list)
    winner_model_id: str = ""
    averages: dict[str, float] = Field(default_factory=dict)
    degraded: bool = False
    error_log: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/core/test_models.py::TestPanelModels -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/models.py tests/core/test_models.py
git commit -m "feat(models): add PanelProposal/PanelScore/PanelResult"
```

---

## Task 2: Wire env vars + startup validation

**Files:**
- Modify: `src/styleclaw/core/config.py` (add panel vars + validation hook)
- Test: `tests/core/test_config.py`

- [ ] **Step 1: Write failing config tests**

Append to `tests/core/test_config.py`:

```python
class TestPanelConfig:
    """STYLECLAW_PANEL_* validation.

    Both toggles default off. Either toggle on requires exactly 3 model ids;
    labels (if given) must match length. Both off → panel env is fully ignored.
    """

    def _reload(self):
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        return config_mod

    def test_both_off_ignores_panel_envs(self, monkeypatch):
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "only-one")
        monkeypatch.setenv("STYLECLAW_PANEL_LABELS", "A,B")  # mismatched len, ignored
        cfg = self._reload()
        assert cfg.PANEL_REFINE_ENABLED is False
        assert cfg.PANEL_MODEL_SELECT_ENABLED is False
        # Models list is parsed but not validated when both toggles off.
        # validate_panel_config() returns no errors.
        assert cfg.validate_panel_config() == []

    def test_refine_on_requires_three_models(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "a,b")
        monkeypatch.delenv("STYLECLAW_PANEL_LABELS", raising=False)
        cfg = self._reload()
        errs = cfg.validate_panel_config()
        assert any("STYLECLAW_PANEL_MODELS" in e and "exactly 3" in e for e in errs)

    def test_select_on_with_three_models_ok(self, monkeypatch):
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        monkeypatch.setenv("STYLECLAW_PANEL_MODEL_SELECT", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "a, b ,c")
        monkeypatch.delenv("STYLECLAW_PANEL_LABELS", raising=False)
        cfg = self._reload()
        assert cfg.PANEL_MODELS == ["a", "b", "c"]
        assert cfg.validate_panel_config() == []

    def test_labels_must_match_length(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "a,b,c")
        monkeypatch.setenv("STYLECLAW_PANEL_LABELS", "Opus,GPT")
        cfg = self._reload()
        errs = cfg.validate_panel_config()
        assert any("STYLECLAW_PANEL_LABELS" in e for e in errs)

    def test_labels_default_to_model_ids(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "a,b,c")
        monkeypatch.delenv("STYLECLAW_PANEL_LABELS", raising=False)
        cfg = self._reload()
        assert cfg.PANEL_LABELS == ["a", "b", "c"]

    def test_validate_env_calls_panel_validator(self, monkeypatch):
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "only-one")
        cfg = self._reload()
        errs = cfg.validate_env()
        assert any("STYLECLAW_PANEL_MODELS" in e for e in errs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/core/test_config.py::TestPanelConfig -v`
Expected: FAIL — `AttributeError` on `PANEL_REFINE_ENABLED` / `validate_panel_config`.

- [ ] **Step 3: Implement panel config**

Insert in `src/styleclaw/core/config.py` after the existing `_bool_env` function and before `MAX_AUTO_ROUNDS`:

```python
def _list_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]
```

Append after the `RH_CLIENT_*` lines and before `env_truthy`:

```python
# --- Three-model panel toggles (default OFF). When either is on,
# STYLECLAW_PANEL_MODELS must list exactly 3 model ids; labels (optional)
# must match length. validate_panel_config() reports problems instead of
# raising at import time so unit tests can still load config_mod cleanly.
PANEL_REFINE_ENABLED: bool = _bool_env("STYLECLAW_PANEL_REFINE", False)
PANEL_MODEL_SELECT_ENABLED: bool = _bool_env("STYLECLAW_PANEL_MODEL_SELECT", False)
PANEL_MODELS: list[str] = _list_env("STYLECLAW_PANEL_MODELS")
_PANEL_LABELS_RAW: list[str] = _list_env("STYLECLAW_PANEL_LABELS")
PANEL_LABELS: list[str] = _PANEL_LABELS_RAW or list(PANEL_MODELS)


def validate_panel_config() -> list[str]:
    """Return error strings if panel envs are inconsistent.

    No-op when both toggles are off.
    """
    errors: list[str] = []
    if not (PANEL_REFINE_ENABLED or PANEL_MODEL_SELECT_ENABLED):
        return errors
    if len(PANEL_MODELS) != 3:
        errors.append(
            "STYLECLAW_PANEL_MODELS must list exactly 3 comma-separated model "
            f"ids when STYLECLAW_PANEL_REFINE or STYLECLAW_PANEL_MODEL_SELECT "
            f"is set (got {len(PANEL_MODELS)}: {PANEL_MODELS!r})."
        )
    if _PANEL_LABELS_RAW and len(_PANEL_LABELS_RAW) != len(PANEL_MODELS):
        errors.append(
            "STYLECLAW_PANEL_LABELS length must match STYLECLAW_PANEL_MODELS "
            f"(got {len(_PANEL_LABELS_RAW)} labels for {len(PANEL_MODELS)} models)."
        )
    return errors
```

Modify `validate_env()` — append before the final `return errors`:

```python
    errors.extend(validate_panel_config())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/core/test_config.py::TestPanelConfig -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/config.py tests/core/test_config.py
git commit -m "feat(config): add STYLECLAW_PANEL_* env vars and validation"
```

---

## Task 3: Implement `core/panel.py::run_panel`

**Files:**
- Create: `src/styleclaw/core/panel.py`
- Test: `tests/core/test_panel.py`

- [ ] **Step 1: Write failing tests for `run_panel`**

Create `tests/core/test_panel.py`:

```python
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
```

Make sure `pytest-asyncio` is on. Check `pyproject.toml`; if `asyncio_mode = "auto"` is set, the `@pytest.mark.asyncio` marker is redundant but harmless. The existing `tests/agents/test_refine_prompt.py` is the reference.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/core/test_panel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'styleclaw.core.panel'`.

- [ ] **Step 3: Implement `core/panel.py`**

Create `src/styleclaw/core/panel.py`:

```python
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

    `llms` and `labels` are positional pairs. `propose` is called once per
    provider; `score` is called for every (evaluator, proposal) pair where
    evaluator != proposal author. Failures in either phase are captured in
    ``error_log`` rather than raised.
    """
    if len(llms) != len(labels):
        raise ValueError(
            f"llms and labels must be the same length (got {len(llms)} vs {len(labels)})"
        )

    error_log: list[str] = []

    # Phase 1: propose.
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
    score_tasks: list[tuple[str, str, asyncio.Future]] = []
    async with asyncio.TaskGroup() as tg:
        for evaluator in llms:
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

    # Phase 3: aggregate.
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

    # Stable tie-break: pick the entry that appears earliest in `llms`.
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
    score: ScoreFn, evaluator: LLMProvider, proposal: PanelProposal,
) -> tuple[float, str] | BaseException:
    try:
        return await score(evaluator, proposal.payload)
    except BaseException as exc:  # noqa: BLE001  — we record and continue
        return exc


def _model_id_of(llm: LLMProvider) -> str:
    """Best-effort identifier for an LLMProvider.

    Real providers expose ``_model_id`` (see ``OpenAICompatProvider``); test
    doubles set the same attribute to keep things uniform.
    """
    return getattr(llm, "_model_id", repr(llm))
```

Note about `TaskGroup`: if any wrapped coroutine raises, `asyncio.TaskGroup` re-raises an `ExceptionGroup` from the `async with` exit. We avoid that by wrapping in `_safe_score` so exceptions are *returned*, not *raised* — keeping all sibling scoring calls alive.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/core/test_panel.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/panel.py tests/core/test_panel.py
git commit -m "feat(core): add domain-agnostic three-model panel coordinator"
```

---

## Task 4: Add `panel_factory.build_panel_providers`

**Files:**
- Create: `src/styleclaw/providers/llm/panel_factory.py`
- Test: `tests/providers/llm/test_panel_factory.py`

- [ ] **Step 1: Write failing factory tests**

Create `tests/providers/llm/test_panel_factory.py`:

```python
from __future__ import annotations

import pytest


def _reload_config():
    import importlib
    import styleclaw.core.config as config_mod
    importlib.reload(config_mod)
    return config_mod


class TestBuildPanelProviders:
    def test_raises_when_neither_toggle_on(self, monkeypatch):
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        _reload_config()
        from styleclaw.providers.llm.panel_factory import build_panel_providers
        with pytest.raises(RuntimeError, match="no panel toggle is enabled"):
            build_panel_providers()

    def test_raises_when_validation_fails(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "a,b")  # only 2
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        _reload_config()
        from styleclaw.providers.llm.panel_factory import build_panel_providers
        with pytest.raises(ValueError, match="STYLECLAW_PANEL_MODELS"):
            build_panel_providers()

    def test_returns_three_providers_with_distinct_model_ids(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.setenv("STYLECLAW_PANEL_LABELS", "One,Two,Three")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        _reload_config()
        from styleclaw.providers.llm.panel_factory import build_panel_providers
        pairs = build_panel_providers()
        assert [label for _, label in pairs] == ["One", "Two", "Three"]
        assert [p._model_id for p, _ in pairs] == ["m1", "m2", "m3"]
        # All share the same base URL.
        assert all(p._base_url == "http://x" for p, _ in pairs)

    def test_labels_fall_back_to_model_ids(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.delenv("STYLECLAW_PANEL_LABELS", raising=False)
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        _reload_config()
        from styleclaw.providers.llm.panel_factory import build_panel_providers
        pairs = build_panel_providers()
        assert [label for _, label in pairs] == ["m1", "m2", "m3"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/providers/llm/test_panel_factory.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the factory**

Create `src/styleclaw/providers/llm/panel_factory.py`:

```python
from __future__ import annotations

import importlib

from styleclaw.providers.llm.base import LLMProvider
from styleclaw.providers.llm.openai_compat import OpenAICompatProvider


def build_panel_providers() -> list[tuple[LLMProvider, str]]:
    """Instantiate three OpenAI-compat providers, one per panel model id.

    Reloads ``styleclaw.core.config`` so callers that flip env vars at runtime
    (tests, repls) see updated values without managing module state by hand.
    All providers share the same base URL + API key; only ``model_id`` differs.
    Caller is responsible for closing the returned providers (httpx clients).
    """
    config_mod = importlib.reload(importlib.import_module("styleclaw.core.config"))

    if not (config_mod.PANEL_REFINE_ENABLED or config_mod.PANEL_MODEL_SELECT_ENABLED):
        raise RuntimeError(
            "build_panel_providers() called but no panel toggle is enabled "
            "(set STYLECLAW_PANEL_REFINE=1 or STYLECLAW_PANEL_MODEL_SELECT=1)."
        )

    errors = config_mod.validate_panel_config()
    if errors:
        raise ValueError("; ".join(errors))

    pairs: list[tuple[LLMProvider, str]] = []
    for model_id, label in zip(config_mod.PANEL_MODELS, config_mod.PANEL_LABELS):
        provider = OpenAICompatProvider(model_id=model_id)
        pairs.append((provider, label))
    return pairs


async def close_panel_providers(pairs: list[tuple[LLMProvider, str]]) -> None:
    """Best-effort close of httpx clients held by the providers."""
    for provider, _ in pairs:
        close = getattr(provider, "close", None)
        if close is not None:
            await close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/providers/llm/test_panel_factory.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/providers/llm/panel_factory.py tests/providers/llm/test_panel_factory.py
git commit -m "feat(providers): add build_panel_providers factory"
```

---

## Task 5: Add storage helpers for `panel.json`

**Files:**
- Modify: `src/styleclaw/storage/project_store.py` (add 4 helpers)
- Test: `tests/core/test_models.py` or a new file — add `tests/storage/test_panel_io.py`

- [ ] **Step 1: Write failing storage tests**

Create `tests/storage/test_panel_io.py`:

```python
from __future__ import annotations

import pytest

from styleclaw.core.models import (
    PanelProposal,
    PanelResult,
    PanelScore,
    ProjectConfig,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


def _make_project(name: str = "p") -> str:
    project_store.create_project(ProjectConfig(name=name))
    return name


def _sample_result() -> PanelResult:
    return PanelResult(
        proposals=[PanelProposal(model_id="a", payload={"x": 1})],
        scores=[PanelScore(evaluator_model_id="a", target_model_id="b", score=7.0)],
        winner_model_id="a",
        averages={"a": 7.0},
    )


class TestPanelStorage:
    def test_save_and_load_round_panel(self):
        name = _make_project()
        result = _sample_result()
        project_store.save_round_panel_result(name, round_num=1, result=result, pass_num=1)
        loaded = project_store.load_round_panel_result(name, round_num=1, pass_num=1)
        assert loaded == result

    def test_save_and_load_model_select_panel(self):
        name = _make_project()
        result = _sample_result()
        project_store.save_model_select_panel_result(name, result=result, pass_num=1)
        loaded = project_store.load_model_select_panel_result(name, pass_num=1)
        assert loaded == result

    def test_load_returns_none_when_missing(self):
        name = _make_project()
        assert project_store.load_round_panel_result(name, round_num=1, pass_num=1) is None
        assert project_store.load_model_select_panel_result(name, pass_num=1) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/storage/test_panel_io.py -v`
Expected: FAIL — `AttributeError: module 'styleclaw.storage.project_store' has no attribute 'save_round_panel_result'`.

- [ ] **Step 3: Implement storage helpers**

Append to `src/styleclaw/storage/project_store.py` (after the existing round/model-select helpers). First, extend the imports at the top to include `PanelResult`:

```python
from styleclaw.core.models import (
    BatchConfig,
    ModelEvaluation,
    PanelResult,
    ProjectConfig,
    ProjectState,
    PromptConfig,
    RoundEvaluation,
    StyleAnalysis,
    TaskRecord,
    UploadRecord,
)
```

Then add (place near `save_round_evaluation` for refine, and near `save_evaluation` for model-select):

```python
def save_round_panel_result(
    name: str, round_num: int, result: PanelResult, pass_num: int = 1,
) -> None:
    _save_model(result, round_dir(name, round_num, pass_num) / "panel.json")


def load_round_panel_result(
    name: str, round_num: int, pass_num: int = 1,
) -> PanelResult | None:
    path = round_dir(name, round_num, pass_num) / "panel.json"
    if not path.exists():
        return None
    return _load_model(PanelResult, path)


def save_model_select_panel_result(
    name: str, result: PanelResult, pass_num: int = 1,
) -> None:
    _save_model(result, model_select_dir(name, pass_num) / "panel.json")


def load_model_select_panel_result(
    name: str, pass_num: int = 1,
) -> PanelResult | None:
    path = model_select_dir(name, pass_num) / "panel.json"
    if not path.exists():
        return None
    return _load_model(PanelResult, path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/storage/test_panel_io.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/storage/project_store.py tests/storage/test_panel_io.py
git commit -m "feat(storage): add panel.json sidecar read/write helpers"
```

---

## Task 6: Add the scoring prompt templates

**Files:**
- Create: `src/styleclaw/providers/llm/prompts/score_refine_proposal.md`
- Create: `src/styleclaw/providers/llm/prompts/score_model_select_proposal.md`

No test in this task — the prompts are exercised by Task 7 / Task 8 adapter tests.

- [ ] **Step 1: Create `score_refine_proposal.md`**

Create `src/styleclaw/providers/llm/prompts/score_refine_proposal.md`:

```markdown
# Score a Candidate Trigger Phrase

You are evaluating one candidate trigger phrase that another model produced.
You did NOT write it — your job is to grade it, not rewrite it.

## Context

- IP info: {ip_info}
- Round number: {round_num}
- Recent evaluation history (for context only, do not re-score images):
{history_scores}

## Candidate trigger phrase

```
{candidate_trigger}
```

Optional adjustment note from the author (may be empty):
{candidate_note}

## Scoring rubric (single 0.0–10.0 score)

Grade the candidate on how well it is likely to reproduce the style shown in
the reference images, weighing all of:

1. **Faithfulness to the visible style** (color, line, lighting, texture, mood).
2. **Generalization** — would this phrase work for subjects beyond the IP, or
   has it baked in too much character-specific content?
3. **Concision and clarity** — vague filler, contradictory cues, or wall-of-text
   weakens the score even if every clause is individually reasonable.

## Output

Return STRICT JSON, no markdown fences, no commentary:

```
{"score": <float 0.0-10.0>, "rationale": "<one or two sentences>"}
```
```

- [ ] **Step 2: Create `score_model_select_proposal.md`**

Create `src/styleclaw/providers/llm/prompts/score_model_select_proposal.md`:

```markdown
# Score a Candidate Model Recommendation

You are evaluating one candidate model recommendation (with per-model scores
and a chosen winner) that another evaluator produced. You did NOT write it —
your job is to grade the recommendation, not redo the evaluation.

## Candidate evaluation (JSON)

```
{candidate_evaluation}
```

## Scoring rubric (single 0.0–10.0 score)

Grade the candidate on:

1. **Alignment with the reference images** — does the chosen model/variant in
   fact reproduce the style best in the supplied generations?
2. **Reasonableness of per-model scores** — are the dimension scores defensible
   given what each model produced, or are they obvious miscalls?
3. **Recommendation quality** — is the chosen variant (prompt-only vs
   prompt-sref) the right call given the generations?

## Output

Return STRICT JSON, no markdown fences, no commentary:

```
{"score": <float 0.0-10.0>, "rationale": "<one or two sentences>"}
```
```

- [ ] **Step 3: Verify file presence**

Run: `ls src/styleclaw/providers/llm/prompts/score_*.md`
Expected: both files listed.

- [ ] **Step 4: Commit**

```bash
git add src/styleclaw/providers/llm/prompts/score_refine_proposal.md src/styleclaw/providers/llm/prompts/score_model_select_proposal.md
git commit -m "feat(prompts): add scoring rubrics for refine/model-select panels"
```

---

## Task 7: Implement `agents/refine_panel.py`

**Files:**
- Create: `src/styleclaw/agents/refine_panel.py`
- Test: `tests/agents/test_refine_panel.py`

- [ ] **Step 1: Write failing adapter tests**

Create `tests/agents/test_refine_panel.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from styleclaw.agents.refine_panel import refine_with_panel
from styleclaw.core.models import PanelResult, PromptConfig


def _stub_provider(model_id: str, propose_payload: str, score_value: float):
    """Build a stub provider with scripted invoke() responses.

    - First invoke() call (propose) returns a JSON PromptConfig-shaped string.
    - Subsequent invoke() calls (score) return a JSON score object.

    The adapter is responsible for choosing which prompt template to render.
    Test relies on call order, not template inspection.
    """
    prov = AsyncMock()
    prov._model_id = model_id
    propose_json = f'{{"trigger_phrase": "{propose_payload}", "adjustment_note": ""}}'
    score_json = f'{{"score": {score_value}, "rationale": "ok"}}'
    prov.invoke.side_effect = [propose_json] + [score_json] * 10  # pad for safety
    return prov


@pytest.mark.asyncio
async def test_refine_with_panel_returns_winner_prompt_and_result(tmp_path):
    refs = [tmp_path / "ref1.png"]
    refs[0].write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    llms = [
        _stub_provider("a", "trigger-A", 8.0),
        _stub_provider("b", "trigger-B", 9.0),
        _stub_provider("c", "trigger-C", 7.0),
    ]
    labels = ["A", "B", "C"]

    prompt_config, panel_result = await refine_with_panel(
        llms, labels, refs,
        current_trigger="prev",
        round_num=2,
        ip_info="anime",
        evaluations=[],
        human_direction="",
    )

    assert isinstance(prompt_config, PromptConfig)
    assert isinstance(panel_result, PanelResult)
    # All three stubs score 8/9/7 respectively for everyone — so b's proposal
    # (scored by a=8 and c=7 -> avg 7.5) ties with c (scored by a=8, b=9 -> avg 8.5)
    # etc. Verify *some* trigger from the candidate set was chosen.
    assert prompt_config.trigger_phrase in {"trigger-A", "trigger-B", "trigger-C"}
    assert panel_result.winner_model_id in {"a", "b", "c"}
    assert prompt_config.round == 2
    # Three propose calls + six score calls = nine total invocations max.
    total_calls = sum(len(p.invoke.call_args_list) for p in llms)
    assert 3 <= total_calls <= 9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/agents/test_refine_panel.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the adapter**

Create `src/styleclaw/agents/refine_panel.py`:

```python
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
                f"- {s.model}: color={d.color_palette} line={d.line_style} "
                f"light={d.lighting} texture={d.texture} mood={d.overall_mood} "
                f"total={s.total:.1f}"
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

    async def score(evaluator: LLMProvider, payload: dict) -> tuple[float, str]:
        rendered = (
            score_template
            .replace("{ip_info}", sanitize_braces(ip_info))
            .replace("{round_num}", str(round_num))
            .replace("{history_scores}", history_text)
            .replace("{candidate_trigger}", sanitize_braces(payload.get("trigger_phrase", "")))
            .replace("{candidate_note}", sanitize_braces(payload.get("adjustment_note", "")))
        )
        blocks = await build_image_blocks_async(list(ref_image_paths))
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
            f"round-{round_num - 1:03d}" if round_num > 1 else "initial-analysis"
        ),
    })
    logger.info(
        "Panel-refined trigger (round %d, winner=%s): %s",
        round_num, panel.winner_model_id, prompt_config.trigger_phrase[:80],
    )
    return prompt_config, panel
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/agents/test_refine_panel.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/agents/refine_panel.py tests/agents/test_refine_panel.py
git commit -m "feat(agents): add refine_with_panel three-model adapter"
```

---

## Task 8: Implement `agents/select_model_panel.py`

**Files:**
- Create: `src/styleclaw/agents/select_model_panel.py`
- Test: `tests/agents/test_select_model_panel.py`

- [ ] **Step 1: Write failing adapter tests**

Create `tests/agents/test_select_model_panel.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from styleclaw.agents.select_model_panel import select_models_with_panel
from styleclaw.core.models import ModelEvaluation, PanelResult


def _stub(model_id: str, recommendation: str, score: float):
    prov = AsyncMock()
    prov._model_id = model_id
    eval_json = json.dumps({
        "evaluations": [],
        "recommendation": recommendation,
        "recommended_variant": "prompt-only",
        "next_direction": "",
    })
    score_json = json.dumps({"score": score, "rationale": "ok"})
    prov.invoke.side_effect = [eval_json] + [score_json] * 10
    return prov


@pytest.mark.asyncio
async def test_select_models_with_panel(tmp_path):
    refs = [tmp_path / "ref1.png"]
    refs[0].write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    (img_dir / "out1.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    model_images = {"mj-v7/prompt-only": [img_dir / "out1.png"]}
    llms = [
        _stub("a", "mj-v7", 9.0),
        _stub("b", "niji7", 7.0),
        _stub("c", "mj-v7", 8.5),
    ]
    labels = ["A", "B", "C"]

    evaluation, panel = await select_models_with_panel(
        llms, labels, refs, model_images,
    )

    assert isinstance(evaluation, ModelEvaluation)
    assert isinstance(panel, PanelResult)
    assert panel.winner_model_id in {"a", "b", "c"}
    # Winner's recommendation should be reflected in the merged ModelEvaluation.
    winner_payload = next(p.payload for p in panel.proposals if p.model_id == panel.winner_model_id)
    assert evaluation.recommendation == winner_payload["recommendation"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/agents/test_select_model_panel.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the adapter**

Create `src/styleclaw/agents/select_model_panel.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/agents/test_select_model_panel.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/agents/select_model_panel.py tests/agents/test_select_model_panel.py
git commit -m "feat(agents): add select_models_with_panel three-model adapter"
```

---

## Task 9: Branch `do_refine` on the refine toggle

**Files:**
- Modify: `src/styleclaw/orchestrator/actions.py:385-461` (the `do_refine` body)
- Test: `tests/orchestrator/test_actions_do.py` (add panel-mode case)

- [ ] **Step 1: Write failing tests**

Append to `tests/orchestrator/test_actions_do.py`:

```python
class TestDoRefinePanel:
    """do_refine should branch on STYLECLAW_PANEL_REFINE."""

    @pytest.mark.asyncio
    async def test_panel_off_routes_to_single_model(self, tmp_path, monkeypatch):
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        name = _create_project(phase=Phase.STYLE_REFINE, selected_models=["mj-v7"])
        # Seed analysis so the round-1 path can read a current_trigger.
        project_store.save_analysis(name, StyleAnalysis(trigger_phrase="seed"))

        with patch(
            "styleclaw.agents.refine_prompt.refine_prompt",
            new=AsyncMock(return_value=PromptConfig(round=1, trigger_phrase="single-model-win")),
        ) as single, patch(
            "styleclaw.agents.refine_panel.refine_with_panel",
            new=AsyncMock(),
        ) as panel:
            result = await do_refine(_ctx(name, llm=AsyncMock()), {})

        assert result.ok
        single.assert_awaited_once()
        panel.assert_not_awaited()
        # No panel.json sidecar.
        round_d = project_store.round_dir(name, 1)
        assert not (round_d / "panel.json").exists()

    @pytest.mark.asyncio
    async def test_panel_on_routes_through_panel_and_writes_sidecar(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        import importlib, styleclaw.core.config as config_mod
        importlib.reload(config_mod)

        from styleclaw.core.models import PanelProposal, PanelResult

        name = _create_project(phase=Phase.STYLE_REFINE, selected_models=["mj-v7"])
        project_store.save_analysis(name, StyleAnalysis(trigger_phrase="seed"))

        panel_result = PanelResult(
            proposals=[PanelProposal(model_id="m1", payload={"trigger_phrase": "panel-win"})],
            scores=[],
            winner_model_id="m1",
            averages={"m1": 9.0},
        )
        panel_prompt = PromptConfig(round=1, trigger_phrase="panel-win", derived_from="initial-analysis")

        with patch(
            "styleclaw.providers.llm.panel_factory.build_panel_providers",
            return_value=[(AsyncMock(_model_id=f"m{i}"), f"L{i}") for i in (1, 2, 3)],
        ), patch(
            "styleclaw.providers.llm.panel_factory.close_panel_providers",
            new=AsyncMock(),
        ), patch(
            "styleclaw.agents.refine_panel.refine_with_panel",
            new=AsyncMock(return_value=(panel_prompt, panel_result)),
        ) as panel_call, patch(
            "styleclaw.agents.refine_prompt.refine_prompt",
            new=AsyncMock(),
        ) as single_call:
            result = await do_refine(_ctx(name, llm=AsyncMock()), {})

        assert result.ok
        panel_call.assert_awaited_once()
        single_call.assert_not_awaited()

        # Main artifact contains winner's trigger phrase (unchanged downstream contract).
        loaded = project_store.load_prompt_config(name, round_num=1)
        assert loaded.trigger_phrase == "panel-win"

        # Sidecar exists.
        loaded_panel = project_store.load_round_panel_result(name, round_num=1)
        assert loaded_panel is not None
        assert loaded_panel.winner_model_id == "m1"

    @pytest.mark.asyncio
    async def test_panel_failure_returns_step_failure(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        import importlib, styleclaw.core.config as config_mod
        importlib.reload(config_mod)

        name = _create_project(phase=Phase.STYLE_REFINE, selected_models=["mj-v7"])
        project_store.save_analysis(name, StyleAnalysis(trigger_phrase="seed"))

        with patch(
            "styleclaw.providers.llm.panel_factory.build_panel_providers",
            return_value=[(AsyncMock(_model_id=f"m{i}"), f"L{i}") for i in (1, 2, 3)],
        ), patch(
            "styleclaw.providers.llm.panel_factory.close_panel_providers",
            new=AsyncMock(),
        ), patch(
            "styleclaw.agents.refine_panel.refine_with_panel",
            new=AsyncMock(side_effect=RuntimeError("Refine panel produced no winner")),
        ):
            result = await do_refine(_ctx(name, llm=AsyncMock()), {})

        assert result.ok is False
        assert "panel" in result.message.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/orchestrator/test_actions_do.py::TestDoRefinePanel -v`
Expected: FAIL — `do_refine` does not yet branch on the toggle.

- [ ] **Step 3: Modify `do_refine` to branch on the toggle**

In `src/styleclaw/orchestrator/actions.py`, replace the body of `do_refine` (currently lines 385-461) with a version that branches on `PANEL_REFINE_ENABLED`. Keep the existing pre-flight code (phase guard, round skipping, history loading, current_trigger resolution) verbatim; only swap the *call* and the *write* sections.

```python
async def do_refine(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    import styleclaw.core.config as _cfg
    from styleclaw.agents.refine_prompt import refine_prompt, refine_prompt_with_thinking
    from styleclaw.core.models import RoundEvaluation

    state = project_store.load_state(ctx.project)

    if state.phase != Phase.STYLE_REFINE:
        return StepResult(
            ok=False,
            message=f"refine requires STYLE_REFINE phase (current: {state.phase}). "
                    f"Run 'select-model' first to advance from MODEL_SELECT."
        )

    config = project_store.load_config(ctx.project)
    root = project_store.project_dir(ctx.project)
    ref_paths = [root / r for r in config.ref_images]

    pass_num = state.current_model_select_pass or 1
    round_num = state.current_round + 1

    while True:
        try:
            project_store.load_prompt_config(ctx.project, round_num, pass_num=pass_num)
            round_num += 1
        except FileNotFoundError:
            break

    if round_num > MAX_AUTO_ROUNDS:
        return StepResult(ok=False, message=f"Max rounds ({MAX_AUTO_ROUNDS}) reached")

    evaluations: list[RoundEvaluation] = []
    for r in range(1, round_num):
        try:
            ev = project_store.load_round_evaluation(ctx.project, r, pass_num=pass_num)
            evaluations.append(ev)
        except FileNotFoundError:
            logger.warning("Evaluation for round %d not found, skipping history entry.", r)

    if round_num == 1:
        analysis = project_store.load_analysis(ctx.project, pass_num=pass_num)
        current_trigger = analysis.trigger_phrase
    else:
        prev_prompt = project_store.load_prompt_config(
            ctx.project, round_num - 1, pass_num=pass_num,
        )
        current_trigger = prev_prompt.trigger_phrase

    direction = args.get("direction", "")

    if _cfg.PANEL_REFINE_ENABLED:
        from styleclaw.agents.refine_panel import refine_with_panel
        from styleclaw.providers.llm.panel_factory import (
            build_panel_providers,
            close_panel_providers,
        )

        pairs = build_panel_providers()
        try:
            llms = [p for p, _ in pairs]
            labels = [label for _, label in pairs]
            try:
                prompt_config, panel_result = await refine_with_panel(
                    llms, labels, ref_paths, current_trigger, round_num,
                    config.ip_info, evaluations, direction,
                )
            except RuntimeError as exc:
                return StepResult(ok=False, message=f"refine panel failed: {exc}")
        finally:
            await close_panel_providers(pairs)

        project_store.save_prompt_config(
            ctx.project, round_num, prompt_config, pass_num=pass_num,
        )
        project_store.save_round_panel_result(
            ctx.project, round_num, panel_result, pass_num=pass_num,
        )

        new_state = state.with_round(round_num)
        project_store.save_state(ctx.project, new_state)

        msg = f"Round {round_num} [panel:{panel_result.winner_model_id}]: {prompt_config.trigger_phrase}"
        if panel_result.degraded:
            msg += f" (degraded; see panel.json — {len(panel_result.error_log)} issue(s))"
        return StepResult(ok=True, message=msg, data={"panel": True, "degraded": panel_result.degraded})

    # Single-model path (unchanged).
    thinking = ""
    if ctx.show_thinking:
        prompt_config, thinking = await refine_prompt_with_thinking(
            ctx.llm, ref_paths, current_trigger, round_num,
            config.ip_info, evaluations, direction,
            thinking_budget=ctx.thinking_budget,
        )
    else:
        prompt_config = await refine_prompt(
            ctx.llm, ref_paths, current_trigger, round_num,
            config.ip_info, evaluations, direction,
        )
    project_store.save_prompt_config(
        ctx.project, round_num, prompt_config, pass_num=pass_num,
    )

    if thinking:
        round_d = project_store.round_dir(ctx.project, round_num, pass_num=pass_num)
        project_store.save_thinking(round_d / "prompt.json", thinking)

    new_state = state.with_round(round_num)
    project_store.save_state(ctx.project, new_state)

    msg = f"Round {round_num}: {prompt_config.trigger_phrase}"
    if thinking:
        msg += f" | thinking saved ({len(thinking)} chars)"
    return StepResult(ok=True, message=msg)
```

- [ ] **Step 4: Run all touched tests**

Run:
```bash
uv run python -m pytest tests/orchestrator/test_actions_do.py::TestDoRefinePanel tests/orchestrator/test_actions_do.py -v -k "refine"
```
Expected: new tests pass; pre-existing `do_refine` tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/orchestrator/actions.py tests/orchestrator/test_actions_do.py
git commit -m "feat(orchestrator): branch do_refine on STYLECLAW_PANEL_REFINE"
```

---

## Task 10: Branch `do_evaluate` (MODEL_SELECT branch) on the model-select toggle

**Files:**
- Modify: `src/styleclaw/orchestrator/actions.py:241-297` (the MODEL_SELECT branch of `do_evaluate`)
- Test: `tests/orchestrator/test_actions_do.py` (add panel-mode case)

- [ ] **Step 1: Write failing tests**

Append to `tests/orchestrator/test_actions_do.py`:

```python
class TestDoEvaluatePanel:
    """do_evaluate (MODEL_SELECT) should branch on STYLECLAW_PANEL_MODEL_SELECT."""

    @pytest.mark.asyncio
    async def test_panel_off_routes_to_single_model(self, monkeypatch):
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        name = _create_project(phase=Phase.MODEL_SELECT)
        # Seed at least one model output so the path doesn't short-circuit.
        results_dir = project_store.model_results_dir(name, "mj-v7", variant="prompt-only")
        (results_dir / "out-1.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        project_store.save_task_record(
            name, "mj-v7",
            TaskRecord(task_id="t", model_id="mj-v7", status="SUCCESS"),
            variant="prompt-only",
        )

        single_eval = ModelEvaluation(recommendation="mj-v7", recommended_variant="prompt-only")
        with patch(
            "styleclaw.agents.select_model.evaluate_models",
            new=AsyncMock(return_value=single_eval),
        ) as single, patch(
            "styleclaw.agents.select_model_panel.select_models_with_panel",
            new=AsyncMock(),
        ) as panel, patch(
            "styleclaw.scripts.report.generate_model_select_report",
            return_value=Path("/tmp/x.html"),
        ):
            result = await do_evaluate(_ctx(name, llm=AsyncMock()), {})

        assert result.ok
        single.assert_awaited_once()
        panel.assert_not_awaited()
        assert project_store.load_model_select_panel_result(name) is None

    @pytest.mark.asyncio
    async def test_panel_on_routes_through_panel_and_writes_sidecar(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_MODEL_SELECT", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        import importlib, styleclaw.core.config as config_mod
        importlib.reload(config_mod)

        from styleclaw.core.models import PanelProposal, PanelResult

        name = _create_project(phase=Phase.MODEL_SELECT)
        results_dir = project_store.model_results_dir(name, "mj-v7", variant="prompt-only")
        (results_dir / "out-1.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        project_store.save_task_record(
            name, "mj-v7",
            TaskRecord(task_id="t", model_id="mj-v7", status="SUCCESS"),
            variant="prompt-only",
        )

        panel_eval = ModelEvaluation(recommendation="mj-v7", recommended_variant="prompt-sref")
        panel_result = PanelResult(
            proposals=[PanelProposal(model_id="m1", payload=panel_eval.model_dump())],
            scores=[],
            winner_model_id="m1",
            averages={"m1": 9.0},
        )

        with patch(
            "styleclaw.providers.llm.panel_factory.build_panel_providers",
            return_value=[(AsyncMock(_model_id=f"m{i}"), f"L{i}") for i in (1, 2, 3)],
        ), patch(
            "styleclaw.providers.llm.panel_factory.close_panel_providers",
            new=AsyncMock(),
        ), patch(
            "styleclaw.agents.select_model_panel.select_models_with_panel",
            new=AsyncMock(return_value=(panel_eval, panel_result)),
        ), patch(
            "styleclaw.scripts.report.generate_model_select_report",
            return_value=Path("/tmp/x.html"),
        ):
            result = await do_evaluate(_ctx(name, llm=AsyncMock()), {})

        assert result.ok
        loaded = project_store.load_evaluation(name)
        assert loaded.recommendation == "mj-v7"
        assert loaded.recommended_variant == "prompt-sref"
        loaded_panel = project_store.load_model_select_panel_result(name)
        assert loaded_panel is not None
        assert loaded_panel.winner_model_id == "m1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/orchestrator/test_actions_do.py::TestDoEvaluatePanel -v`
Expected: FAIL — `do_evaluate` does not yet branch on the toggle.

- [ ] **Step 3: Modify `do_evaluate` MODEL_SELECT branch**

In `src/styleclaw/orchestrator/actions.py`, edit only the `if state.phase == Phase.MODEL_SELECT:` block inside `do_evaluate` (around line 247). The block currently:
1. Loads `model_images`,
2. Picks single-model `evaluate_models` / `evaluate_models_with_thinking`,
3. Saves evaluation, optional thinking, generates report.

Replace it with:

```python
    if state.phase == Phase.MODEL_SELECT:
        import styleclaw.core.config as _cfg
        from styleclaw.agents.select_model import (
            evaluate_models,
            evaluate_models_with_thinking,
        )
        from styleclaw.scripts.report import generate_model_select_report
        from styleclaw.storage.image_store import list_output_images

        pass_num = state.current_model_select_pass or 1

        model_images: dict[str, list[Path]] = {}
        records = project_store.load_all_task_records(ctx.project, pass_num=pass_num)
        for key in records:
            if "/" in key:
                model_id, variant = key.split("/", 1)
                results_dir = project_store.model_results_dir(
                    ctx.project, model_id, variant=variant, pass_num=pass_num,
                )
            else:
                results_dir = project_store.model_results_dir(
                    ctx.project, key, pass_num=pass_num,
                )
            images = list_output_images(results_dir)
            if images:
                model_images[key] = images

        if not model_images:
            return StepResult(ok=False, message="No generated images found")

        if _cfg.PANEL_MODEL_SELECT_ENABLED:
            from styleclaw.agents.select_model_panel import select_models_with_panel
            from styleclaw.providers.llm.panel_factory import (
                build_panel_providers,
                close_panel_providers,
            )

            pairs = build_panel_providers()
            try:
                llms = [p for p, _ in pairs]
                labels = [label for _, label in pairs]
                try:
                    evaluation, panel_result = await select_models_with_panel(
                        llms, labels, ref_paths, model_images,
                    )
                except RuntimeError as exc:
                    return StepResult(ok=False, message=f"model-select panel failed: {exc}")
            finally:
                await close_panel_providers(pairs)

            project_store.save_evaluation(ctx.project, evaluation, pass_num=pass_num)
            project_store.save_model_select_panel_result(
                ctx.project, panel_result, pass_num=pass_num,
            )
            generate_model_select_report(ctx.project, pass_num=pass_num)

            msg = (
                f"Recommendation: {evaluation.recommendation} "
                f"[panel:{panel_result.winner_model_id}] (pass {pass_num})"
            )
            if panel_result.degraded:
                msg += f" (degraded; see panel.json — {len(panel_result.error_log)} issue(s))"
            return StepResult(
                ok=True, message=msg,
                data={
                    "recommendation": evaluation.recommendation,
                    "pass_num": pass_num,
                    "panel": True,
                    "degraded": panel_result.degraded,
                },
            )

        # Single-model path (unchanged).
        thinking = ""
        if ctx.show_thinking:
            evaluation, thinking = await evaluate_models_with_thinking(
                ctx.llm, ref_paths, model_images, thinking_budget=ctx.thinking_budget,
            )
        else:
            evaluation = await evaluate_models(ctx.llm, ref_paths, model_images)
        project_store.save_evaluation(ctx.project, evaluation, pass_num=pass_num)
        if thinking:
            project_store.save_thinking(
                project_store.model_select_dir(ctx.project, pass_num) / "evaluation.json",
                thinking,
            )
        generate_model_select_report(ctx.project, pass_num=pass_num)

        msg = f"Recommendation: {evaluation.recommendation} (pass {pass_num})"
        if thinking:
            msg += f" | thinking saved ({len(thinking)} chars)"
        return StepResult(
            ok=True, message=msg,
            data={"recommendation": evaluation.recommendation, "pass_num": pass_num},
        )
```

The STYLE_REFINE branch of `do_evaluate` is **not** touched — `evaluate_result` is out of scope per the spec.

- [ ] **Step 4: Run all touched tests**

Run:
```bash
uv run python -m pytest tests/orchestrator/test_actions_do.py::TestDoEvaluatePanel tests/orchestrator/test_actions_do.py -v -k "evaluate"
```
Expected: new tests pass; existing `do_evaluate` tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/orchestrator/actions.py tests/orchestrator/test_actions_do.py
git commit -m "feat(orchestrator): branch do_evaluate (MODEL_SELECT) on STYLECLAW_PANEL_MODEL_SELECT"
```

---

## Task 11: Surface panel data in HTML reports

**Files:**
- Modify: `src/styleclaw/scripts/report.py` (load `panel.json` and pass to template context)
- Modify: `src/styleclaw/reports/templates/style_refine.html` (add gated block)
- Modify: `src/styleclaw/reports/templates/model_select.html` (add gated block)
- Test: extend an existing report test or add a small one

- [ ] **Step 1: Find the report functions**

Run:
```bash
uv run python -c "import inspect, styleclaw.scripts.report as r; print(inspect.getsource(r.generate_style_refine_report))"
uv run python -c "import inspect, styleclaw.scripts.report as r; print(inspect.getsource(r.generate_model_select_report))"
```
Inspect to confirm the exact context dict each report builds. The change in step 3 must reuse the existing variable names.

- [ ] **Step 2: Write a failing test**

Append to `tests/orchestrator/test_actions_do.py` (or a dedicated `tests/scripts/test_report_panel.py` if reports already have one). Minimal smoke test:

```python
def test_style_refine_report_includes_panel_block_when_sidecar_present(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")
    name = _create_project(phase=Phase.STYLE_REFINE, current_round=1)
    project_store.save_prompt_config(
        name, 1, PromptConfig(round=1, trigger_phrase="t"),
    )
    project_store.save_round_evaluation(
        name, 1, RoundEvaluation(round=1, evaluations=[]),
    )
    from styleclaw.core.models import PanelProposal, PanelResult
    project_store.save_round_panel_result(
        name, 1,
        PanelResult(
            proposals=[PanelProposal(model_id="a", label="Opus", payload={"trigger_phrase": "t"})],
            scores=[],
            winner_model_id="a",
            averages={"a": 8.0},
        ),
    )

    from styleclaw.scripts.report import generate_style_refine_report
    path = generate_style_refine_report(name, round_num=1)
    html = Path(path).read_text(encoding="utf-8")
    assert "Panel review" in html
    assert "Opus" in html


def test_style_refine_report_omits_panel_block_when_no_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")
    name = _create_project(phase=Phase.STYLE_REFINE, current_round=1)
    project_store.save_prompt_config(
        name, 1, PromptConfig(round=1, trigger_phrase="t"),
    )
    project_store.save_round_evaluation(
        name, 1, RoundEvaluation(round=1, evaluations=[]),
    )

    from styleclaw.scripts.report import generate_style_refine_report
    path = generate_style_refine_report(name, round_num=1)
    html = Path(path).read_text(encoding="utf-8")
    assert "Panel review" not in html
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run python -m pytest tests/orchestrator/test_actions_do.py -v -k "panel_block"`
Expected: FAIL — template has no panel block / context misses `panel`.

- [ ] **Step 4: Update `report.py`**

For each report function, after loading the main JSON, attempt to load the panel sidecar and inject it into the template context as `panel`:

```python
# In generate_style_refine_report, before render:
panel = project_store.load_round_panel_result(name, round_num=round_num, pass_num=pass_num)
context["panel"] = panel.model_dump() if panel is not None else None
```

```python
# In generate_model_select_report, before render:
panel = project_store.load_model_select_panel_result(name, pass_num=pass_num)
context["panel"] = panel.model_dump() if panel is not None else None
```

Use whatever variable name the existing function calls its render dict — match the surrounding code.

- [ ] **Step 5: Append panel block to each template**

In `src/styleclaw/reports/templates/style_refine.html`, add near the top of the body:

```jinja
{% if panel %}
<section class="panel-review">
  <h2>Panel review {% if panel.degraded %}<span class="warn">(degraded)</span>{% endif %}</h2>
  <div class="proposals">
    {% for p in panel.proposals %}
      <div class="proposal {% if p.model_id == panel.winner_model_id %}winner{% endif %}">
        <h3>{{ p.label or p.model_id }}{% if p.model_id == panel.winner_model_id %} — winner{% endif %}</h3>
        <pre>{{ p.payload | tojson(indent=2) }}</pre>
      </div>
    {% endfor %}
  </div>
  <table class="scores">
    <thead>
      <tr><th>Evaluator \\ Target</th>{% for p in panel.proposals %}<th>{{ p.label or p.model_id }}</th>{% endfor %}</tr>
    </thead>
    <tbody>
      {% for ev in panel.proposals %}
        <tr>
          <th>{{ ev.label or ev.model_id }}</th>
          {% for tg in panel.proposals %}
            {% if ev.model_id == tg.model_id %}
              <td class="self">—</td>
            {% else %}
              {% set match = panel.scores | selectattr('evaluator_model_id', 'equalto', ev.model_id) | selectattr('target_model_id', 'equalto', tg.model_id) | list %}
              <td>{% if match %}{{ '%.1f' | format(match[0].score) }}{% else %}—{% endif %}</td>
            {% endif %}
          {% endfor %}
        </tr>
      {% endfor %}
    </tbody>
  </table>
  {% if panel.error_log %}
    <details><summary>Errors ({{ panel.error_log | length }})</summary>
      <ul>{% for e in panel.error_log %}<li>{{ e }}</li>{% endfor %}</ul>
    </details>
  {% endif %}
</section>
{% endif %}
```

Add the same block (copy-paste verbatim) into `src/styleclaw/reports/templates/model_select.html`. The two templates are independent — the engineer should not refactor them into a shared partial unless one already exists in the project.

- [ ] **Step 6: Run all report tests**

Run: `uv run python -m pytest tests/ -v -k "report"`
Expected: new tests pass; existing report tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/styleclaw/scripts/report.py src/styleclaw/reports/templates/style_refine.html src/styleclaw/reports/templates/model_select.html tests/orchestrator/test_actions_do.py
git commit -m "feat(report): render Panel review block when panel.json sidecar exists"
```

---

## Task 12: Document env vars in `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md` (Environment Variables and/or Runtime Tunables tables)

- [ ] **Step 1: Add a new row group to the env var documentation**

In `CLAUDE.md`, in the "Runtime Tunables (optional)" table, insert the following rows (keep the table format):

```
| `STYLECLAW_PANEL_REFINE` | unset | When truthy, `do_refine` runs a three-model panel (propose + cross-score + winner) instead of a single-model call |
| `STYLECLAW_PANEL_MODEL_SELECT` | unset | When truthy, `do_evaluate` in MODEL_SELECT runs the same three-model panel |
| `STYLECLAW_PANEL_MODELS` | unset | Required when either panel toggle is on — exactly 3 comma-separated OpenAI-compat model ids |
| `STYLECLAW_PANEL_LABELS` | unset | Optional human-readable labels (same length as `STYLECLAW_PANEL_MODELS`); falls back to model ids in reports/logs |
```

In the "Conventions" section, append a bullet:

```
- **Panel mode**: When `STYLECLAW_PANEL_REFINE` or `STYLECLAW_PANEL_MODEL_SELECT` is on, the corresponding orchestrator action routes through `core.panel.run_panel` (3 proposals + ≤6 cross-evaluations, no self-scoring). The winner's payload still lands in the existing main artifact (`prompt.json` / `evaluation.json`), so downstream code is unchanged; full `PanelResult` is persisted alongside as `panel.json`. See `docs/superpowers/specs/2026-05-14-three-model-panel-design.md`.
```

- [ ] **Step 2: Sanity check the file**

Run: `uv run python -m pytest tests/ -v` — full suite should still pass, since `CLAUDE.md` is documentation only.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document STYLECLAW_PANEL_* env vars and panel routing convention"
```

---

## Task 13: Regression sweep + final verification

**Files:** none (verification only).

- [ ] **Step 1: Confirm panel-off bytes match baseline**

This step is the "panel disabled regression" assertion the spec calls out. With both toggles off, neither sidecar file may appear and the main artifacts must be byte-identical to a non-panel run.

The existing tests in `tests/orchestrator/test_actions_do.py` already cover the panel-off path; the new `test_panel_off_routes_to_single_model` cases assert that no `panel.json` is produced. Verify by:

Run:
```bash
uv run python -m pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 2: Confirm coverage holds**

Run:
```bash
uv run python -m pytest tests/ --cov=src --cov-fail-under=80
```
Expected: coverage ≥ 80%. If below, add a targeted test in the area that dropped (most likely the new branches in `do_refine` / `do_evaluate`).

- [ ] **Step 3: Run a real CLI smoke (optional, requires API keys)**

If credentials are available:

```bash
STYLECLAW_PANEL_REFINE=0 STYLECLAW_PANEL_MODEL_SELECT=0 uv run styleclaw --help
```
Expected: no startup error.

```bash
STYLECLAW_PANEL_REFINE=1 STYLECLAW_PANEL_MODELS="only-one" uv run styleclaw --help
```
Expected: startup error mentioning `STYLECLAW_PANEL_MODELS`.

- [ ] **Step 4: Final commit (only if anything outstanding)**

If steps 1-3 surfaced fixes, group them into one commit; otherwise no commit is needed.

```bash
git status
# If clean:
echo "Implementation complete."
```
