# Three-Model Review Panel — Design

**Status**: Approved, pending implementation plan
**Date**: 2026-05-14
**Scope**: StyleClaw `STYLE_REFINE.refine` and `MODEL_SELECT.evaluate`

## §1 Overview and Non-Goals

Upgrade two LLM-driven decision points to a **three-model review panel**: three LLMs independently produce a candidate output, then cross-score each other's candidates (no self-scoring). The highest average score wins and becomes the canonical output. Both touch points are gated by independent env switches and default to OFF, so existing single-model projects regress to zero behavior change.

### In scope

- `STYLE_REFINE.refine` (trigger phrase competition)
- `MODEL_SELECT.evaluate` (image-gen model recommendation)

### Out of scope (this spec)

- `evaluate_result` (per-round image scoring)
- `analyze_style`, `design_cases`
- The `LLMProvider` Protocol itself (not modified)
- New providers — reuse the existing OpenAI-compat client

### Goals

- Reduce single-model bias on the two highest-leverage decisions in the pipeline
- Preserve the existing single-model code path verbatim when the panel is off
- Keep panel artifacts auditable on disk for post-hoc review

### Non-goals

- Real-time tie-breaking arbitration via a fourth "judge" model
- Replacing every LLM call with a panel
- Automatic prompt iteration based on panel disagreement

## §2 Configuration

### Env variables

```
# Independent toggles per phase
STYLECLAW_PANEL_REFINE=1           # STYLE_REFINE.refine via panel
STYLECLAW_PANEL_MODEL_SELECT=1     # MODEL_SELECT.evaluate via panel

# Required when either toggle is on: exactly 3 comma-separated model ids,
# all reachable through the existing OpenAI-compat endpoint.
STYLECLAW_PANEL_MODELS=claude-opus-4-7,gpt-5.5,gemini-3-pro

# Optional human-readable labels (same length as models). Falls back to model
# ids when omitted. Used in report HTML and CLI logs.
STYLECLAW_PANEL_LABELS=Opus,GPT,Gemini
```

`OPENAI_COMPAT_BASE_URL` and `OPENAI_COMPAT_API_KEY` continue to drive the underlying transport. Phases whose toggle is 0 keep using `LLM_MODEL` (single-model).

### Startup validation

In `core/config.py`:

- If any panel toggle is 1 → `STYLECLAW_PANEL_MODELS` must produce exactly 3 non-empty trimmed ids; otherwise raise `ValueError` with a message naming the env var.
- `STYLECLAW_PANEL_LABELS`, if set, must have the same length as models; mismatched length raises.
- When both toggles are 0, `STYLECLAW_PANEL_MODELS` and `STYLECLAW_PANEL_LABELS` are ignored without warning.

### Combination matrix

| Scenario | `PANEL_REFINE` | `PANEL_MODEL_SELECT` |
|---|:---:|:---:|
| Default (no panel) | 0 | 0 |
| Panel for trigger phrase only | 1 | 0 |
| Panel for image-gen model pick only | 0 | 1 |
| Both | 1 | 1 |

## §3 Data Model Additions

Three new frozen Pydantic models in `core/models.py`:

```python
class PanelProposal(_FrozenModel):
    """One participant's submitted artifact."""
    model_id: str                 # who produced it
    label: str = ""               # human-readable name (falls back to model_id)
    payload: dict[str, Any]       # domain-specific output (trigger phrase, ModelEvaluation, ...)
    thinking: str = ""            # captured reasoning when --show-thinking is on


class PanelScore(_FrozenModel):
    """One evaluator's score on one proposal."""
    evaluator_model_id: str
    target_model_id: str          # whose proposal is being scored
    score: float                  # 0.0–10.0
    rationale: str = ""


class PanelResult(_FrozenModel):
    """Aggregate output of a panel competition."""
    proposals: list[PanelProposal]
    scores: list[PanelScore]                         # up to 6 entries (3 evaluators × 2 others)
    winner_model_id: str                             # empty when degraded with no usable winner
    averages: dict[str, float]                       # model_id → mean of scores received
    degraded: bool = False                           # true when any participant failed
    error_log: list[str] = Field(default_factory=list)
```

### Reuse, not replace

- `PromptConfig` (output of `refine`) and `ModelEvaluation` (output of `select_model.evaluate`) keep their schemas **untouched**.
- Panel mode writes the **winner's** payload into the same existing artifact (`prompt.json` / `evaluation.json`). Downstream code (`generate`, `evaluate_result`, `report`) cannot tell the difference between single-model and panel mode.
- Full panel detail is persisted in a sibling `panel.json` next to the main artifact. See §6.

## §4 Architecture and Call Flow

```
do_refine (orchestrator/actions.py)
  │
  ├─ panel_enabled_for("refine")?
  │    └── False → agents/refine_prompt.py (unchanged)
  │    └── True  → agents/refine_panel.py::refine_with_panel(ctx, ...)
  │                  │
  │                  └── core/panel.py::run_panel(
  │                          llms=[opus, gpt, gemini],
  │                          labels=[...],
  │                          propose=_refine_proposer,
  │                          score=_refine_scorer,
  │                          ...
  │                      )
  │                      ├── Phase 1: concurrent propose (≤3 LLM calls)
  │                      ├── Phase 2: concurrent cross-eval (≤6 LLM calls, no self)
  │                      └── Phase 3: aggregate → PanelResult
  │
  └─ write winner.payload into prompt.json, write panel.json sidecar

do_evaluate (MODEL_SELECT path)
  │
  └─ symmetric: agents/select_model_panel.py + same run_panel(...)
                  propose=_select_proposer, score=_select_scorer
```

### Module boundaries

- `core/panel.py` — **domain-agnostic** orchestrator. Knows nothing about triggers, scores, or model_select. Pure async coordination + aggregation.
- `agents/refine_panel.py` — domain wiring for the refine case: a `propose` function that calls `refine_prompt()` once, a `score` function that reads `prompts/score_refine_proposal.md`.
- `agents/select_model_panel.py` — domain wiring for the model-select case: `propose` calls the existing `evaluate_model_select` once, `score` reads `prompts/score_model_select_proposal.md`.
- `LLMProvider` Protocol — **unchanged**.

### Provider instantiation

A new helper `providers/llm/panel_factory.py::build_panel_providers()` reads `STYLECLAW_PANEL_MODELS` + `STYLECLAW_PANEL_LABELS` and returns a `list[tuple[OpenAICompatProvider, str]]` (provider, label). All three share `OPENAI_COMPAT_BASE_URL` / `OPENAI_COMPAT_API_KEY`; only the `model_id` differs.

### Concurrency

- Phase 1: 3 concurrent proposals via `asyncio.gather(..., return_exceptions=True)`.
- Phase 2: up to 6 concurrent scoring calls.
- All requests still flow through each provider's own `LLM_CONCURRENCY_LIMIT` semaphore, so the panel never floods the underlying endpoint.
- Worst-case 9 LLM calls per panel round, ≈3× the latency and token cost of a single-model round.

## §5 Algorithm Details

### `core/panel.py::run_panel`

```python
async def run_panel(
    llms: list[LLMProvider],                   # exactly 3
    labels: list[str],                         # exactly 3, display only
    propose: Callable[[LLMProvider], Awaitable[dict[str, Any]]],
    score: Callable[[LLMProvider, dict[str, Any]], Awaitable[tuple[float, str]]],
    min_proposals: int = 2,
    min_scores_per_proposal: int = 1,
) -> PanelResult: ...
```

**Phase 1 — propose (concurrent).** `asyncio.gather(*[propose(llm) for llm in llms], return_exceptions=True)`.

- Each success becomes a `PanelProposal` with `model_id` + `label`.
- Each exception is appended to `error_log`; no re-raise.
- If surviving proposals < `min_proposals` → return `PanelResult(degraded=True, winner_model_id="")` and let the caller decide whether to fail the step.

**Phase 2 — cross-evaluation (concurrent, no self-scoring).**

```
for evaluator in llms:
    for proposal in proposals:
        if proposal.model_id == evaluator.model_id:
            continue                              # no self-scoring
        schedule score(evaluator, proposal.payload)
```

At most 6 scheduled calls (3 evaluators × 2 others). Any individual failure is logged to `error_log`; the corresponding cell in the scoring matrix stays empty.

**Phase 3 — aggregate.**

```python
for proposal in proposals:
    received = [s.score for s in scores if s.target_model_id == proposal.model_id]
    if len(received) < min_scores_per_proposal:
        error_log.append(f"proposal {proposal.model_id}: insufficient scores")
        continue
    averages[proposal.model_id] = sum(received) / len(received)

if not averages:
    return PanelResult(..., degraded=True, winner_model_id="")

winner = max(averages, key=averages.get)
# Tie-break: stable — pick the entry that appears earliest in STYLECLAW_PANEL_MODELS.
```

### Domain wiring

**`agents/refine_panel.py`**
- `propose`: calls `refine_prompt()` once with the bound `LLMProvider`; returns `{"trigger_phrase": ..., "adjustment_note": ...}`.
- `score`: renders `prompts/score_refine_proposal.md` (refs + round history + candidate trigger phrase) and parses `{"score": float, "rationale": str}`.

**`agents/select_model_panel.py`**
- `propose`: calls the existing `evaluate_model_select()` once; returns `ModelEvaluation.model_dump()`.
- `score`: renders `prompts/score_model_select_proposal.md` (refs + per-model output samples + candidate recommendation) and parses `{"score": float, "rationale": str}`.

### New prompt templates

Two files added under `providers/llm/prompts/`, structurally aligned with the existing `evaluate.md`:

- `score_refine_proposal.md` — scoring rubric for a candidate trigger phrase (single 0–10 score + rationale).
- `score_model_select_proposal.md` — scoring rubric for a candidate model recommendation.

Both must explicitly require JSON output with no markdown fences, list the scoring dimensions inline, and forbid the evaluator from rewriting the proposal.

## §6 Persistence and Reporting

### New sidecar artifacts

```
data/projects/<name>/style-refine/pass-NNN/round-NNN/panel.json
data/projects/<name>/model-select/pass-NNN/panel.json
```

Each file holds the full `PanelResult.model_dump()` — 3 proposals, up to 6 scores, winner id, averages map, `degraded` flag, and `error_log`. This is the audit record; nothing downstream depends on it.

### Main artifacts (unchanged)

- **Refine**: the winner's `trigger_phrase` is written into `prompt.json` exactly as the single-model path does. `generate` / `evaluate` / `report` are oblivious.
- **Model select**: the winner's `ModelEvaluation` is written into `evaluation.json` with no schema change.

### Report templates

`reports/templates/style_refine.html` and `reports/templates/model_select.html` get an additional **Panel review** block, gated by `{% if panel %}`:

- Side-by-side display of the 3 proposals (label + payload summary + captured `thinking` when available).
- Scoring matrix as a small table (rows = evaluators, columns = targets, diagonal blank).
- Winner row highlighted; `degraded=True` renders a warning banner at the top of the block.
- When no `panel.json` exists, the block is omitted entirely — single-model reports look exactly as before.

## §7 Failure Handling

| Situation | Behavior |
|---|---|
| Panel toggle on but `STYLECLAW_PANEL_MODELS` does not yield exactly 3 ids | Startup raises `ValueError` in `core/config.py`; CLI refuses to run |
| `STYLECLAW_PANEL_LABELS` length mismatch | Startup raises `ValueError` |
| One `propose` call raises | Logged to `error_log`; if ≥ 2 proposals survive, continue |
| Surviving proposals < 2 | `do_refine` / `do_evaluate` returns `StepResult(ok=False)` with a message naming the failing models and suggesting the user disable the relevant panel toggle or check the provider |
| One `score` call raises | Logged to `error_log`; the missing cell does not block aggregation |
| A given proposal receives 0 valid scores | That proposal is dropped from `averages`; `error_log` records the cause |
| All proposals dropped during aggregation | `degraded=True`, `winner_model_id=""`, main artifact is **not** written, step fails |
| Partial degradation but a winner exists | `degraded=True`, main artifact written normally, CLI prints a yellow warning pointing at `panel.json` |

429 / 5xx retries continue to be handled by the existing provider-level logic (no extra retry layer is added inside `run_panel`).

## §8 Testing Strategy

### Unit

- `tests/core/test_panel.py` — `run_panel()` pure logic:
  - all 3 proposals succeed → correct winner, full 6-cell matrix, `degraded=False`
  - 1 proposal raises → 2 survivors, 2 scoring calls (1 evaluator × 2 targets becomes 2 evaluators × 1 target = 2 calls), winner computed
  - 2 proposals raise → step fails (`min_proposals` floor)
  - some scores raise → proposal still aggregated as long as it has ≥ 1 valid score
  - all scores for one proposal missing → proposal dropped from `averages`
  - tie on averages → stable winner determined by position in `llms`
- `tests/agents/test_refine_panel.py` — `propose` / `score` adapters with mocked `LLMProvider`s; assert prompt template selection and JSON parsing.
- `tests/agents/test_select_model_panel.py` — symmetric coverage for the model-select adapter.
- `tests/core/test_config.py` — env validation:
  - either panel toggle on + fewer than 3 models → raises
  - labels length mismatch → raises
  - both toggles off → `STYLECLAW_PANEL_MODELS` / `STYLECLAW_PANEL_LABELS` ignored without warning

### Integration

- `tests/orchestrator/test_actions_do.py`:
  - `do_refine` with `STYLECLAW_PANEL_REFINE=0` → routes to existing `refine_prompt()`; no `panel.json` is written.
  - `do_refine` with `STYLECLAW_PANEL_REFINE=1` → routes to `refine_with_panel`; the winner's `trigger_phrase` lands in `prompt.json`; `panel.json` sibling exists with 3 proposals.
  - Same coverage for `do_evaluate` × `STYLECLAW_PANEL_MODEL_SELECT`.

### Regression protection

All existing single-model tests stay untouched. A new explicit assertion is added per affected test path: when both panel toggles are 0, **no** `panel.json` is created and the main artifact bytes match the pre-panel baseline.

### Mock strategy

Three mock `LLMProvider` instances with distinct `model_id`s; `invoke()` / `invoke_with_thinking()` return scripted JSON. No `respx` needed — `run_panel` is pure orchestration on top of the Protocol, so HTTP mocking belongs only to the provider-layer tests that already exist.
