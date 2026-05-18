# Per-Role LLM Routing

**Status**: Approved, ready for implementation plan
**Date**: 2026-05-18
**Scope**: Add a routing layer so different LLM call sites in StyleClaw can use different model IDs (e.g. `gemini-2.5-pro-preview` for image analysis, `claude-sonnet-4-6` for character design), while preserving today's single-`LLM_MODEL` behavior for users who do not opt in.

## Motivation

Today every LLM call in StyleClaw goes through a single `LLMProvider` instance built from `LLM_MODEL`. The five agent call sites and the orchestrator planner have very different capability and context-window needs:

| Tier | Call sites | What it needs |
|---|---|---|
| A — vision-heavy judge | `select_model`, `evaluate_result` (+ their panel scorers) | Top-tier multimodal, 20-40 images in one prompt for `select_model`, strict comparison reasoning |
| A — vision-heavy creator | `analyze_style`, `refine_prompt` (+ refine panel scorer) | Strong multimodal + linguistic creativity (translates visual deltas into prompt-engineering vocabulary) |
| B — text-only long output | `design_cases` | No vision; produces 100 structured cases (`max_tokens=16384`); needs "stays-on-task at length" |
| C — lightweight structured | `orchestrator.planner.plan` | Tiny input, ≤5-step JSON output, runs on every `styleclaw run` |

Forcing all four tiers onto one model wastes either capability (when the model is small) or money/latency (when the model is the strongest available). Per-role routing fixes both.

## Goals

1. Map each LLM call site to one of four roles: `vision_critic`, `vision_analyst`, `writer`, `planner`.
2. Allow the operator to pin a different model ID per role via env vars.
3. Keep the existing `LLM_MODEL` and `STYLECLAW_PANEL_MODELS` semantics as fallbacks so existing setups need zero changes.
4. Persist the actual model ID used into each LLM-derived artifact on disk, for debugging and experiment reproducibility.
5. Leave a clean extension point for cross-provider routing (different `base_url` / `api_key` per role) without committing to it now.

## Non-Goals

- Cross-provider routing (different gateways per role). The internal `RoleConfig` carries `base_url` / `api_key` fields, but they are always `None` in this iteration. A future change can populate them; downstream code will not need to change.
- Per-project routing overrides (`config.json` extension). YAGNI for v1.
- Letting the LLM pick a model dynamically per request.

## Roles

```python
class Role(str, Enum):
    VISION_CRITIC   = "vision_critic"    # select_model, evaluate_result, + panel scorers
    VISION_ANALYST  = "vision_analyst"   # analyze_style, refine_prompt, + refine panel scorer
    WRITER          = "writer"           # design_cases
    PLANNER         = "planner"          # orchestrator.planner.plan
```

`str, Enum` matches the existing `Phase` pattern — JSON-serializable for free, IDE autocomplete prevents typos.

### Call-site → role map

| Call site | Role | Notes |
|---|---|---|
| `agents/analyze_style.py::analyze_style[_with_thinking]` | VISION_ANALYST | |
| `agents/select_model.py::evaluate_models[_with_thinking]` | VISION_CRITIC | |
| `agents/evaluate_result.py::evaluate_round[_with_thinking]` | VISION_CRITIC | |
| `agents/refine_prompt.py::refine_prompt[_with_thinking]` | VISION_ANALYST | |
| `agents/design_cases.py::design_cases` | WRITER | |
| `agents/refine_panel.py::refine_with_panel` | VISION_ANALYST | both propose + cross-score use this role's panel pool |
| `agents/select_model_panel.py::select_models_with_panel` | VISION_CRITIC | both propose + cross-score use this role's panel pool |
| `orchestrator/planner.py::plan` | PLANNER | |

## Environment Variables

All new envs are **optional**. Empty config keeps today's behavior.

### Single-model role routing

```bash
STYLECLAW_MODEL_VISION_CRITIC=gemini-2.5-pro-preview-05-06
STYLECLAW_MODEL_VISION_ANALYST=claude-sonnet-4-6
STYLECLAW_MODEL_WRITER=claude-sonnet-4-6
STYLECLAW_MODEL_PLANNER=gemini-2.5-flash
```

Each role's resolution order:

1. `STYLECLAW_MODEL_<ROLE>` if set
2. Existing `LLM_MODEL`

### Panel pool routing (vision_critic + vision_analyst only)

```bash
STYLECLAW_PANEL_MODELS_VISION_CRITIC=model-a,model-b,model-c
STYLECLAW_PANEL_MODELS_VISION_ANALYST=model-d,model-e,model-f
```

Each panel role's resolution order:

1. `STYLECLAW_PANEL_MODELS_<ROLE>` if set
2. Existing global `STYLECLAW_PANEL_MODELS`

Per-role panel **labels** are *not* added in v1. The global `STYLECLAW_PANEL_LABELS` continues to label the global pool; per-role panels label proposals by `model_id`. This keeps v1 lean and revisitable.

## Architecture

### New module: `src/styleclaw/core/llm_routing.py`

```python
@dataclass(frozen=True)
class RoleConfig:
    """Resolved config for one role.

    base_url / api_key are extension hooks for cross-provider routing — always
    None in v1. Plumbing them through RoleRouter today means a future change
    only edits llm_routing.py, not every call site.
    """
    model_id: str
    base_url: str | None = None
    api_key:  str | None = None


class RoleRouter:
    """Lazy-build LLMProvider instances, scoped by Role.

    Lifecycle is owned by ExecutionContext: built in cli._build_context(),
    closed in cli._close_resource(). Construction is lazy so test runs and
    read-only commands (status, rollback, ...) don't open HTTP clients.
    """
    @classmethod
    def from_env(cls) -> "RoleRouter":
        """Parse all relevant env vars. Does not instantiate any provider."""

    def get(self, role: Role) -> LLMProvider:
        """Return a single-model provider for the role. Cached after first call."""

    def get_panel(self, role: Role) -> tuple[list[LLMProvider], list[str]]:
        """Return (3 providers, 3 labels) for the role's panel pool.

        Caches after first call. Each provider is its own instance (mirrors the
        existing panel pattern — separate semaphores allow 3x concurrent calls).
        """

    async def close(self) -> None:
        """Close every provider built so far. Idempotent."""
```

Provider class selection (OpenAICompat > RunningHubLLM > Bedrock) reuses the existing precedence in `cli.py::build_llm()`. The router only overrides each instance's `model_id` constructor argument.

### Wiring: `ExecutionContext`

`cli.py::_build_context()`:

```python
# before
llm = build_llm()
ctx = ExecutionContext(client=client, llm=llm, ...)

# after
router = RoleRouter.from_env()
ctx = ExecutionContext(client=client, llm_router=router, ...)
```

`cli.py::_close_resource()` adds `await ctx.llm_router.close()`.

### Per-action injection

Every orchestrator action in `orchestrator/actions.py` and every CLI command that calls an agent gains exactly one new line:

```python
async def do_analyze(ctx, args):
    llm = ctx.llm_router.get(Role.VISION_ANALYST)
    return await analyze_style_with_thinking(llm, ...)

async def do_evaluate(ctx, args):
    if ctx.state.phase == Phase.MODEL_SELECT:
        if PANEL_MODEL_SELECT_ENABLED:
            llms, labels = ctx.llm_router.get_panel(Role.VISION_CRITIC)
            return await select_models_with_panel(llms, labels, ...)
        llm = ctx.llm_router.get(Role.VISION_CRITIC)
        return await evaluate_models_with_thinking(llm, ...)
    # STYLE_REFINE branch: also Role.VISION_CRITIC (evaluate_round is a critic)
    ...
```

**Agent function signatures do not change.** They still take a single `llm: LLMProvider` (or `llms: list[LLMProvider]` for panels). The router decides which instance to hand over.

## Validation

Extend `core/config.py::validate_env()` (already called at CLI startup, skippable via `STYLECLAW_SKIP_ENV_CHECK`):

- For each `Role`: if `STYLECLAW_MODEL_<ROLE>` is unset **and** `LLM_MODEL` is unset → emit `"no model resolvable for role <X>"`.
- For each panel-capable role (`vision_critic`, `vision_analyst`):
  - If the corresponding panel toggle (`STYLECLAW_PANEL_MODEL_SELECT` for critic, `STYLECLAW_PANEL_REFINE` for analyst) is on:
    - Effective pool = `STYLECLAW_PANEL_MODELS_<ROLE>` if set else `STYLECLAW_PANEL_MODELS`
    - Pool must have exactly 3 entries; otherwise emit `"panel pool for <X> must have exactly 3 models, got N"`.
    - If neither env is set → `"panel for <X> is enabled but no pool is configured"`.

Existing provider-credential checks (one of OpenAI-compat / RunningHub LLM / Bedrock) stay unchanged.

Validation only inspects env vars; it never builds providers. First-use construction happens lazily inside `RoleRouter.get()` / `get_panel()`.

## Artifact Recording

Add an optional `model_id: str = ""` field to each Pydantic model that gets persisted as an LLM-derived artifact:

| Model | Disk path |
|---|---|
| `StyleAnalysis` | `model-select/pass-NNN/initial-analysis.json` |
| `ModelEvaluation` | `model-select/pass-NNN/evaluation.json` |
| `RoundEvaluation` | `style-refine/pass-NNN/round-NNN/evaluation.json` |
| `PromptConfig` | `style-refine/pass-NNN/round-NNN/prompt.json` |
| `BatchConfig` | `batch-t2i/batch-NNN/cases.json`, `batch-i2i/batch-NNN/cases.json` |

The action layer fills the field post-parse via `obj.model_copy(update={"model_id": resolved_model_id})`. Agents stay model-agnostic. Default `""` (matches project convention of empty-string-over-None) so old on-disk files load without error.

`PanelResult.proposals[].model_id` already carries the per-proposal model — no change needed for panel artifacts.

`ActionPlan` (from `orchestrator/planner.py`) is transient and not persisted; no field added.

## Backwards Compatibility

Zero migration required for existing users:

| Existing config | After this change |
|---|---|
| Only `LLM_MODEL` set | All four roles resolve to `LLM_MODEL` — single client behaves identically to today |
| `LLM_MODEL` + `STYLECLAW_PANEL_REFINE=1` + `STYLECLAW_PANEL_MODELS=a,b,c` | vision_analyst panel pool falls back to global `STYLECLAW_PANEL_MODELS` — identical to today |
| Bedrock or RunningHub LLM as the active provider | Provider-selection precedence (OpenAICompat > RunningHubLLM > Bedrock) unchanged |
| On-disk artifact JSON without `model_id` | Loads fine — Pydantic defaults to `""` |

## Test Plan

### New: `tests/test_llm_routing.py`

- `from_env()` parses each of the four `STYLECLAW_MODEL_<ROLE>` envs.
- Missing role env falls back to `LLM_MODEL`.
- `from_env()` parses each panel role pool; missing role pool falls back to global `STYLECLAW_PANEL_MODELS`.
- `get(Role.X)` is idempotent: two calls return the same provider instance.
- `get_panel(Role.X)` returns 3 distinct providers with the three configured `model_id`s and 3 corresponding labels.
- `close()` is idempotent; closes every provider built so far (single + panel).
- `validate_env()` reports `"no model resolvable for role <X>"` when both role env and `LLM_MODEL` are absent.
- `validate_env()` reports a panel error when a panel toggle is on but no pool is configured.
- `validate_env()` reports a panel error when pool length ≠ 3.

### Updated: existing orchestrator + CLI tests

- `ExecutionContext.llm` → `ExecutionContext.llm_router` (mechanical rename).
- Each action test that stubbed an `LLMProvider` now stubs a minimal `RoleRouter` returning the same stub provider for `get(...)`.
- Existing panel tests (`tests/test_panel*.py` or equivalent) switch their pool source to per-role envs and verify the fallback chain.

### Smoke

- All role envs + `LLM_MODEL` unset → `validate_env()` returns a non-empty error list. CLI exits before any provider is constructed (verifies the lazy boundary).

## Documentation

- `CLAUDE.md`:
  - "Runtime Tunables" table gains the 4 `STYLECLAW_MODEL_<ROLE>` and 2 `STYLECLAW_PANEL_MODELS_<ROLE>` entries.
  - "Architecture" section gets a short "Per-role LLM routing" paragraph pointing to the call-site → role map above.
- `.env.example`: 6 commented-out example envs.
- No new top-level docs; CLAUDE.md remains the single source of truth.

## Risks and Open Questions

- **Panel pool with cross-provider models**: when v1 ships, all 3 models in a panel pool share the active provider class (OpenAI-compat / RunningHub / Bedrock). Mixing providers inside one panel pool waits for the cross-provider extension.
- **Test churn**: the `llm` → `llm_router` rename touches many tests. Mitigation: do it as a single mechanical commit before semantic changes, so review can verify the diff is rename-only.
- **Operator surprise** if `LLM_MODEL` is the only env set: behavior matches today, but the operator might assume role envs are required. CLAUDE.md and `.env.example` need to call this out explicitly.
