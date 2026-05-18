# Per-Role LLM Routing — Part 2: Validation + Pydantic `model_id` Field

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add startup validation for the new routing env vars, and give the 5 LLM-derived Pydantic models an optional `model_id` field so future code can persist which model produced each artifact.

**Architecture:** Two concerns merged because they're both small. (1) `validate_routing_env()` lives in `llm_routing.py` (owns its own validation surface); `config.validate_env()` calls it via late import to avoid the config→llm_routing circular. (2) `model_id: str = ""` is added to `StyleAnalysis`, `ModelEvaluation`, `RoundEvaluation`, `PromptConfig`, `BatchConfig`. Default `""` so old on-disk JSON loads without error.

**Tech Stack:** Same as Part 1. Pydantic v2 `_FrozenModel`, env vars via `monkeypatch.setenv` / `importlib.reload`.

**Reference:** [spec](../specs/2026-05-18-per-role-llm-routing-design.md) — sections "Validation" and "Artifact Recording".

**Dependencies:** Part 1 must be complete (`llm_routing.py` exists with `Role` enum and `RoleRouter.from_env`).

---

## File Structure

- **Modify:** `src/styleclaw/core/llm_routing.py` — add `_PANEL_TOGGLE_FOR_ROLE` mapping + `validate_routing_env()` function
- **Modify:** `src/styleclaw/core/config.py` — `validate_env()` calls `validate_routing_env()` via late import
- **Modify:** `src/styleclaw/core/models.py` — add `model_id: str = ""` to 5 frozen models
- **Modify:** `tests/core/test_llm_routing.py` — add `TestValidateRoutingEnv` class
- **Modify:** `tests/core/test_config.py` — add cases that go through `validate_env()` end to end
- **Modify:** `tests/core/test_models.py` — add `TestModelIdField` class covering the 5 models

---

## Task 1: `validate_routing_env()` — per-role missing-model check

**Files:**
- Modify: `src/styleclaw/core/llm_routing.py` (add `validate_routing_env`)
- Modify: `src/styleclaw/core/config.py` (call `validate_routing_env` from `validate_env`)
- Modify: `tests/core/test_llm_routing.py` (add `TestValidateRoutingEnvSingle` class)
- Modify: `tests/core/test_config.py` (one end-to-end case)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_llm_routing.py`:

```python
class TestValidateRoutingEnvSingle:
    """Per-role missing-model check.

    If both STYLECLAW_MODEL_<ROLE> and LLM_MODEL are unset, the role is not
    resolvable and validation must emit one error per such role.
    """

    @pytest.fixture(autouse=True)
    def _clear_envs(self, monkeypatch):
        for role in Role:
            monkeypatch.delenv(f"STYLECLAW_MODEL_{role.value.upper()}", raising=False)
            monkeypatch.delenv(f"STYLECLAW_PANEL_MODELS_{role.value.upper()}", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODELS", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        import importlib, styleclaw.core.config as cfg
        importlib.reload(cfg)
        yield

    def test_all_unset_emits_one_error_per_role(self):
        from styleclaw.core.llm_routing import Role, validate_routing_env

        errors = validate_routing_env()
        # One error per role — 4 total.
        for role in Role:
            assert any(role.value in e for e in errors), f"missing error for {role}"
        assert len(errors) >= 4

    def test_llm_model_set_satisfies_all_roles(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "fallback-model")

        from styleclaw.core.llm_routing import validate_routing_env
        assert validate_routing_env() == []

    def test_role_env_satisfies_just_that_role(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_MODEL_VISION_CRITIC", "critic-only")

        from styleclaw.core.llm_routing import Role, validate_routing_env
        errors = validate_routing_env()

        # vision_critic resolved; other 3 still missing.
        assert not any("vision_critic" in e for e in errors)
        assert any("vision_analyst" in e for e in errors)
        assert any("writer" in e for e in errors)
        assert any("planner" in e for e in errors)
```

Append to `tests/core/test_config.py` (inside `TestValidateEnv` class):

```python
    def test_validate_env_reports_missing_role_models(self, monkeypatch):
        # All provider creds set so the existing checks pass, but no LLM_MODEL
        # and no role envs — validate_env should surface the routing errors.
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        from styleclaw.core.models import Phase  # noqa: F401 (forces import order)
        import importlib, styleclaw.core.config as cfg_mod
        importlib.reload(cfg_mod)
        errs = cfg_mod.validate_env()
        assert any("vision_critic" in e for e in errs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/core/test_llm_routing.py::TestValidateRoutingEnvSingle tests/core/test_config.py::TestValidateEnv::test_validate_env_reports_missing_role_models -v`
Expected: FAIL with `ImportError: cannot import name 'validate_routing_env' from 'styleclaw.core.llm_routing'`

- [ ] **Step 3: Write the implementation**

Append to `src/styleclaw/core/llm_routing.py`:

```python
def validate_routing_env() -> list[str]:
    """Return human-readable error strings for misconfigured routing envs.

    Empty list when everything is fine. Called by ``config.validate_env()`` at
    CLI startup. Never raises — aggregates errors so the user sees them all
    in one pass.
    """
    errors: list[str] = []
    for role in Role:
        env_name = f"STYLECLAW_MODEL_{role.value.upper()}"
        if not os.getenv(env_name) and not os.getenv("LLM_MODEL"):
            errors.append(
                f"no model resolvable for role '{role.value}': "
                f"set {env_name} or LLM_MODEL"
            )
    return errors
```

Modify `src/styleclaw/core/config.py::validate_env()` — append a single line just before `return errors`:

```python
    errors.extend(validate_panel_config())

    # Per-role routing checks (vision_critic / vision_analyst / writer / planner).
    # Late import to avoid circular: llm_routing imports from config.
    from styleclaw.core.llm_routing import validate_routing_env
    errors.extend(validate_routing_env())

    return errors
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run python -m pytest tests/core/test_llm_routing.py tests/core/test_config.py -v`
Expected: PASS — new tests green; existing tests in `TestValidateEnv` that set `LLM_MODEL` or have no provider creds still behave as before.

If the existing `test_runninghub_llm_satisfies_llm_requirement` now fails because `LLM_MODEL` is unset, that's expected — patch it:

```python
    def test_runninghub_llm_satisfies_llm_requirement(self, monkeypatch) -> None:
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("RUNNINGHUB_LLM", "1")
        monkeypatch.setenv("LLM_MODEL", "dummy-model")  # NEW: satisfy routing check
        monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        assert validate_env() == []
```

Re-run to confirm green.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/llm_routing.py src/styleclaw/core/config.py \
        tests/core/test_llm_routing.py tests/core/test_config.py
git commit -m "$(cat <<'EOF'
feat(llm_routing): validate_routing_env — per-role missing-model check

validate_env now reports an error per role with neither STYLECLAW_MODEL_<ROLE>
nor LLM_MODEL set. Late import keeps config <-> llm_routing acyclic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `validate_routing_env()` — per-role panel pool length check

**Files:**
- Modify: `src/styleclaw/core/llm_routing.py` (add `_PANEL_TOGGLE_FOR_ROLE` mapping + extend `validate_routing_env`)
- Modify: `tests/core/test_llm_routing.py` (add `TestValidateRoutingEnvPanel` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_llm_routing.py`:

```python
class TestValidateRoutingEnvPanel:
    """Per-role panel pool length check.

    When a panel toggle is on, the effective pool for that role (role-specific
    env > global STYLECLAW_PANEL_MODELS) must have exactly 3 entries.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        for role in Role:
            monkeypatch.delenv(f"STYLECLAW_MODEL_{role.value.upper()}", raising=False)
            monkeypatch.delenv(f"STYLECLAW_PANEL_MODELS_{role.value.upper()}", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODELS", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        monkeypatch.setenv("LLM_MODEL", "dummy")  # silence role-missing errors
        import importlib, styleclaw.core.config as cfg
        importlib.reload(cfg)
        yield

    def _reload(self):
        import importlib, styleclaw.core.config as cfg
        importlib.reload(cfg)

    def test_refine_on_no_pool_emits_error(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        self._reload()
        from styleclaw.core.llm_routing import validate_routing_env
        errors = validate_routing_env()
        assert any("vision_analyst" in e and "no pool" in e for e in errors)

    def test_select_on_wrong_size_role_pool_emits_error(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_MODEL_SELECT", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS_VISION_CRITIC", "a,b")  # only 2
        self._reload()
        from styleclaw.core.llm_routing import validate_routing_env
        errors = validate_routing_env()
        assert any("vision_critic" in e and "exactly 3" in e for e in errors)

    def test_refine_on_global_fallback_size_3_ok(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "x,y,z")
        self._reload()
        from styleclaw.core.llm_routing import validate_routing_env
        assert validate_routing_env() == []

    def test_refine_off_no_pool_check(self, monkeypatch):
        # Panel toggle OFF — pool length is irrelevant.
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "only-one")
        self._reload()
        from styleclaw.core.llm_routing import validate_routing_env
        assert validate_routing_env() == []

    def test_role_pool_overrides_global_validation(self, monkeypatch):
        # Global pool wrong size, but role pool correct — must pass.
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "bad")  # wrong size at global
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS_VISION_ANALYST", "a1,a2,a3")
        self._reload()
        from styleclaw.core.llm_routing import validate_routing_env
        # The role override is what counts; global isn't checked when role is set.
        errors = validate_routing_env()
        assert not any("vision_analyst" in e for e in errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/core/test_llm_routing.py::TestValidateRoutingEnvPanel -v`
Expected: FAIL — the panel-related assertions trip because `validate_routing_env` doesn't yet check pools.

- [ ] **Step 3: Write the implementation**

In `src/styleclaw/core/llm_routing.py`, add the toggle mapping at module level (above `validate_routing_env`):

```python
# Which env var turns on the panel for each role.
_PANEL_TOGGLE_FOR_ROLE: dict[Role, str] = {
    Role.VISION_CRITIC: "STYLECLAW_PANEL_MODEL_SELECT",
    Role.VISION_ANALYST: "STYLECLAW_PANEL_REFINE",
}
```

Then extend `validate_routing_env`:

```python
def validate_routing_env() -> list[str]:
    """Return human-readable error strings for misconfigured routing envs."""
    errors: list[str] = []

    # 1. Per-role single-model resolvability.
    for role in Role:
        env_name = f"STYLECLAW_MODEL_{role.value.upper()}"
        if not os.getenv(env_name) and not os.getenv("LLM_MODEL"):
            errors.append(
                f"no model resolvable for role '{role.value}': "
                f"set {env_name} or LLM_MODEL"
            )

    # 2. Per-role panel pool length when the matching toggle is on.
    from styleclaw.core.config import env_truthy, PANEL_MODELS as GLOBAL_PANEL_MODELS
    for role, toggle_env in _PANEL_TOGGLE_FOR_ROLE.items():
        if not env_truthy(toggle_env):
            continue
        role_env = f"STYLECLAW_PANEL_MODELS_{role.value.upper()}"
        role_raw = os.getenv(role_env, "").strip()
        if role_raw:
            pool = [m.strip() for m in role_raw.split(",") if m.strip()]
            source = role_env
        else:
            pool = list(GLOBAL_PANEL_MODELS)
            source = "STYLECLAW_PANEL_MODELS"
        if not pool:
            errors.append(
                f"panel for '{role.value}' is enabled ({toggle_env}=1) "
                f"but no pool is configured: set {role_env} or {source}"
            )
        elif len(pool) != 3:
            errors.append(
                f"panel pool for '{role.value}' must have exactly 3 models "
                f"(got {len(pool)} from {source})"
            )

    return errors
```

Note that this duplicates the resolution logic from `RoleRouter._resolve_panel_pool`. That's fine — the validator runs without instantiating the router; refactoring to share the helper can wait until a real second caller emerges. Tag the duplication in a comment if it bothers reviewers.

The existing `config.validate_panel_config()` still runs (called by `validate_env()`) and still enforces the global-pool length. When a per-role pool is set, both run — but the per-role check is what's load-bearing; the global-pool check now just guards the "old config style" path. Test `test_role_pool_overrides_global_validation` covers this overlap (when role overrides global, the global pool is still permitted to be wrong, but it's never used).

Wait — read again: if PANEL_REFINE is on AND PANEL_MODELS is "bad" (wrong size) AND PANEL_MODELS_VISION_ANALYST is set correctly, then `validate_panel_config()` will still complain about PANEL_MODELS. That's a false positive.

Fix: update `config.validate_panel_config()` to skip the global check when both role pools are independently set OR the corresponding role pool overrides cover both panel toggles. Simplest version:

In `src/styleclaw/core/config.py::validate_panel_config()`, replace the body:

```python
def validate_panel_config() -> list[str]:
    """Return error strings if panel envs are inconsistent.

    Only checks the global STYLECLAW_PANEL_MODELS when per-role pools are
    NOT set. Per-role pool validation lives in
    llm_routing.validate_routing_env (called separately from validate_env).
    """
    errors: list[str] = []
    if not (PANEL_REFINE_ENABLED or PANEL_MODEL_SELECT_ENABLED):
        return errors
    # If both panel toggles have role-specific pools, skip the global check.
    refine_overridden = bool(os.getenv("STYLECLAW_PANEL_MODELS_VISION_ANALYST"))
    select_overridden = bool(os.getenv("STYLECLAW_PANEL_MODELS_VISION_CRITIC"))
    refine_needs_global = PANEL_REFINE_ENABLED and not refine_overridden
    select_needs_global = PANEL_MODEL_SELECT_ENABLED and not select_overridden
    if refine_needs_global or select_needs_global:
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

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run python -m pytest tests/core/test_llm_routing.py tests/core/test_config.py -v`
Expected: PASS — all new tests green; existing panel tests still green (they don't set role-specific envs, so global-check still kicks in).

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/llm_routing.py src/styleclaw/core/config.py \
        tests/core/test_llm_routing.py
git commit -m "$(cat <<'EOF'
feat(llm_routing): per-role panel pool length validation

validate_routing_env now resolves the effective pool per role (role-specific
env > global) and requires exactly 3 entries when the matching toggle is on.
validate_panel_config skips the global-pool check when role overrides exist.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `model_id: str = ""` to 5 Pydantic models

**Files:**
- Modify: `src/styleclaw/core/models.py` (5 small field additions)
- Modify: `tests/core/test_models.py` (add `TestModelIdField` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_models.py`:

```python
from styleclaw.core.models import (
    BatchConfig,
    ModelEvaluation,
    PromptConfig,
    RoundEvaluation,
    StyleAnalysis,
)


class TestModelIdField:
    """All 5 LLM-derived artifact models carry an optional model_id field.

    Default "" lets old on-disk JSON (without the field) round-trip cleanly,
    and lets action code fill it post-parse via model_copy(update=...).
    """

    def test_style_analysis_defaults_empty(self):
        assert StyleAnalysis().model_id == ""

    def test_model_evaluation_defaults_empty(self):
        assert ModelEvaluation().model_id == ""

    def test_round_evaluation_defaults_empty(self):
        assert RoundEvaluation().model_id == ""

    def test_prompt_config_defaults_empty(self):
        assert PromptConfig().model_id == ""

    def test_batch_config_defaults_empty(self):
        assert BatchConfig().model_id == ""

    def test_old_json_without_field_loads(self):
        # Simulate a legacy file written before the field was added.
        legacy_json = '{"trigger_phrase": "x"}'
        analysis = StyleAnalysis.model_validate_json(legacy_json)
        assert analysis.trigger_phrase == "x"
        assert analysis.model_id == ""

    def test_model_copy_update_sets_field(self):
        analysis = StyleAnalysis(trigger_phrase="t")
        updated = analysis.model_copy(update={"model_id": "claude-sonnet-4-6"})
        assert updated.model_id == "claude-sonnet-4-6"
        # Original frozen instance unchanged.
        assert analysis.model_id == ""

    def test_round_evaluation_with_round_still_works(self):
        # RoundEvaluation has helper methods — make sure model_id doesn't
        # interfere with them.
        ev = RoundEvaluation(round=1, model_id="m").model_copy(
            update={"evaluations": []}
        )
        assert ev.model_id == "m"
        assert ev.should_approve() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/core/test_models.py::TestModelIdField -v`
Expected: FAIL — `AttributeError: 'StyleAnalysis' object has no attribute 'model_id'`

- [ ] **Step 3: Write the implementation**

Edit `src/styleclaw/core/models.py`. For each of the 5 models, append a `model_id: str = ""` field at the bottom of its body (after the existing fields, before any methods).

Edit `StyleAnalysis` (currently ends with `trigger_variants: list[str] = Field(default_factory=list)` near line 80):

```python
class StyleAnalysis(_FrozenModel):
    # 7个核心维度
    visual_style: str = ""
    color_science: str = ""
    lighting_quality: str = ""
    material_texture: str = ""
    post_processing: str = ""
    spatial_perspective: str = ""
    dynamic_state: str = ""
    # 输出
    trigger_phrase: str = ""
    trigger_variants: list[str] = Field(default_factory=list)
    # Which LLM model produced this analysis. Filled post-parse by the action
    # layer; default "" so legacy on-disk JSON loads cleanly.
    model_id: str = ""
```

Edit `ModelEvaluation` (around line 114):

```python
class ModelEvaluation(_FrozenModel):
    evaluations: list[ModelScore] = Field(default_factory=list)
    recommendation: str = ""
    recommended_variant: str = ""
    next_direction: str = ""
    model_id: str = ""
```

Edit `PromptConfig` (around line 121):

```python
class PromptConfig(_FrozenModel):
    round: int = 0
    trigger_phrase: str = ""
    model_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    derived_from: str = ""
    adjustment_note: str = ""
    model_id: str = ""
```

Edit `RoundEvaluation` (around line 138). Add the field **after** `next_direction` and **before** the methods (`should_approve`, `needs_human`):

```python
class RoundEvaluation(_FrozenModel):
    round: int = 0
    evaluations: list[RoundScore] = Field(default_factory=list)
    recommendation: str = ""
    next_direction: str = ""
    model_id: str = ""

    def should_approve(self) -> bool:
        # ... existing body unchanged ...
```

Edit `BatchConfig` (around line 167):

```python
class BatchConfig(_FrozenModel):
    batch: int = 0
    trigger_phrase: str = ""
    cases: list[BatchCase] = Field(default_factory=list)
    model_id: str = ""
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run python -m pytest tests/core/test_models.py -v`
Expected: PASS — all `TestModelIdField` tests green, all existing tests still green.

Also run the broader suite to confirm nothing else relies on these models being closed-form:

Run: `uv run python -m pytest tests/ -x -q`
Expected: PASS — full suite green. Pydantic's extra-fields default is "ignore" for these `_FrozenModel`s, so existing JSON parsers don't break.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/models.py tests/core/test_models.py
git commit -m "$(cat <<'EOF'
feat(models): add model_id field to LLM-derived artifact models

StyleAnalysis, ModelEvaluation, RoundEvaluation, PromptConfig, BatchConfig
each get a default-empty model_id field. Action code (Part 3) will fill it
post-parse so each artifact records which LLM produced it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wrap-up — full test sweep

**Files:** None modified. Verification only.

- [ ] **Step 1: Full test suite**

Run: `uv run python -m pytest tests/ -v`
Expected: PASS — all tests green. New tests added in Part 2: ~8 in `TestValidateRoutingEnvSingle` / `TestValidateRoutingEnvPanel` + ~8 in `TestModelIdField` + 1 patched test in `TestValidateEnv`.

- [ ] **Step 2: Coverage sanity check**

Run: `uv run python -m pytest tests/ --cov=src/styleclaw/core/llm_routing --cov=src/styleclaw/core/models --cov-report=term-missing 2>&1 | tail -20`
Expected: high coverage on `llm_routing.py` and `models.py`. Missing lines should be limited to the `RoleRouter.get()` / `get_panel()` / `close()` branches that are exercised in Part 1 tests (already covered there).

- [ ] **Step 3: Done — no further commit needed**

Part 2 is complete. The routing module now has validation, the 5 artifact models carry `model_id`, and nothing downstream uses either yet. Part 3 (integration) is the next step.
