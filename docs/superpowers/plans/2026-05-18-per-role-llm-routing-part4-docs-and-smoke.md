# Per-Role LLM Routing — Part 4: Docs + End-to-End Smoke

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the new routing knobs in the project's user-facing docs and verify end-to-end that role envs actually take effect.

**Architecture:** Three small documentation edits + one functional smoke test. No production source changes.

**Tech Stack:** Markdown editing for CLAUDE.md and `.env.example`. The smoke test is a single pytest case that constructs a `RoleRouter` with role envs set and confirms each role hands back a provider with the right `_model_id`.

**Reference:** [spec](../specs/2026-05-18-per-role-llm-routing-design.md) — sections "Environment Variables" and "Documentation".

**Dependencies:** Parts 1, 2, 3 must be complete (the module, the validation, and the integration must all be in place).

---

## File Structure

- **Modify:** `CLAUDE.md` — Runtime Tunables table + Architecture note
- **Modify:** `.env.example` — 6 commented examples
- **Create:** `tests/test_routing_smoke.py` — single end-to-end smoke

---

## Task 1: Update `CLAUDE.md` — Runtime Tunables table

**Files:**
- Modify: `CLAUDE.md` (one table addition)

- [ ] **Step 1: Find the "Runtime Tunables" section**

The table begins with `| Variable | Default | Purpose |` and lists `STYLECLAW_*` envs in order. Find the last `STYLECLAW_PANEL_LABELS` row.

- [ ] **Step 2: Append the new rows**

Insert directly **after** the row for `STYLECLAW_PANEL_LABELS`:

```markdown
| `STYLECLAW_MODEL_VISION_CRITIC` | unset | Model ID for the **vision_critic** role (`select_model` + `evaluate_result`). Falls back to `LLM_MODEL` when unset. |
| `STYLECLAW_MODEL_VISION_ANALYST` | unset | Model ID for the **vision_analyst** role (`analyze_style` + `refine_prompt`). Falls back to `LLM_MODEL` when unset. |
| `STYLECLAW_MODEL_WRITER` | unset | Model ID for the **writer** role (`design_cases`). Falls back to `LLM_MODEL` when unset. |
| `STYLECLAW_MODEL_PLANNER` | unset | Model ID for the **planner** role (orchestrator `plan()` calls). Falls back to `LLM_MODEL` when unset. |
| `STYLECLAW_PANEL_MODELS_VISION_CRITIC` | unset | 3 comma-separated model IDs for the vision_critic panel pool when `STYLECLAW_PANEL_MODEL_SELECT=1`. Falls back to `STYLECLAW_PANEL_MODELS`. |
| `STYLECLAW_PANEL_MODELS_VISION_ANALYST` | unset | 3 comma-separated model IDs for the vision_analyst panel pool when `STYLECLAW_PANEL_REFINE=1`. Falls back to `STYLECLAW_PANEL_MODELS`. |
```

- [ ] **Step 3: Verify rendering**

Open `CLAUDE.md` in your editor and confirm the table still renders as a markdown table (column count matches; no stray `|` characters break the grid).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(CLAUDE.md): document per-role LLM routing envs in Runtime Tunables

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Update `CLAUDE.md` — Architecture note

**Files:**
- Modify: `CLAUDE.md` (one short subsection)

- [ ] **Step 1: Find the "Conventions" section**

Toward the bottom of `CLAUDE.md`, there's a bulleted "Conventions" list with entries like `**Variant routing**: ...` and `**Prompt building**: ...`. We'll add one more entry in the same style.

- [ ] **Step 2: Append a "Per-role LLM routing" bullet**

Right after the existing `**Variant routing**` bullet, insert:

```markdown
- **Per-role LLM routing**: Every LLM call site is tagged with one of four roles — `vision_critic` (select_model + evaluate_result + their panel scorers), `vision_analyst` (analyze_style + refine_prompt + refine panel scorer), `writer` (design_cases), `planner` (orchestrator.planner.plan). `core.llm_routing.RoleRouter.from_env()` resolves each role to a `model_id` via `STYLECLAW_MODEL_<ROLE>` with fallback to `LLM_MODEL`. Panel pools use `STYLECLAW_PANEL_MODELS_<ROLE>` with fallback to the global `STYLECLAW_PANEL_MODELS`. The router is built once per CLI invocation in `cli._build_context` and disposed via `_close_resource`. Each persisted artifact (`initial-analysis.json`, `evaluation.json`, `prompt.json`, `cases.json`) records the `model_id` that produced it. See `docs/superpowers/specs/2026-05-18-per-role-llm-routing-design.md`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(CLAUDE.md): add Per-role LLM routing convention

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update `.env.example` with role samples

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Append routing examples**

At the end of `.env.example`, add a new commented block:

```bash

# --- Per-role LLM routing (optional; all envs are commented out by default) ---
#
# Each role falls back to LLM_MODEL when its env is unset. Set only the roles
# you want to override. Same provider/gateway as LLM_MODEL — only the model_id
# differs per role in this iteration.
# STYLECLAW_MODEL_VISION_CRITIC=gemini-2.5-pro-preview-05-06
# STYLECLAW_MODEL_VISION_ANALYST=claude-sonnet-4-6
# STYLECLAW_MODEL_WRITER=claude-sonnet-4-6
# STYLECLAW_MODEL_PLANNER=gemini-2.5-flash

# Panel pools per role (used when the matching panel toggle is on). Each pool
# must list exactly 3 model IDs. Falls back to STYLECLAW_PANEL_MODELS.
# STYLECLAW_PANEL_MODELS_VISION_CRITIC=model-a,model-b,model-c
# STYLECLAW_PANEL_MODELS_VISION_ANALYST=model-d,model-e,model-f
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "$(cat <<'EOF'
docs(.env.example): show per-role routing env samples

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: End-to-end smoke test

**Files:**
- Create: `tests/test_routing_smoke.py`

This test stands on top of the full integration. It builds a `RoleRouter.from_env()` with role envs set, then asserts that each role's resolved provider carries the expected `_model_id`. Catches plumbing regressions (e.g. someone renames a Role enum value but forgets the env-var lookup string).

- [ ] **Step 1: Write the test**

Create `tests/test_routing_smoke.py`:

```python
"""End-to-end smoke for per-role LLM routing.

Exercises the full env-var → RoleRouter → provider pipeline without making
any network call. Provider construction is real (creates an httpx client),
but `close()` tears it down promptly via the existing teardown path.
"""
from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip all routing envs before each test so leaks from other tests
    don't poison the smoke."""
    from styleclaw.core.llm_routing import Role

    for role in Role:
        monkeypatch.delenv(f"STYLECLAW_MODEL_{role.value.upper()}", raising=False)
        monkeypatch.delenv(f"STYLECLAW_PANEL_MODELS_{role.value.upper()}", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("STYLECLAW_PANEL_MODELS", raising=False)
    monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
    monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
    # Force OpenAI-compat path; no real request will be made.
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:9/v1")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "smoke-test-key")
    monkeypatch.delenv("RUNNINGHUB_LLM", raising=False)
    yield


def test_smoke_each_role_resolves_to_its_env_model():
    """Set a distinct model per role; verify provider._model_id matches."""
    import os
    os.environ["STYLECLAW_MODEL_VISION_CRITIC"] = "model-critic"
    os.environ["STYLECLAW_MODEL_VISION_ANALYST"] = "model-analyst"
    os.environ["STYLECLAW_MODEL_WRITER"] = "model-writer"
    os.environ["STYLECLAW_MODEL_PLANNER"] = "model-planner"
    try:
        from styleclaw.core.llm_routing import Role, RoleRouter
        router = RoleRouter.from_env()
        try:
            assert router.get(Role.VISION_CRITIC)._model_id == "model-critic"
            assert router.get(Role.VISION_ANALYST)._model_id == "model-analyst"
            assert router.get(Role.WRITER)._model_id == "model-writer"
            assert router.get(Role.PLANNER)._model_id == "model-planner"
        finally:
            asyncio.run(router.close())
    finally:
        for env in ("STYLECLAW_MODEL_VISION_CRITIC", "STYLECLAW_MODEL_VISION_ANALYST",
                    "STYLECLAW_MODEL_WRITER", "STYLECLAW_MODEL_PLANNER"):
            os.environ.pop(env, None)


def test_smoke_fallback_to_llm_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "global-default")

    from styleclaw.core.llm_routing import Role, RoleRouter
    router = RoleRouter.from_env()
    try:
        for role in Role:
            assert router.get(role)._model_id == "global-default", role
    finally:
        asyncio.run(router.close())


def test_smoke_panel_pool_resolves_per_role(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "default")  # silences single-model validation
    monkeypatch.setenv("STYLECLAW_PANEL_MODELS_VISION_CRITIC", "c1,c2,c3")
    monkeypatch.setenv("STYLECLAW_PANEL_MODELS_VISION_ANALYST", "a1,a2,a3")

    from styleclaw.core.llm_routing import Role, RoleRouter
    router = RoleRouter.from_env()
    try:
        critic_llms, _ = router.get_panel(Role.VISION_CRITIC)
        analyst_llms, _ = router.get_panel(Role.VISION_ANALYST)
        assert [p._model_id for p in critic_llms] == ["c1", "c2", "c3"]
        assert [p._model_id for p in analyst_llms] == ["a1", "a2", "a3"]
    finally:
        asyncio.run(router.close())


def test_smoke_validate_env_passes_with_all_role_envs(monkeypatch):
    """A realistic happy-path config: role envs set, no panel mode, all
    provider creds present. validate_env should return [] (no errors)."""
    monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
    monkeypatch.setenv("STYLECLAW_MODEL_VISION_CRITIC", "x")
    monkeypatch.setenv("STYLECLAW_MODEL_VISION_ANALYST", "y")
    monkeypatch.setenv("STYLECLAW_MODEL_WRITER", "z")
    monkeypatch.setenv("STYLECLAW_MODEL_PLANNER", "w")

    import importlib, styleclaw.core.config as cfg
    importlib.reload(cfg)
    assert cfg.validate_env() == []
```

- [ ] **Step 2: Run the smoke**

Run: `uv run python -m pytest tests/test_routing_smoke.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 3: Full suite verification**

Run: `uv run python -m pytest tests/ -q`
Expected: full green. The smoke test is additive; nothing else should break.

- [ ] **Step 4: Commit**

```bash
git add tests/test_routing_smoke.py
git commit -m "$(cat <<'EOF'
test: end-to-end smoke for per-role LLM routing

Verifies env -> RoleRouter -> provider mapping for all 4 roles, the
LLM_MODEL fallback, per-role panel pools, and validate_env's happy path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final review

**Files:** None modified. Verification only.

- [ ] **Step 1: Confirm spec coverage**

Open the spec one more time: [docs/superpowers/specs/2026-05-18-per-role-llm-routing-design.md](../specs/2026-05-18-per-role-llm-routing-design.md). Walk through each section and tick off:

- "Roles" — implemented in `core.llm_routing.Role` ✓
- "Environment Variables" — `STYLECLAW_MODEL_<ROLE>` × 4, `STYLECLAW_PANEL_MODELS_<ROLE>` × 2; documented in CLAUDE.md and `.env.example` ✓
- "Architecture / RoleRouter API" — `from_env`, `get`, `get_panel`, `close` all present ✓
- "Wiring: ExecutionContext" — `llm_router` field; cli builds router ✓
- "Per-action injection" — all 5 LLM call sites (analyze, evaluate ×2, refine, design_cases, planner) routed ✓
- "Validation" — `validate_routing_env` + panel pool length checks ✓
- "Artifact Recording" — `model_id` field on 5 Pydantic models; populated by each action ✓
- "Backwards Compatibility" — covered by smoke test + per-task running on `LLM_MODEL` only ✓
- "Test Plan" — every bullet in the spec has a corresponding test ✓
- "Documentation" — CLAUDE.md table + convention, `.env.example` samples ✓

- [ ] **Step 2: Run the full suite one last time**

Run: `uv run python -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: green summary.

- [ ] **Step 3: Manual CLI smoke (optional but recommended)**

If you have an OpenAI-compat key handy:

```bash
# Without any role envs — behaves exactly like before, all roles use LLM_MODEL.
uv run styleclaw status

# With role envs — every artifact records the model that produced it.
STYLECLAW_MODEL_PLANNER=gemini-2.5-flash uv run styleclaw status
```

Expected: identical output in both cases for a read-only command. Routing differences only matter for commands that actually call LLMs.

- [ ] **Step 4: Done — feature shipped**

The four parts together complete the per-role LLM routing feature. From this point on the operator can set any of the role envs to specialize models per call site, or leave them all unset and keep today's single-`LLM_MODEL` behavior.
