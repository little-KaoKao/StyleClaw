# StyleClaw

AI-powered style trigger word exploration system for image generation. Given a set of reference images representing an IP's visual style, StyleClaw uses LLM analysis + batch image generation to iteratively discover and validate a concise "trigger phrase" that reliably reproduces that style across diverse subjects.

## Quick Start

```bash
# 1. Install dependencies (requires Python 3.11+ and uv)
uv sync

# 2. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your keys (see "Environment Variables" below)

# 3. Verify setup
uv run styleclaw --help
uv run python -m pytest tests/ -v
```

## Environment Variables

Copy `.env.example` to `.env` and fill in. Choose ONE LLM provider path (precedence below):

| Variable | Required | Description |
|----------|----------|-------------|
| `RUNNINGHUB_API_KEY` | Yes | RunningHub API key for image generation |
| **Option A: OpenAI-compatible (recommended, e.g. gptproto.com)** | | |
| `OPENAI_COMPAT_API_KEY` | A | API key for the OpenAI-compatible provider |
| `OPENAI_COMPAT_BASE_URL` | A | Provider base URL (e.g. `https://api.gptproto.com/v1`) |
| `LLM_MODEL` | A | Model ID (e.g. `gemini-2.5-pro-preview-05-06`) |
| **Option B: AWS Bedrock (legacy)** | | |
| `AWS_REGION` | B | AWS region (e.g. `us-east-1`) |
| `AWS_BEARER_TOKEN_BEDROCK` | B | Bearer token sent as `Authorization: Bearer ...` to a pre-configured proxy/gateway that forwards to Bedrock after SigV4 signing. **Not a standard AWS credential.** Treat as a secret. |
| `LLM_MODEL` | B | Bedrock model ID (e.g. `anthropic.Codex-sonnet-4-20250514`) |
| **Option C: RunningHub LLM** | | |
| `RUNNINGHUB_LLM` | C | Set to `1` / `true` / `yes` / `on` to use RunningHub's OpenAI-compatible LLM API (same key as image gen). |
| `RUNNINGHUB_LLM_BASE_URL` | C | Default `https://llm.runninghub.cn/v1`. |
| `LLM_MODEL` | C | e.g. `rh-llm-a/rh-c-o-47` (defaults in code if unset). |
| `RUNNINGHUB_LLM_REASONING_EFFORT` | C | Optional; default `high` for `invoke_with_thinking`; use `off` to omit `reasoning_effort` in the request body. |

**Precedence:** `OPENAI_COMPAT_API_KEY` → OpenAI-compat provider; else if `RUNNINGHUB_LLM` is truthy → RunningHub LLM; else → Bedrock.

### Runtime Tunables (optional)

All defined in `core/config.py` via `_int_env` / `_float_env`. Invalid values raise immediately on import.

| Variable | Default | Purpose |
|----------|--------:|---------|
| `STYLECLAW_DATA_ROOT` | `data/projects` | Root directory for all project storage (also honored in tests via monkeypatch) |
| `STYLECLAW_LOG_LEVEL` | `INFO` | Default log level (e.g. `DEBUG`, `WARNING`); `--verbose / -v` is a per-invocation shortcut to DEBUG |
| `STYLECLAW_SKIP_ENV_CHECK` | unset | Truthy value disables the `validate_env()` gate at CLI startup (for offline tooling) |
| `STYLECLAW_MAX_ROUNDS` | `5` | Cap on auto refine rounds in STYLE_REFINE |
| `STYLECLAW_CONCURRENCY` | `5` | Async semaphore for image-gen submissions |
| `STYLECLAW_LLM_CONCURRENCY` | `4` | Async semaphore for parallel LLM calls |
| `STYLECLAW_TASK_TIMEOUT` | `300` | Per-task poll timeout (seconds) |
| `STYLECLAW_POLL_INTERVAL` | `3` | Inner poll loop interval (seconds) — single task |
| `STYLECLAW_POLL_MAX_CONSEC_FAIL` | `5` | Consecutive poll failures before giving up on a task |
| `STYLECLAW_ORCH_POLL_INTERVAL` | `30` | Orchestrator-driven outer poll cycle interval (seconds) |
| `STYLECLAW_MAX_POLL_CYCLES` | `60` | Cap on orchestrator poll cycles before reporting timeout |
| `STYLECLAW_STREAM_DISPLAY` | `1` | Print LLM response deltas to stdout as they arrive (`  ↓ ...`); set to `0` / `false` to silence (e.g. in CI) |
| `STYLECLAW_LLM_WRITE_TIMEOUT` | `300` | httpx write timeout (seconds) for LLM requests. Evaluate POSTs many base64 images at once — raise on slow upload links (e.g. `600`). |
| `STYLECLAW_LLM_READ_TIMEOUT` | `300` | httpx read timeout (seconds) for LLM streaming responses. |
| `STYLECLAW_LLM_CONNECT_TIMEOUT` | `30` | httpx connect timeout (seconds) for LLM connections. |

## Tech Stack

- **Language**: Python 3.11+ with uv package manager
- **HTTP**: httpx (async) — RunningHub client and LLM provider
- **LLM**: OpenAI-compatible API, RunningHub LLM (`llm.runninghub.cn`), or AWS Bedrock (legacy), all via httpx
- **Models**: Pydantic v2 (immutable state via `model_copy(update=...)`)
- **CLI**: Typer
- **Reports**: Jinja2 HTML templates
- **Image**: Pillow (resize to 1024px long-edge, WebP encoding)
- **Config**: python-dotenv

## Architecture

### State Machine Pipeline

```
INIT → MODEL_SELECT → STYLE_REFINE → BATCH_T2I → BATCH_I2I → COMPLETED
```

Each phase has a fixed set of commands. The state machine enforces valid transitions — you cannot skip phases.

### Directory Structure

```
src/styleclaw/
├── core/           # Pure logic, no IO
│   ├── models.py           # All Pydantic models (Phase, ProjectState, ActionPlan, LoopConfig, ProjectConfig, TaskRecord, RoundEvaluation, BatchCase, ...)
│   ├── state_machine.py    # Phase transitions: advance() / rollback() + TRANSITIONS table + per-phase hints
│   ├── prompt_builder.py   # Build API params: trigger + character_desc concatenation, aspect ratio → width/height
│   ├── case_generator.py   # Generate 100 empty BatchCase skeletons across 10 categories
│   ├── image_utils.py      # resize_for_llm(), encode_image_for_llm(), verify_ref_image(), build_image_blocks_async()
│   ├── text_utils.py       # clean_json(), parse_llm_response(), sanitize_braces() — strip markdown fences, validate via Pydantic
│   ├── time_utils.py       # utcnow_iso() — single source of UTC ISO timestamps
│   ├── checkpoint.py       # Atomic JSON KV checkpoint (.checkpoint_<phase>.json) for resumable batch ops
│   └── config.py           # Env-driven constants (MAX_AUTO_ROUNDS, CONCURRENCY_LIMIT, …) + validate_env() + env_truthy()
├── storage/
│   ├── project_store.py    # All filesystem persistence (JSON read/write under DATA_ROOT)
│   └── image_store.py      # download_image() with retry, list_output_images() — handles png/jpg/webp/gif
├── orchestrator/   # Natural-language plan-and-execute layer (drives `styleclaw run`)
│   ├── actions.py          # ACTION_REGISTRY: maps action name → ActionDef(fn, needs_client, needs_llm, requires_confirmation). PHASE_ACTIONS: per-phase whitelist.
│   ├── planner.py          # plan(): LLM(plan.md) → ActionPlan, with one auto-retry on disallowed action names; handles no-project case via _plan_init_only
│   ├── executor.py         # execute(): runs steps sequentially, supports loops, on_confirm / on_step_start / on_step_done hooks, _should_continue_loop checks evaluation pass/needs_human
│   └── suggestions.py      # suggest_next_steps(): phase-aware example `styleclaw run "..."` lines shown after `run` and in `status`
├── providers/
│   ├── llm/
│   │   ├── base.py         # LLMProvider Protocol — invoke() + invoke_with_thinking()
│   │   ├── openai_compat.py # OpenAICompatProvider — gptproto & similar (priority if key set)
│   │   ├── runninghub_llm.py # RunningHubLLMProvider — llm.runninghub.cn, RUNNINGHUB_LLM=1
│   │   ├── bedrock.py      # BedrockProvider — legacy AWS Bedrock fallback
│   │   └── prompts/        # Markdown prompt templates: analyze.md / select_model.md / evaluate.md / refine.md / design_cases.md / plan.md
│   └── runninghub/
│       ├── client.py       # RunningHubClient — async httpx for image gen API
│       ├── models.py       # MODEL_REGISTRY: mj-v7, niji7, nb2, seedream, gpt-image-2 + SrefMode (param|prompt)
│       ├── tasks.py        # submit_task() with 3x retry, poll_task() with timeout
│       └── upload.py       # Upload files to RunningHub
├── agents/          # LLM-powered creative work — each has plain + `*_with_thinking` variants
│   ├── analyze_style.py    # Analyze ref images → StyleAnalysis + initial trigger phrase
│   ├── select_model.py     # Compare models' outputs → ModelEvaluation (recommendation + recommended_variant)
│   ├── evaluate_result.py  # Score round results on 5 dimensions → RoundEvaluation (should_approve / needs_human)
│   ├── refine_prompt.py    # Refine trigger phrase based on evaluation feedback
│   └── design_cases.py     # Design 100 diverse test case descriptions, accepts optional `feedback` string
├── scripts/         # Mechanical work — no LLM calls
│   ├── init_project.py     # Create project dir, copy refs, upload to RunningHub
│   ├── generate.py         # Submit image gen tasks (model-select or style-refine); model-select supports `models=` filter
│   ├── poll.py             # Poll task status, download completed images; exposes retry_failed_model_select / retry_failed_style_refine
│   ├── batch_submit.py     # Submit batch t2i/i2i tasks (100 cases), checkpointed via core.checkpoint
│   ├── report.py           # Generate Jinja2 HTML reports
│   └── migrate.py          # Migrate pre-pass layout to pass-001 layout
├── reports/templates/      # Jinja2 HTML templates: model_select / style_refine / batch_t2i / batch_i2i
└── cli.py           # Typer CLI — all user-facing commands; thin wrapper that delegates to ACTION_REGISTRY via _run_action()
```

### Data Storage Layout

```
data/projects/<project-name>/
├── config.json                          # ProjectConfig
├── state.json                           # ProjectState (current phase, round, batch, selected models)
├── refs/                                # Reference images + upload records
│   ├── *.png / *.jpg
│   └── uploads.json                     # UploadRecord[]
├── model-select/
│   └── pass-001/
│       ├── initial-analysis.json        # StyleAnalysis from LLM
│       ├── evaluation.json              # ModelEvaluation from LLM
│       ├── report.html
│       └── results/<model-id>/<variant>/
│           ├── task.json                # TaskRecord
│           └── output-*.png
├── style-refine/
│   └── pass-001/
│       └── round-001/
│           ├── prompt.json              # PromptConfig (trigger phrase for this round)
│           ├── evaluation.json          # RoundEvaluation
│           ├── report.html
│           └── results/<model-id>/
│               ├── task.json
│               └── output-*.png
├── batch-t2i/
│   └── batch-001/
│       ├── cases.json                   # BatchConfig (100 cases with descriptions)
│       ├── report.html
│       └── results/<case-id>/
│           ├── task.json
│           └── output-*.png
└── batch-i2i/
    └── batch-001/
        ├── source-images/
        ├── uploads.json
        ├── cases.json
        ├── report.html
        └── results/<case-id>/
            ├── task.json
            └── output-*.png
```

## CLI Commands & Full Pipeline Walkthrough

### Natural-Language Orchestrator (`styleclaw run`)

Above the per-phase CLI commands sits a plan-and-execute layer that turns free-form Chinese/English intent into an `ActionPlan` and runs it step by step.

```bash
# Pick a project (with multiple projects under DATA_ROOT, -p is required)
uv run styleclaw run "<intent>" -p <project> [--yes] [--dry-run]
```

Flow inside `cli.run`:

1. **`orchestrator.planner.plan(llm, project, intent)`** loads `state.json` + `config.json`, computes the available action whitelist for the current phase (from `PHASE_ACTIONS`, extended with the next phase's actions when the current phase is in `CROSS_PHASE_PLANNABLE_FROM = {INIT, STYLE_REFINE, BATCH_T2I, BATCH_I2I}` — `MODEL_SELECT` is deliberately excluded so `select-model` never gets autoplanned across a boundary). `GATED_CROSS_PHASE_ACTIONS = {select-model, approve, retest-models, add-refs}` are never lifted in via the cross-phase mechanism. The prompt at `providers/llm/prompts/plan.md` is rendered with state context; the LLM returns an `ActionPlan` JSON which gets one auto-retry if any step name is disallowed.
2. **`executor.display_plan()`** prints the plan summary, ordered steps, optional loop, and a `stop_summary` line ("停在哪").
3. After user confirmation (skippable with `--yes`; `--dry-run` exits here), **`executor.execute()`** runs steps sequentially:
   - Validates `needs_client` / `needs_llm` against the `ExecutionContext`.
   - For `requires_confirmation` actions (`init`, `select-model`, `add-refs`), invokes the `on_confirm` callback in `cli.py` to gather missing fields (`ref_dir`, models+variant, `image_dir`) or let the user override LLM-recommended models.
   - If `plan.loop` is set, after step `end_step` the executor calls `_should_continue_loop(ctx)`: loads the latest `RoundEvaluation`; stops when it `should_approve()` (all dims ≥ 7.0, total ≥ 7.5), or when `needs_human()` (any dim < 5) prints a `!! needs_human` hint identifying the weakest dimension with a suggested redirection phrase; otherwise loops back to `start_step` until `max_iterations`.
4. **`suggestions.suggest_next_steps()`** prints 1–5 example follow-up `styleclaw run "..." -p ...` lines, picked from a per-phase dispatch table.

### ACTION_REGISTRY

All plannable / CLI-callable actions are defined in `orchestrator/actions.py`. Each entry is an `ActionDef(fn, needs_client, needs_llm, requires_confirmation)`.

| Action | needs_client | needs_llm | requires_confirmation | What it does |
|--------|:---:|:---:|:---:|---|
| `init` | ✓ | ✗ | ✓ | Create project from a directory of refs (args: `ref_dir`, `ip_info`, `description`, `force`) |
| `analyze` | ✗ | ✓ | ✗ | LLM extracts style + trigger; advances INIT → MODEL_SELECT |
| `generate` | ✓ | ✗ | ✗ | Submit gen tasks; in MODEL_SELECT accepts `models` (list or comma-string) filter, `force`. `force=true` refuses when the target pass/round already has SUCCESS data (prevents silent overwrite) |
| `poll` | ✓ | ✗ | ✗ | Wait until terminal; auto-retries FAILED tasks once for model-select / style-refine |
| `evaluate` | ✗ | ✓ | ✗ | LLM scoring (MODEL_SELECT compares variants; STYLE_REFINE 5-dim) |
| `select-model` | ✗ | ✗ | ✓ | Requires `models` (comma string) + optional `variant` ∈ {prompt-sref, prompt-only}; advances MODEL_SELECT → STYLE_REFINE |
| `refine` | ✗ | ✓ | ✗ | One round (capped at `MAX_AUTO_ROUNDS`); auto-skips round numbers that already have `prompt.json` (non-destructive rollback support); optional `direction` |
| `approve` | ✗ | ✗ | ✗ | `target` ∈ {batch-t2i, completed} |
| `design-cases` | ✗ | ✓ | ✗ | Always creates a new batch (`current_batch + 1`); optional `feedback` is folded into the design prompt |
| `batch-submit` | ✓ | ✗ | ✗ | Submits pending cases; optional `model` override |
| `report` | ✗ | ✗ | ✗ | HTML report for current batch |
| `retest-models` | ✗ | ✗ | ✗ | Opens MODEL_SELECT pass-(N+1) seeded with current trigger; preserves all prior passes |
| `back-to-t2i` | ✗ | ✗ | ✗ | BATCH_I2I → BATCH_T2I |
| `set-sref` | ✗ | ✗ | ✗ | `index` (int, 0-based). In MODEL_SELECT, auto-bumps to a new pass (copying analysis forward) when the current pass already has any SUCCESS task — previous pass preserved on disk. Outside MODEL_SELECT, just updates `sref_index` in place |
| `set-pass` | ✗ | ✗ | ✗ | `pass_num` (int ≥ 1) |
| `add-refs` | ✓ | ✗ | ✓ | `image_dir`; advances BATCH_T2I → BATCH_I2I if needed; appends to i2i uploads |

`PHASE_ACTIONS` controls which actions are surfaced to the planner for each phase. `cli.py` exposes one-shot CLI wrappers that go through `_run_action()` → `ExecutionContext` → the same registry, so behavior is identical between manual and orchestrated modes.

### Global CLI flags

| Flag | Scope | Description |
|------|-------|-------------|
| `--verbose / -v` | global | Lift the root logger to DEBUG for this invocation |
| `--show-thinking / --no-show-thinking` | `analyze`, `evaluate`, `refine`, `run` | Capture LLM reasoning content (default on); saved as `*.thinking.md` siblings of the JSON output |
| `--thinking-budget <int>` | same as above | Forward `thinking_budget` to `invoke_with_thinking`; default `5000` |
| `--dry-run` | `run`, `generate`, `batch-submit` | Print the plan / estimated task counts and exit without side effects |

The `@app.callback()` in `cli.py` skips env validation for read-only / local commands (`status`, `rollback`, `set-sref`, `set-pass`, `migrate`, `archive`, `clean`) so they keep working without API keys.

### Phase 1: INIT

```bash
# Initialize project with reference images and IP description
styleclaw init <project-name> \
  --ref img1.png --ref img2.png --ref img3.png \
  --info "Spider-Verse animation style" \
  --desc "Testing Spider-Verse visual style extraction"

# Analyze reference images (LLM extracts style dimensions + initial trigger phrase)
styleclaw analyze <project-name>
# → Phase advances to MODEL_SELECT
```

### Phase 2: MODEL_SELECT

Each model is tested under two conditions (variants):
- **prompt-only**: Only the trigger phrase, no style reference image
- **prompt-sref**: Trigger phrase + style reference image (MJ via `--sref` param, others via `参考图1的风格：` prompt prefix + `imageUrls`)

If prompt-only is sufficient (total ≥ 7.0), prefer it for flexibility.

```bash
# Generate test images across all models (2 variants × 2 genders each)
styleclaw generate <project-name>
# → Skips SUCCESS tasks, retries FAILED tasks automatically

# Force re-submit (only useful when current pass has NO SUCCESS data;
# refuses with an error if any SUCCESS task already exists in the current pass)
styleclaw generate <project-name> --force

# Limit submission to a subset of models (MODEL_SELECT only)
styleclaw generate <project-name> --models mj-v7,niji7

# Preview what would be submitted (no API calls)
styleclaw generate <project-name> --dry-run

# Poll until images are ready (auto-retries FAILED once, then tolerates partial failure)
styleclaw poll <project-name>

# LLM evaluates which models best reproduce the style (compares both variants)
styleclaw evaluate <project-name>
# → Outputs scores + HTML report with sref image and test subject descriptions
# → recommended_variant ∈ {prompt-sref, prompt-only} is propagated to STYLE_REFINE

# Confirm model selection (advances to STYLE_REFINE)
# --variant locks the prompt-construction mode used during refinement
styleclaw select-model <project-name> --models mj-v7 --variant prompt-only
```

**Changing the sref image (auto-bumps the pass):**
```bash
# List ref images (0-based index)
styleclaw status <project-name>

# Switch sref to a different ref image (e.g. ref-003 = index 2).
# If the current pass already has SUCCESS results, this auto-bumps to the
# next pass (analysis is copied forward; previous pass-NNN is preserved
# untouched on disk). If the current pass is empty, sref_index changes in place.
styleclaw set-sref <project-name> 2

# Generate with the new sref in the (possibly new) current pass.
# No --force needed: the new pass starts empty so generate just submits everything.
styleclaw generate <project-name>
styleclaw poll <project-name>
styleclaw evaluate <project-name>
```

**Re-running model selection with the SAME sref (explicit new pass):**
```bash
# Rollback only changes the state pointer; it does NOT touch pass data.
# After rollback, generate still targets state.current_model_select_pass —
# if that pass has SUCCESS data, generate (no flag) skips it and generate --force
# refuses. To get a fresh comparison set, use retest-models.
styleclaw rollback <project-name> --to MODEL_SELECT

# Open a fresh pass (pass-NNN+1), copying analysis forward. Same sref.
styleclaw retest-models <project-name>
styleclaw generate <project-name>
styleclaw poll <project-name>
styleclaw evaluate <project-name>

# Navigate between existing passes (read-only / for inspection):
styleclaw set-pass <project-name> 2

# If a bad pass was created and you want to discard it before continuing:
rm -rf data/projects/<name>/model-select/pass-003
styleclaw set-pass <project-name> 2   # switch active pass back
```

**When to use which (MODEL_SELECT mental model):**

A `pass-NNN` directory is a frozen experiment with one fixed `(analysis, sref)` tuple. The rules:

| Goal | Command |
|------|---------|
| Submit the missing/failed tasks in the current pass | `generate` (no flag) |
| Re-test with a different sref | `set-sref <i>` → `generate` (auto-bumps pass when needed) |
| Re-test with the same sref but fresh data (e.g. different models, or you edited the trigger) | `retest-models` → `generate` |
| Jump back to an earlier pass to look at it | `set-pass <n>` |
| Throw away a botched pass before continuing | `rm -rf .../pass-NNN` then `set-pass <prev>` |
| Redo a pass that contains only FAILED/QUEUED (no SUCCESS) tasks | `generate --force` |

`generate --force` refuses if the current pass has any SUCCESS task — preventing silent overwrite. To redo a pass that already has SUCCESS data, open a new pass via `retest-models` (same sref) or `set-sref` (new sref).

### Phase 3: STYLE_REFINE (iterative loop)

```bash
# LLM refines the trigger phrase (each call = one round)
styleclaw refine <project-name>
# Optional: provide human direction
styleclaw refine <project-name> --direction "increase contrast, add halftone dots"

# Generate images with the refined trigger
styleclaw generate <project-name>

# Poll for results
styleclaw poll <project-name>

# Evaluate this round (LLM scores on 5 dimensions: color, line, lighting, texture, mood)
styleclaw evaluate <project-name>
# → If all scores ≥ 7.0 and total ≥ 7.5: ready to approve
# → Otherwise: continue refining or adjust manually

# Repeat refine → generate → poll → evaluate until satisfied (max 5 auto rounds)

# When satisfied, approve to advance to batch testing
styleclaw approve <project-name>
# → Phase advances to BATCH_T2I
```

**Mid-refinement commands:**
```bash
# Switch models during STYLE_REFINE (no phase change)
styleclaw select-model <project-name> --models niji7,mj-v7

# Give specific adjustment direction
styleclaw adjust <project-name> --direction "warmer colors, less chromatic aberration"

# Rollback to earlier round (soft rollback - preserves all data)
styleclaw rollback <project-name> --to STYLE_REFINE --round 2
# → Only changes state.json (current_round pointer), keeps all round directories
# → Next refine auto-skips rounds that already have a prompt, creates new round
```

**Rollback behavior (non-destructive)**:
- Only updates `state.json` (current_round pointer)
- Preserves all existing round directories on disk
- Next `refine` skips rounds that already have a prompt config, creates next available round
- Example: rollback to round 2 → refine creates round 4 (skips existing round 3)
- All history preserved for comparison

### Phase 4: BATCH_T2I (100-case generalization test)

```bash
# LLM designs 100 diverse test cases (10 categories × 10 each)
styleclaw design-cases <project-name>
# Re-design with feedback on the previous batch (creates batch-002+, never overwrites)
styleclaw design-cases <project-name> --feedback "上一批群像太少，重做一批"
# → Edit data/projects/<name>/batch-t2i/batch-001/cases.json if needed

# Submit all 100 cases for image generation
styleclaw batch-submit <project-name>
# Optional: specify model
styleclaw batch-submit <project-name> --model mj-v7
# Preview submission plan
styleclaw batch-submit <project-name> --dry-run

# Poll until all tasks complete (100 tasks, each produces 4 images = 400 total)
styleclaw poll <project-name>

# Generate HTML report for visual review
styleclaw report <project-name>
```

**10 test categories** (10 cases each, 100 total):
`adult_male`, `adult_female`, `shota`, `loli`, `elderly_male`, `elderly_female`, `creature`, `outdoor_scene`, `indoor_scene`, `group`

**Generalization rule**: Only 1-2 out of 100 cases may reference IP-specific elements. The remaining 98+ must be completely original subjects to test style trigger generalization.

### Phase 5: BATCH_I2I (image-to-image testing)

```bash
# Add reference images for i2i testing (advances to BATCH_I2I)
styleclaw add-refs <project-name> --images ref1.png --images ref2.png

# Submit i2i batch
styleclaw batch-submit <project-name> --i2i

# Poll and report
styleclaw poll <project-name>
styleclaw report <project-name> --i2i

# Mark project as completed
styleclaw approve <project-name> --phase completed
```

### Utility commands

```bash
# Check project status
styleclaw status                    # List all projects (name + phase)
styleclaw status <project-name>     # Detailed status + phase-aware "建议下一步" hints

# Generate report (auto-detects phase)
styleclaw report <project-name>

# Lifecycle / housekeeping (no API keys needed)
styleclaw archive <project-name>           # Move project to data/projects/.archive/<ts>-<name>/ (non-destructive)
styleclaw clean --stalled                  # Dry-run: list projects with last_updated > 7 days and phase != COMPLETED
styleclaw clean --stalled --days 14        # Custom threshold
styleclaw clean --stalled --yes            # Actually archive matches

# Pre-pass storage migration (idempotent)
styleclaw migrate <project-name>

# Rollback honors --round only when that round dir exists on disk for STYLE_REFINE.
styleclaw rollback <project-name> --to STYLE_REFINE --round 2
```

## Available Image Generation Models

| Model ID | Name | sref Mode | Notes |
|----------|------|:---------:|-------|
| `mj-v7` | Midjourney v7 | `param` | `--sref` + `sw=100`, stylize=200, returns 4 images per task |
| `niji7` | Midjourney niji7 | `param` | `--sref` + `sw=100`, anime-focused, stylize=200 |
| `nb2` | NanoBanana2 | `prompt` | `参考图1的风格：` + `imageUrls`, resolution=2k, max 20000 char |
| `seedream` | Seedream v5-lite | `prompt` | `参考图1的风格：` + `imageUrls`, width×height, max 2000 char |
| `gpt-image-2` | GPT-Image-2 | `prompt` | `参考图1的风格：` + `imageUrls`, resolution=2k, max 20000 char |

**sref modes**: `param` = style ref via API parameter (`sref`+`sw`); `prompt` = style ref via prompt prefix + `imageUrls` parameter

## Conventions

- **Immutability**: All Pydantic models use `model_copy(update=...)` — never mutate in place. Time-stamped writes go through `core.time_utils.utcnow_iso()`.
- **Async**: `asyncio.TaskGroup` for parallel work, semaphore `CONCURRENCY_LIMIT` (default 5) for image-gen, `LLM_CONCURRENCY_LIMIT` (default 4) for parallel LLM calls.
- **Client lifecycle**: Use `_build_context()` / `_close_resource()` in CLI to ensure proper httpx client cleanup. Same context is reused by the orchestrator across all steps in one `run`.
- **Storage**: JSON files under `DATA_ROOT` (default `data/projects/`, overridable via `STYLECLAW_DATA_ROOT`); monkeypatch `DATA_ROOT` in tests.
- **LLM output**: Always strip markdown fences and parse via `core.text_utils.parse_llm_response(raw, ModelCls)` before use — it surfaces a useful error preview on failure.
- **Variant routing**: `ProjectState.selected_variant` (`"prompt-sref"` or `"prompt-only"`) decides whether STYLE_REFINE generation passes `sref_url`. Variant is locked via `select-model --variant` (or the orchestrator's confirmation prompt) and defaults to the LLM-recommended variant.
- **Prompt building**: Final prompt = `trigger_phrase + ", " + character_desc`; for prompt-sref variant with `SrefMode.PROMPT` models: `参考图1的风格：trigger_phrase + ", " + character_desc` + `imageUrls` param.
- **Image encoding**: `encode_image_for_llm()` returns `(base64_str, media_type)` — resizes to 1024px long-edge, format determined by image mode (RGBA→PNG, else→JPEG).
- **Submit retry**: RunningHub submit retries up to 3 times on empty `taskId` response.
- **Poll**: Skips tasks with status `SUCCESS` or `FAILED`, skips tasks with no `task_id`. The orchestrator's `do_poll` adds one automatic retry of FAILED tasks for MODEL_SELECT / STYLE_REFINE (batch retries stay explicit because of cost).
- **Thinking traces**: When `--show-thinking` is on, each `*_with_thinking` agent writes `<artifact>.thinking.md` alongside the JSON it produced (e.g. `evaluation.thinking.md` next to `evaluation.json`).
- **Checkpointing**: Long-running batch ops use `core.checkpoint.Checkpoint` (atomic write via `.tmp` + `replace`) so interrupted runs can resume by skipping already-recorded items.
- **Cross-phase planning**: The planner extends action whitelist only from `INIT / STYLE_REFINE / BATCH_T2I / BATCH_I2I`; `MODEL_SELECT` stays gated so `select-model` cannot be autoplanned in. `select-model / approve / retest-models / add-refs` are excluded from cross-phase extension regardless.

## Commands

```bash
uv run python -m pytest tests/ -v          # Run tests
uv run python -m pytest tests/ --cov=src   # With coverage (fail_under=80%)
uv run styleclaw --help                    # CLI help
```
