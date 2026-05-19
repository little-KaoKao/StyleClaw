# StyleClaw

AI-powered style trigger word exploration system for image generation.

Given a set of reference images representing an IP's visual style, StyleClaw uses LLM analysis + batch image generation to iteratively discover and validate a concise **trigger phrase** that reliably reproduces that style across diverse subjects.

## How It Works

```
Reference Images ──▶ LLM Analysis ──▶ Model Selection ──▶ Iterative Refinement ──▶ 100-Case Validation
```

StyleClaw runs a state-machine pipeline:

```
INIT → MODEL_SELECT → STYLE_REFINE → BATCH_T2I → BATCH_I2I → COMPLETED
```

1. **INIT** — Provide reference images; LLM extracts style dimensions and an initial trigger phrase
2. **MODEL_SELECT** — Generate test images across multiple models; LLM picks the best one
3. **STYLE_REFINE** — Iteratively refine the trigger phrase (up to 5 rounds, scored on 5 dimensions)
4. **BATCH_T2I** — Validate generalization with 100 diverse test cases (10 categories × 10)
5. **BATCH_I2I** — Image-to-image testing for further validation

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **RunningHub** API key (for image generation)
- **LLM provider** — pick one (precedence below):
  - An **OpenAI-compatible** provider like [gptproto.com](https://gptproto.com) (recommended), or
  - **RunningHub LLM** at `https://llm.runninghub.cn/v1` (same `RUNNINGHUB_API_KEY` as image generation), or
  - **AWS Bedrock** access with a bearer token (legacy)

## Installation

```bash
git clone https://github.com/little-KaoKao/StyleClaw.git
cd StyleClaw

# Install all dependencies
uv sync

# Set up environment variables
cp .env.example .env
```

Edit `.env` with your credentials. Choose **one** LLM provider path (do not mix credentials; precedence below):

```env
RUNNINGHUB_API_KEY=<your-runninghub-api-key>

# Option A: OpenAI-compatible (recommended; wins over RunningHub LLM and Bedrock if set)
OPENAI_COMPAT_API_KEY=<your-api-key>
OPENAI_COMPAT_BASE_URL=https://api.gptproto.com/v1
LLM_MODEL=gemini-2.5-pro-preview-05-06

# Option B: AWS Bedrock (legacy)
# AWS_REGION=us-east-1
# AWS_BEARER_TOKEN_BEDROCK=<your-bedrock-token>
# LLM_MODEL=anthropic.claude-sonnet-4-20250514

# Option C: RunningHub LLM (same key as images; do not enable alongside Option A)
# RUNNINGHUB_LLM=1
# RUNNINGHUB_LLM_BASE_URL=https://llm.runninghub.cn/v1
# LLM_MODEL=rh-llm-a/rh-c-o-47
# RUNNINGHUB_LLM_REASONING_EFFORT=high
```

| Variable | Required | Description |
|----------|:--------:|-------------|
| `RUNNINGHUB_API_KEY` | Yes | RunningHub API key for image generation |
| `OPENAI_COMPAT_API_KEY` | A | API key for OpenAI-compatible provider (e.g. gptproto) |
| `OPENAI_COMPAT_BASE_URL` | A | Provider base URL (e.g. `https://api.gptproto.com/v1`) |
| `LLM_MODEL` | Yes | Model ID for the chosen provider |
| `RUNNINGHUB_LLM` | C | Set to `1` / `true` / `yes` / `on` to use RunningHub LLM |
| `RUNNINGHUB_LLM_BASE_URL` | No | LLM base URL; default `https://llm.runninghub.cn/v1` |
| `RUNNINGHUB_LLM_REASONING_EFFORT` | No | Passed on `invoke_with_thinking`; default `high`; use `off` to omit |
| `AWS_REGION` | B | AWS region (only if using Bedrock) |
| `AWS_BEARER_TOKEN_BEDROCK` | B | Bearer token for Bedrock proxy/gateway (only if using Bedrock) |

**Precedence:** If `OPENAI_COMPAT_API_KEY` is set, the OpenAI-compatible provider is used. Otherwise, if `RUNNINGHUB_LLM` is truthy, RunningHub LLM is used. Otherwise Bedrock.

#### Optional runtime tunables

All have sensible defaults; you only need to set these to tune performance or override paths.

| Variable | Default | Purpose |
|----------|--------:|---------|
| `STYLECLAW_DATA_ROOT` | `data/projects` | Root for project storage |
| `STYLECLAW_LOG_LEVEL` | `INFO` | Default log level (`DEBUG` / `WARNING` / …); `-v` is the per-call shortcut |
| `STYLECLAW_SKIP_ENV_CHECK` | — | Truthy disables CLI-startup env validation |
| `STYLECLAW_MAX_ROUNDS` | `5` | Cap on auto refine rounds |
| `STYLECLAW_CONCURRENCY` | `5` | Image-gen concurrency |
| `STYLECLAW_LLM_CONCURRENCY` | `4` | LLM call concurrency |
| `STYLECLAW_TASK_TIMEOUT` | `300` | Single-task poll timeout (seconds) |
| `STYLECLAW_POLL_INTERVAL` | `3` | Inner poll interval (seconds) |
| `STYLECLAW_ORCH_POLL_INTERVAL` | `30` | Outer orchestrator poll interval (seconds) |
| `STYLECLAW_MAX_POLL_CYCLES` | `60` | Max orchestrator poll cycles before timeout |

Verify the installation:

```bash
uv run styleclaw --help
```

## Quick Start

### Natural Language Mode (Recommended)

Use `styleclaw run` with a natural language description — the system plans and executes automatically:

```bash
# Create a project first
uv run styleclaw init spider-verse \
  --ref ref1.png --ref ref2.png --ref ref3.png \
  --info "Spider-Verse animation style"

# Then describe what you want in natural language
# Use the same name as in `init`. With multiple projects under data/projects, `-p` is required.
uv run styleclaw run "analyze style and select the best model" -p spider-verse
uv run styleclaw run "refine trigger phrase until scores pass" -p spider-verse
uv run styleclaw run "design test cases and run batch generation" -p spider-verse
```

The `run` command converts your intent into an execution plan via LLM, displays it for confirmation, then executes step by step. The plan includes:

- A **summary** describing what will happen
- An ordered list of **steps** (action name + Chinese description + args)
- An optional **loop** for iterative work (e.g., refine → generate → poll → evaluate until pass)
- A **stop_summary** ("停在哪") telling you where execution stops and what comes next

Loop exit logic: stops when the round evaluation passes (all 5 dimensions ≥ 7.0 and total ≥ 7.5); also stops with a `!! needs_human` diagnostic — naming the weakest dimension and suggesting a redirection phrase — when any dimension drops below 5.0.

After execution, `run` prints 1–5 example follow-up commands tailored to the new phase ("下一步可以这样说"). `status <project>` shows the same hints whenever you want to see them.

Actions that change the world (`init`, `select-model`, `add-refs`) pause for an interactive confirmation prompt where you fine-tune arguments — for example, overriding the LLM-recommended model or picking the prompt-sref / prompt-only variant. `--yes` skips both the top-level "Execute?" prompt and these inline confirmations.

```bash
# Options
uv run styleclaw run "<intent>" -p <project>           # Required if multiple projects; optional if exactly one
uv run styleclaw run "<intent>" --yes                  # Skip all confirmation prompts
uv run styleclaw run "<intent>" --dry-run              # Print the plan and exit without executing
uv run styleclaw run "<intent>" --no-show-thinking     # Suppress saved LLM reasoning files
uv run styleclaw run "<intent>" --thinking-budget 8000 # Tune reasoning-budget token cap (default 5000)
```

### Step-by-Step Mode

You can also run each command manually for finer control:

```bash
# 1. Create a project with reference images
uv run styleclaw init spider-verse \
  --ref ref1.png --ref ref2.png --ref ref3.png \
  --info "Spider-Verse animation style"
# Or auto-discover from a directory:
uv run styleclaw init spider-verse --ref-dir /path/to/refs --info "Spider-Verse animation style"

# 2. Analyze references (LLM extracts style + initial trigger)
uv run styleclaw analyze spider-verse

# 3. Generate test images across all models (2 variants × 2 genders)
uv run styleclaw generate spider-verse
uv run styleclaw poll spider-verse
# Optional: restrict to specific models, e.g. styleclaw generate spider-verse --models mj-v7,niji7

# 4. Evaluate and select the best model + variant (prompt-sref or prompt-only)
uv run styleclaw evaluate spider-verse
uv run styleclaw select-model spider-verse --models mj-v7 --variant prompt-sref

# 5. Refine trigger phrase (repeat until satisfied)
uv run styleclaw refine spider-verse
uv run styleclaw generate spider-verse
uv run styleclaw poll spider-verse
uv run styleclaw evaluate spider-verse

# 6. Approve and run 100-case batch test
uv run styleclaw approve spider-verse --yes        # --yes skips the interactive "Proceed?" prompt
uv run styleclaw design-cases spider-verse
uv run styleclaw batch-submit spider-verse
uv run styleclaw poll spider-verse
uv run styleclaw report spider-verse
```

## CLI Reference

### Orchestrator

| Command | Description |
|---------|-------------|
| `run "<intent>"` | Natural language execution — LLM plans, you confirm, system executes |
| `run "<intent>" -p <name>` | Specify project explicitly |
| `run "<intent>" --yes` | Skip confirmation and execute immediately |

### Core Pipeline Commands

| Command | Phase | Description |
|---------|-------|-------------|
| `init <name> --ref <img>...` | — | Create project with reference images |
| `analyze <name>` | INIT | LLM analyzes references, extracts initial trigger |
| `generate <name>` | MODEL_SELECT / STYLE_REFINE | Submit image generation tasks |
| `poll <name>` | Any active | Poll task status, download completed images |
| `evaluate <name>` | MODEL_SELECT / STYLE_REFINE | LLM scores generated images |
| `select-model <name> --models <ids>` | MODEL_SELECT / STYLE_REFINE | Choose model(s) to use |
| `refine <name>` | STYLE_REFINE | LLM refines the trigger phrase |
| `approve <name>` | STYLE_REFINE / BATCH_I2I | Advance to next phase |
| `design-cases <name>` | BATCH_T2I | LLM designs 100 test case descriptions |
| `batch-submit <name>` | BATCH_T2I / BATCH_I2I | Submit batch generation tasks |
| `report <name>` | BATCH_T2I / BATCH_I2I | Generate HTML visual report |

### Utility Commands

| Command | Description |
|---------|-------------|
| `status` | List all projects with their current phase |
| `status <name>` | Show detailed project status + phase-aware "next step" hints |
| `adjust <name> --direction <text>` | Provide manual direction for refinement (shortcut for `refine --direction`) |
| `rollback <name> --to <phase> --round <n>` | Roll back to an earlier phase/round (non-destructive) |
| `retest-models <name>` | Open a new MODEL_SELECT pass (pass-002+) without destroying prior data |
| `back-to-t2i <name>` | Return from BATCH_I2I to BATCH_T2I |
| `set-sref <name> <index>` | Set which ref image to use as style reference (0-based) |
| `set-pass <name> <pass>` | Switch active model-select pass number |
| `add-refs <name> --images <img>...` | Add reference images for i2i testing (advances BATCH_T2I → BATCH_I2I) |
| `archive <name>` | Move a project under `data/projects/.archive/<ts>-<name>/` (non-destructive) |
| `clean --stalled [--days N] [--yes]` | List (default) or archive projects idle > N days and not COMPLETED |
| `migrate <name>` | Move pre-pass storage layout into the pass-scoped layout (idempotent) |

### Global Flags

| Flag | Where it applies | What it does |
|------|------------------|---------------|
| `--verbose / -v` | any command | Lift the root logger to DEBUG for this invocation |
| `--show-thinking / --no-show-thinking` | `analyze` / `evaluate` / `refine` / `run` | Save LLM reasoning to `*.thinking.md` siblings (default on) |
| `--thinking-budget <int>` | same as above | Token budget passed to `invoke_with_thinking` (default 5000) |
| `--dry-run` | `run` / `generate` / `batch-submit` | Print the plan / estimated task counts and exit without API calls |

### Options

```bash
# For multi-line commands, every continued line must end with \ before the newline;
# otherwise the next line is executed as a new shell command.
uv run styleclaw init <name> \
  --ref <image-path> \
  --ref-dir <dir> \
  --info <text> \
  --desc <text> \
  --force
# Repeat --ref for multiple images; use --ref-dir or --ref as needed; --force overwrites.

uv run styleclaw generate <name> \
  --force \                # Only allowed when the current pass has no SUCCESS tasks
                           # (e.g. all FAILED/QUEUED). Refuses if any task in the pass
                           # already succeeded — prevents silent overwrite. To redo a
                           # pass that already has successes, use `retest-models`
                           # (new pass, same sref) or `set-sref` (new pass, new ref).
  --models mj-v7,niji7 \   # MODEL_SELECT only — limit submission to a subset
  --dry-run                # Show planned operations and exit

uv run styleclaw refine <name> \
  --direction <text>

uv run styleclaw select-model <name> \
  --models <model-ids> \   # Comma-separated, e.g. "mj-v7"
  --variant prompt-sref    # or prompt-only — locks the variant used in STYLE_REFINE.
                           # Omit to use evaluation.json's recommended_variant.

uv run styleclaw design-cases <name> \
  --feedback "<text>"      # Feedback on previous batch; design-cases always creates a NEW batch

uv run styleclaw batch-submit <name> \
  --i2i \
  --model <model-id> \
  --dry-run

uv run styleclaw approve <name> \
  --phase completed \
  --yes

uv run styleclaw report <name> \
  --i2i
```

## Available Models

| Model ID | Name | Style Ref Mode | Notes |
|----------|------|:--------------:|-------|
| `mj-v7` | Midjourney v7 | `param` | Default; style ref via `--sref` + `sw=100`; stylize=200, 4 images per task |
| `niji7` | Midjourney niji7 | `param` | Style ref via `--sref` + `sw=100`; anime-focused, stylize=200 |
| `nb2` | NanoBanana2 | `prompt` | Style ref via prompt prefix `参考图1的风格：` + `imageUrls`; 2K resolution, max 20K char prompt |
| `seedream` | Seedream v5-lite | `prompt` | Style ref via prompt prefix `参考图1的风格：` + `imageUrls`; uses width×height, max 2K char prompt |
| `gpt-image-2` | GPT-Image-2 | `prompt` | Style ref via prompt prefix `参考图1的风格：` + `imageUrls`; 2K resolution, quality=medium, max 20K char prompt |

All models support style reference; the only difference is how the reference is passed (`param` vs `prompt` + `imageUrls`).

### Variants

During MODEL_SELECT, each model is tested under two variants:

- **prompt-only** — only the trigger phrase, no style reference image
- **prompt-sref** — trigger phrase + style reference image (MJ via `--sref` param, others via prompt prefix `参考图1的风格：` + `imageUrls`)

If prompt-only is already sufficient (total ≥ 7.0), prefer it — it generalizes better without being anchored to one specific reference frame, and is faster on prompt-mode models. The `evaluate` step writes its `recommended_variant` to `evaluation.json`, and `select-model` uses it by default when `--variant` is omitted.

## Style Refinement Scoring

During STYLE_REFINE, the LLM evaluates generated images on 5 dimensions:

| Dimension | Description |
|-----------|-------------|
| Color Palette | How well colors match the reference style |
| Line Style | Stroke weight, edge treatment, linework |
| Lighting | Light direction, contrast, shadow style |
| Texture | Surface detail, grain, material feel |
| Overall Mood | Emotional tone and atmospheric consistency |

**Pass criteria**: all dimensions ≥ 7.0 and total score ≥ 7.5 (out of 10).

### Iteration loop handling

After each `evaluate`, the LLM writes one of three `recommendation` values into `evaluation.json`:

| Recommendation | When | What to do |
|---|---|---|
| `continue_refine` | Scores climbing but not yet at the bar | `refine` (auto-reads `next_direction`) |
| `needs_human` | Any dimension dropped below 5, or scores regressed | `refine --direction "<concrete correction>"` — or `rollback --to STYLE_REFINE --round <better-round>` then re-refine |
| `approve` | All 5 ≥ 7 and total ≥ 7.5 | `approve --yes` |

The auto loop is capped at `STYLECLAW_MAX_ROUNDS` (default 5). `rollback` is non-destructive — earlier rounds stay on disk; the next `refine` skips occupied round numbers and creates a new one.

## Batch Test Categories

The 100-case generalization test covers 10 categories (10 cases each):

`adult_male` · `adult_female` · `shota` · `loli` · `elderly_male` · `elderly_female` · `creature` · `outdoor_scene` · `indoor_scene` · `group`

Only 1–2 cases may reference the original IP. The remaining 98+ must be completely original subjects to test trigger generalization.

## Project Data

All project data is stored under `data/projects/<name>/`:

```
data/projects/<name>/
├── config.json              # Project configuration
├── state.json               # Current phase, round, batch, selected models
├── refs/                    # Reference images + upload records
├── model-select/pass-NNN/   # Model comparison results + report
├── style-refine/pass-NNN/round-NNN/  # Per-round results + evaluations
├── batch-t2i/batch-NNN/     # 100-case t2i results + report
└── batch-i2i/batch-NNN/     # i2i results + report
```

Older projects may still use the pre-pass layout. Run `uv run styleclaw migrate <name>` to move them into the pass-scoped layout.

## Development

```bash
# Run tests
uv run python -m pytest tests/ -v

# Run tests with coverage (minimum 80%)
uv run python -m pytest tests/ --cov=src

# Skip slow integration tests
uv run python -m pytest tests/ -m "not integration"
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Package Manager | uv |
| HTTP Client | httpx (async) |
| LLM | OpenAI-compatible API, RunningHub LLM, or AWS Bedrock (legacy) |
| Data Models | Pydantic v2 |
| CLI | Typer |
| Reports | Jinja2 HTML templates |
| Image Processing | Pillow |
| Config | python-dotenv |

## License

See [LICENSE](LICENSE) for details.
