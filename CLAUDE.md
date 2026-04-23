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

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `RUNNINGHUB_API_KEY` | Yes | RunningHub API key for image generation |
| `AWS_REGION` | Yes | AWS region for Bedrock (e.g. `us-east-1`) |
| `AWS_BEARER_TOKEN_BEDROCK` | Yes | Bearer token for AWS Bedrock authentication |
| `CLAUDE_MODEL` | No | Bedrock model ID (default: `anthropic.claude-sonnet-4-20250514`) |

## Tech Stack

- **Language**: Python 3.11+ with uv package manager
- **HTTP**: httpx (async) — both RunningHub client and Bedrock LLM provider
- **LLM**: Claude via AWS Bedrock (httpx + bearer token auth)
- **Models**: Pydantic v2 (immutable state via `model_copy(update=...)`)
- **CLI**: Typer
- **Reports**: Jinja2 HTML templates
- **Image**: Pillow (resize to 1024px long-edge before sending to LLM)
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
│   ├── models.py           # All Pydantic models (Phase, ProjectState, TaskRecord, BatchCase, etc.)
│   ├── state_machine.py    # Phase transitions: advance() and rollback()
│   ├── prompt_builder.py   # Build API params: trigger + character_desc concatenation, aspect ratio → width/height
│   ├── case_generator.py   # Generate 100 empty BatchCase skeletons across 10 categories
│   └── image_utils.py      # resize_for_llm(), encode_image_for_llm() → (base64, media_type)
├── storage/
│   └── project_store.py    # All filesystem persistence (JSON read/write under data/projects/<name>/)
├── providers/
│   ├── llm/
│   │   ├── bedrock.py      # BedrockProvider — async httpx calls to AWS Bedrock
│   │   └── prompts/        # Markdown prompt templates for LLM agents
│   └── runninghub/
│       ├── client.py       # RunningHubClient — async httpx for image gen API
│       ├── models.py       # MODEL_REGISTRY: mj-v7, niji7, nb2, seedream configs
│       ├── tasks.py        # submit_task() with 3x retry, poll_task() with timeout
│       └── upload.py       # Upload files to RunningHub
├── agents/          # LLM-powered creative work (each calls BedrockProvider)
│   ├── analyze_style.py    # Analyze ref images → StyleAnalysis + initial trigger phrase
│   ├── select_model.py     # Compare models' outputs → ModelEvaluation
│   ├── evaluate_result.py  # Score round results on 5 dimensions → RoundEvaluation
│   ├── refine_prompt.py    # Refine trigger phrase based on evaluation feedback
│   └── design_cases.py     # Design 100 diverse test case descriptions
├── scripts/         # Mechanical work — no LLM calls
│   ├── init_project.py     # Create project dir, copy refs, upload to RunningHub
│   ├── generate.py         # Submit image gen tasks (model-select or style-refine)
│   ├── poll.py             # Poll task status, download completed images
│   ├── batch_submit.py     # Submit batch t2i/i2i tasks (100 cases)
│   └── report.py           # Generate Jinja2 HTML reports
├── reports/templates/      # Jinja2 HTML templates for reports
└── cli.py           # Typer CLI — all user-facing commands
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
│   ├── initial-analysis.json            # StyleAnalysis from LLM
│   ├── evaluation.json                  # ModelEvaluation from LLM
│   ├── report.html
│   └── results/<model-id>/
│       ├── task.json                    # TaskRecord
│       └── output-*.png
├── style-refine/
│   └── round-001/
│       ├── prompt.json                  # PromptConfig (trigger phrase for this round)
│       ├── evaluation.json              # RoundEvaluation
│       ├── report.html
│       └── results/<model-id>/
│           ├── task.json
│           └── output-*.png
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

```bash
# Generate test images across all models
styleclaw generate <project-name>

# Poll until images are ready
styleclaw poll <project-name>

# LLM evaluates which models best reproduce the style
styleclaw evaluate <project-name>
# → Outputs scores + HTML report

# Confirm model selection (advances to STYLE_REFINE)
styleclaw select-model <project-name> --models mj-v7
```

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

# Rollback to earlier phase if needed
styleclaw rollback <project-name> --to STYLE_REFINE --round 2
```

### Phase 4: BATCH_T2I (100-case generalization test)

```bash
# LLM designs 100 diverse test cases (10 categories × 10 each)
styleclaw design-cases <project-name>
# → Edit data/projects/<name>/batch-t2i/batch-001/cases.json if needed

# Submit all 100 cases for image generation
styleclaw batch-submit <project-name>
# Optional: specify model
styleclaw batch-submit <project-name> --model mj-v7

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
styleclaw status                    # List all projects
styleclaw status <project-name>     # Detailed status

# Generate report (auto-detects phase)
styleclaw report <project-name>
```

## Available Image Generation Models

| Model ID | Name | sref Support | Notes |
|----------|------|:---:|-------|
| `mj-v7` | Midjourney v7 | Yes | Default stylize=200, returns 4 images per task |
| `niji7` | Midjourney niji7 | Yes | Anime-focused, stylize=200 |
| `nb2` | NanoBanana2 | No | resolution=2k, max 20000 char prompt |
| `seedream` | Seedream v5-lite | No | Uses width×height instead of aspectRatio, max 2000 char prompt |

## Conventions

- **Immutability**: All Pydantic models use `model_copy(update=...)` — never mutate in place
- **Async**: `asyncio.TaskGroup` for parallel work, semaphore(5) for concurrency
- **Client lifecycle**: Use `_run_with_client()` in CLI to ensure proper httpx client cleanup
- **Storage**: JSON files under `data/projects/<name>/`, monkeypatch `DATA_ROOT` in tests
- **LLM output**: Always strip markdown fences via `_clean_json()` before parsing
- **Prompt building**: Final prompt = `trigger_phrase + ", " + character_desc` (via `build_params()` in `prompt_builder.py`)
- **Image encoding**: `encode_image_for_llm()` returns `(base64_str, media_type)` — resizes to 1024px long-edge, format determined by image mode (RGBA→PNG, else→JPEG)
- **Submit retry**: RunningHub submit retries up to 3 times on empty `taskId` response
- **Poll**: Skips tasks with status `SUCCESS` or `FAILED`, skips tasks with no `task_id`

## Commands

```bash
uv run python -m pytest tests/ -v          # Run tests
uv run python -m pytest tests/ --cov=src   # With coverage (fail_under=80%)
uv run styleclaw --help                    # CLI help
```
