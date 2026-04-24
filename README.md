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
- **AWS Bedrock** access with bearer token (for Claude LLM calls)

## Installation

```bash
git clone https://github.com/your-org/styleclaw.git
cd styleclaw

# Install all dependencies
uv sync

# Set up environment variables
cp .env.example .env
```

Edit `.env` with your credentials:

```env
RUNNINGHUB_API_KEY=<your-runninghub-api-key>
AWS_REGION=us-east-1
AWS_BEARER_TOKEN_BEDROCK=<your-bedrock-token>
CLAUDE_MODEL=anthropic.claude-sonnet-4-20250514    # optional
```

| Variable | Required | Description |
|----------|:--------:|-------------|
| `RUNNINGHUB_API_KEY` | Yes | RunningHub API key for image generation |
| `AWS_REGION` | Yes | AWS region for Bedrock (e.g. `us-east-1`) |
| `AWS_BEARER_TOKEN_BEDROCK` | Yes | Bearer token for AWS Bedrock authentication |
| `CLAUDE_MODEL` | No | Bedrock model ID (default: `anthropic.claude-sonnet-4-20250514`) |

Verify the installation:

```bash
uv run styleclaw --help
```

## Quick Start

```bash
# 1. Create a project with reference images
uv run styleclaw init spider-verse \
  --ref ref1.png --ref ref2.png --ref ref3.png \
  --info "Spider-Verse animation style"

# 2. Analyze references (LLM extracts style + initial trigger)
uv run styleclaw analyze spider-verse

# 3. Generate test images across all models
uv run styleclaw generate spider-verse
uv run styleclaw poll spider-verse

# 4. Evaluate and select the best model
uv run styleclaw evaluate spider-verse
uv run styleclaw select-model spider-verse --models mj-v7

# 5. Refine trigger phrase (repeat until satisfied)
uv run styleclaw refine spider-verse
uv run styleclaw generate spider-verse
uv run styleclaw poll spider-verse
uv run styleclaw evaluate spider-verse

# 6. Approve and run 100-case batch test
uv run styleclaw approve spider-verse
uv run styleclaw design-cases spider-verse
uv run styleclaw batch-submit spider-verse
uv run styleclaw poll spider-verse
uv run styleclaw report spider-verse
```

## CLI Reference

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
| `status` | List all projects |
| `status <name>` | Show detailed project status |
| `adjust <name> --direction <text>` | Provide manual direction for refinement |
| `rollback <name> --to <phase> --round <n>` | Roll back to an earlier phase |
| `add-refs <name> --images <img>...` | Add reference images for i2i testing |

### Options

```bash
uv run styleclaw init <name> \
  --ref <image-path>         # Reference image (repeatable)
  --info <text>              # IP description
  --desc <text>              # Project description

uv run styleclaw refine <name> \
  --direction <text>         # Optional: human guidance for refinement

uv run styleclaw batch-submit <name> \
  --i2i                      # Submit image-to-image instead of text-to-image
  --model <model-id>         # Specify model (default: project's selected model)

uv run styleclaw approve <name> \
  --phase completed          # For BATCH_I2I → COMPLETED transition
  --yes                      # Skip confirmation prompt

uv run styleclaw report <name> \
  --i2i                      # Generate i2i report instead of t2i
```

## Available Models

| Model ID | Name | Style Ref | Notes |
|----------|------|:---------:|-------|
| `mj-v7` | Midjourney v7 | Yes | Default; stylize=200, 4 images per task |
| `niji7` | Midjourney niji7 | Yes | Anime-focused, stylize=200 |
| `nb2` | NanoBanana2 | No | 2K resolution, max 20K char prompt |
| `seedream` | Seedream v5-lite | No | Uses width×height, max 2K char prompt |

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
├── model-select/            # Model comparison results + report
├── style-refine/round-NNN/  # Per-round results + evaluations
├── batch-t2i/batch-NNN/     # 100-case t2i results + report
└── batch-i2i/batch-NNN/     # i2i results + report
```

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
| LLM | Claude via AWS Bedrock |
| Data Models | Pydantic v2 |
| CLI | Typer |
| Reports | Jinja2 HTML templates |
| Image Processing | Pillow |
| Config | python-dotenv |

## License

See [LICENSE](LICENSE) for details.
