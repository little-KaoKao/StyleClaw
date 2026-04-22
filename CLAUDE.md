# StyleClaw

AI-powered style trigger word exploration system for image generation.

## Tech Stack

- **Language**: Python 3.11+ with uv package manager
- **HTTP**: httpx (async)
- **LLM**: Claude via AWS Bedrock (boto3)
- **Models**: Pydantic v2
- **CLI**: Typer
- **Reports**: Jinja2 HTML templates
- **Image**: Pillow (resize to 1024px for LLM)
- **Config**: python-dotenv

## Architecture

State machine pipeline: `INIT → MODEL_SELECT → STYLE_REFINE → BATCH_T2I → BATCH_I2I → COMPLETED`

Key directories under `src/styleclaw/`:
- `core/` — Pure logic, no IO (models, state machine, prompt builder, case generator)
- `storage/` — File-system persistence (JSON + images)
- `providers/` — External service adapters (RunningHub API, Bedrock LLM)
- `scripts/` — Mechanical work (generate, poll, batch submit, reports) — no LLM calls
- `agents/` — LLM-powered creative work (analyze, evaluate, refine, design cases)
- `reports/templates/` — Jinja2 HTML templates

## Conventions

- **Immutability**: All Pydantic models use `model_copy(update=...)` — never mutate in place
- **Async**: `asyncio.TaskGroup` for parallel work, semaphore(5) for concurrency
- **Storage**: JSON files under `data/projects/<name>/`, monkeypatch `DATA_ROOT` in tests
- **LLM output**: Always strip markdown fences via `_clean_json()` before parsing
- **Tests**: pytest with `tmp_path` fixtures, no real API calls in unit tests

## Commands

```bash
uv run python -m pytest tests/ -v          # Run tests
uv run python -m pytest tests/ --cov=src   # With coverage
uv run styleclaw --help                    # CLI help
```

## Environment Variables

See `.env.example` for required variables:
- `RUNNINGHUB_API_KEY` — RunningHub API authentication
- `AWS_REGION` — AWS region for Bedrock
- `CLAUDE_MODEL` — Bedrock model ID
