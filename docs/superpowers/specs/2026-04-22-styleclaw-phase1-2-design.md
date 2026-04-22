# StyleClaw Phase 1-2 Build Design

## Scope

Vertical slice covering INIT and MODEL_SELECT phases. Builds the thinnest path through all layers, validates real API integration early.

## What's Out of Scope

- Phase 3 (STYLE_REFINE), Phase 4 (BATCH_T2I), Phase 5 (BATCH_I2I)
- HTML report generation
- `rollback` CLI command (state machine supports it, but CLI deferred)
- `design-cases`, `batch-submit`, `approve`, `adjust` CLI commands

## Section 1: Foundation + Phase 1 (INIT)

### Project Scaffolding

- `pyproject.toml` with uv, src layout `src/styleclaw/`
- Dependencies: httpx, boto3, pydantic, typer, pillow, python-dotenv
- `.env` for `RUNNINGHUB_API_KEY`; AWS credentials from system path (boto3 auto-discovers)
- Runtime data under `data/projects/<name>/`, gitignored

### Modules

| Module | Purpose |
|---|---|
| `core/models.py` | Pydantic models: `ProjectConfig`, `ProjectState`, `Phase` enum, `HistoryEntry` |
| `core/state_machine.py` | Phase transition validation + rollback logic. Full TRANSITIONS map (pure logic, no IO) |
| `providers/runninghub/client.py` | httpx.AsyncClient with auth header, exponential backoff retry (3x), concurrency semaphore (5) |
| `providers/runninghub/upload.py` | `upload_file(path) -> url`, multipart/form-data |
| `storage/project_store.py` | Create project directory tree, read/write config.json and state.json |
| `storage/image_store.py` | Download image from URL to local path |
| `cli.py` | Typer app, unified CLI entry point |
| `scripts/init_project.py` | Init logic: upload refs, create dirs, write config + state |

### CLI Commands (Phase 1)

```
styleclaw init <name> --ref <image>... --info "description"
styleclaw status [<name>]
```

## Section 2: Phase 2 (MODEL_SELECT)

### Modules

| Module | Purpose |
|---|---|
| `providers/runninghub/models.py` | `MODEL_REGISTRY`: 4 models' endpoints, prompt limits, aspect ratio format, special params |
| `providers/runninghub/tasks.py` | `submit_task`, `poll_status`, `get_outputs`. Handles sync (results in response) and async (poll) paths |
| `core/prompt_builder.py` | trigger phrase + model config → final API params. Seedream 2000-char truncation, MJ sref injection, aspect ratio format mapping |
| `providers/llm/base.py` | `LLMProvider` Protocol: `invoke(system, messages, max_tokens, temperature) -> str` |
| `providers/llm/bedrock.py` | boto3 Bedrock implementation, multimodal (base64 image input) |
| `providers/llm/prompts/analyze.md` | System prompt for style analysis agent |
| `providers/llm/prompts/select_model.md` | System prompt for model selection agent |
| `agents/analyze_style.py` | Ref images + IP info → 6-dimension analysis + initial trigger phrase → `initial-analysis.json` |
| `agents/select_model.py` | Ref images + 4 models' generated images → 5-dimension scoring + recommendation → `evaluation.json` |
| `scripts/generate.py` | Batch submit generation tasks |
| `scripts/poll.py` | Poll pending tasks + download completed images |

### Image Compression (`core/image_utils.py`)

Resize to 1024px long edge (Pillow) before sending to LLM. Saves tokens. Also provides base64 encoding for multimodal API input.

### CLI Commands (Phase 2)

```
styleclaw analyze <name>
styleclaw generate <name> --phase model-select
styleclaw poll <name>
styleclaw evaluate <name> --phase model-select
styleclaw select-model <name> --models mj-v7,niji7
```

### Key Decisions

- Agents are single-shot LLM calls, not multi-turn. Input assembled in Python, output parsed as JSON.
- `poll` is idempotent — skips completed tasks, only processes pending.
- `generate` and `poll` are separate (submit then poll), allowing intermediate state inspection.
- All 4 models run in parallel during MODEL_SELECT via asyncio + semaphore(5).

## Section 3: Testing Strategy

| Category | Scope | Mocking |
|---|---|---|
| Unit tests | State machine, prompt builder, model registry | None (pure logic) |
| Integration (mocked) | RunningHub client (retry/auth/semaphore), LLM provider, project store | httpx mock, boto3 mock, real tmp filesystem |
| Integration (real API) | Upload image, submit one generation, call Bedrock | None — marked `@pytest.mark.integration`, not in CI |

Tools: pytest + pytest-asyncio. Coverage target 80% on `core/` and `storage/`.

No E2E tests yet — full pipeline validated manually.
