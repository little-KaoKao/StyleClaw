from __future__ import annotations

import asyncio
import inspect
import logging
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import typer
from dotenv import load_dotenv

from styleclaw.core.config import MAX_AUTO_ROUNDS, env_truthy, validate_env
from styleclaw.core.models import Phase, ProjectState, TaskStatus
from styleclaw.core.state_machine import advance
from styleclaw.orchestrator.actions import ExecutionContext, StepResult
from styleclaw.storage import project_store

load_dotenv()

app = typer.Typer(name="styleclaw", help="AI style trigger word exploration system")

# Log level is INFO by default; --verbose/-v on any command lifts it to DEBUG.
# STYLECLAW_LOG_LEVEL can also set it (e.g. DEBUG, WARNING) for persistent
# overrides in CI or scripts.
_default_level_name = os.getenv("STYLECLAW_LOG_LEVEL", "INFO").upper()
_default_level = getattr(logging, _default_level_name, logging.INFO)
logging.basicConfig(level=_default_level, format="%(levelname)s: %(message)s")


@app.callback()
def _global_options(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show DEBUG-level logs from all subsystems",
    ),
) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Skip env validation for commands that don't touch any external service.
    _skip_validation = {
        "status", "rollback", "set-sref", "set-pass", "migrate",
        "archive", "clean",
    }
    if (
        os.getenv("STYLECLAW_SKIP_ENV_CHECK")
        or ctx.invoked_subcommand in _skip_validation
        or ctx.resilient_parsing
    ):
        return
    errors = validate_env()
    if errors:
        for err in errors:
            typer.echo(f"Error: {err}", err=True)
        typer.echo(
            "Hint: copy .env.example to .env and fill in the required keys.",
            err=True,
        )
        raise typer.Exit(1)


def _get_api_key() -> str:
    key = os.getenv("RUNNINGHUB_API_KEY")
    if not key:
        typer.echo("Error: RUNNINGHUB_API_KEY not set in environment.", err=True)
        raise typer.Exit(1)
    return key


def _build_llm_provider() -> Any:
    if os.getenv("OPENAI_COMPAT_API_KEY"):
        from styleclaw.providers.llm.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider()
    if env_truthy("RUNNINGHUB_LLM"):
        from styleclaw.providers.llm.runninghub_llm import RunningHubLLMProvider
        return RunningHubLLMProvider()
    from styleclaw.providers.llm.bedrock import BedrockProvider
    return BedrockProvider()


async def _close_resource(resource: Any, label: str) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await asyncio.wait_for(result, timeout=5.0)
    except asyncio.TimeoutError:
        logging.getLogger(__name__).warning(
            "Timed out closing %s after 5s.", label,
        )
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "Error closing %s: %s", label, exc,
        )


@asynccontextmanager
async def _build_context(
    project: str,
    needs_client: bool = False,
    needs_llm: bool = False,
    show_thinking: bool = False,
    thinking_budget: int = 5000,
) -> AsyncIterator[ExecutionContext]:
    from styleclaw.providers.runninghub.client import RunningHubClient

    client = None
    llm = None
    try:
        if needs_client:
            client = RunningHubClient(api_key=_get_api_key())
        if needs_llm:
            llm = _build_llm_provider()
        yield ExecutionContext(
            project=project, client=client, llm=llm,
            show_thinking=show_thinking, thinking_budget=thinking_budget,
        )
    finally:
        for label, resource in (("client", client), ("llm", llm)):
            if resource is None:
                continue
            await _close_resource(resource, label)


def _run_action(
    project: str,
    action_name: str,
    args: dict[str, Any] | None = None,
    show_thinking: bool = False,
    thinking_budget: int = 5000,
) -> StepResult:
    import httpx

    from styleclaw.orchestrator.actions import ACTION_REGISTRY

    action_def = ACTION_REGISTRY.get(action_name)
    if action_def is None:
        raise ValueError(f"Unknown action: {action_name}")

    async def _exec() -> StepResult:
        async with _build_context(
            project,
            needs_client=action_def.needs_client,
            needs_llm=action_def.needs_llm,
            show_thinking=show_thinking,
            thinking_budget=thinking_budget,
        ) as ctx:
            return await action_def.fn(ctx, args or {})

    try:
        return asyncio.run(_exec())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except httpx.HTTPStatusError as exc:
        typer.echo(f"API error ({exc.response.status_code}): {exc}", err=True)
        raise typer.Exit(1) from exc
    except httpx.TransportError as exc:
        typer.echo(f"Network error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name"),
    ref: list[Path] = typer.Option(None, "--ref", help="Reference image paths"),
    ref_dir: Path = typer.Option(None, "--ref-dir", help="Directory containing reference images"),
    info: str = typer.Option(None, "--info", help="IP/style description"),
    description: str = typer.Option("", "--desc", help="Project description"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing project"),
) -> None:
    """Initialize a new project with reference images."""
    # Auto-discover images
    if not ref:
        image_exts = {".png", ".jpg", ".jpeg", ".webp"}
        search_dir = ref_dir if ref_dir else Path.cwd()

        if ref_dir and not ref_dir.is_dir():
            typer.echo(f"Error: Directory not found: {ref_dir}", err=True)
            raise typer.Exit(1)

        discovered = [p for p in search_dir.iterdir() if p.suffix.lower() in image_exts]
        if not discovered:
            typer.echo(f"Error: No images found in {search_dir}. Use --ref to specify paths.", err=True)
            raise typer.Exit(1)
        typer.echo(f"Auto-discovered {len(discovered)} images from {search_dir}: {', '.join(p.name for p in discovered)}")
        ref = discovered

    for r in ref:
        if not r.exists():
            typer.echo(f"Error: Reference image not found: {r}", err=True)
            raise typer.Exit(1)

    # Prompt for info if not provided
    if not info:
        info = typer.prompt("IP/style description (e.g., 'Spider-Verse animation style')")

    from styleclaw.providers.runninghub.client import RunningHubClient
    from styleclaw.scripts.init_project import init_project

    async def _exec() -> Path:
        async with RunningHubClient(api_key=_get_api_key()) as client:
            return await init_project(name, ref, info, description, client, force=force)

    root = asyncio.run(_exec())
    typer.echo(f"Project initialized at {root}")


@app.command()
def status(
    name: Optional[str] = typer.Argument(None, help="Project name (omit to list all)"),
) -> None:
    """Show project status."""
    if name is None:
        projects = project_store.list_projects()
        if not projects:
            typer.echo("No projects found.")
            return
        for p in projects:
            state = project_store.load_state(p)
            typer.echo(f"  {p}: {state.phase}")
        return

    config = project_store.load_config(name)
    state = project_store.load_state(name)
    typer.echo(f"Project: {config.name}")
    typer.echo(f"Phase:   {state.phase}")
    typer.echo(f"Models:  {', '.join(state.selected_models) or '(none)'}")
    typer.echo(f"Round:   {state.current_round}")
    typer.echo(f"Pass:    {state.current_model_select_pass}")
    typer.echo(f"Updated: {state.last_updated}")
    if config.ip_info:
        typer.echo(f"IP Info: {config.ip_info[:100]}")

    from styleclaw.orchestrator.suggestions import suggest_next_steps
    suggestions = suggest_next_steps(name)
    if suggestions:
        typer.echo("\n建议下一步：")
        for line in suggestions:
            typer.echo(f"  {line}")


def _archive_project(name: str) -> Path:
    """Move a project's directory under DATA_ROOT/.archive/<timestamp>-<name>/.

    Verifies the project exists by loading state.json before moving. The move
    is non-destructive — the project is renamed, not deleted.
    """
    project_store.load_state(name)
    src = project_store.project_dir(name)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive_root = project_store.DATA_ROOT / ".archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / f"{timestamp}-{name}"
    shutil.move(str(src), str(target))
    return target


def _find_stalled_projects(days: int) -> list[tuple[str, ProjectState]]:
    """Return (name, state) for projects whose last_updated is older than
    `days` days and whose phase is not COMPLETED. Projects with unreadable
    state or unparseable timestamps are skipped.
    """
    threshold = datetime.now(timezone.utc) - timedelta(days=days)
    stalled: list[tuple[str, ProjectState]] = []
    for name in project_store.list_projects():
        try:
            state = project_store.load_state(name)
        except (FileNotFoundError, ValueError):
            continue
        if state.phase == Phase.COMPLETED:
            continue
        try:
            last = datetime.fromisoformat(state.last_updated)
        except ValueError:
            continue
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if last < threshold:
            stalled.append((name, state))
    return stalled


@app.command()
def archive(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Move a project to the archive directory (non-destructive)."""
    try:
        target = _archive_project(name)
    except FileNotFoundError as exc:
        typer.echo(f"Error: project '{name}' not found ({exc})", err=True)
        raise typer.Exit(1) from exc
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Archived {name} -> {target}")


@app.command()
def clean(
    stalled: bool = typer.Option(
        False, "--stalled", help="Find projects stuck for >--days days",
    ),
    days: int = typer.Option(
        7, "--days", help="Stalled threshold in days (default: 7)",
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Actually archive matches (default: dry-run)",
    ),
) -> None:
    """List or archive stalled projects.

    Without --yes this is a dry run. Stalled projects are those whose
    last_updated is more than `--days` days old and whose phase is not
    COMPLETED. Action is always archive (non-destructive); no data is deleted.
    """
    if not stalled:
        typer.echo(
            "Error: --stalled is required (no other selection modes yet).",
            err=True,
        )
        raise typer.Exit(1)

    matches = _find_stalled_projects(days)
    if not matches:
        typer.echo(f"No stalled projects (>{days} days, not COMPLETED).")
        return

    action_label = "archive" if yes else "would archive"
    header = f"Found {len(matches)} stalled project(s) (>{days} days, not COMPLETED):"
    typer.echo(header)
    for proj_name, state in matches:
        typer.echo(
            f"  {proj_name}\tphase={state.phase}"
            f"\tlast_updated={state.last_updated}\t-> {action_label}"
        )

    if not yes:
        typer.echo("\nDry-run. Pass --yes to archive.")
        return

    typer.echo("")
    for proj_name, _state in matches:
        target = _archive_project(proj_name)
        typer.echo(f"  archived {proj_name} -> {target}")


@app.command()
def migrate(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Migrate a project from pre-pass storage layout to pass-scoped layout.

    Moves `model-select/{initial-analysis.json, evaluation.json, results/...}`
    under `model-select/pass-001/`, and each `style-refine/round-NNN/` under
    `style-refine/pass-001/round-NNN/`. Safe to re-run.
    """
    from styleclaw.scripts.migrate import migrate_project

    try:
        result = migrate_project(name)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not result.anything_migrated:
        typer.echo(f"{name}: nothing to migrate.")
        return

    if result.model_select_migrated:
        typer.echo(f"{name}: migrated model-select/ → model-select/pass-001/")
    if result.style_refine_rounds_migrated:
        rounds = ", ".join(f"round-{r:03d}" for r in result.style_refine_rounds_migrated if r > 0)
        typer.echo(f"{name}: migrated style-refine rounds → pass-001/ ({rounds})")


@app.command()
def analyze(
    name: str = typer.Argument(..., help="Project name"),
    show_thinking: bool = typer.Option(
        True, "--show-thinking/--no-show-thinking", help="Show LLM reasoning process (default: on)",
    ),
    thinking_budget: int = typer.Option(
        5000, "--thinking-budget", help="Thinking token budget",
    ),
) -> None:
    """Analyze reference images and generate initial trigger phrase."""
    state = project_store.load_state(name)
    if state.phase != Phase.INIT:
        typer.echo(f"Error: Project must be in INIT phase (current: {state.phase})", err=True)
        raise typer.Exit(1)

    result = _run_action(
        name, "analyze",
        show_thinking=show_thinking, thinking_budget=thinking_budget,
    )
    typer.echo(f"Analysis complete. {result.message}")
    if show_thinking:
        md = (
            project_store.project_dir(name)
            / "model-select" / "pass-001" / "initial-analysis.thinking.md"
        )
        if md.exists():
            typer.echo("\n--- LLM thinking ---")
            typer.echo(md.read_text(encoding="utf-8"))
            typer.echo("--- end thinking ---\n")
    state = project_store.load_state(name)
    typer.echo(f"Phase advanced to: {state.phase}")


@app.command()
def generate(
    name: str = typer.Argument(..., help="Project name"),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="Retry only failed tasks"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-submit even if SUCCESS record exists"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show planned operations and exit"),
) -> None:
    """Submit generation tasks (auto-detects phase)."""
    state = project_store.load_state(name)

    if state.phase == Phase.STYLE_REFINE and state.current_round < 1:
        typer.echo("Error: Run 'refine' first to set up a round.", err=True)
        raise typer.Exit(1)

    if state.phase not in (Phase.MODEL_SELECT, Phase.STYLE_REFINE):
        typer.echo(f"Error: Cannot generate in {state.phase} phase.", err=True)
        raise typer.Exit(1)

    if dry_run:
        from styleclaw.providers.runninghub.models import MODEL_REGISTRY
        if state.phase == Phase.MODEL_SELECT:
            models = list(MODEL_REGISTRY.keys())
            # 2 variants × 2 genders per model
            est_tasks = len(models) * 2 * 2
            typer.echo("[dry-run] generate (MODEL_SELECT)")
            typer.echo(f"  Models: {', '.join(models)}")
            typer.echo(f"  Estimated tasks: {est_tasks}")
        else:
            typer.echo("[dry-run] generate (STYLE_REFINE)")
            typer.echo(f"  Models: {', '.join(state.selected_models) or '(none)'}")
            typer.echo(f"  Round:  {state.current_round}")
            typer.echo(f"  Estimated tasks: {len(state.selected_models)}")
        return

    result = _run_action(name, "generate", {"retry_failed": retry_failed, "force": force})
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)


@app.command()
def poll(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Poll pending tasks and download completed images (auto-detects phase)."""
    state = project_store.load_state(name)

    valid_phases = (Phase.MODEL_SELECT, Phase.STYLE_REFINE, Phase.BATCH_T2I, Phase.BATCH_I2I)
    if state.phase not in valid_phases:
        typer.echo(f"Error: Nothing to poll in {state.phase} phase.", err=True)
        raise typer.Exit(1)

    result = _run_action(name, "poll")
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)


@app.command()
def evaluate(
    name: str = typer.Argument(..., help="Project name"),
    show_thinking: bool = typer.Option(
        True, "--show-thinking/--no-show-thinking", help="Show LLM reasoning process (default: on)",
    ),
    thinking_budget: int = typer.Option(
        5000, "--thinking-budget", help="Thinking token budget",
    ),
) -> None:
    """Evaluate generated images against reference style (auto-detects phase)."""
    state = project_store.load_state(name)

    if state.phase not in (Phase.MODEL_SELECT, Phase.STYLE_REFINE):
        typer.echo(f"Error: Cannot evaluate in {state.phase} phase.", err=True)
        raise typer.Exit(1)

    result = _run_action(
        name, "evaluate",
        show_thinking=show_thinking, thinking_budget=thinking_budget,
    )
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)

    if show_thinking:
        project_dir = project_store.project_dir(name)
        if state.phase == Phase.MODEL_SELECT:
            pass_num = state.current_model_select_pass or 1
            md = project_dir / "model-select" / f"pass-{pass_num:03d}" / "evaluation.thinking.md"
        else:
            md = (
                project_dir / "style-refine"
                / f"round-{state.current_round:03d}" / "evaluation.thinking.md"
            )
        if md.exists():
            typer.echo("\n--- LLM thinking ---")
            typer.echo(md.read_text(encoding="utf-8"))
            typer.echo("--- end thinking ---\n")


@app.command(name="select-model")
def select_model(
    name: str = typer.Argument(..., help="Project name"),
    models: str = typer.Option(..., "--models", help="Comma-separated model IDs"),
    variant: str = typer.Option("", "--variant", help="Variant to use in STYLE_REFINE: prompt-sref or prompt-only"),
) -> None:
    """Confirm selected models for style refinement. Works in MODEL_SELECT or STYLE_REFINE phase."""
    state = project_store.load_state(name)
    if state.phase not in (Phase.MODEL_SELECT, Phase.STYLE_REFINE):
        typer.echo(
            f"Error: Project must be in MODEL_SELECT or STYLE_REFINE phase (current: {state.phase})",
            err=True,
        )
        raise typer.Exit(1)

    if variant and variant not in ("prompt-sref", "prompt-only"):
        typer.echo(f"Error: --variant must be 'prompt-sref' or 'prompt-only', got '{variant}'", err=True)
        raise typer.Exit(1)

    from styleclaw.providers.runninghub.models import MODEL_REGISTRY

    selected = [m.strip() for m in models.split(",")]
    for m in selected:
        if m not in MODEL_REGISTRY:
            typer.echo(f"Error: Unknown model '{m}'. Available: {list(MODEL_REGISTRY.keys())}", err=True)
            raise typer.Exit(1)

    result = _run_action(name, "select-model", {"models": models, "variant": variant})
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)


@app.command()
def refine(
    name: str = typer.Argument(..., help="Project name"),
    direction: str = typer.Option("", "--direction", help="Human direction for refinement"),
    show_thinking: bool = typer.Option(
        True, "--show-thinking/--no-show-thinking", help="Show LLM reasoning process (default: on)",
    ),
    thinking_budget: int = typer.Option(
        5000, "--thinking-budget", help="Thinking token budget",
    ),
) -> None:
    """Refine trigger phrase using LLM (one round)."""
    state = project_store.load_state(name)
    if state.phase != Phase.STYLE_REFINE:
        typer.echo(f"Error: Project must be in STYLE_REFINE phase (current: {state.phase})", err=True)
        raise typer.Exit(1)

    if state.current_round + 1 > MAX_AUTO_ROUNDS:
        typer.echo(
            f"Error: Reached max auto rounds ({MAX_AUTO_ROUNDS}). "
            f"Use 'approve' to advance or 'adjust --direction ...' to continue manually.",
            err=True,
        )
        raise typer.Exit(1)

    result = _run_action(
        name, "refine", {"direction": direction},
        show_thinking=show_thinking, thinking_budget=thinking_budget,
    )
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)

    if show_thinking:
        new_state = project_store.load_state(name)
        md = (
            project_store.project_dir(name) / "style-refine"
            / f"round-{new_state.current_round:03d}" / "prompt.thinking.md"
        )
        if md.exists():
            typer.echo("\n--- LLM thinking ---")
            typer.echo(md.read_text(encoding="utf-8"))
            typer.echo("--- end thinking ---\n")


@app.command()
def approve(
    name: str = typer.Argument(..., help="Project name"),
    phase: str = typer.Option("batch-t2i", "--phase", help="Target phase: batch-t2i or completed"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Approve current style and advance to next phase."""
    state = project_store.load_state(name)

    if phase == "batch-t2i":
        if state.phase != Phase.STYLE_REFINE:
            typer.echo(f"Error: Must be in STYLE_REFINE (current: {state.phase})", err=True)
            raise typer.Exit(1)

        trigger = _get_current_trigger(name, state)
        typer.echo("=== Approve & Start Batch Testing ===")
        typer.echo(f"  Models:  {', '.join(state.selected_models)}")
        typer.echo(f"  Trigger: {trigger}")
        typer.echo(f"  Next:    BATCH_T2I (100 test cases)")

        if not yes and not typer.confirm("Proceed?"):
            typer.echo("Cancelled.")
            raise typer.Exit(0)

        result = _run_action(name, "approve", {"target": "batch-t2i"})
    elif phase == "completed":
        if state.phase != Phase.BATCH_I2I:
            typer.echo(f"Error: Must be in BATCH_I2I (current: {state.phase})", err=True)
            raise typer.Exit(1)

        typer.echo("=== Mark Project Completed ===")
        typer.echo(f"  Models: {', '.join(state.selected_models)}")

        if not yes and not typer.confirm("Proceed?"):
            typer.echo("Cancelled.")
            raise typer.Exit(0)

        result = _run_action(name, "approve", {"target": "completed"})
    else:
        typer.echo(f"Error: Unknown target phase '{phase}'", err=True)
        raise typer.Exit(1)

    typer.echo(f"Phase advanced to: {project_store.load_state(name).phase}")


def _get_current_trigger(name: str, state: ProjectState) -> str:
    pass_num = state.current_model_select_pass or 1
    if state.current_round >= 1:
        prompt_config = project_store.load_prompt_config(
            name, state.current_round, pass_num=pass_num,
        )
        return prompt_config.trigger_phrase
    try:
        analysis = project_store.load_analysis(name, pass_num=pass_num)
        return analysis.trigger_phrase
    except FileNotFoundError:
        logging.getLogger(__name__).warning("Analysis file not found for project '%s'", name)
        return "(not found — run 'analyze' first)"


@app.command()
def adjust(
    name: str = typer.Argument(..., help="Project name"),
    direction: str = typer.Option(..., "--direction", help="Adjustment direction"),
    show_thinking: bool = typer.Option(
        True, "--show-thinking/--no-show-thinking", help="Show LLM reasoning process (default: on)",
    ),
    thinking_budget: int = typer.Option(
        5000, "--thinking-budget", help="Thinking token budget",
    ),
) -> None:
    """Give adjustment direction then refine (shortcut for refine --direction)."""
    refine(
        name=name, direction=direction,
        show_thinking=show_thinking, thinking_budget=thinking_budget,
    )


@app.command()
def rollback(
    name: str = typer.Argument(..., help="Project name"),
    to: str = typer.Option(..., "--to", help="Target phase to rollback to"),
    round_num: Optional[int] = typer.Option(None, "--round", help="Target round number"),
) -> None:
    """Rollback project to an earlier phase."""
    from styleclaw.core.state_machine import rollback as do_rollback

    state = project_store.load_state(name)
    try:
        target = Phase(to.upper())
    except ValueError:
        valid = ", ".join(p.value for p in Phase)
        typer.echo(f"Error: Invalid phase '{to}'. Valid phases: {valid}", err=True)
        raise typer.Exit(1)

    new_state = do_rollback(state, target)
    if round_num is not None:
        if round_num < 0:
            typer.echo(f"Error: Round number must be non-negative, got {round_num}", err=True)
            raise typer.Exit(1)
        if target == Phase.STYLE_REFINE and round_num > 0:
            style_refine_root = project_store.project_dir(name) / "style-refine"
            pass_dirs = sorted(style_refine_root.glob(f"pass-*/round-{round_num:03d}"))
            if not pass_dirs:
                typer.echo(f"Error: Round {round_num} does not exist on disk.", err=True)
                raise typer.Exit(1)
        new_state = new_state.with_round(round_num)
    project_store.save_state(name, new_state)

    typer.echo(f"Rolled back to {new_state.phase} (round={new_state.current_round})")


@app.command(name="set-sref")
def set_sref(
    name: str = typer.Argument(..., help="Project name"),
    index: int = typer.Argument(..., help="0-based index of ref image to use as sref"),
) -> None:
    """Set which reference image to use as style reference (sref)."""
    config = project_store.load_config(name)
    if index < 0 or index >= len(config.ref_images):
        typer.echo(f"Error: index {index} out of range (0–{len(config.ref_images)-1})", err=True)
        raise typer.Exit(1)
    new_config = config.model_copy(update={"sref_index": index})
    project_store.save_config(name, new_config)
    typer.echo(f"sref set to ref-{index+1:03d}: {config.ref_images[index]}")
    state = project_store.load_state(name)
    if state.phase == Phase.MODEL_SELECT:
        typer.echo(
            "Hint: existing model-select SUCCESS tasks are not auto-invalidated. "
            f"To regenerate with this sref: styleclaw generate {name} --force",
        )


@app.command(name="set-pass")
def set_pass(
    name: str = typer.Argument(..., help="Project name"),
    pass_num: int = typer.Argument(..., help="Pass number to switch to (1-based)"),
) -> None:
    """Switch the active model-select pass (e.g. after deleting a bad pass)."""
    state = project_store.load_state(name)
    new_state = state.with_model_select_pass(pass_num)
    project_store.save_state(name, new_state)
    typer.echo(f"Active pass set to {pass_num}")


@app.command(name="retest-models")
def retest_models_cmd(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Re-enter MODEL_SELECT to re-test all models with the current trigger."""
    state = project_store.load_state(name)
    if state.phase not in (Phase.STYLE_REFINE, Phase.BATCH_T2I):
        typer.echo(
            f"Error: retest-models requires STYLE_REFINE or BATCH_T2I "
            f"(current: {state.phase})",
            err=True,
        )
        raise typer.Exit(1)

    result = _run_action(name, "retest-models")
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)
    typer.echo(
        "Next: run 'generate', 'poll', 'evaluate', then 'select-model' to pick a model."
    )


@app.command(name="back-to-t2i")
def back_to_t2i_cmd(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Return from BATCH_I2I to BATCH_T2I when i2i results are unsatisfying."""
    state = project_store.load_state(name)
    if state.phase != Phase.BATCH_I2I:
        typer.echo(
            f"Error: back-to-t2i requires BATCH_I2I phase (current: {state.phase})",
            err=True,
        )
        raise typer.Exit(1)

    result = _run_action(name, "back-to-t2i")
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)


@app.command(name="design-cases")
def design_cases_cmd(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Design 100 test cases using LLM."""
    state = project_store.load_state(name)
    if state.phase != Phase.BATCH_T2I:
        typer.echo(f"Error: Must be in BATCH_T2I phase (current: {state.phase})", err=True)
        raise typer.Exit(1)

    result = _run_action(name, "design-cases")
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)


@app.command(name="batch-submit")
def batch_submit_cmd(
    name: str = typer.Argument(..., help="Project name"),
    i2i: bool = typer.Option(False, "--i2i", help="Submit image-to-image batch"),
    model: Optional[str] = typer.Option(None, "--model", help="Model ID (defaults to first selected)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show planned operations and exit"),
) -> None:
    """Submit batch generation tasks."""
    state = project_store.load_state(name)

    if i2i:
        if state.phase != Phase.BATCH_I2I:
            typer.echo(f"Error: Must be in BATCH_I2I phase (current: {state.phase})", err=True)
            raise typer.Exit(1)
    else:
        if state.phase != Phase.BATCH_T2I:
            typer.echo(f"Error: Must be in BATCH_T2I phase (current: {state.phase})", err=True)
            raise typer.Exit(1)

    model_id = model or (state.selected_models[0] if state.selected_models else None)
    if not model_id:
        typer.echo("Error: No model selected.", err=True)
        raise typer.Exit(1)

    if dry_run:
        if i2i:
            uploads = project_store.load_i2i_uploads(name, state.current_batch or 1)
            typer.echo("[dry-run] batch-submit --i2i")
            typer.echo(f"  Model: {model_id}")
            typer.echo(f"  Batch: {state.current_batch}")
            typer.echo(f"  Reference uploads: {len(uploads)}")
        else:
            try:
                cfg = project_store.load_batch_config(name, state.current_batch or 1)
                pending = sum(1 for c in cfg.cases if c.status == "pending")
                total = len(cfg.cases)
            except FileNotFoundError:
                pending = 0
                total = 0
            typer.echo("[dry-run] batch-submit")
            typer.echo(f"  Model: {model_id}")
            typer.echo(f"  Batch: {state.current_batch}")
            typer.echo(f"  Cases: {pending} pending of {total} total")
        return

    result = _run_action(name, "batch-submit", {"model": model_id})
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)


@app.command()
def report(
    name: str = typer.Argument(..., help="Project name"),
    i2i: bool = typer.Option(False, "--i2i", help="Generate i2i report instead of t2i"),
) -> None:
    """Generate HTML report (auto-detects phase)."""
    from styleclaw.scripts.report import (
        generate_batch_i2i_report,
        generate_batch_t2i_report,
        generate_model_select_report,
        generate_style_refine_report,
    )

    state = project_store.load_state(name)

    if state.phase == Phase.MODEL_SELECT:
        path = generate_model_select_report(name)
        typer.echo(f"Model-select report generated: {path}")
    elif state.phase == Phase.STYLE_REFINE:
        path = generate_style_refine_report(name, state.current_round)
        typer.echo(f"Report generated: {path}")
    elif state.phase == Phase.BATCH_I2I or i2i:
        path = generate_batch_i2i_report(name, state.current_batch)
        typer.echo(f"I2I report generated: {path}")
    elif state.phase == Phase.BATCH_T2I:
        path = generate_batch_t2i_report(name, state.current_batch)
        typer.echo(f"T2I report generated: {path}")
    else:
        typer.echo(f"Error: No report available in {state.phase} phase.", err=True)
        raise typer.Exit(1)


@app.command(name="add-refs")
def add_refs(
    name: str = typer.Argument(..., help="Project name"),
    images: list[Path] = typer.Option(..., "--images", help="Reference image paths for i2i"),
) -> None:
    """Add reference images for image-to-image batch testing."""
    import shutil

    from styleclaw.core.models import UploadRecord
    from styleclaw.providers.runninghub.client import RunningHubClient
    from styleclaw.providers.runninghub.upload import upload_file

    state = project_store.load_state(name)
    if state.phase not in (Phase.BATCH_T2I, Phase.BATCH_I2I):
        typer.echo(f"Error: Must be in BATCH_T2I or BATCH_I2I phase (current: {state.phase})", err=True)
        raise typer.Exit(1)

    from styleclaw.core.image_utils import verify_ref_image

    invalid: list[tuple[Path, str]] = []
    for p in images:
        try:
            verify_ref_image(p)
        except ValueError as exc:
            invalid.append((p, str(exc)))
    if invalid:
        for _, msg in invalid:
            typer.echo(f"Error: {msg}", err=True)
        raise typer.Exit(1)

    if state.phase == Phase.BATCH_T2I:
        new_state = advance(state, Phase.BATCH_I2I)
        project_store.save_state(name, new_state)
        state = new_state

    batch_num = state.current_batch or 1
    source_dir = project_store.batch_i2i_dir(name, batch_num) / "source-images"
    source_dir.mkdir(parents=True, exist_ok=True)

    local_dests: list[Path] = []
    for img_path in images:
        dest = source_dir / img_path.name
        shutil.copy2(img_path, dest)
        local_dests.append(dest)

    async def _upload_all() -> tuple[dict[int, UploadRecord], list[tuple[int, str]]]:
        results: dict[int, UploadRecord] = {}
        errors: list[tuple[int, str]] = []

        async with RunningHubClient(api_key=_get_api_key()) as client:
            async def _one(idx: int, dest: Path) -> None:
                try:
                    results[idx] = await upload_file(client, dest)
                    typer.echo(f"  Uploaded {idx + 1}/{len(local_dests)}: {dest.name}")
                except (RuntimeError, ValueError) as exc:
                    errors.append((idx, str(exc)))
                    typer.echo(f"  Failed   {idx + 1}/{len(local_dests)}: {dest.name} ({exc})", err=True)

            async with asyncio.TaskGroup() as tg:
                for idx, dest in enumerate(local_dests):
                    tg.create_task(_one(idx, dest))
        return results, errors

    results, errors = asyncio.run(_upload_all())
    new_records = [results[i] for i in sorted(results)]

    if new_records:
        existing_records = project_store.load_i2i_uploads(name, batch_num)
        upload_records = existing_records + new_records
        project_store.save_i2i_uploads(name, batch_num, upload_records)
    else:
        upload_records = project_store.load_i2i_uploads(name, batch_num)

    if state.current_batch != batch_num:
        new_state = state.with_batch(batch_num)
        project_store.save_state(name, new_state)

    typer.echo(
        f"Added {len(new_records)}/{len(local_dests)} reference images for i2i batch {batch_num}."
    )
    if errors:
        typer.echo(f"Warning: {len(errors)} uploads failed — see messages above.", err=True)
        raise typer.Exit(1)


def _confirm_select_model(
    action_name: str,
    args: dict[str, Any],
    ctx: "ExecutionContext",
) -> dict[str, Any] | None:
    """Prompt user to confirm or override model selection."""
    from styleclaw.providers.runninghub.models import MODEL_REGISTRY

    try:
        evaluation = project_store.load_evaluation(ctx.project)
    except FileNotFoundError:
        evaluation = None

    typer.echo("\n=== 模型选择确认 ===")
    if evaluation:
        typer.echo(f"  LLM 推荐: {evaluation.recommendation}")
        if evaluation.recommended_variant:
            typer.echo(f"  推荐方案: {evaluation.recommended_variant}")
        typer.echo("  各模型评分:")
        for ev in evaluation.evaluations:
            label = f"{ev.model}"
            if ev.variant:
                label += f" [{ev.variant}]"
            typer.echo(f"    {label:30s} total={ev.total:.1f}")
        typer.echo("")

    default_models = args.get("models", "")
    if not default_models and evaluation:
        default_models = evaluation.recommendation

    available = list(MODEL_REGISTRY.keys())
    typer.echo(f"  可选模型: {', '.join(available)}")

    while True:
        user_input = typer.prompt(
            "  选择模型 (逗号分隔, 回车使用推荐)",
            default=default_models,
        )

        if not user_input or not user_input.strip():
            typer.echo("  已取消。")
            return None

        selected = [m.strip() for m in user_input.strip().split(",")]
        invalid = [m for m in selected if m not in MODEL_REGISTRY]
        if invalid:
            typer.echo(f"  ✗ 无效的模型 ID: {', '.join(invalid)}")
            typer.echo(f"  提示: 只填模型 ID（如 mj-v7），不需要带 prompt-sref / prompt-only")
            typer.echo(f"  可选模型: {', '.join(available)}")
            continue
        break

    # Ask which variant to use in STYLE_REFINE
    default_variant = evaluation.recommended_variant if evaluation and evaluation.recommended_variant else "prompt-sref"
    while True:
        variant_input = typer.prompt(
            "  出图方案 (prompt-sref / prompt-only, 回车使用推荐)",
            default=default_variant,
        )
        variant = variant_input.strip()
        if variant in ("prompt-sref", "prompt-only"):
            break
        typer.echo("  ✗ 请输入 prompt-sref 或 prompt-only")

    return {**args, "models": ", ".join(selected), "variant": variant}


def _confirm_init(
    action_name: str,
    args: dict[str, Any],
    ctx: "ExecutionContext",
) -> dict[str, Any] | None:
    """Prompt user to confirm new-project parameters: ref_dir, ip_info, etc."""
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}

    typer.echo("\n=== 创建新项目 ===")
    typer.echo(f"  项目名: {ctx.project}")

    while True:
        ref_dir_str = typer.prompt(
            "  参考图目录 (绝对路径或~开头)",
            default=args.get("ref_dir", "") or "",
        )
        ref_dir = Path(ref_dir_str.strip()).expanduser()
        if not ref_dir.is_dir():
            typer.echo(f"  ✗ 目录不存在: {ref_dir}")
            continue
        found = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in image_exts)
        if not found:
            typer.echo(f"  ✗ 目录里没有支持的图片 (.png/.jpg/.jpeg/.webp): {ref_dir}")
            continue
        typer.echo(f"  ✓ 发现 {len(found)} 张图片: {', '.join(p.name for p in found)}")
        break

    ip_info = typer.prompt(
        "  IP / 风格描述",
        default=args.get("ip_info", "") or "",
    ).strip()
    if not ip_info:
        typer.echo("  ✗ IP 描述不能为空。")
        return None

    description = args.get("description", "") or ""
    return {
        **args,
        "ref_dir": str(ref_dir),
        "ip_info": ip_info,
        "description": description,
    }


def _confirm_add_refs(
    action_name: str,
    args: dict[str, Any],
    ctx: "ExecutionContext",
) -> dict[str, Any] | None:
    """Prompt user for the directory of i2i reference images to add."""
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}

    typer.echo("\n=== 添加图生图参考图 ===")
    typer.echo(f"  项目: {ctx.project}")

    while True:
        image_dir_str = typer.prompt(
            "  i2i 参考图目录 (绝对路径或~开头)",
            default=args.get("image_dir", "") or "",
        )
        image_dir = Path(image_dir_str.strip()).expanduser()
        if not image_dir.is_dir():
            typer.echo(f"  ✗ 目录不存在: {image_dir}")
            continue
        found = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in image_exts)
        if not found:
            typer.echo(f"  ✗ 目录里没有支持的图片: {image_dir}")
            continue
        typer.echo(f"  ✓ 发现 {len(found)} 张图片: {', '.join(p.name for p in found)}")
        break

    return {**args, "image_dir": str(image_dir)}


def _confirm_dispatch(
    action_name: str,
    args: dict[str, Any],
    ctx: "ExecutionContext",
) -> dict[str, Any] | None:
    """Route to the per-action confirmation callback. Unknown actions pass
    through unchanged so future ``requires_confirmation`` actions don't
    silently lose their args."""
    if action_name == "select-model":
        return _confirm_select_model(action_name, args, ctx)
    if action_name == "init":
        return _confirm_init(action_name, args, ctx)
    if action_name == "add-refs":
        return _confirm_add_refs(action_name, args, ctx)
    return args


@app.command()
def run(
    intent: str = typer.Argument(..., help="Natural language description of what to do"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan only; print the plan and exit without executing",
    ),
    show_thinking: bool = typer.Option(
        True, "--show-thinking/--no-show-thinking", help="Show LLM reasoning process (default: on)",
    ),
    thinking_budget: int = typer.Option(
        5000, "--thinking-budget", help="Thinking token budget",
    ),
) -> None:
    """Run actions from natural language intent (plan-then-execute)."""
    if project is None:
        projects = project_store.list_projects()
        if len(projects) == 1:
            project = projects[0]
        elif not projects:
            typer.echo(
                "Error: No projects yet. Pass --project NAME to create one via natural language.",
                err=True,
            )
            raise typer.Exit(1)
        else:
            typer.echo("Error: Multiple projects found. Specify --project.", err=True)
            typer.echo(f"  Available: {', '.join(projects)}", err=True)
            raise typer.Exit(1)

    from styleclaw.orchestrator.actions import ACTION_REGISTRY
    from styleclaw.orchestrator.executor import display_plan, execute
    from styleclaw.orchestrator.planner import plan

    async def _plan_and_execute() -> None:
        llm = _build_llm_provider()
        try:
            action_plan = await plan(llm, project, intent)
        finally:
            await _close_resource(llm, "llm")

        display_plan(action_plan, project)

        if dry_run:
            typer.echo("(dry-run) 未执行；去掉 --dry-run 后再跑即可")
            return

        if not yes and not typer.confirm("Execute?"):
            typer.echo("Cancelled.")
            raise typer.Exit(0)

        needs_client = any(
            ACTION_REGISTRY.get(s.name) and ACTION_REGISTRY[s.name].needs_client
            for s in action_plan.steps
        )
        needs_llm = any(
            ACTION_REGISTRY.get(s.name) and ACTION_REGISTRY[s.name].needs_llm
            for s in action_plan.steps
        )

        def _on_start(i: int, name: str, desc: str) -> None:
            typer.echo(f"\n  [{i + 1}/{len(action_plan.steps)}] {name} — {desc}")

        def _on_done(i: int, name: str, result: StepResult) -> None:
            if result.ok:
                typer.echo(f"  -> {result.message}")
            else:
                typer.echo(f"  x  {result.message}", err=True)

        confirm_fn = None if yes else _confirm_dispatch

        async with _build_context(
            project, needs_client, needs_llm,
            show_thinking=show_thinking, thinking_budget=thinking_budget,
        ) as ctx:
            results = await execute(
                action_plan, ctx,
                on_step_start=_on_start,
                on_step_done=_on_done,
                on_confirm=confirm_fn,
            )
            if results and not results[-1].ok:
                raise typer.Exit(1)

    asyncio.run(_plan_and_execute())

    if not dry_run:
        from styleclaw.orchestrator.suggestions import suggest_next_steps
        try:
            suggestions = suggest_next_steps(project)
        except FileNotFoundError:
            suggestions = []
        if suggestions:
            typer.echo("\n下一步可以这样说：")
            for line in suggestions:
                typer.echo(f"  {line}")
        typer.echo("\nDone.")


if __name__ == "__main__":
    app()
