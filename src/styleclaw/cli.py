from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import typer
from dotenv import load_dotenv

from styleclaw.core.config import MAX_AUTO_ROUNDS
from styleclaw.core.models import Phase, ProjectState, TaskStatus
from styleclaw.core.state_machine import advance
from styleclaw.orchestrator.actions import ExecutionContext, StepResult
from styleclaw.storage import project_store

load_dotenv()

app = typer.Typer(name="styleclaw", help="AI style trigger word exploration system")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _get_api_key() -> str:
    key = os.getenv("RUNNINGHUB_API_KEY")
    if not key:
        typer.echo("Error: RUNNINGHUB_API_KEY not set in environment.", err=True)
        raise typer.Exit(1)
    return key


@asynccontextmanager
async def _build_context(
    project: str,
    needs_client: bool = False,
    needs_llm: bool = False,
    show_thinking: bool = False,
    thinking_budget: int = 5000,
) -> AsyncIterator[ExecutionContext]:
    from styleclaw.providers.llm.bedrock import BedrockProvider
    from styleclaw.providers.runninghub.client import RunningHubClient

    client = None
    llm = None
    try:
        if needs_client:
            client = RunningHubClient(api_key=_get_api_key())
        if needs_llm:
            llm = BedrockProvider()
        yield ExecutionContext(
            project=project, client=client, llm=llm,
            show_thinking=show_thinking, thinking_budget=thinking_budget,
        )
    finally:
        if client:
            await client.close()
        if llm:
            await llm.close()


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
    ref: list[Path] = typer.Option(..., "--ref", help="Reference image paths"),
    info: str = typer.Option("", "--info", help="IP/style description"),
    description: str = typer.Option("", "--desc", help="Project description"),
) -> None:
    """Initialize a new project with reference images."""
    for r in ref:
        if not r.exists():
            typer.echo(f"Error: Reference image not found: {r}", err=True)
            raise typer.Exit(1)

    from styleclaw.providers.runninghub.client import RunningHubClient
    from styleclaw.scripts.init_project import init_project

    async def _exec() -> Path:
        async with RunningHubClient(api_key=_get_api_key()) as client:
            return await init_project(name, ref, info, description, client)

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


@app.command()
def analyze(
    name: str = typer.Argument(..., help="Project name"),
    show_thinking: bool = typer.Option(
        False, "--show-thinking", help="Capture and save LLM reasoning alongside output",
    ),
    thinking_budget: int = typer.Option(
        5000, "--thinking-budget", help="Thinking token budget (when --show-thinking)",
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
) -> None:
    """Submit generation tasks (auto-detects phase)."""
    state = project_store.load_state(name)

    if state.phase == Phase.STYLE_REFINE and state.current_round < 1:
        typer.echo("Error: Run 'refine' first to set up a round.", err=True)
        raise typer.Exit(1)

    if state.phase not in (Phase.MODEL_SELECT, Phase.STYLE_REFINE):
        typer.echo(f"Error: Cannot generate in {state.phase} phase.", err=True)
        raise typer.Exit(1)

    result = _run_action(name, "generate")
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
        False, "--show-thinking", help="Capture and save LLM reasoning alongside output",
    ),
    thinking_budget: int = typer.Option(
        5000, "--thinking-budget", help="Thinking token budget (when --show-thinking)",
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
) -> None:
    """Confirm selected models for style refinement. Works in MODEL_SELECT or STYLE_REFINE phase."""
    state = project_store.load_state(name)
    if state.phase not in (Phase.MODEL_SELECT, Phase.STYLE_REFINE):
        typer.echo(
            f"Error: Project must be in MODEL_SELECT or STYLE_REFINE phase (current: {state.phase})",
            err=True,
        )
        raise typer.Exit(1)

    from styleclaw.providers.runninghub.models import MODEL_REGISTRY

    selected = [m.strip() for m in models.split(",")]
    for m in selected:
        if m not in MODEL_REGISTRY:
            typer.echo(f"Error: Unknown model '{m}'. Available: {list(MODEL_REGISTRY.keys())}", err=True)
            raise typer.Exit(1)

    result = _run_action(name, "select-model", {"models": models})
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)


@app.command()
def refine(
    name: str = typer.Argument(..., help="Project name"),
    direction: str = typer.Option("", "--direction", help="Human direction for refinement"),
    show_thinking: bool = typer.Option(
        False, "--show-thinking", help="Capture and save LLM reasoning alongside output",
    ),
    thinking_budget: int = typer.Option(
        5000, "--thinking-budget", help="Thinking token budget (when --show-thinking)",
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
    if state.current_round >= 1:
        prompt_config = project_store.load_prompt_config(name, state.current_round)
        return prompt_config.trigger_phrase
    try:
        analysis = project_store.load_analysis(name)
        return analysis.trigger_phrase
    except FileNotFoundError:
        logging.getLogger(__name__).warning("Analysis file not found for project '%s'", name)
        return "(not found — run 'analyze' first)"


@app.command()
def adjust(
    name: str = typer.Argument(..., help="Project name"),
    direction: str = typer.Option(..., "--direction", help="Adjustment direction"),
    show_thinking: bool = typer.Option(
        False, "--show-thinking", help="Capture and save LLM reasoning alongside output",
    ),
    thinking_budget: int = typer.Option(
        5000, "--thinking-budget", help="Thinking token budget (when --show-thinking)",
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
        round_dir = project_store.project_dir(name) / "style-refine" / f"round-{round_num:03d}"
        if target == Phase.STYLE_REFINE and round_num > 0 and not round_dir.exists():
            typer.echo(f"Error: Round {round_num} does not exist on disk.", err=True)
            raise typer.Exit(1)
        new_state = new_state.with_round(round_num)
    project_store.save_state(name, new_state)

    typer.echo(f"Rolled back to {new_state.phase} (round={new_state.current_round})")


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

    from styleclaw.providers.runninghub.client import RunningHubClient
    from styleclaw.providers.runninghub.upload import upload_file

    state = project_store.load_state(name)
    if state.phase not in (Phase.BATCH_T2I, Phase.BATCH_I2I):
        typer.echo(f"Error: Must be in BATCH_T2I or BATCH_I2I phase (current: {state.phase})", err=True)
        raise typer.Exit(1)

    if state.phase == Phase.BATCH_T2I:
        new_state = advance(state, Phase.BATCH_I2I)
        project_store.save_state(name, new_state)
        state = new_state

    batch_num = state.current_batch or 1
    source_dir = project_store.batch_i2i_dir(name, batch_num) / "source-images"
    source_dir.mkdir(parents=True, exist_ok=True)

    async def _upload_all() -> list:
        async with RunningHubClient(api_key=_get_api_key()) as client:
            records = []
            for i, img_path in enumerate(images, 1):
                if not img_path.exists():
                    typer.echo(f"Error: Image not found: {img_path}", err=True)
                    raise typer.Exit(1)
                dest = source_dir / img_path.name
                shutil.copy2(img_path, dest)
                record = await upload_file(client, dest)
                records.append(record)
                typer.echo(f"  Uploaded {i}/{len(images)}: {img_path.name}")
            return records

    new_records = asyncio.run(_upload_all())
    existing_records = project_store.load_i2i_uploads(name, batch_num)
    upload_records = existing_records + new_records
    project_store.save_i2i_uploads(name, batch_num, upload_records)

    if state.current_batch != batch_num:
        new_state = state.with_batch(batch_num)
        project_store.save_state(name, new_state)

    typer.echo(f"Added {len(upload_records)} reference images for i2i batch {batch_num}.")


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
    user_input = typer.prompt(
        "  选择模型 (逗号分隔, 回车使用推荐)",
        default=default_models,
    )

    if not user_input or not user_input.strip():
        typer.echo("  已取消。")
        return None

    return {**args, "models": user_input.strip()}


@app.command()
def run(
    intent: str = typer.Argument(..., help="Natural language description of what to do"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    show_thinking: bool = typer.Option(
        False, "--show-thinking", help="Capture and save LLM reasoning alongside output",
    ),
    thinking_budget: int = typer.Option(
        5000, "--thinking-budget", help="Thinking token budget (when --show-thinking)",
    ),
) -> None:
    """Run actions from natural language intent (plan-then-execute)."""
    if project is None:
        projects = project_store.list_projects()
        if len(projects) == 1:
            project = projects[0]
        elif not projects:
            typer.echo("Error: No projects found. Run 'init' first.", err=True)
            raise typer.Exit(1)
        else:
            typer.echo("Error: Multiple projects found. Specify --project.", err=True)
            typer.echo(f"  Available: {', '.join(projects)}", err=True)
            raise typer.Exit(1)

    from styleclaw.orchestrator.actions import ACTION_REGISTRY
    from styleclaw.orchestrator.executor import display_plan, execute
    from styleclaw.orchestrator.planner import plan
    from styleclaw.providers.llm.bedrock import BedrockProvider
    from styleclaw.providers.runninghub.client import RunningHubClient

    async def _plan_and_execute() -> None:
        async with BedrockProvider() as llm:
            action_plan = await plan(llm, project, intent)

        display_plan(action_plan, project)

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

        confirm_fn = None if yes else _confirm_select_model

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
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
