from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from styleclaw.core.models import Phase, ProjectState
from styleclaw.core.state_machine import advance
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


def _get_client():
    from styleclaw.providers.runninghub.client import RunningHubClient
    return RunningHubClient(api_key=_get_api_key())


async def _run_with_client(coro_fn):
    client = _get_client()
    try:
        return await coro_fn(client)
    finally:
        await client.close()


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

    from styleclaw.scripts.init_project import init_project

    root = asyncio.run(_run_with_client(
        lambda c: init_project(name, ref, info, description, c)
    ))
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
    typer.echo(f"Updated: {state.last_updated}")
    if config.ip_info:
        typer.echo(f"IP Info: {config.ip_info[:100]}")


@app.command()
def analyze(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Analyze reference images and generate initial trigger phrase."""
    state = project_store.load_state(name)
    if state.phase != Phase.INIT:
        typer.echo(f"Error: Project must be in INIT phase (current: {state.phase})", err=True)
        raise typer.Exit(1)

    config = project_store.load_config(name)
    root = project_store.project_dir(name)
    ref_paths = [root / r for r in config.ref_images]

    from styleclaw.agents.analyze_style import analyze_style
    from styleclaw.providers.llm.bedrock import BedrockProvider

    llm = BedrockProvider()
    analysis = asyncio.run(analyze_style(llm, ref_paths, config.ip_info))
    project_store.save_analysis(name, analysis)

    new_state = advance(state, Phase.MODEL_SELECT)
    project_store.save_state(name, new_state)

    typer.echo(f"Analysis complete. Trigger phrase: {analysis.trigger_phrase}")
    typer.echo(f"Phase advanced to: {new_state.phase}")


@app.command()
def generate(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Submit generation tasks (auto-detects phase)."""
    state = project_store.load_state(name)

    if state.phase == Phase.MODEL_SELECT:
        _generate_model_select(name)
    elif state.phase == Phase.STYLE_REFINE:
        _generate_style_refine(name, state)
    else:
        typer.echo(f"Error: Cannot generate in {state.phase} phase.", err=True)
        raise typer.Exit(1)


def _generate_model_select(name: str) -> None:
    analysis = project_store.load_analysis(name)
    uploads = project_store.load_uploads(name)
    sref_url = uploads[0].url if uploads else ""

    from styleclaw.scripts.generate import generate_model_select

    records = asyncio.run(_run_with_client(
        lambda c: generate_model_select(name, c, analysis.trigger_phrase, sref_url=sref_url)
    ))

    for mid, rec in records.items():
        typer.echo(f"  {mid}: task_id={rec.task_id} status={rec.status}")


def _generate_style_refine(name: str, state: "ProjectState") -> None:
    round_num = state.current_round
    if round_num < 1:
        typer.echo("Error: Run 'refine' first to set up a round.", err=True)
        raise typer.Exit(1)

    prompt_config = project_store.load_prompt_config(name, round_num)
    uploads = project_store.load_uploads(name)
    sref_url = uploads[0].url if uploads else ""

    from styleclaw.scripts.generate import generate_style_refine

    records = asyncio.run(_run_with_client(
        lambda c: generate_style_refine(
            name, c, round_num, prompt_config.trigger_phrase,
            sref_url=sref_url,
            extra_model_params=prompt_config.model_params,
        )
    ))

    for mid, rec in records.items():
        typer.echo(f"  {mid}: task_id={rec.task_id} status={rec.status}")


@app.command()
def poll(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Poll pending tasks and download completed images (auto-detects phase)."""
    state = project_store.load_state(name)

    if state.phase == Phase.MODEL_SELECT:
        _poll_model_select(name)
    elif state.phase == Phase.STYLE_REFINE:
        _poll_style_refine(name, state)
    elif state.phase in (Phase.BATCH_T2I, Phase.BATCH_I2I):
        _poll_batch(name, state)
    else:
        typer.echo(f"Error: Nothing to poll in {state.phase} phase.", err=True)
        raise typer.Exit(1)


def _poll_model_select(name: str) -> None:
    from styleclaw.scripts.poll import poll_model_select

    records = asyncio.run(_run_with_client(
        lambda c: poll_model_select(name, c)
    ))

    for mid, rec in records.items():
        n_results = len(rec.results)
        typer.echo(f"  {mid}: status={rec.status} images={n_results}")


def _poll_style_refine(name: str, state: "ProjectState") -> None:
    from styleclaw.scripts.poll import poll_style_refine

    records = asyncio.run(_run_with_client(
        lambda c: poll_style_refine(name, c, state.current_round)
    ))

    for mid, rec in records.items():
        typer.echo(f"  {mid}: status={rec.status} images={len(rec.results)}")


def _poll_batch(name: str, state: "ProjectState") -> None:
    from styleclaw.scripts.poll import poll_batch

    phase = "i2i" if state.phase == Phase.BATCH_I2I else "t2i"
    records = asyncio.run(_run_with_client(
        lambda c: poll_batch(name, c, state.current_batch, phase=phase)
    ))

    completed = sum(1 for r in records.values() if r.status == "SUCCESS")
    typer.echo(f"Batch poll: {completed}/{len(records)} completed.")


@app.command()
def evaluate(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Evaluate generated images against reference style (auto-detects phase)."""
    state = project_store.load_state(name)

    if state.phase == Phase.MODEL_SELECT:
        _evaluate_model_select(name)
    elif state.phase == Phase.STYLE_REFINE:
        _evaluate_style_refine(name, state)
    else:
        typer.echo(f"Error: Cannot evaluate in {state.phase} phase.", err=True)
        raise typer.Exit(1)


def _evaluate_model_select(name: str) -> None:
    config = project_store.load_config(name)
    root = project_store.project_dir(name)
    ref_paths = [root / r for r in config.ref_images]

    model_images: dict[str, list[Path]] = {}
    records = project_store.load_all_task_records(name)
    for mid in records:
        results_dir = project_store.model_results_dir(name, mid)
        images = sorted(results_dir.glob("output-*.png"))
        if images:
            model_images[mid] = images

    if not model_images:
        typer.echo("Error: No generated images found. Run 'generate' and 'poll' first.", err=True)
        raise typer.Exit(1)

    from styleclaw.agents.select_model import evaluate_models
    from styleclaw.providers.llm.bedrock import BedrockProvider

    llm = BedrockProvider()
    evaluation = asyncio.run(evaluate_models(llm, ref_paths, model_images))
    project_store.save_evaluation(name, evaluation)

    typer.echo("Evaluation results:")
    for ev in evaluation.evaluations:
        typer.echo(f"  {ev.model}: total={ev.total:.1f} {ev.analysis[:60]}")
    typer.echo(f"Recommendation: {evaluation.recommendation}")

    from styleclaw.scripts.report import generate_model_select_report

    report_path = generate_model_select_report(name)
    typer.echo(f"\nReport generated: {report_path}")
    typer.echo("Review the report, then run 'select-model' to confirm which models to use.")
    typer.echo(f"  Example: styleclaw select-model {name} --models mj-v7,niji7")


def _evaluate_style_refine(name: str, state: "ProjectState") -> None:
    round_num = state.current_round

    config = project_store.load_config(name)
    root = project_store.project_dir(name)
    ref_paths = [root / r for r in config.ref_images]

    model_images: dict[str, list[Path]] = {}
    records = project_store.load_all_round_task_records(name, round_num)
    for mid in records:
        results_dir = project_store.round_results_dir(name, round_num, mid)
        images = sorted(results_dir.glob("output-*.png"))
        if images:
            model_images[mid] = images

    if not model_images:
        typer.echo("Error: No generated images for this round. Run 'generate' and 'poll' first.", err=True)
        raise typer.Exit(1)

    from styleclaw.agents.evaluate_result import evaluate_round
    from styleclaw.providers.llm.bedrock import BedrockProvider

    llm = BedrockProvider()
    evaluation = asyncio.run(evaluate_round(llm, ref_paths, model_images, round_num))
    project_store.save_round_evaluation(name, round_num, evaluation)

    typer.echo(f"Round {round_num} evaluation:")
    for ev in evaluation.evaluations:
        s = ev.scores
        typer.echo(
            f"  {ev.model}: color={s.color_palette} line={s.line_style} "
            f"light={s.lighting} texture={s.texture} mood={s.overall_mood} "
            f"total={ev.total:.1f}"
        )
    typer.echo(f"Recommendation: {evaluation.recommendation}")

    from styleclaw.scripts.report import generate_style_refine_report

    report_path = generate_style_refine_report(name, round_num)
    typer.echo(f"\nReport generated: {report_path}")

    if evaluation.should_approve():
        typer.echo("All scores meet threshold. Review the report, then run 'approve' to advance to BATCH_T2I.")
    elif evaluation.needs_human():
        typer.echo("Some scores are too low. Review the report and decide:")
        typer.echo(f"  - Adjust trigger: styleclaw adjust {name} --direction '...'")
        typer.echo(f"  - Switch models:  styleclaw select-model {name} --models ...")
        typer.echo(f"  - Approve anyway: styleclaw approve {name}")
    else:
        typer.echo(f"Direction: {evaluation.next_direction}")
        typer.echo("Review the report, then:")
        typer.echo(f"  - Continue refining: styleclaw refine {name}")
        typer.echo(f"  - Switch models:     styleclaw select-model {name} --models ...")
        typer.echo(f"  - Approve:           styleclaw approve {name}")


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

    if state.phase == Phase.MODEL_SELECT:
        new_state = advance(state, Phase.STYLE_REFINE)
        new_state = new_state.with_selected_models(selected)
        project_store.save_state(name, new_state)
        typer.echo(f"Selected models: {', '.join(selected)}")
        typer.echo(f"Phase advanced to: {new_state.phase}")
    else:
        new_state = state.with_selected_models(selected)
        project_store.save_state(name, new_state)
        typer.echo(f"Updated models: {', '.join(selected)} (staying in STYLE_REFINE)")
        typer.echo(f"Run 'refine' to generate new trigger phrases for these models.")


# --- Phase 3: STYLE_REFINE commands ---


MAX_AUTO_ROUNDS = 5


@app.command()
def refine(
    name: str = typer.Argument(..., help="Project name"),
    direction: str = typer.Option("", "--direction", help="Human direction for refinement"),
) -> None:
    """Refine trigger phrase using LLM (one round)."""
    state = project_store.load_state(name)
    if state.phase != Phase.STYLE_REFINE:
        typer.echo(f"Error: Project must be in STYLE_REFINE phase (current: {state.phase})", err=True)
        raise typer.Exit(1)

    config = project_store.load_config(name)
    root = project_store.project_dir(name)
    ref_paths = [root / r for r in config.ref_images]

    round_num = state.current_round + 1

    if round_num > MAX_AUTO_ROUNDS:
        typer.echo(
            f"Error: Reached max auto rounds ({MAX_AUTO_ROUNDS}). "
            f"Use 'approve' to advance or 'adjust --direction ...' to continue manually.",
            err=True,
        )
        raise typer.Exit(1)

    from styleclaw.agents.refine_prompt import refine_prompt
    from styleclaw.core.models import RoundEvaluation
    from styleclaw.providers.llm.bedrock import BedrockProvider

    evaluations: list[RoundEvaluation] = []
    for r in range(1, round_num):
        try:
            ev = project_store.load_round_evaluation(name, r)
            evaluations.append(ev)
        except FileNotFoundError:
            pass

    if round_num == 1:
        analysis = project_store.load_analysis(name)
        current_trigger = analysis.trigger_phrase
    else:
        prev_prompt = project_store.load_prompt_config(name, round_num - 1)
        current_trigger = prev_prompt.trigger_phrase

    llm = BedrockProvider()
    prompt_config = asyncio.run(refine_prompt(
        llm, ref_paths, current_trigger, round_num,
        config.ip_info, evaluations, direction,
    ))
    project_store.save_prompt_config(name, round_num, prompt_config)

    new_state = state.with_round(round_num)
    project_store.save_state(name, new_state)

    typer.echo(f"Round {round_num} trigger: {prompt_config.trigger_phrase}")
    if prompt_config.adjustment_note:
        typer.echo(f"Adjustments: {prompt_config.adjustment_note}")


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

        new_state = advance(state, Phase.BATCH_T2I)
    elif phase == "completed":
        if state.phase != Phase.BATCH_I2I:
            typer.echo(f"Error: Must be in BATCH_I2I (current: {state.phase})", err=True)
            raise typer.Exit(1)

        typer.echo("=== Mark Project Completed ===")
        typer.echo(f"  Models: {', '.join(state.selected_models)}")

        if not yes and not typer.confirm("Proceed?"):
            typer.echo("Cancelled.")
            raise typer.Exit(0)

        new_state = advance(state, Phase.COMPLETED)
    else:
        typer.echo(f"Error: Unknown target phase '{phase}'", err=True)
        raise typer.Exit(1)

    project_store.save_state(name, new_state)
    typer.echo(f"Phase advanced to: {new_state.phase}")


def _get_current_trigger(name: str, state: ProjectState) -> str:
    if state.current_round >= 1:
        prompt_config = project_store.load_prompt_config(name, state.current_round)
        return prompt_config.trigger_phrase
    try:
        analysis = project_store.load_analysis(name)
        return analysis.trigger_phrase
    except FileNotFoundError:
        return "(unknown)"


@app.command()
def adjust(
    name: str = typer.Argument(..., help="Project name"),
    direction: str = typer.Option(..., "--direction", help="Adjustment direction"),
) -> None:
    """Give adjustment direction then refine (shortcut for refine --direction)."""
    refine(name=name, direction=direction)


@app.command()
def rollback(
    name: str = typer.Argument(..., help="Project name"),
    to: str = typer.Option(..., "--to", help="Target phase to rollback to"),
    round_num: Optional[int] = typer.Option(None, "--round", help="Target round number"),
) -> None:
    """Rollback project to an earlier phase."""
    from styleclaw.core.state_machine import rollback as do_rollback

    state = project_store.load_state(name)
    target = Phase(to.upper())

    new_state = do_rollback(state, target)
    if round_num is not None:
        new_state = new_state.with_round(round_num)
    project_store.save_state(name, new_state)

    typer.echo(f"Rolled back to {new_state.phase} (round={new_state.current_round})")


# --- Phase 4: BATCH_T2I commands ---


@app.command(name="design-cases")
def design_cases_cmd(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Design 100 test cases using LLM."""
    state = project_store.load_state(name)
    if state.phase != Phase.BATCH_T2I:
        typer.echo(f"Error: Must be in BATCH_T2I phase (current: {state.phase})", err=True)
        raise typer.Exit(1)

    config = project_store.load_config(name)
    batch_num = state.current_batch + 1

    last_round = state.current_round
    prompt_config = project_store.load_prompt_config(name, last_round)

    from styleclaw.agents.design_cases import design_cases
    from styleclaw.providers.llm.bedrock import BedrockProvider

    llm = BedrockProvider()
    batch_config = asyncio.run(design_cases(
        llm, config.ip_info, prompt_config.trigger_phrase, batch_num,
    ))
    project_store.save_batch_config(name, batch_num, batch_config)

    new_state = state.with_batch(batch_num)
    project_store.save_state(name, new_state)

    typer.echo(f"Designed {len(batch_config.cases)} cases for batch {batch_num}.")
    typer.echo(f"Review/edit: {project_store.batch_t2i_dir(name, batch_num) / 'cases.json'}")


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

        model_id = model or (state.selected_models[0] if state.selected_models else None)
        if not model_id:
            typer.echo("Error: No model selected.", err=True)
            raise typer.Exit(1)

        last_round = state.current_round
        prompt_config = project_store.load_prompt_config(name, last_round)

        from styleclaw.scripts.batch_submit import batch_submit_i2i

        records = asyncio.run(_run_with_client(
            lambda c: batch_submit_i2i(
                name, c, state.current_batch, model_id, prompt_config.trigger_phrase,
            )
        ))

        typer.echo(f"Submitted {len(records)} i2i tasks.")
    else:
        if state.phase != Phase.BATCH_T2I:
            typer.echo(f"Error: Must be in BATCH_T2I phase (current: {state.phase})", err=True)
            raise typer.Exit(1)

        model_id = model or (state.selected_models[0] if state.selected_models else None)
        if not model_id:
            typer.echo("Error: No model selected.", err=True)
            raise typer.Exit(1)

        uploads = project_store.load_uploads(name)
        sref_url = uploads[0].url if uploads else ""

        from styleclaw.scripts.batch_submit import batch_submit_t2i

        records = asyncio.run(_run_with_client(
            lambda c: batch_submit_t2i(
                name, c, state.current_batch, model_id, sref_url=sref_url,
            )
        ))

        typer.echo(f"Submitted {len(records)} t2i tasks for batch {state.current_batch}.")


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


# --- Phase 5: BATCH_I2I commands ---


@app.command(name="add-refs")
def add_refs(
    name: str = typer.Argument(..., help="Project name"),
    images: list[Path] = typer.Option(..., "--images", help="Reference image paths for i2i"),
) -> None:
    """Add reference images for image-to-image batch testing."""
    import shutil

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

    from styleclaw.providers.runninghub.upload import upload_file

    async def _upload_all(client):
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

    upload_records = asyncio.run(_run_with_client(_upload_all))
    project_store.save_i2i_uploads(name, batch_num, upload_records)

    if state.current_batch != batch_num:
        new_state = state.with_batch(batch_num)
        project_store.save_state(name, new_state)

    typer.echo(f"Added {len(upload_records)} reference images for i2i batch {batch_num}.")


if __name__ == "__main__":
    app()
