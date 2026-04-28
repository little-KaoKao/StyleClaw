from __future__ import annotations

import logging
from typing import Any, Callable

import typer

from styleclaw.core.models import ActionPlan, LoopConfig
from styleclaw.orchestrator.actions import ACTION_REGISTRY, ExecutionContext, StepResult
from styleclaw.storage import project_store

logger = logging.getLogger(__name__)

ConfirmCallback = Callable[[str, dict[str, Any], ExecutionContext], dict[str, Any] | None]


def _should_continue_loop(ctx: ExecutionContext) -> bool:
    state = project_store.load_state(ctx.project)
    if state.current_round < 1:
        return False
    try:
        evaluation = project_store.load_round_evaluation(ctx.project, state.current_round)
    except FileNotFoundError:
        logger.warning("No evaluation found for round %d, stopping loop.", state.current_round)
        return False
    return not evaluation.should_approve()


def display_plan(plan: ActionPlan, project: str) -> None:
    state = project_store.load_state(project)
    typer.echo(f"\n  Plan: {plan.summary}")
    typer.echo(f"  Project: {project} | Phase: {state.phase}\n")

    for i, step in enumerate(plan.steps):
        prefix = f"  {i + 1}."
        typer.echo(f"{prefix} {step.name:15s} — {step.description}")

    if plan.loop:
        s, e = plan.loop.start_step + 1, plan.loop.end_step + 1
        typer.echo(f"\n  Loop: steps {s}-{e} repeat until pass (max {plan.loop.max_iterations}x)")

    typer.echo("")


async def execute(
    plan: ActionPlan,
    ctx: ExecutionContext,
    on_step_start: Callable[[int, str, str], None] | None = None,
    on_step_done: Callable[[int, str, StepResult], None] | None = None,
    on_confirm: ConfirmCallback | None = None,
) -> list[StepResult]:
    results: list[StepResult] = []
    steps = plan.steps
    i = 0
    loop_count = 0

    while i < len(steps):
        step = steps[i]

        action_def = ACTION_REGISTRY.get(step.name)
        if action_def is None:
            result = StepResult(ok=False, message=f"Unknown action: {step.name}")
            results.append(result)
            if on_step_done:
                on_step_done(i, step.name, result)
            return results

        if action_def.needs_client and ctx.client is None:
            result = StepResult(ok=False, message=f"Action '{step.name}' requires an HTTP client but none was provided")
            results.append(result)
            if on_step_done:
                on_step_done(i, step.name, result)
            return results

        if action_def.needs_llm and ctx.llm is None:
            result = StepResult(ok=False, message=f"Action '{step.name}' requires an LLM provider but none was provided")
            results.append(result)
            if on_step_done:
                on_step_done(i, step.name, result)
            return results

        if action_def.requires_confirmation and on_confirm:
            confirmed_args = on_confirm(step.name, step.args, ctx)
            if confirmed_args is None:
                result = StepResult(ok=False, message=f"User cancelled '{step.name}'")
                results.append(result)
                if on_step_done:
                    on_step_done(i, step.name, result)
                return results
            step = step.model_copy(update={"args": confirmed_args})

        if on_step_start:
            on_step_start(i, step.name, step.description)

        result = await action_def.fn(ctx, step.args)
        results.append(result)

        if on_step_done:
            on_step_done(i, step.name, result)

        if not result.ok:
            return results

        if plan.loop and i == plan.loop.end_step:
            if _should_continue_loop(ctx) and loop_count < plan.loop.max_iterations:
                loop_count += 1
                logger.info("Loop iteration %d/%d", loop_count, plan.loop.max_iterations)
                i = plan.loop.start_step
                continue

        i += 1

    return results
