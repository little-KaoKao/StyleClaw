from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import typer

from styleclaw.core.models import ActionPlan, LoopConfig, RoundEvaluation
from styleclaw.orchestrator.actions import (
    ACTION_REGISTRY,
    ExecutionContext,
    StepResult,
    _validate_action_args,
)
from styleclaw.storage import project_store

logger = logging.getLogger(__name__)

ConfirmCallback = Callable[[str, dict[str, Any], ExecutionContext], dict[str, Any] | None]

_DIMENSION_LABELS: dict[str, str] = {
    "color_palette": "色彩调性",
    "line_style": "线条风格",
    "lighting": "光影",
    "texture": "材质",
    "overall_mood": "整体氛围",
}

_DIMENSION_HINTS: dict[str, str] = {
    "color_palette": "提高色彩饱和度、减弱色差光",
    "line_style": "调整线条粗细、加强轮廓",
    "lighting": "增强光影对比、加强立体感",
    "texture": "增加材质细节、强化笔触",
    "overall_mood": "调整整体氛围、强化情绪表达",
}


def _find_weakest_dimension(evaluation: RoundEvaluation) -> tuple[str, float] | None:
    weakest: tuple[str, float] | None = None
    for ev in evaluation.evaluations:
        for dim in _DIMENSION_LABELS:
            score = getattr(ev.scores, dim)
            if weakest is None or score < weakest[1]:
                weakest = (dim, score)
    return weakest


def _format_report_path(project: str, round_num: int, pass_num: int) -> str:
    report_path = (
        project_store.project_dir(project)
        / "style-refine"
        / f"pass-{pass_num:03d}"
        / f"round-{round_num:03d}"
        / "report.html"
    )
    try:
        return str(report_path.relative_to(Path.cwd()))
    except ValueError:
        return str(report_path)


def _format_needs_human_hint(
    project: str, round_num: int, pass_num: int, evaluation: RoundEvaluation,
) -> str:
    weakest = _find_weakest_dimension(evaluation)
    report_path = _format_report_path(project, round_num, pass_num)

    if weakest is None:
        diagnosis = "评分缺失，无法定位最弱维度"
        hint_phrase = "调整方向"
    else:
        dim, score = weakest
        label = _DIMENSION_LABELS.get(dim, dim)
        diagnosis = f"{label}得分 {score:.1f} 最弱"
        hint_phrase = _DIMENSION_HINTS.get(dim, "调整方向")

    return (
        "\n  !! needs_human: 某维度得分 < 5，自动循环已停止。\n"
        f"  {diagnosis}；可以告诉我方向，例如：\n"
        f'    styleclaw run "{hint_phrase}" -p {project}\n'
        f"  报告：{report_path}\n"
    )


def _should_continue_loop(ctx: ExecutionContext) -> bool:
    state = project_store.load_state(ctx.project)
    if state.current_round < 1:
        return False
    pass_num = state.current_model_select_pass or 1
    try:
        evaluation = project_store.load_round_evaluation(
            ctx.project, state.current_round, pass_num=pass_num,
        )
    except FileNotFoundError:
        logger.warning("No evaluation found for round %d, stopping loop.", state.current_round)
        return False
    if evaluation.needs_human():
        typer.echo(
            _format_needs_human_hint(ctx.project, state.current_round, pass_num, evaluation),
            err=True,
        )
        return False
    return not evaluation.should_approve()


def display_plan(plan: ActionPlan, project: str) -> None:
    state = project_store.load_state(project)
    sep = "  " + "─" * 60

    typer.echo("")
    typer.echo(sep)
    typer.echo(f"  Plan: {plan.summary}")
    typer.echo(f"  Project: {project} | Phase: {state.phase}")
    typer.echo(sep)

    typer.echo("  Steps:")
    for i, step in enumerate(plan.steps):
        typer.echo(f"    {i + 1}. {step.name:15s} — {step.description}")

    if plan.loop:
        s, e = plan.loop.start_step + 1, plan.loop.end_step + 1
        typer.echo("")
        typer.echo("  Loop:")
        typer.echo(
            f"    repeat steps {s}-{e} until pass (max {plan.loop.max_iterations}x)"
        )

    from styleclaw.orchestrator.cost_estimate import estimate_plan, format_plan_estimate
    cost_lines = format_plan_estimate(estimate_plan(plan, project))
    if cost_lines:
        typer.echo("")
        typer.echo("  Estimate:")
        for line in cost_lines:
            typer.echo(f"    {line}")

    if plan.stop_summary:
        typer.echo("")
        typer.echo(f"  执行完后：{plan.stop_summary}")

    typer.echo(sep)
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

        # Validate args first — it's the cheapest gate and catches the most
        # LLM mistakes (unknown keys, out-of-range ints, oversized strings).
        validated_args, invalid = _validate_action_args(step.name, step.args)
        if invalid is not None:
            results.append(invalid)
            if on_step_done:
                on_step_done(i, step.name, invalid)
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
            confirmed_args = on_confirm(step.name, validated_args, ctx)
            if confirmed_args is None:
                result = StepResult(ok=False, message=f"User cancelled '{step.name}'")
                results.append(result)
                if on_step_done:
                    on_step_done(i, step.name, result)
                return results
            # Re-validate any args the confirmation callback rewrote.
            validated_args, invalid = _validate_action_args(step.name, confirmed_args)
            if invalid is not None:
                results.append(invalid)
                if on_step_done:
                    on_step_done(i, step.name, invalid)
                return results
            step = step.model_copy(update={"args": validated_args})

        if on_step_start:
            on_step_start(i, step.name, step.description)

        result = await action_def.fn(ctx, validated_args)
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
