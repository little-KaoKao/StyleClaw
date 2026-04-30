from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

from styleclaw.core.config import MAX_AUTO_ROUNDS, MAX_POLL_CYCLES, ORCHESTRATOR_POLL_INTERVAL
from styleclaw.core.models import Phase, TaskStatus
from styleclaw.providers.llm.base import LLMProvider
from styleclaw.providers.runninghub.client import RunningHubClient
from styleclaw.storage import project_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepResult:
    ok: bool
    message: str = ""
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExecutionContext:
    project: str
    client: RunningHubClient | None = None
    llm: LLMProvider | None = None
    poll_interval: float = ORCHESTRATOR_POLL_INTERVAL
    show_thinking: bool = False
    thinking_budget: int = 5000


@dataclass(frozen=True)
class ActionDef:
    fn: Callable[[ExecutionContext, dict[str, Any]], Awaitable[StepResult]]
    needs_client: bool = False
    needs_llm: bool = False
    requires_confirmation: bool = False


async def do_analyze(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.agents.analyze_style import analyze_style, analyze_style_with_thinking
    from styleclaw.core.state_machine import advance

    config = project_store.load_config(ctx.project)
    root = project_store.project_dir(ctx.project)
    ref_paths = [root / r for r in config.ref_images]

    thinking = ""
    if ctx.show_thinking:
        analysis, thinking = await analyze_style_with_thinking(
            ctx.llm, ref_paths, config.ip_info, thinking_budget=ctx.thinking_budget,
        )
    else:
        analysis = await analyze_style(ctx.llm, ref_paths, config.ip_info)

    pass_num = 1
    project_store.save_analysis(ctx.project, analysis, pass_num=pass_num)
    if thinking:
        project_store.save_thinking(
            project_store.model_select_dir(ctx.project, pass_num) / "initial-analysis.json",
            thinking,
        )

    state = project_store.load_state(ctx.project)
    new_state = advance(state, Phase.MODEL_SELECT).with_model_select_pass(pass_num)
    project_store.save_state(ctx.project, new_state)

    msg = f"Trigger: {analysis.trigger_phrase}"
    if thinking:
        msg += f" | thinking saved ({len(thinking)} chars)"
    return StepResult(ok=True, message=msg)


async def do_generate(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.scripts.generate import generate_model_select, generate_style_refine

    state = project_store.load_state(ctx.project)

    if state.phase == Phase.MODEL_SELECT:
        pass_num = state.current_model_select_pass or 1
        if pass_num > 1 and state.current_round >= 1:
            prompt_cfg = project_store.load_prompt_config(ctx.project, state.current_round)
            trigger = prompt_cfg.trigger_phrase
        else:
            analysis = project_store.load_analysis(ctx.project, pass_num=pass_num)
            trigger = analysis.trigger_phrase
        uploads = project_store.load_uploads(ctx.project)
        sref_url = uploads[0].url if uploads else ""
        records = await generate_model_select(
            ctx.project, ctx.client, trigger,
            sref_url=sref_url, pass_num=pass_num,
        )
        return StepResult(ok=True, message=f"Submitted {len(records)} model tasks (pass {pass_num})")

    if state.phase == Phase.STYLE_REFINE:
        round_num = state.current_round
        prompt_config = project_store.load_prompt_config(ctx.project, round_num)
        uploads = project_store.load_uploads(ctx.project)
        sref_url = uploads[0].url if uploads else ""
        records = await generate_style_refine(
            ctx.project, ctx.client, round_num, prompt_config.trigger_phrase,
            sref_url=sref_url, extra_model_params=prompt_config.model_params,
        )
        return StepResult(ok=True, message=f"Submitted {len(records)} refine tasks")

    return StepResult(ok=False, message=f"Cannot generate in {state.phase}")


async def do_poll(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.scripts.poll import poll_batch, poll_model_select, poll_style_refine

    max_cycles = args.get("max_cycles", MAX_POLL_CYCLES)
    state = project_store.load_state(ctx.project)

    for cycle in range(max_cycles):
        state = project_store.load_state(ctx.project)
        if state.phase == Phase.MODEL_SELECT:
            pass_num = state.current_model_select_pass or 1
            records = await poll_model_select(ctx.project, ctx.client, pass_num=pass_num)
        elif state.phase == Phase.STYLE_REFINE:
            records = await poll_style_refine(ctx.project, ctx.client, state.current_round)
        elif state.phase in (Phase.BATCH_T2I, Phase.BATCH_I2I):
            phase_str = "i2i" if state.phase == Phase.BATCH_I2I else "t2i"
            records = await poll_batch(ctx.project, ctx.client, state.current_batch, phase=phase_str)
        else:
            return StepResult(ok=False, message=f"Nothing to poll in {state.phase}")

        pending = [r for r in records.values() if r.status not in ("SUCCESS", "FAILED")]
        succeeded = sum(1 for r in records.values() if r.status == TaskStatus.SUCCESS)
        failed = sum(1 for r in records.values() if r.status == TaskStatus.FAILED)
        if not pending:
            msg = f"{succeeded}/{len(records)} succeeded"
            if failed:
                msg += f" ({failed} failed)"
            return StepResult(ok=failed == 0, message=msg)

        logger.info("Waiting... %d/%d completed (cycle %d/%d)", succeeded + failed, len(records), cycle + 1, max_cycles)
        await asyncio.sleep(ctx.poll_interval)

    return StepResult(ok=False, message=f"Poll timed out after {max_cycles} cycles")


async def do_evaluate(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    state = project_store.load_state(ctx.project)
    config = project_store.load_config(ctx.project)
    root = project_store.project_dir(ctx.project)
    ref_paths = [root / r for r in config.ref_images]

    if state.phase == Phase.MODEL_SELECT:
        from styleclaw.agents.select_model import (
            evaluate_models,
            evaluate_models_with_thinking,
        )
        from styleclaw.scripts.report import generate_model_select_report

        pass_num = state.current_model_select_pass or 1

        model_images: dict[str, list[Path]] = {}
        records = project_store.load_all_task_records(ctx.project, pass_num=pass_num)
        for key in records:
            if "/" in key:
                model_id, variant = key.split("/", 1)
                results_dir = project_store.model_results_dir(
                    ctx.project, model_id, variant=variant, pass_num=pass_num,
                )
            else:
                results_dir = project_store.model_results_dir(
                    ctx.project, key, pass_num=pass_num,
                )
            images = sorted(results_dir.glob("output-*.png"))
            if images:
                model_images[key] = images

        if not model_images:
            return StepResult(ok=False, message="No generated images found")

        thinking = ""
        if ctx.show_thinking:
            evaluation, thinking = await evaluate_models_with_thinking(
                ctx.llm, ref_paths, model_images, thinking_budget=ctx.thinking_budget,
            )
        else:
            evaluation = await evaluate_models(ctx.llm, ref_paths, model_images)
        project_store.save_evaluation(ctx.project, evaluation, pass_num=pass_num)
        if thinking:
            project_store.save_thinking(
                project_store.model_select_dir(ctx.project, pass_num) / "evaluation.json",
                thinking,
            )
        generate_model_select_report(ctx.project, pass_num=pass_num)

        msg = f"Recommendation: {evaluation.recommendation} (pass {pass_num})"
        if thinking:
            msg += f" | thinking saved ({len(thinking)} chars)"
        return StepResult(
            ok=True, message=msg,
            data={"recommendation": evaluation.recommendation, "pass_num": pass_num},
        )

    if state.phase == Phase.STYLE_REFINE:
        from styleclaw.agents.evaluate_result import (
            evaluate_round,
            evaluate_round_with_thinking,
        )
        from styleclaw.scripts.report import generate_style_refine_report

        round_num = state.current_round
        model_images = {}
        records = project_store.load_all_round_task_records(ctx.project, round_num)
        for mid in records:
            results_dir = project_store.round_results_dir(ctx.project, round_num, mid)
            images = sorted(results_dir.glob("output-*.png"))
            if images:
                model_images[mid] = images

        if not model_images:
            return StepResult(ok=False, message="No generated images for this round")

        thinking = ""
        if ctx.show_thinking:
            evaluation, thinking = await evaluate_round_with_thinking(
                ctx.llm, ref_paths, model_images, round_num,
                thinking_budget=ctx.thinking_budget,
            )
        else:
            evaluation = await evaluate_round(ctx.llm, ref_paths, model_images, round_num)
        project_store.save_round_evaluation(ctx.project, round_num, evaluation)
        if thinking:
            round_d = project_store.round_dir(ctx.project, round_num)
            project_store.save_thinking(round_d / "evaluation.json", thinking)
        generate_style_refine_report(ctx.project, round_num)

        passed = evaluation.should_approve()
        scores_msg = ", ".join(
            f"{e.model}={e.total:.1f}" for e in evaluation.evaluations
        )
        msg = f"Scores: [{scores_msg}] {'PASS' if passed else 'needs refinement'}"
        if thinking:
            msg += f" | thinking saved ({len(thinking)} chars)"
        return StepResult(ok=True, message=msg, data={"passed": passed})

    return StepResult(ok=False, message=f"Cannot evaluate in {state.phase}")


async def do_select_model(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.core.state_machine import advance
    from styleclaw.providers.runninghub.models import MODEL_REGISTRY

    models_str = args.get("models", "")
    if not models_str:
        return StepResult(ok=False, message="No models specified")

    selected = [m.strip() for m in models_str.split(",")]
    for m in selected:
        if m not in MODEL_REGISTRY:
            return StepResult(ok=False, message=f"Unknown model: {m}")

    state = project_store.load_state(ctx.project)
    if state.phase == Phase.MODEL_SELECT:
        new_state = advance(state, Phase.STYLE_REFINE)
        new_state = new_state.with_selected_models(selected)
        project_store.save_state(ctx.project, new_state)
        return StepResult(ok=True, message=f"Selected {', '.join(selected)}, advanced to STYLE_REFINE")
    elif state.phase == Phase.STYLE_REFINE:
        new_state = state.with_selected_models(selected)
        project_store.save_state(ctx.project, new_state)
        return StepResult(ok=True, message=f"Updated models: {', '.join(selected)}")

    return StepResult(ok=False, message=f"Cannot select model in {state.phase}")


async def do_refine(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.agents.refine_prompt import refine_prompt, refine_prompt_with_thinking
    from styleclaw.core.models import RoundEvaluation

    state = project_store.load_state(ctx.project)
    config = project_store.load_config(ctx.project)
    root = project_store.project_dir(ctx.project)
    ref_paths = [root / r for r in config.ref_images]

    round_num = state.current_round + 1
    if round_num > MAX_AUTO_ROUNDS:
        return StepResult(ok=False, message=f"Max rounds ({MAX_AUTO_ROUNDS}) reached")

    evaluations: list[RoundEvaluation] = []
    for r in range(1, round_num):
        try:
            ev = project_store.load_round_evaluation(ctx.project, r)
            evaluations.append(ev)
        except FileNotFoundError:
            logger.warning("Evaluation for round %d not found, skipping history entry.", r)

    if round_num == 1:
        analysis = project_store.load_analysis(ctx.project)
        current_trigger = analysis.trigger_phrase
    else:
        prev_prompt = project_store.load_prompt_config(ctx.project, round_num - 1)
        current_trigger = prev_prompt.trigger_phrase

    direction = args.get("direction", "")
    thinking = ""
    if ctx.show_thinking:
        prompt_config, thinking = await refine_prompt_with_thinking(
            ctx.llm, ref_paths, current_trigger, round_num,
            config.ip_info, evaluations, direction,
            thinking_budget=ctx.thinking_budget,
        )
    else:
        prompt_config = await refine_prompt(
            ctx.llm, ref_paths, current_trigger, round_num,
            config.ip_info, evaluations, direction,
        )
    project_store.save_prompt_config(ctx.project, round_num, prompt_config)

    if thinking:
        round_d = project_store.round_dir(ctx.project, round_num)
        project_store.save_thinking(round_d / "prompt.json", thinking)

    new_state = state.with_round(round_num)
    project_store.save_state(ctx.project, new_state)

    msg = f"Round {round_num}: {prompt_config.trigger_phrase}"
    if thinking:
        msg += f" | thinking saved ({len(thinking)} chars)"
    return StepResult(ok=True, message=msg)


async def do_approve(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.core.state_machine import advance

    state = project_store.load_state(ctx.project)
    target = args.get("target", "batch-t2i")

    if target == "completed":
        if state.phase != Phase.BATCH_I2I:
            return StepResult(ok=False, message=f"Must be in BATCH_I2I (current: {state.phase})")
        new_state = advance(state, Phase.COMPLETED)
    else:
        if state.phase != Phase.STYLE_REFINE:
            return StepResult(ok=False, message=f"Must be in STYLE_REFINE (current: {state.phase})")
        new_state = advance(state, Phase.BATCH_T2I)

    project_store.save_state(ctx.project, new_state)
    return StepResult(ok=True, message=f"Advanced to {new_state.phase}")


async def do_design_cases(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.agents.design_cases import design_cases

    state = project_store.load_state(ctx.project)
    config = project_store.load_config(ctx.project)
    batch_num = state.current_batch + 1

    prompt_config = project_store.load_prompt_config(ctx.project, state.current_round)
    batch_config = await design_cases(
        ctx.llm, config.ip_info, prompt_config.trigger_phrase, batch_num,
    )
    project_store.save_batch_config(ctx.project, batch_num, batch_config)

    new_state = state.with_batch(batch_num)
    project_store.save_state(ctx.project, new_state)

    return StepResult(ok=True, message=f"Designed {len(batch_config.cases)} cases")


async def do_batch_submit(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.scripts.batch_submit import batch_submit_i2i, batch_submit_t2i

    state = project_store.load_state(ctx.project)
    model_id = args.get("model") or (state.selected_models[0] if state.selected_models else None)
    if not model_id:
        return StepResult(ok=False, message="No model selected")

    if state.phase == Phase.BATCH_T2I:
        uploads = project_store.load_uploads(ctx.project)
        sref_url = uploads[0].url if uploads else ""
        records = await batch_submit_t2i(
            ctx.project, ctx.client, state.current_batch, model_id, sref_url=sref_url,
        )
        return StepResult(ok=True, message=f"Submitted {len(records)} t2i tasks")

    if state.phase == Phase.BATCH_I2I:
        prompt_config = project_store.load_prompt_config(ctx.project, state.current_round)
        records = await batch_submit_i2i(
            ctx.project, ctx.client, state.current_batch, model_id, prompt_config.trigger_phrase,
        )
        return StepResult(ok=True, message=f"Submitted {len(records)} i2i tasks")

    return StepResult(ok=False, message=f"Cannot batch-submit in {state.phase}")


async def do_report(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.scripts.report import (
        generate_batch_i2i_report,
        generate_batch_t2i_report,
    )

    state = project_store.load_state(ctx.project)
    if state.phase == Phase.BATCH_T2I:
        path = generate_batch_t2i_report(ctx.project, state.current_batch)
        return StepResult(ok=True, message=f"Report: {path}")
    if state.phase == Phase.BATCH_I2I:
        path = generate_batch_i2i_report(ctx.project, state.current_batch)
        return StepResult(ok=True, message=f"Report: {path}")

    return StepResult(ok=False, message=f"No report for {state.phase}")


async def do_retest_models(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.core.state_machine import advance

    state = project_store.load_state(ctx.project)
    if state.phase not in (Phase.STYLE_REFINE, Phase.BATCH_T2I):
        return StepResult(
            ok=False,
            message=f"retest-models requires STYLE_REFINE or BATCH_T2I (current: {state.phase})",
        )

    new_pass = (state.current_model_select_pass or 0) + 1
    new_state = (
        advance(state, Phase.MODEL_SELECT)
        .with_model_select_pass(new_pass)
    )
    project_store.save_state(ctx.project, new_state)
    return StepResult(
        ok=True,
        message=f"Entered MODEL_SELECT pass {new_pass} for re-test",
        data={"pass_num": new_pass},
    )


async def do_back_to_t2i(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.core.state_machine import advance

    state = project_store.load_state(ctx.project)
    if state.phase != Phase.BATCH_I2I:
        return StepResult(
            ok=False,
            message=f"back-to-t2i requires BATCH_I2I (current: {state.phase})",
        )
    new_state = advance(state, Phase.BATCH_T2I)
    project_store.save_state(ctx.project, new_state)
    return StepResult(ok=True, message="Returned to BATCH_T2I")


ACTION_REGISTRY: dict[str, ActionDef] = {
    "analyze":       ActionDef(fn=do_analyze,       needs_client=False, needs_llm=True),
    "generate":      ActionDef(fn=do_generate,      needs_client=True,  needs_llm=False),
    "poll":          ActionDef(fn=do_poll,          needs_client=True,  needs_llm=False),
    "evaluate":      ActionDef(fn=do_evaluate,      needs_client=False, needs_llm=True),
    "select-model":  ActionDef(fn=do_select_model,  needs_client=False, needs_llm=False, requires_confirmation=True),
    "refine":        ActionDef(fn=do_refine,        needs_client=False, needs_llm=True),
    "approve":       ActionDef(fn=do_approve,       needs_client=False, needs_llm=False),
    "design-cases":  ActionDef(fn=do_design_cases,  needs_client=False, needs_llm=True),
    "batch-submit":  ActionDef(fn=do_batch_submit,  needs_client=True,  needs_llm=False),
    "report":        ActionDef(fn=do_report,        needs_client=False, needs_llm=False),
    "retest-models": ActionDef(fn=do_retest_models, needs_client=False, needs_llm=False),
    "back-to-t2i":   ActionDef(fn=do_back_to_t2i,   needs_client=False, needs_llm=False),
}


PHASE_ACTIONS: dict[Phase, list[str]] = {
    Phase.INIT:         ["analyze"],
    Phase.MODEL_SELECT: ["generate", "poll", "evaluate", "select-model"],
    Phase.STYLE_REFINE: ["refine", "generate", "poll", "evaluate", "approve", "select-model", "retest-models"],
    Phase.BATCH_T2I:    ["design-cases", "batch-submit", "poll", "report", "approve", "retest-models"],
    Phase.BATCH_I2I:    ["batch-submit", "poll", "report", "approve", "back-to-t2i"],
    Phase.COMPLETED:    [],
}
