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


# Per-action allowed arg keys. Any key in step.args that isn't in this set
# is rejected with a clear error — protects against the LLM hallucinating
# extra parameters (or a typo silently dropping into "ignored args").
# Actions without entries here accept no args.
ACTION_ALLOWED_ARGS: dict[str, frozenset[str]] = {
    "init":          frozenset({"ref_dir", "ip_info", "description", "force"}),
    "generate":      frozenset({"force", "models"}),
    "poll":          frozenset({"max_cycles"}),
    "select-model":  frozenset({"models", "variant"}),
    "refine":        frozenset({"direction"}),
    "approve":       frozenset({"target"}),
    "design-cases":  frozenset({"feedback"}),
    "batch-submit":  frozenset({"model"}),
    "set-sref":      frozenset({"index"}),
    "set-pass":      frozenset({"pass_num"}),
    "add-refs":      frozenset({"image_dir"}),
}


def _reject_unknown_args(action_name: str, args: dict[str, Any]) -> StepResult | None:
    """Return a failure StepResult if args has keys not allowed for this
    action, else None. Keys are case-sensitive."""
    allowed = ACTION_ALLOWED_ARGS.get(action_name, frozenset())
    unknown = sorted(set(args) - allowed)
    if not unknown:
        return None
    return StepResult(
        ok=False,
        message=(
            f"{action_name}: unknown args {unknown}. "
            f"Allowed: {sorted(allowed) or '(none)'}."
        ),
    )


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

        # Ensure analysis exists for this pass (copy from previous if needed)
        try:
            analysis = project_store.load_analysis(ctx.project, pass_num=pass_num)
        except FileNotFoundError:
            prev_analysis = project_store.load_analysis(ctx.project, pass_num=pass_num - 1)
            project_store.save_analysis(ctx.project, prev_analysis, pass_num=pass_num)
            analysis = prev_analysis
        trigger = analysis.trigger_phrase
        uploads = project_store.load_uploads(ctx.project)
        config = project_store.load_config(ctx.project)
        sref_url = uploads[config.sref_index].url if uploads else ""

        force = bool(args.get("force", False))
        if force:
            existing_records = project_store.load_all_task_records(ctx.project, pass_num=pass_num)
            if any(r.status == TaskStatus.SUCCESS for r in existing_records.values()):
                return StepResult(
                    ok=False,
                    message=(
                        f"`generate --force` would overwrite SUCCESS data in pass-{pass_num:03d}. "
                        f"You usually don't need --force: a plain `generate` already retries "
                        f"FAILED tasks and skips SUCCESS ones. "
                        f"If you really want a fresh pass, run `retest-models` (same sref) "
                        f"or `set-sref` (new sref) — both open a new pass and preserve this one."
                    ),
                )

        models_arg = args.get("models")
        models: list[str] | None
        if isinstance(models_arg, str):
            models = [m.strip() for m in models_arg.split(",") if m.strip()]
        elif isinstance(models_arg, list):
            models = [str(m).strip() for m in models_arg if str(m).strip()]
        else:
            models = None
        from styleclaw.providers.runninghub.models import MODEL_REGISTRY
        if models is not None:
            unknown = [m for m in models if m not in MODEL_REGISTRY]
            if unknown:
                return StepResult(
                    ok=False,
                    message=f"Unknown model(s): {', '.join(unknown)}. "
                            f"Choose from: {', '.join(MODEL_REGISTRY.keys())}",
                )
            if not models:
                models = None

        records = await generate_model_select(
            ctx.project, ctx.client, trigger,
            sref_url=sref_url, models=models,
            pass_num=pass_num, force=force,
        )
        msg = f"Submitted {len(records)} model tasks (pass {pass_num})"
        if models:
            msg += f" [filtered: {', '.join(models)}]"
        return StepResult(ok=True, message=msg)

    if state.phase == Phase.STYLE_REFINE:
        pass_num = state.current_model_select_pass or 1
        round_num = state.current_round
        prompt_config = project_store.load_prompt_config(
            ctx.project, round_num, pass_num=pass_num,
        )
        uploads = project_store.load_uploads(ctx.project)
        config = project_store.load_config(ctx.project)
        use_sref = state.selected_variant != "prompt-only"
        sref_url = (uploads[config.sref_index].url if uploads else "") if use_sref else ""

        force = bool(args.get("force", False))
        if force:
            existing_records = project_store.load_all_round_task_records(
                ctx.project, round_num, pass_num=pass_num,
            )
            if any(r.status == TaskStatus.SUCCESS for r in existing_records.values()):
                return StepResult(
                    ok=False,
                    message=(
                        f"`generate --force` would overwrite SUCCESS data in round-{round_num:03d}. "
                        f"You usually don't need --force: a plain `generate` already retries "
                        f"FAILED tasks and skips SUCCESS ones. "
                        f"To start a clean round, run `refine` first — it opens the next round."
                    ),
                )

        records = await generate_style_refine(
            ctx.project, ctx.client, round_num, prompt_config.trigger_phrase,
            sref_url=sref_url, extra_model_params=prompt_config.model_params,
            pass_num=pass_num, force=force,
        )
        return StepResult(ok=True, message=f"Submitted {len(records)} refine tasks (variant: {state.selected_variant})")

    return StepResult(ok=False, message=f"Cannot generate in {state.phase}")


async def do_poll(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.scripts.poll import (
        poll_batch,
        poll_model_select,
        poll_style_refine,
        retry_failed_model_select,
        retry_failed_style_refine,
    )

    max_cycles_raw = args.get("max_cycles", MAX_POLL_CYCLES)
    try:
        max_cycles = int(max_cycles_raw)
    except (TypeError, ValueError):
        return StepResult(
            ok=False,
            message=f"poll args.max_cycles must be an integer (got {max_cycles_raw!r})",
        )
    if max_cycles < 1:
        return StepResult(ok=False, message=f"poll args.max_cycles must be >= 1 (got {max_cycles})")
    if max_cycles > MAX_POLL_CYCLES:
        logger.warning(
            "poll args.max_cycles=%d exceeds STYLECLAW_MAX_POLL_CYCLES=%d; clamping.",
            max_cycles, MAX_POLL_CYCLES,
        )
        max_cycles = MAX_POLL_CYCLES
    state = project_store.load_state(ctx.project)

    for cycle in range(max_cycles):
        state = project_store.load_state(ctx.project)
        if state.phase == Phase.MODEL_SELECT:
            pass_num = state.current_model_select_pass or 1
            records = await poll_model_select(ctx.project, ctx.client, pass_num=pass_num)
        elif state.phase == Phase.STYLE_REFINE:
            pass_num = state.current_model_select_pass or 1
            records = await poll_style_refine(
                ctx.project, ctx.client, state.current_round, pass_num=pass_num,
            )
        elif state.phase in (Phase.BATCH_T2I, Phase.BATCH_I2I):
            phase_str = "i2i" if state.phase == Phase.BATCH_I2I else "t2i"
            records = await poll_batch(ctx.project, ctx.client, state.current_batch, phase=phase_str)
        else:
            return StepResult(ok=False, message=f"Nothing to poll in {state.phase}")

        pending = [r for r in records.values() if r.status not in ("SUCCESS", "FAILED")]
        succeeded = sum(1 for r in records.values() if r.status == TaskStatus.SUCCESS)
        failed = sum(1 for r in records.values() if r.status == TaskStatus.FAILED)
        if not pending:
            # All tasks reached a terminal state. Auto-retry FAILED ones once
            # (model-select and style-refine only — batch retries are too
            # costly with 100 cases and should stay explicit). After retry,
            # always continue downstream: partial failures are tolerated so
            # evaluate/select-model don't get blocked by a few timeouts.
            if failed:
                if state.phase == Phase.MODEL_SELECT:
                    pass_num = state.current_model_select_pass or 1
                    records = await retry_failed_model_select(
                        ctx.project, ctx.client, pass_num=pass_num,
                    )
                elif state.phase == Phase.STYLE_REFINE:
                    pass_num = state.current_model_select_pass or 1
                    records = await retry_failed_style_refine(
                        ctx.project, ctx.client, state.current_round, pass_num=pass_num,
                    )
                succeeded = sum(1 for r in records.values() if r.status == TaskStatus.SUCCESS)
                failed = sum(1 for r in records.values() if r.status == TaskStatus.FAILED)

            msg = f"{succeeded}/{len(records)} succeeded"
            if failed:
                msg += f" ({failed} failed, skipping)"
            return StepResult(ok=True, message=msg, data={"succeeded": succeeded, "failed": failed})

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
        from styleclaw.storage.image_store import list_output_images

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
            images = list_output_images(results_dir)
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
        from styleclaw.storage.image_store import list_output_images

        pass_num = state.current_model_select_pass or 1
        round_num = state.current_round
        model_images = {}
        records = project_store.load_all_round_task_records(
            ctx.project, round_num, pass_num=pass_num,
        )
        for mid in records:
            results_dir = project_store.round_results_dir(
                ctx.project, round_num, mid, pass_num=pass_num,
            )
            images = list_output_images(results_dir)
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
        project_store.save_round_evaluation(
            ctx.project, round_num, evaluation, pass_num=pass_num,
        )
        if thinking:
            round_d = project_store.round_dir(ctx.project, round_num, pass_num=pass_num)
            project_store.save_thinking(round_d / "evaluation.json", thinking)
        generate_style_refine_report(ctx.project, round_num, pass_num=pass_num)

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

    variant = args.get("variant", "").strip()
    if variant and variant not in ("prompt-sref", "prompt-only"):
        return StepResult(ok=False, message=f"Unknown variant: {variant}. Use 'prompt-sref' or 'prompt-only'.")

    state = project_store.load_state(ctx.project)
    if state.phase == Phase.MODEL_SELECT:
        new_state = advance(state, Phase.STYLE_REFINE)
        new_state = new_state.with_selected_models(selected, variant=variant)
        project_store.save_state(ctx.project, new_state)
        v_info = f" [{variant}]" if variant else ""
        return StepResult(ok=True, message=f"Selected {', '.join(selected)}{v_info}, advanced to STYLE_REFINE")
    elif state.phase == Phase.STYLE_REFINE:
        new_state = state.with_selected_models(selected, variant=variant)
        project_store.save_state(ctx.project, new_state)
        v_info = f" [{variant}]" if variant else ""
        return StepResult(ok=True, message=f"Updated models: {', '.join(selected)}{v_info}")

    return StepResult(ok=False, message=f"Cannot select model in {state.phase}")


async def do_refine(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.agents.refine_prompt import refine_prompt, refine_prompt_with_thinking
    from styleclaw.core.models import RoundEvaluation

    state = project_store.load_state(ctx.project)
    
    # Phase guard: refine only works in STYLE_REFINE
    if state.phase != Phase.STYLE_REFINE:
        return StepResult(
            ok=False,
            message=f"refine requires STYLE_REFINE phase (current: {state.phase}). "
                    f"Run 'select-model' first to advance from MODEL_SELECT."
        )
    
    config = project_store.load_config(ctx.project)
    root = project_store.project_dir(ctx.project)
    ref_paths = [root / r for r in config.ref_images]

    pass_num = state.current_model_select_pass or 1
    round_num = state.current_round + 1

    # Skip rounds that already have a prompt config (non-destructive rollback support)
    while True:
        try:
            project_store.load_prompt_config(ctx.project, round_num, pass_num=pass_num)
            round_num += 1
        except FileNotFoundError:
            break

    if round_num > MAX_AUTO_ROUNDS:
        return StepResult(ok=False, message=f"Max rounds ({MAX_AUTO_ROUNDS}) reached")

    evaluations: list[RoundEvaluation] = []
    for r in range(1, round_num):
        try:
            ev = project_store.load_round_evaluation(ctx.project, r, pass_num=pass_num)
            evaluations.append(ev)
        except FileNotFoundError:
            logger.warning("Evaluation for round %d not found, skipping history entry.", r)

    if round_num == 1:
        analysis = project_store.load_analysis(ctx.project, pass_num=pass_num)
        current_trigger = analysis.trigger_phrase
    else:
        prev_prompt = project_store.load_prompt_config(
            ctx.project, round_num - 1, pass_num=pass_num,
        )
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
    project_store.save_prompt_config(
        ctx.project, round_num, prompt_config, pass_num=pass_num,
    )

    if thinking:
        round_d = project_store.round_dir(ctx.project, round_num, pass_num=pass_num)
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

    if state.phase != Phase.BATCH_T2I:
        return StepResult(
            ok=False,
            message=f"design-cases requires BATCH_T2I phase (current: {state.phase}). "
                    f"Run 'approve' from STYLE_REFINE first.",
        )
    if state.current_round < 1:
        return StepResult(
            ok=False,
            message="design-cases requires a refined trigger phrase — "
                    "no STYLE_REFINE round has been completed (current_round=0).",
        )

    config = project_store.load_config(ctx.project)
    batch_num = state.current_batch + 1

    pass_num = state.current_model_select_pass or 1
    prompt_config = project_store.load_prompt_config(
        ctx.project, state.current_round, pass_num=pass_num,
    )
    feedback = str(args.get("feedback", "") or "").strip()
    batch_config = await design_cases(
        ctx.llm, config.ip_info, prompt_config.trigger_phrase, batch_num,
        feedback=feedback,
    )
    project_store.save_batch_config(ctx.project, batch_num, batch_config)

    new_state = state.with_batch(batch_num)
    project_store.save_state(ctx.project, new_state)

    msg = f"Designed {len(batch_config.cases)} cases (batch {batch_num})"
    if feedback:
        msg += " [applied feedback]"
    return StepResult(ok=True, message=msg)


async def do_batch_submit(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.providers.runninghub.models import MODEL_REGISTRY
    from styleclaw.scripts.batch_submit import batch_submit_i2i, batch_submit_t2i

    state = project_store.load_state(ctx.project)
    model_id = args.get("model") or (state.selected_models[0] if state.selected_models else None)
    if not model_id:
        return StepResult(ok=False, message="No model selected")
    if model_id not in MODEL_REGISTRY:
        return StepResult(
            ok=False,
            message=f"Unknown model '{model_id}'. Choose from: {', '.join(MODEL_REGISTRY.keys())}",
        )

    if state.phase == Phase.BATCH_T2I:
        uploads = project_store.load_uploads(ctx.project)
        config = project_store.load_config(ctx.project)
        sref_url = uploads[config.sref_index].url if uploads else ""
        records = await batch_submit_t2i(
            ctx.project, ctx.client, state.current_batch, model_id, sref_url=sref_url,
        )
        return StepResult(ok=True, message=f"Submitted {len(records)} t2i tasks")

    if state.phase == Phase.BATCH_I2I:
        pass_num = state.current_model_select_pass or 1
        prompt_config = project_store.load_prompt_config(
            ctx.project, state.current_round, pass_num=pass_num,
        )
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
    from styleclaw.core.models import StyleAnalysis
    from styleclaw.core.state_machine import advance

    state = project_store.load_state(ctx.project)
    if state.phase not in (Phase.MODEL_SELECT, Phase.STYLE_REFINE, Phase.BATCH_T2I):
        return StepResult(
            ok=False,
            message=f"retest-models requires MODEL_SELECT, STYLE_REFINE, or BATCH_T2I (current: {state.phase})",
        )

    old_pass = state.current_model_select_pass or 1
    current_trigger = ""
    if state.current_round >= 1:
        try:
            prompt_cfg = project_store.load_prompt_config(
                ctx.project, state.current_round, pass_num=old_pass,
            )
            current_trigger = prompt_cfg.trigger_phrase
        except FileNotFoundError:
            pass
    if not current_trigger:
        try:
            prev_analysis = project_store.load_analysis(ctx.project, pass_num=old_pass)
            current_trigger = prev_analysis.trigger_phrase
        except FileNotFoundError:
            pass

    new_pass = old_pass + 1
    project_store.save_analysis(
        ctx.project, StyleAnalysis(trigger_phrase=current_trigger), pass_num=new_pass,
    )

    # advance() only allows forward transitions, so only call it when leaving
    # a later phase. From MODEL_SELECT we stay in MODEL_SELECT and just bump
    # the pass counter.
    if state.phase == Phase.MODEL_SELECT:
        new_state = state.with_model_select_pass(new_pass)
    else:
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


async def do_init(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.scripts.init_project import init_project

    ref_dir_str = args.get("ref_dir", "").strip()
    if not ref_dir_str:
        return StepResult(ok=False, message="init requires args.ref_dir")
    ref_dir = Path(ref_dir_str).expanduser()
    if not ref_dir.is_dir():
        return StepResult(ok=False, message=f"ref_dir is not a directory: {ref_dir}")

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    refs = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in image_exts)
    if not refs:
        return StepResult(ok=False, message=f"No images found in {ref_dir}")

    ip_info = args.get("ip_info", "").strip()
    description = args.get("description", "")
    force = bool(args.get("force", False))

    root = await init_project(
        ctx.project, refs, ip_info, description, ctx.client, force=force,
    )
    return StepResult(
        ok=True,
        message=f"Created project '{ctx.project}' with {len(refs)} ref images at {root}",
    )


async def do_set_sref(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    """Set which reference image is used as the style reference (0-based).

    In MODEL_SELECT, changing the sref is a hard experimental boundary — if the
    current pass already has SUCCESS results, auto-bump to a new pass first so
    the previous experiment is preserved on disk.
    """
    from styleclaw.core.models import StyleAnalysis

    if "index" not in args:
        return StepResult(ok=False, message="set-sref requires args.index")
    try:
        index = int(args["index"])
    except (TypeError, ValueError):
        return StepResult(ok=False, message=f"set-sref args.index must be an integer (got {args['index']!r})")

    config = project_store.load_config(ctx.project)
    if index < 0 or index >= len(config.ref_images):
        return StepResult(
            ok=False,
            message=f"index {index} out of range (0–{len(config.ref_images) - 1})",
        )

    state = project_store.load_state(ctx.project)
    bumped_to: int | None = None
    if state.phase == Phase.MODEL_SELECT:
        old_pass = state.current_model_select_pass or 1
        existing = project_store.load_all_task_records(ctx.project, pass_num=old_pass)
        if any(r.status == TaskStatus.SUCCESS for r in existing.values()):
            new_pass = old_pass + 1
            try:
                prev_analysis = project_store.load_analysis(ctx.project, pass_num=old_pass)
            except FileNotFoundError:
                prev_analysis = StyleAnalysis()
            project_store.save_analysis(ctx.project, prev_analysis, pass_num=new_pass)
            project_store.save_state(ctx.project, state.with_model_select_pass(new_pass))
            bumped_to = new_pass

    new_config = config.model_copy(update={"sref_index": index})
    project_store.save_config(ctx.project, new_config)

    msg = f"sref set to ref-{index + 1:03d}: {config.ref_images[index]}"
    if bumped_to is not None:
        msg += f" (auto-bumped to pass-{bumped_to:03d}; previous pass preserved)"
    return StepResult(ok=True, message=msg, data={"pass_num": bumped_to} if bumped_to else None)


async def do_set_pass(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    """Switch the active model-select pass number."""
    if "pass_num" not in args:
        return StepResult(ok=False, message="set-pass requires args.pass_num")
    try:
        pass_num = int(args["pass_num"])
    except (TypeError, ValueError):
        return StepResult(ok=False, message=f"set-pass args.pass_num must be an integer (got {args['pass_num']!r})")
    if pass_num < 1:
        return StepResult(ok=False, message=f"pass_num must be >= 1 (got {pass_num})")

    state = project_store.load_state(ctx.project)
    new_state = state.with_model_select_pass(pass_num)
    project_store.save_state(ctx.project, new_state)
    return StepResult(ok=True, message=f"Active pass set to {pass_num}")


async def do_add_refs(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    """Add reference images for image-to-image batch testing.

    Required args (collected via the confirmation callback):
      - image_dir: directory containing reference images for i2i

    Behavior mirrors the ``add-refs`` CLI command: when called from
    BATCH_T2I, advances the project to BATCH_I2I; uploads new images and
    appends them to the i2i uploads list for the current batch.
    """
    import shutil

    from styleclaw.core.image_utils import verify_ref_image
    from styleclaw.core.models import UploadRecord
    from styleclaw.core.state_machine import advance
    from styleclaw.providers.runninghub.upload import upload_file

    image_dir_str = args.get("image_dir", "").strip()
    if not image_dir_str:
        return StepResult(ok=False, message="add-refs requires args.image_dir")
    image_dir = Path(image_dir_str).expanduser()
    if not image_dir.is_dir():
        return StepResult(ok=False, message=f"image_dir is not a directory: {image_dir}")

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in image_exts)
    if not images:
        return StepResult(ok=False, message=f"No images found in {image_dir}")

    state = project_store.load_state(ctx.project)
    if state.phase not in (Phase.BATCH_T2I, Phase.BATCH_I2I):
        return StepResult(
            ok=False,
            message=f"add-refs requires BATCH_T2I or BATCH_I2I (current: {state.phase})",
        )

    for img in images:
        try:
            verify_ref_image(img)
        except ValueError as exc:
            return StepResult(ok=False, message=f"invalid ref image {img.name}: {exc}")

    if state.phase == Phase.BATCH_T2I:
        state = advance(state, Phase.BATCH_I2I)
        project_store.save_state(ctx.project, state)

    batch_num = state.current_batch or 1
    source_dir = project_store.batch_i2i_dir(ctx.project, batch_num) / "source-images"
    source_dir.mkdir(parents=True, exist_ok=True)

    local_dests: list[Path] = []
    for img in images:
        dest = source_dir / img.name
        shutil.copy2(img, dest)
        local_dests.append(dest)

    new_records: dict[int, UploadRecord] = {}

    async def _one(idx: int, dest: Path) -> None:
        new_records[idx] = await upload_file(ctx.client, dest)

    async with asyncio.TaskGroup() as tg:
        for idx, dest in enumerate(local_dests):
            tg.create_task(_one(idx, dest))

    appended = [new_records[i] for i in sorted(new_records)]
    if appended:
        existing = project_store.load_i2i_uploads(ctx.project, batch_num)
        project_store.save_i2i_uploads(ctx.project, batch_num, existing + appended)

    if state.current_batch != batch_num:
        new_state = state.with_batch(batch_num)
        project_store.save_state(ctx.project, new_state)

    return StepResult(
        ok=True,
        message=f"Added {len(appended)} ref images to BATCH_I2I batch {batch_num}",
    )


ACTION_REGISTRY: dict[str, ActionDef] = {
    "init":          ActionDef(fn=do_init,          needs_client=True,  needs_llm=False, requires_confirmation=True),
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
    "set-sref":      ActionDef(fn=do_set_sref,      needs_client=False, needs_llm=False),
    "set-pass":      ActionDef(fn=do_set_pass,      needs_client=False, needs_llm=False),
    "add-refs":      ActionDef(fn=do_add_refs,      needs_client=True,  needs_llm=False, requires_confirmation=True),
}


PHASE_ACTIONS: dict[Phase, list[str]] = {
    Phase.INIT:         ["analyze"],
    Phase.MODEL_SELECT: ["generate", "poll", "evaluate", "select-model", "retest-models", "set-sref", "set-pass"],
    Phase.STYLE_REFINE: ["refine", "generate", "poll", "evaluate", "approve", "select-model", "retest-models", "set-sref"],
    Phase.BATCH_T2I:    ["design-cases", "batch-submit", "poll", "report", "approve", "retest-models", "add-refs"],
    Phase.BATCH_I2I:    ["batch-submit", "poll", "report", "approve", "back-to-t2i", "add-refs"],
    Phase.COMPLETED:    [],
}
