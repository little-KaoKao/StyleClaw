"""Estimate the API cost of an ActionPlan before execution.

The user's main concern is "save credits" — submitting too many image-gen
tasks by mistake. This module walks an :class:`ActionPlan` against the
current :class:`ProjectState` and produces a human-readable summary of:

- how many image-generation tasks will be submitted
- how many output images those tasks are expected to return
- which steps are LLM-only (cheap) or no-cost (state mutations only)

Estimates are deliberately conservative — RunningHub may return fewer
images than requested, but the planner should not under-promise costs.

The module has no side effects: it only reads project_store data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from styleclaw.core.models import ActionPlan, Phase, ProjectState
from styleclaw.providers.runninghub.models import MODEL_REGISTRY, SrefMode, get_model
from styleclaw.storage import project_store

# Test matrix used by generate_model_select: 2 variants × 2 genders.
MODEL_SELECT_VARIANTS = 2
MODEL_SELECT_GENDERS = 2
MODEL_SELECT_TASKS_PER_MODEL = MODEL_SELECT_VARIANTS * MODEL_SELECT_GENDERS

# Mirrors generate.py IMAGES_PER_MODEL_REFINE for PROMPT-mode models.
REFINE_MAX_IMAGES = 3


def _images_per_task(model_id: str, *, refine: bool = False) -> int:
    """Approximate output images per submitted task.

    PARAM-mode models (mj-v7, niji7) always return 4 images.
    PROMPT-mode models return 1 by default; REFINE bumps maxImages to
    REFINE_MAX_IMAGES.
    """
    try:
        config = get_model(model_id)
    except ValueError:
        return 1
    if config.sref_mode == SrefMode.PARAM:
        return 4
    return REFINE_MAX_IMAGES if refine else 1


@dataclass(frozen=True)
class StepEstimate:
    label: str
    tasks: int = 0
    images: int = 0
    note: str = ""


@dataclass(frozen=True)
class PlanEstimate:
    steps: list[StepEstimate] = field(default_factory=list)
    loop_iterations_max: int = 1

    @property
    def total_tasks(self) -> int:
        per_iter = sum(s.tasks for s in self.steps)
        return per_iter * max(1, self.loop_iterations_max)

    @property
    def total_images(self) -> int:
        per_iter = sum(s.images for s in self.steps)
        return per_iter * max(1, self.loop_iterations_max)


def _resolve_models(args: dict[str, Any]) -> list[str] | None:
    raw = args.get("models")
    if isinstance(raw, str):
        return [m.strip() for m in raw.split(",") if m.strip()] or None
    if isinstance(raw, list):
        return [str(m).strip() for m in raw if str(m).strip()] or None
    return None


def _estimate_generate(args: dict[str, Any], state: ProjectState) -> StepEstimate:
    if state.phase == Phase.MODEL_SELECT:
        chosen = _resolve_models(args) or list(MODEL_REGISTRY.keys())
        tasks = len(chosen) * MODEL_SELECT_TASKS_PER_MODEL
        images = sum(
            _images_per_task(m) * MODEL_SELECT_TASKS_PER_MODEL for m in chosen
        )
        note = f"{len(chosen)} 模型 × {MODEL_SELECT_VARIANTS} 变体 × {MODEL_SELECT_GENDERS} 性别"
        return StepEstimate("generate", tasks=tasks, images=images, note=note)

    if state.phase == Phase.STYLE_REFINE:
        models = state.selected_models or []
        tasks = len(models)
        images = sum(_images_per_task(m, refine=True) for m in models)
        note = f"{len(models)} 选定模型，每个一次"
        return StepEstimate("generate", tasks=tasks, images=images, note=note)

    return StepEstimate("generate", note=f"(在 {state.phase} 不会真的提交)")


def _estimate_batch_submit(
    project: str, args: dict[str, Any], state: ProjectState,
) -> StepEstimate:
    batch_num = state.current_batch
    if batch_num < 1:
        return StepEstimate("batch-submit", note="(尚未 design-cases)")

    try:
        cfg = project_store.load_batch_config(project, batch_num)
    except FileNotFoundError:
        return StepEstimate(
            "batch-submit",
            note=f"({project_store.batch_label(batch_num)} 还没有 cases.json)",
        )

    pending = sum(1 for c in cfg.cases if c.status == "pending")
    model_id = args.get("model") or (state.selected_models[0] if state.selected_models else None)
    if model_id is None:
        return StepEstimate("batch-submit", tasks=pending, note=f"{pending} 个 pending case，模型未指定")

    refine_like = state.phase == Phase.BATCH_I2I  # i2i jobs request maxImages too
    images = pending * _images_per_task(model_id, refine=refine_like)
    return StepEstimate(
        "batch-submit", tasks=pending, images=images,
        note=f"{pending} pending × {model_id}",
    )


def _estimate_step(
    project: str, name: str, args: dict[str, Any], state: ProjectState,
) -> StepEstimate:
    if name == "generate":
        return _estimate_generate(args, state)
    if name == "batch-submit":
        return _estimate_batch_submit(project, args, state)
    if name in ("analyze", "evaluate", "refine", "design-cases"):
        return StepEstimate(name, note="LLM 调用，不提交图像任务")
    if name == "poll":
        return StepEstimate("poll", note="等待已提交任务，不新增")
    if name == "init":
        return StepEstimate("init", note="只上传参考图")
    if name == "add-refs":
        return StepEstimate("add-refs", note="只上传 i2i 参考图")
    return StepEstimate(name, note="(无 API 成本)")


def estimate_plan(plan: ActionPlan, project: str) -> PlanEstimate:
    try:
        state = project_store.load_state(project)
    except FileNotFoundError:
        # No project yet (init mode); nothing to estimate.
        return PlanEstimate(steps=[], loop_iterations_max=1)

    estimates = [_estimate_step(project, s.name, s.args, state) for s in plan.steps]
    loop_max = plan.loop.max_iterations if plan.loop else 1
    return PlanEstimate(steps=estimates, loop_iterations_max=loop_max)


def format_plan_estimate(estimate: PlanEstimate) -> list[str]:
    """Render a PlanEstimate as a list of lines for typer.echo."""
    if not estimate.steps:
        return []

    lines: list[str] = []
    for s in estimate.steps:
        if s.tasks or s.images:
            chunk = f"{s.label:14s} → {s.tasks} 任务"
            if s.images:
                chunk += f"，约 {s.images} 张图"
            if s.note:
                chunk += f"（{s.note}）"
        elif s.note:
            chunk = f"{s.label:14s} → {s.note}"
        else:
            continue
        lines.append(chunk)

    if estimate.total_tasks or estimate.total_images:
        suffix = (
            f" × 最多 {estimate.loop_iterations_max} 轮 = {estimate.total_tasks} 任务"
            if estimate.loop_iterations_max > 1
            else ""
        )
        lines.append(
            f"合计：{sum(s.tasks for s in estimate.steps)} 任务"
            f"，约 {sum(s.images for s in estimate.steps)} 张图{suffix}"
        )
    return lines
