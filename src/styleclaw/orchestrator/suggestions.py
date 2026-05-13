"""Phase-aware natural-language next-step suggestions for the `styleclaw run` UX.

Pure read-only helpers: load state via :mod:`styleclaw.storage.project_store`
and return formatted example commands. The intent is to bridge the gap between
finishing a phase and knowing what to say next when interacting with the
plan-and-execute orchestrator.
"""
from __future__ import annotations

from styleclaw.core.models import Phase, ProjectState
from styleclaw.storage import project_store


def _fmt(intent: str, project: str) -> str:
    return f'styleclaw run "{intent}" -p {project}'


def _init_suggestions(project: str, _state: ProjectState) -> list[str]:
    return [
        _fmt("分析参考图片，进入模型选型", project),
    ]


def _model_select_suggestions(project: str, state: ProjectState) -> list[str]:
    has_models = bool(state.selected_models)
    pick_intent = (
        "用 mj-v7 prompt-only 进入精炼"
        if not has_models
        else f"用 {state.selected_models[0]} {state.selected_variant} 进入精炼"
    )
    return [
        _fmt(pick_intent, project),
        _fmt("换 sref 到第 2 张重测", project),
        _fmt("只重测 mj-v7 和 niji7", project),
        _fmt("用全部模型新开一个 pass 再测一次", project),
    ]


def _style_refine_suggestions(project: str, state: ProjectState) -> list[str]:
    rollback_round = max(1, state.current_round - 1) if state.current_round > 1 else 1
    return [
        _fmt("继续精炼一轮", project),
        _fmt("给方向：增加对比度，加点半色调", project),
        _fmt(f"回退到第 {rollback_round} 轮重新精炼", project),
        _fmt("效果可以了，approve 进入批量测试", project),
    ]


def _batch_t2i_suggestions(project: str, _state: ProjectState) -> list[str]:
    return [
        _fmt("设计 100 个测试用例", project),
        _fmt("提交批量生成", project),
        _fmt("出一份 HTML 报告", project),
        _fmt("加几张参考图，进入图生图阶段", project),
    ]


def _batch_i2i_suggestions(project: str, _state: ProjectState) -> list[str]:
    return [
        _fmt("提交 i2i 批量任务", project),
        _fmt("出 i2i 报告", project),
        _fmt("全部完成，标记项目结束", project),
    ]


def _completed_suggestions(project: str, _state: ProjectState) -> list[str]:
    return [
        _fmt("查看最终报告", project),
        _fmt("回到批量阶段再调一调", project),
    ]


_DISPATCH = {
    Phase.INIT: _init_suggestions,
    Phase.MODEL_SELECT: _model_select_suggestions,
    Phase.STYLE_REFINE: _style_refine_suggestions,
    Phase.BATCH_T2I: _batch_t2i_suggestions,
    Phase.BATCH_I2I: _batch_i2i_suggestions,
    Phase.COMPLETED: _completed_suggestions,
}


def suggest_next_steps(project: str) -> list[str]:
    """Return 1-5 example `styleclaw run "..." -p <project>` lines for the project's current phase."""
    state = project_store.load_state(project)
    builder = _DISPATCH.get(state.phase)
    if builder is None:
        return []
    return builder(project, state)
