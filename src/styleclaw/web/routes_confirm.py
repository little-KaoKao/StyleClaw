from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from styleclaw.core.models import Action, ActionPlan
from styleclaw.orchestrator.actions import ACTION_REGISTRY, StepResult
from styleclaw.orchestrator.executor import execute
from styleclaw.web.context import build_context

router = APIRouter(prefix="/api/projects", tags=["confirm"])

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


async def _run_single(project: str, action: str, args: dict) -> StepResult:
    action_def = ACTION_REGISTRY[action]
    plan = ActionPlan(
        summary=action,
        steps=[Action(name=action, description=action, args=args)],
        loop=None,
        stop_summary="",
    )
    async with build_context(
        project, needs_client=action_def.needs_client, needs_llm=action_def.needs_llm,
    ) as ctx:
        results = await execute(plan, ctx)
    return results[-1] if results else StepResult(ok=False, message="no result")


def _result_payload(result: StepResult) -> dict:
    return {"ok": result.ok, "message": result.message, "data": result.data}


async def _save_uploads(files: list[UploadFile]) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="styleclaw-upload-"))
    saved = 0
    for f in files:
        fname = Path(f.filename or "").name
        if not fname or Path(fname).suffix.lower() not in _IMAGE_EXTS:
            continue
        dest = tmp_dir / fname
        dest.write_bytes(await f.read())
        saved += 1
    if saved == 0:
        raise HTTPException(status_code=400, detail="no valid image files uploaded")
    return tmp_dir


@router.post("")
async def create_project(
    name: str = Form(...),
    ip_info: str = Form(""),
    description: str = Form(""),
    force: bool = Form(False),
    files: list[UploadFile] = File(...),
) -> dict:
    tmp_dir = await _save_uploads(files)
    result = await _run_single(
        name, "init",
        {"ref_dir": str(tmp_dir), "ip_info": ip_info, "description": description, "force": force},
    )
    return _result_payload(result)


class SelectModelRequest(BaseModel):
    models: str
    variant: str = ""


@router.post("/{name}/select-model")
async def select_model(name: str, req: SelectModelRequest) -> dict:
    result = await _run_single(
        name, "select-model", {"models": req.models, "variant": req.variant},
    )
    return _result_payload(result)


@router.post("/{name}/refs")
async def add_refs(name: str, files: list[UploadFile] = File(...)) -> dict:
    tmp_dir = await _save_uploads(files)
    result = await _run_single(name, "add-refs", {"image_dir": str(tmp_dir)})
    return _result_payload(result)


class RollbackRequest(BaseModel):
    to: str          # phase name, case-insensitive, e.g. "STYLE_REFINE"
    round: int | None = None


@router.post("/{name}/rollback")
async def rollback(name: str, req: RollbackRequest) -> dict:
    from styleclaw.core.models import Phase
    from styleclaw.core.state_machine import can_rollback
    from styleclaw.core.state_machine import rollback as do_rollback
    from styleclaw.storage import project_store

    # 1. Parse target phase (case-insensitive); 400 on invalid.
    try:
        target = Phase(req.to.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid phase: {req.to}")

    with project_store.project_lock(name):
        try:
            state = project_store.load_state(name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"project '{name}' not found")

        # Same-phase round navigation: project is already IN STYLE_REFINE and caller
        # wants to switch to a different round within the same phase.
        if target == Phase.STYLE_REFINE and state.phase == Phase.STYLE_REFINE and req.round is not None:
            sr_root = project_store.project_dir(name) / "style-refine"
            matches = sorted(sr_root.glob(project_store.round_glob_across_passes(req.round)))
            if not matches:
                return {"ok": False, "message": f"round {req.round} 不存在", "data": None}
            new_state = state.with_round(req.round)
            project_store.save_state(name, new_state)
            return {
                "ok": True,
                "message": f"已切换到 round {req.round}",
                "data": {"phase": state.phase.value, "round": req.round},
            }

        if not can_rollback(state, target):
            return {
                "ok": False,
                "message": f"无法回滚到 {target.value}（必须是更早且访问过的阶段）",
                "data": None,
            }

        # If rolling back to STYLE_REFINE with a specific round > 0, verify it exists on disk.
        if target == Phase.STYLE_REFINE and req.round is not None and req.round > 0:
            sr_root = project_store.project_dir(name) / "style-refine"
            matches = sorted(sr_root.glob(project_store.round_glob_across_passes(req.round)))
            if not matches:
                return {"ok": False, "message": f"round {req.round} 不存在", "data": None}

        new_state = do_rollback(state, target)
        if req.round is not None:
            new_state = new_state.with_round(req.round)
        project_store.save_state(name, new_state)

    return {
        "ok": True,
        "message": (
            f"已回滚到 {target.value}"
            + (f" round {req.round}" if req.round is not None else "")
        ),
        "data": {"phase": target.value, "round": req.round},
    }
