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
