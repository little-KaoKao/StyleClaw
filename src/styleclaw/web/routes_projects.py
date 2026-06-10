from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from styleclaw.orchestrator.suggestions import suggest_next_steps
from styleclaw.storage import project_store
from styleclaw.web.gallery import build_gallery

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("")
async def list_projects() -> dict:
    out = []
    for name in project_store.list_projects():
        try:
            state = project_store.load_state(name)
        except FileNotFoundError:
            continue
        out.append({
            "name": name,
            "phase": state.phase.value,
            "current_round": state.current_round,
            "current_batch": state.current_batch,
            "last_updated": state.last_updated,
        })
    return {"projects": out}


@router.get("/{name}")
async def project_detail(name: str) -> dict:
    try:
        state = project_store.load_state(name)
        config = project_store.load_config(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"project '{name}' not found")
    return {
        "state": state.model_dump(),
        "config": config.model_dump(),
        "suggestions": suggest_next_steps(name),
    }


@router.get("/{name}/history")
async def project_history(name: str) -> dict:
    from styleclaw.web.history import build_history
    try:
        return build_history(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"project '{name}' not found")


@router.get("/{name}/gallery")
async def project_gallery(
    name: str,
    phase: str | None = Query(None),
    pass_: int | None = Query(None, alias="pass"),
    round_: int | None = Query(None, alias="round"),
    batch: int | None = Query(None),
) -> dict:
    try:
        return build_gallery(name, phase=phase, pass_num=pass_, round_num=round_, batch_num=batch)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"project '{name}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


media_router = APIRouter(prefix="/media", tags=["media"])


def safe_media_path(name: str, file_path: str) -> Path | None:
    """Resolve a media path under the project dir, or None if it escapes."""
    base = project_store.project_dir(name).resolve()
    target = (base / file_path).resolve()
    if not target.is_relative_to(base):
        return None
    return target


@media_router.get("/{name}/{file_path:path}")
async def media(name: str, file_path: str) -> FileResponse:
    target = safe_media_path(name, file_path)
    if target is None:
        raise HTTPException(status_code=400, detail="invalid path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(target)
