from __future__ import annotations

from fastapi import APIRouter, HTTPException

from styleclaw.orchestrator.suggestions import suggest_next_steps
from styleclaw.storage import project_store

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
