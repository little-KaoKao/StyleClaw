from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from styleclaw.core.models import Action, ActionPlan, LoopConfig
from styleclaw.orchestrator.actions import ACTION_REGISTRY
from styleclaw.orchestrator.planner import plan
from styleclaw.web.run_manager import RunConflict

router = APIRouter(prefix="/api/projects", tags=["runs"])


class PlanRequest(BaseModel):
    intent: str


def _planner_llm():
    """Build an LLM for the planner role. Isolated for easy test override."""
    from styleclaw.core.llm_routing import Role, RoleRouter

    router_obj = RoleRouter.from_env()
    return router_obj.get(Role.PLANNER)


@router.post("/{name}/plan")
async def preview_plan(name: str, req: PlanRequest) -> dict:
    from styleclaw.storage import project_store

    try:
        project_store.load_state(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"project '{name}' not found")
    llm = _planner_llm()
    action_plan = await plan(llm, name, req.intent)
    return action_plan.model_dump()


class StepIn(BaseModel):
    name: str
    description: str = ""
    args: dict[str, Any] = {}


class RunRequest(BaseModel):
    steps: list[StepIn]
    loop: dict[str, Any] | None = None
    summary: str = ""
    stop_summary: str = ""


@router.post("/{name}/run")
async def start_run(name: str, req: RunRequest, request: Request) -> dict:
    from styleclaw.storage import project_store

    try:
        project_store.load_state(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"project '{name}' not found")
    if not req.steps:
        raise HTTPException(status_code=400, detail="steps must not be empty")
    for s in req.steps:
        if s.name not in ACTION_REGISTRY:
            raise HTTPException(status_code=400, detail=f"unknown action: {s.name}")
        if ACTION_REGISTRY[s.name].requires_confirmation:
            raise HTTPException(
                status_code=400,
                detail=f"action '{s.name}' requires a dedicated confirm endpoint, not /run",
            )
    action_plan = ActionPlan(
        summary=req.summary or req.steps[0].name,
        steps=[Action(name=s.name, description=s.description or s.name, args=s.args) for s in req.steps],
        loop=LoopConfig(**req.loop) if req.loop else None,
        stop_summary=req.stop_summary,
    )
    mgr = request.app.state.run_manager
    kind = "plan" if len(req.steps) > 1 or action_plan.loop else "action"
    try:
        run_id = await mgr.start(name, action_plan, kind=kind)
    except RunConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"run_id": run_id}


@router.get("/{name}/runs/{run_id}")
async def get_run(name: str, run_id: str, request: Request) -> dict:
    mgr = request.app.state.run_manager
    try:
        return mgr.get(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
