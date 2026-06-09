from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from styleclaw.orchestrator.planner import plan

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
