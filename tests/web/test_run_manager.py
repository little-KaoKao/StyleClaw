import asyncio

import pytest

from styleclaw.core.models import Action, ActionPlan, Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store
from styleclaw.web.run_manager import RunConflict, RunManager


@pytest.fixture
def refine_project(data_root):
    """A project parked in STYLE_REFINE so `approve` (no client/llm) runs cleanly."""
    config = ProjectConfig(name="p", ip_info="anime", ref_images=["refs/ref-001.png"])
    project_store.create_project(config)
    project_store.save_state("p", ProjectState(phase=Phase.STYLE_REFINE, current_round=1))
    return "p"


def _approve_plan() -> ActionPlan:
    return ActionPlan(
        summary="approve",
        steps=[Action(name="approve", description="approve", args={"target": "batch-t2i"})],
        loop=None,
        stop_summary="",
    )


@pytest.mark.asyncio
async def test_run_emits_step_and_done_events(refine_project):
    mgr = RunManager()
    run_id = await mgr.start(refine_project, _approve_plan(), kind="action")
    # wait for completion
    for _ in range(100):
        snap = mgr.get(run_id)
        if snap["status"] in ("done", "error"):
            break
        await asyncio.sleep(0.02)
    snap = mgr.get(run_id)
    assert snap["status"] == "done"
    types = [e["type"] for e in snap["events"]]
    assert "run_started" in types
    assert "step_start" in types
    assert "step_done" in types
    assert types[-1] == "done"
    # phase actually advanced
    assert project_store.load_state(refine_project).phase == Phase.BATCH_T2I


@pytest.mark.asyncio
async def test_second_run_while_active_conflicts(refine_project, monkeypatch):
    mgr = RunManager()

    # Make the action slow so the first run is still active when the second starts.
    import styleclaw.web.run_manager as rm

    real_execute = rm.execute

    async def slow_execute(plan, ctx, **kw):
        await asyncio.sleep(0.3)
        return await real_execute(plan, ctx, **kw)

    monkeypatch.setattr(rm, "execute", slow_execute)

    run_id = await mgr.start(refine_project, _approve_plan(), kind="action")
    with pytest.raises(RunConflict):
        await mgr.start(refine_project, _approve_plan(), kind="action")
    # let the first finish to avoid a dangling task
    for _ in range(100):
        if mgr.get(run_id)["status"] in ("done", "error"):
            break
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_subscribe_receives_live_events(refine_project):
    mgr = RunManager()
    run_id = await mgr.start(refine_project, _approve_plan(), kind="action")
    queue, replay = mgr.subscribe(run_id)
    seen = [e["type"] for e in replay]
    try:
        while "done" not in seen:
            ev = await asyncio.wait_for(queue.get(), timeout=2.0)
            seen.append(ev["type"])
    finally:
        mgr.unsubscribe(run_id, queue)
    assert "done" in seen
