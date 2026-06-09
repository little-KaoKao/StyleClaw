import pytest

from styleclaw.core.models import Action, ActionPlan, Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


@pytest.fixture
def planned_project(data_root):
    project_store.create_project(
        ProjectConfig(name="p", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("p", ProjectState(phase=Phase.STYLE_REFINE, current_round=1))
    return "p"


def test_plan_endpoint(client, planned_project, monkeypatch):
    fake_plan = ActionPlan(
        summary="精炼一轮",
        steps=[Action(name="refine", description="refine", args={})],
        loop=None,
        stop_summary="停在评分后",
    )

    async def fake_plan_fn(llm, project, intent):
        return fake_plan

    monkeypatch.setattr("styleclaw.web.routes_runs.plan", fake_plan_fn)
    # avoid building a real RoleRouter
    monkeypatch.setattr(
        "styleclaw.web.routes_runs._planner_llm",
        lambda: object(),
    )

    resp = client.post("/api/projects/p/plan", json={"intent": "帮我精炼一轮"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"] == "精炼一轮"
    assert body["steps"][0]["name"] == "refine"
