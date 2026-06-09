import time

from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


def _refine_project(name: str) -> None:
    project_store.create_project(
        ProjectConfig(name=name, ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state(name, ProjectState(phase=Phase.STYLE_REFINE, current_round=1))


def test_run_single_action_and_poll(client, data_root):
    _refine_project("p")
    resp = client.post(
        "/api/projects/p/run",
        json={"steps": [{"name": "approve", "args": {"target": "batch-t2i"}}]},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    status = None
    for _ in range(100):
        snap = client.get(f"/api/projects/p/runs/{run_id}").json()
        status = snap["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.02)
    assert status == "done"
    types = [e["type"] for e in snap["events"]]
    assert types[-1] == "done"
    assert project_store.load_state("p").phase == Phase.BATCH_T2I


def test_run_unknown_run_id_404(client, data_root):
    _refine_project("p")
    resp = client.get("/api/projects/p/runs/nonexistent")
    assert resp.status_code == 404


def test_run_rejects_empty_steps(client, data_root):
    _refine_project("p")
    resp = client.post("/api/projects/p/run", json={"steps": []})
    assert resp.status_code == 400
