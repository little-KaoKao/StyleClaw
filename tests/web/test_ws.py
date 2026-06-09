import time

from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


def _refine_project(name: str) -> None:
    project_store.create_project(
        ProjectConfig(name=name, ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state(name, ProjectState(phase=Phase.STYLE_REFINE, current_round=1))


def test_ws_streams_until_done(client, data_root):
    _refine_project("p")
    run_id = client.post(
        "/api/projects/p/run",
        json={"steps": [{"name": "approve", "args": {"target": "batch-t2i"}}]},
    ).json()["run_id"]

    received = []
    with client.websocket_connect(f"/api/projects/p/events?run_id={run_id}") as ws:
        for _ in range(50):
            ev = ws.receive_json()
            received.append(ev["type"])
            if ev["type"] in ("done", "error"):
                break
    assert "done" in received


def test_ws_no_run_closes(client, data_root):
    _refine_project("p")
    # no active run, no run_id -> server should close promptly
    with client.websocket_connect("/api/projects/p/events") as ws:
        ev = ws.receive_json()
        assert ev["type"] == "error"
