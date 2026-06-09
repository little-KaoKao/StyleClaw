import time

from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


def test_full_run_then_gallery(client, data_root):
    # Park a project in STYLE_REFINE, run `approve` via /run, watch WS to done,
    # then read the gallery for the new phase. Exercises the whole M1 spine
    # with no network (approve needs neither client nor llm).
    project_store.create_project(
        ProjectConfig(name="p", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("p", ProjectState(phase=Phase.STYLE_REFINE, current_round=1))

    run_id = client.post(
        "/api/projects/p/run",
        json={"steps": [{"name": "approve", "args": {"target": "batch-t2i"}}]},
    ).json()["run_id"]

    seen = []
    with client.websocket_connect(f"/api/projects/p/events?run_id={run_id}") as ws:
        for _ in range(50):
            ev = ws.receive_json()
            seen.append(ev["type"])
            if ev["type"] in ("done", "error"):
                break
    assert seen[-1] == "done"

    detail = client.get("/api/projects/p").json()
    assert detail["state"]["phase"] == "BATCH_T2I"

    gallery = client.get("/api/projects/p/gallery").json()
    assert gallery["phase"] == "BATCH_T2I"
