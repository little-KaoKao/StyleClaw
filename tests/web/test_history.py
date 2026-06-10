from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store
from styleclaw.web.history import build_history


def _setup(name: str) -> None:
    project_store.create_project(
        ProjectConfig(name=name, ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state(name, ProjectState(phase=Phase.STYLE_REFINE, current_model_select_pass=2, current_round=1))
    root = project_store.project_dir(name)
    # model-select pass 1 + 2
    for pnum in (1, 2):
        (root / "model-select" / project_store.pass_label(pnum)).mkdir(parents=True, exist_ok=True)
    # style-refine pass 1 with rounds 1,2,3
    for rnum in (1, 2, 3):
        (root / "style-refine" / project_store.pass_label(1) / project_store.round_label(rnum)).mkdir(parents=True, exist_ok=True)
    # batch-t2i batches 1,2
    for bnum in (1, 2):
        (root / "batch-t2i" / project_store.batch_label(bnum)).mkdir(parents=True, exist_ok=True)


def test_build_history_enumerates(data_root):
    _setup("h")
    hist = build_history("h")
    assert hist["model_select"]["passes"] == [1, 2]
    assert hist["style_refine"]["passes"]["1"] == [1, 2, 3]
    assert hist["batch_t2i"]["batches"] == [1, 2]
    assert hist["batch_i2i"]["batches"] == []
    assert hist["current"]["phase"] == "STYLE_REFINE"
    assert hist["current"]["pass"] == 2


def test_history_empty_project(data_root):
    project_store.create_project(
        ProjectConfig(name="empty", ip_info="x", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("empty", ProjectState(phase=Phase.INIT))
    hist = build_history("empty")
    assert hist["model_select"]["passes"] == []
    assert hist["style_refine"]["passes"] == {}
    assert hist["batch_t2i"]["batches"] == []


def test_history_endpoint(client, data_root):
    _setup("h2")
    resp = client.get("/api/projects/h2/history")
    assert resp.status_code == 200
    assert resp.json()["model_select"]["passes"] == [1, 2]


def test_history_endpoint_404(client, data_root):
    resp = client.get("/api/projects/ghost/history")
    assert resp.status_code == 404
