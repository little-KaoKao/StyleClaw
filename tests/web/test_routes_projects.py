from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


def _make_project(name: str, phase: Phase = Phase.INIT) -> None:
    project_store.create_project(
        ProjectConfig(name=name, ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state(name, ProjectState(phase=phase))


def test_list_projects_empty(client):
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert resp.json() == {"projects": []}


def test_list_projects(client, data_root):
    _make_project("alpha", Phase.MODEL_SELECT)
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    names = [p["name"] for p in resp.json()["projects"]]
    assert "alpha" in names
    alpha = next(p for p in resp.json()["projects"] if p["name"] == "alpha")
    assert alpha["phase"] == "MODEL_SELECT"


def test_project_detail(client, data_root):
    _make_project("beta", Phase.STYLE_REFINE)
    resp = client.get("/api/projects/beta")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"]["phase"] == "STYLE_REFINE"
    assert body["config"]["ip_info"] == "anime"
    assert isinstance(body["suggestions"], list)


def test_project_detail_not_found(client, data_root):
    resp = client.get("/api/projects/ghost")
    assert resp.status_code == 404
