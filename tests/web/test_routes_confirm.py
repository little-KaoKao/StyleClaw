import io

from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store


def _png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
        b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_select_model_advances_phase(client, data_root):
    project_store.create_project(
        ProjectConfig(name="p", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("p", ProjectState(phase=Phase.MODEL_SELECT, current_model_select_pass=1))
    resp = client.post(
        "/api/projects/p/select-model",
        json={"models": "mj-v7", "variant": "prompt-only"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert project_store.load_state("p").phase == Phase.STYLE_REFINE


def test_select_model_rejects_unknown(client, data_root):
    project_store.create_project(
        ProjectConfig(name="p", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("p", ProjectState(phase=Phase.MODEL_SELECT, current_model_select_pass=1))
    resp = client.post("/api/projects/p/select-model", json={"models": "no-such-model"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


def test_init_creates_project(client, data_root, monkeypatch):
    async def fake_init_project(name, refs, ip_info, description, client_, force=False):
        from styleclaw.core.models import ProjectConfig as PC
        project_store.create_project(
            PC(name=name, ip_info=ip_info, ref_images=[f"refs/{p.name}" for p in refs]),
            force=force,
        )
        return project_store.project_dir(name)

    monkeypatch.setattr(
        "styleclaw.scripts.init_project.init_project", fake_init_project
    )

    files = [("files", ("ref-001.png", io.BytesIO(_png_bytes()), "image/png"))]
    resp = client.post(
        "/api/projects",
        data={"name": "newproj", "ip_info": "anime", "description": "d"},
        files=files,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True
    assert "newproj" in project_store.list_projects()
