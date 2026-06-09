from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_static(tmp_path, monkeypatch):
    # Build a fake static dir so the SPA-serving branch activates.
    import styleclaw.web.app as app_mod

    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html><body>SPA</body></html>", encoding="utf-8")
    (static / "assets" / "app.js").write_text("console.log('x')", encoding="utf-8")

    # Point the app factory at our fake static dir.
    monkeypatch.setattr(app_mod, "_STATIC_DIR", static)
    return app_mod.create_app()


def test_root_serves_index(app_with_static):
    client = TestClient(app_with_static)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "SPA" in resp.text


def test_client_route_serves_index(app_with_static):
    client = TestClient(app_with_static)
    resp = client.get("/some/client/route")
    assert resp.status_code == 200
    assert "SPA" in resp.text


def test_api_still_json(app_with_static):
    client = TestClient(app_with_static)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_unknown_api_path_is_404_not_html(app_with_static):
    client = TestClient(app_with_static)
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404


def test_assets_served(app_with_static):
    client = TestClient(app_with_static)
    resp = client.get("/assets/app.js")
    assert resp.status_code == 200
    assert "console.log" in resp.text
