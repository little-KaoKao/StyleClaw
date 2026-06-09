from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from styleclaw.storage import project_store


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Isolate DATA_ROOT per test (mirrors tests/test_cli.py::use_tmp_data_root)."""
    root = tmp_path / "projects"
    monkeypatch.setattr(project_store, "DATA_ROOT", root)
    monkeypatch.setenv("RUNNINGHUB_API_KEY", "test-key")
    monkeypatch.setenv("STYLECLAW_SKIP_ENV_CHECK", "1")
    return root


@pytest.fixture
def client(data_root):
    from styleclaw.web.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
