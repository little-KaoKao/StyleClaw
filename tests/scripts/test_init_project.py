from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from styleclaw.core.models import UploadRecord
from styleclaw.scripts.init_project import init_project
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def ref_images(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(3):
        p = tmp_path / f"input-{i}.png"
        p.write_bytes(b"fake image")
        paths.append(p)
    return paths


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    call_count = 0

    async def fake_upload(path, file_path):
        nonlocal call_count
        call_count += 1
        return {
            "code": 0,
            "data": {
                "download_url": f"https://cdn.example.com/ref-{call_count}.png",
                "fileName": f"ref-{call_count}.png",
            },
        }

    client.upload = fake_upload
    return client


class TestInitProject:
    async def test_creates_project_and_copies_refs(self, ref_images, mock_client) -> None:
        root = await init_project("test-proj", ref_images, "anime", "test desc", mock_client)
        assert root.exists()
        assert (root / "config.json").exists()
        refs = list((root / "refs").glob("ref-*.png"))
        assert len(refs) == 3

    async def test_saves_upload_records(self, ref_images, mock_client) -> None:
        await init_project("test-proj", ref_images, "anime", "test desc", mock_client)
        records = project_store.load_uploads("test-proj")
        assert len(records) == 3
        assert all("cdn.example.com" in r.url for r in records)

    async def test_updates_config_with_ref_paths(self, ref_images, mock_client) -> None:
        await init_project("test-proj", ref_images, "anime", "desc", mock_client)
        config = project_store.load_config("test-proj")
        assert len(config.ref_images) == 3
        assert all(r.startswith("refs/ref-") for r in config.ref_images)
