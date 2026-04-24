from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from styleclaw.core.models import (
    BatchCase,
    BatchConfig,
    Phase,
    ProjectConfig,
    ProjectState,
    TaskRecord,
    UploadRecord,
)
from styleclaw.scripts.batch_submit import batch_submit_i2i, batch_submit_t2i
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def setup_project():
    config = ProjectConfig(name="test-proj", ip_info="anime")
    project_store.create_project(config)


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.post.return_value = {"taskId": "t1", "status": "QUEUED"}
    return client


class TestBatchSubmitT2i:
    async def test_submits_pending_cases(self, setup_project, mock_client) -> None:
        cases = [
            BatchCase(id="am-001", category="adult_male", description="desc 1"),
            BatchCase(id="am-002", category="adult_male", description="desc 2"),
            BatchCase(id="am-003", category="adult_male", description="desc 3", status="submitted"),
        ]
        batch_config = BatchConfig(batch=1, trigger_phrase="bold anime", cases=cases)
        project_store.save_batch_config("test-proj", 1, batch_config)

        records = await batch_submit_t2i("test-proj", mock_client, 1, "mj-v7")
        assert len(records) == 2  # only pending cases
        assert "am-001" in records
        assert "am-002" in records
        assert "am-003" not in records

    async def test_updates_case_status_to_submitted(self, setup_project, mock_client) -> None:
        cases = [
            BatchCase(id="am-001", category="adult_male", description="desc 1"),
        ]
        batch_config = BatchConfig(batch=1, trigger_phrase="bold anime", cases=cases)
        project_store.save_batch_config("test-proj", 1, batch_config)

        await batch_submit_t2i("test-proj", mock_client, 1, "mj-v7")
        updated = project_store.load_batch_config("test-proj", 1)
        assert updated.cases[0].status == "submitted"


class TestBatchSubmitI2i:
    async def test_submits_for_uploads(self, setup_project, mock_client) -> None:
        uploads = [
            UploadRecord(local_path="ref.png", url="https://cdn.example.com/1.png", file_name="1.png"),
            UploadRecord(local_path="ref2.png", url="https://cdn.example.com/2.png", file_name="2.png"),
        ]
        project_store.save_i2i_uploads("test-proj", 1, uploads)

        records = await batch_submit_i2i("test-proj", mock_client, 1, "mj-v7", "bold anime")
        assert len(records) == 2
        assert "i2i-001" in records
        assert "i2i-002" in records

    async def test_saves_i2i_batch_config(self, setup_project, mock_client) -> None:
        uploads = [
            UploadRecord(local_path="ref.png", url="https://cdn.example.com/1.png", file_name="1.png"),
        ]
        project_store.save_i2i_uploads("test-proj", 1, uploads)

        await batch_submit_i2i("test-proj", mock_client, 1, "nb2", "bold anime")
        config = project_store.load_i2i_batch_config("test-proj", 1)
        assert config.batch == 1
        assert len(config.cases) == 1
        assert config.cases[0].category == "i2i"

    async def test_mj_uses_image_url_param(self, setup_project, mock_client) -> None:
        uploads = [
            UploadRecord(local_path="ref.png", url="https://cdn.example.com/1.png", file_name="1.png"),
        ]
        project_store.save_i2i_uploads("test-proj", 1, uploads)

        await batch_submit_i2i("test-proj", mock_client, 1, "mj-v7", "bold anime")
        call_args = mock_client.post.call_args
        params = call_args[0][1]
        assert "imageUrl" in params
        assert params["iw"] == 0.5
