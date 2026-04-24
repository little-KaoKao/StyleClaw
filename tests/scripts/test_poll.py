from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from styleclaw.core.models import Phase, ProjectConfig, ProjectState, TaskRecord
from styleclaw.scripts.poll import poll_batch, poll_model_select, poll_style_refine
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
    client.post.return_value = {"status": "SUCCESS", "results": [{"url": "http://img.png"}]}
    return client


class TestPollModelSelect:
    async def test_raises_when_no_records(self, setup_project, mock_client) -> None:
        with pytest.raises(RuntimeError, match="No task records"):
            await poll_model_select("test-proj", mock_client)

    @patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock)
    async def test_polls_and_downloads(self, mock_download, setup_project, mock_client) -> None:
        record = TaskRecord(task_id="t1", model_id="mj-v7", status="QUEUED")
        project_store.save_task_record("test-proj", "mj-v7", record)

        mock_download.return_value = None
        results = await poll_model_select("test-proj", mock_client)
        assert results["mj-v7"].status == "SUCCESS"
        assert mock_download.called

    @patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock)
    async def test_skips_already_succeeded(self, mock_download, setup_project, mock_client) -> None:
        record = TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS")
        project_store.save_task_record("test-proj", "mj-v7", record)
        results = await poll_model_select("test-proj", mock_client)
        assert results["mj-v7"].status == "SUCCESS"
        mock_client.post.assert_not_called()

    @patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock)
    async def test_skips_no_task_id(self, mock_download, setup_project, mock_client) -> None:
        record = TaskRecord(task_id="", model_id="mj-v7", status="FAILED")
        project_store.save_task_record("test-proj", "mj-v7", record)
        results = await poll_model_select("test-proj", mock_client)
        assert results["mj-v7"].status == "FAILED"
        mock_client.post.assert_not_called()


class TestPollStyleRefine:
    async def test_raises_when_no_records(self, setup_project, mock_client) -> None:
        with pytest.raises(RuntimeError, match="No task records"):
            await poll_style_refine("test-proj", mock_client, 1)

    @patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock)
    async def test_polls_round_records(self, mock_download, setup_project, mock_client) -> None:
        record = TaskRecord(task_id="t1", model_id="mj-v7", status="QUEUED")
        project_store.save_round_task_record("test-proj", 1, "mj-v7", record)

        results = await poll_style_refine("test-proj", mock_client, 1)
        assert results["mj-v7"].status == "SUCCESS"


class TestPollBatch:
    async def test_raises_when_no_records(self, setup_project, mock_client) -> None:
        with pytest.raises(RuntimeError, match="No task records"):
            await poll_batch("test-proj", mock_client, 1, phase="t2i")

    @patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock)
    async def test_polls_t2i_records(self, mock_download, setup_project, mock_client) -> None:
        record = TaskRecord(task_id="t1", model_id="mj-v7", status="QUEUED")
        project_store.save_batch_task_record("test-proj", 1, "am-001", record)

        results = await poll_batch("test-proj", mock_client, 1, phase="t2i")
        assert results["am-001"].status == "SUCCESS"

    @patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock)
    async def test_polls_i2i_records(self, mock_download, setup_project, mock_client) -> None:
        record = TaskRecord(task_id="t1", model_id="mj-v7", status="QUEUED")
        project_store.save_i2i_task_record("test-proj", 1, "i2i-001", record)

        results = await poll_batch("test-proj", mock_client, 1, phase="i2i")
        assert results["i2i-001"].status == "SUCCESS"

    @patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock)
    async def test_skips_completed_and_failed(self, mock_download, setup_project, mock_client) -> None:
        r1 = TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS")
        r2 = TaskRecord(task_id="t2", model_id="mj-v7", status="FAILED")
        project_store.save_batch_task_record("test-proj", 1, "am-001", r1)
        project_store.save_batch_task_record("test-proj", 1, "am-002", r2)

        results = await poll_batch("test-proj", mock_client, 1, phase="t2i")
        assert results["am-001"].status == "SUCCESS"
        assert results["am-002"].status == "FAILED"
        mock_client.post.assert_not_called()

    @patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock)
    async def test_skips_no_task_id(self, mock_download, setup_project, mock_client) -> None:
        record = TaskRecord(task_id="", model_id="mj-v7", status="QUEUED")
        project_store.save_batch_task_record("test-proj", 1, "am-001", record)

        results = await poll_batch("test-proj", mock_client, 1, phase="t2i")
        assert results["am-001"].status == "QUEUED"
        mock_client.post.assert_not_called()
