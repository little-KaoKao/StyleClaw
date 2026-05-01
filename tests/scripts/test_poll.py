from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, call, patch

import pytest

from styleclaw.core.models import Phase, ProjectConfig, ProjectState, TaskRecord
from styleclaw.scripts.poll import (
    _download_results,
    poll_batch,
    poll_model_select,
    poll_style_refine,
)
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

        mock_download.return_value = Path("/tmp/output-001.png")
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


class TestDownloadResults:
    async def test_downloads_each_result_url(self, tmp_path) -> None:
        results = [
            {"url": "http://example.com/img1.png"},
            {"url": "http://example.com/img2.png"},
        ]
        with patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock) as mock_dl:
            await _download_results(results, tmp_path)
            assert mock_dl.call_count == 2
            assert mock_dl.call_args_list[0] == call(
                "http://example.com/img1.png", tmp_path / "output-001.png"
            )
            assert mock_dl.call_args_list[1] == call(
                "http://example.com/img2.png", tmp_path / "output-002.png"
            )

    async def test_skips_empty_urls(self, tmp_path) -> None:
        results = [{"url": ""}, {"other_key": "value"}, {"url": "http://ok.png"}]
        with patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock) as mock_dl:
            await _download_results(results, tmp_path)
            assert mock_dl.call_count == 1

    async def test_empty_results_list(self, tmp_path) -> None:
        with patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock) as mock_dl:
            stats = await _download_results([], tmp_path)
            mock_dl.assert_not_called()
            assert stats.attempted == 0 and stats.succeeded == 0

    async def test_returns_stats_on_success(self, tmp_path) -> None:
        results = [
            {"url": "http://example.com/img1.png"},
            {"url": "http://example.com/img2.png"},
        ]
        with patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock):
            stats = await _download_results(results, tmp_path)
        assert stats.attempted == 2 and stats.succeeded == 2 and stats.failed == 0

    async def test_returns_stats_counting_failures(self, tmp_path) -> None:
        results = [
            {"url": "http://example.com/a.png"},
            {"url": "http://example.com/b.png"},
            {"url": "http://example.com/c.png"},
        ]
        side_effects = [tmp_path / "a.png", RuntimeError("boom"), tmp_path / "c.png"]
        with patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock, side_effect=side_effects):
            stats = await _download_results(results, tmp_path)
        assert stats.attempted == 3
        assert stats.succeeded == 2
        assert stats.failed == 1

    async def test_skipped_urls_not_counted_as_attempt(self, tmp_path) -> None:
        results = [{"url": ""}, {"url": "http://example.com/ok.png"}]
        with patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock):
            stats = await _download_results(results, tmp_path)
        assert stats.attempted == 1 and stats.succeeded == 1


class TestConcurrentPoll:
    @patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock)
    async def test_poll_model_select_runs_concurrently(
        self, mock_download, setup_project, mock_client
    ) -> None:
        for mid in ("mj-v7", "niji7"):
            record = TaskRecord(task_id=f"t-{mid}", model_id=mid, status="QUEUED")
            project_store.save_task_record("test-proj", mid, record)

        poll_start_times: list[float] = []

        original_poll_and_update = None

        async def _tracking_poll(client, record):
            poll_start_times.append(asyncio.get_event_loop().time())
            from styleclaw.providers.runninghub.tasks import poll_and_update as real_fn
            return record.model_copy(update={
                "status": "SUCCESS",
                "results": [{"url": "http://img.png"}],
            })

        with patch("styleclaw.scripts.poll.poll_and_update", side_effect=_tracking_poll):
            results = await poll_model_select("test-proj", mock_client)

        assert len(results) == 2
        assert all(r.status == "SUCCESS" for r in results.values())

    @patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock)
    async def test_poll_batch_runs_concurrently(
        self, mock_download, setup_project, mock_client
    ) -> None:
        for i in range(3):
            cid = f"am-{i:03d}"
            record = TaskRecord(task_id=f"t-{cid}", model_id="mj-v7", status="QUEUED")
            project_store.save_batch_task_record("test-proj", 1, cid, record)

        results = await poll_batch("test-proj", mock_client, 1, phase="t2i")
        assert len(results) == 3
        assert all(r.status == "SUCCESS" for r in results.values())

    @patch("styleclaw.scripts.poll.download_image", new_callable=AsyncMock)
    async def test_poll_style_refine_runs_concurrently(
        self, mock_download, setup_project, mock_client
    ) -> None:
        for mid in ("mj-v7", "niji7"):
            record = TaskRecord(task_id=f"t-{mid}", model_id=mid, status="QUEUED")
            project_store.save_round_task_record("test-proj", 1, mid, record)

        results = await poll_style_refine("test-proj", mock_client, 1)
        assert len(results) == 2
        assert all(r.status == "SUCCESS" for r in results.values())
