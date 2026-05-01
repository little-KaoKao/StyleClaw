from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from styleclaw.core.models import TaskRecord
from styleclaw.providers.runninghub.tasks import (
    poll_and_update,
    poll_task,
    query_task,
    submit_task,
)


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


class TestSubmitTask:
    async def test_success_on_first_attempt(self, mock_client: AsyncMock) -> None:
        mock_client.post.return_value = {"taskId": "t1", "status": "QUEUED"}
        record = await submit_task(mock_client, "/api/gen", {"prompt": "test"}, "mj-v7")
        assert record.task_id == "t1"
        assert record.model_id == "mj-v7"
        assert record.status == "QUEUED"

    async def test_retries_on_empty_task_id(self, mock_client: AsyncMock) -> None:
        mock_client.post.side_effect = [
            {"taskId": "", "errorCode": "BUSY"},
            {"taskId": "", "errorCode": "BUSY"},
            {"taskId": "t2", "status": "QUEUED"},
        ]
        record = await submit_task(mock_client, "/api/gen", {"prompt": "test"}, "mj-v7")
        assert record.task_id == "t2"
        assert mock_client.post.call_count == 3

    async def test_fails_after_max_retries(self, mock_client: AsyncMock) -> None:
        mock_client.post.return_value = {"taskId": "", "errorCode": "BUSY", "errorMessage": "server busy"}
        with pytest.raises(RuntimeError, match="failed after 3 retries"):
            await submit_task(mock_client, "/api/gen", {"prompt": "test"}, "mj-v7")

    async def test_captures_prompt_in_record(self, mock_client: AsyncMock) -> None:
        mock_client.post.return_value = {"taskId": "t1"}
        record = await submit_task(mock_client, "/api/gen", {"prompt": "hello"}, "nb2")
        assert record.prompt == "hello"


class TestQueryTask:
    async def test_query_posts_task_id(self, mock_client: AsyncMock) -> None:
        mock_client.post.return_value = {"status": "RUNNING"}
        result = await query_task(mock_client, "t1")
        mock_client.post.assert_called_once_with("/openapi/v2/query", {"taskId": "t1"})
        assert result["status"] == "RUNNING"


class TestPollTask:
    async def test_returns_on_success(self, mock_client: AsyncMock) -> None:
        mock_client.post.side_effect = [
            {"status": "RUNNING"},
            {"status": "SUCCESS", "results": [{"url": "http://img.png"}]},
        ]
        result = await poll_task(mock_client, "t1", interval=0.01, timeout=1)
        assert result["status"] == "SUCCESS"

    async def test_raises_on_failure(self, mock_client: AsyncMock) -> None:
        mock_client.post.return_value = {"status": "FAILED", "errorMessage": "bad input"}
        with pytest.raises(RuntimeError, match="bad input"):
            await poll_task(mock_client, "t1", interval=0.01, timeout=1)

    async def test_raises_on_timeout(self, mock_client: AsyncMock) -> None:
        mock_client.post.return_value = {"status": "RUNNING"}
        with pytest.raises(TimeoutError, match="timed out"):
            await poll_task(mock_client, "t1", interval=0.01, timeout=0.03)

    async def test_aborts_on_consecutive_network_failures(self, mock_client: AsyncMock) -> None:
        import httpx

        mock_client.post.side_effect = httpx.ConnectError("refused")
        with pytest.raises(RuntimeError, match="consecutive network failures"):
            await poll_task(
                mock_client, "t1", interval=0.01, timeout=10,
                max_consecutive_failures=3,
            )
        assert mock_client.post.call_count == 3

    async def test_resets_failure_counter_on_success(self, mock_client: AsyncMock) -> None:
        import httpx

        mock_client.post.side_effect = [
            httpx.ConnectError("1"),
            httpx.ConnectError("2"),
            {"status": "RUNNING"},
            httpx.ConnectError("3"),
            httpx.ConnectError("4"),
            {"status": "SUCCESS", "results": []},
        ]
        result = await poll_task(
            mock_client, "t1", interval=0.01, timeout=10,
            max_consecutive_failures=3,
        )
        assert result["status"] == "SUCCESS"


class TestPollAndUpdate:
    async def test_skips_already_succeeded(self, mock_client: AsyncMock) -> None:
        record = TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS")
        result = await poll_and_update(mock_client, record)
        assert result is record
        mock_client.post.assert_not_called()

    async def test_skips_already_failed(self, mock_client: AsyncMock) -> None:
        record = TaskRecord(task_id="t1", model_id="mj-v7", status="FAILED")
        result = await poll_and_update(mock_client, record)
        assert result is record

    async def test_updates_on_success(self, mock_client: AsyncMock) -> None:
        record = TaskRecord(task_id="t1", model_id="mj-v7", status="QUEUED")
        mock_client.post.return_value = {
            "status": "SUCCESS",
            "results": [{"url": "http://img.png"}],
        }
        result = await poll_and_update(mock_client, record)
        assert result.status == "SUCCESS"
        assert result.completed_at != ""
        assert len(result.results) == 1

    async def test_updates_on_poll_failure(self, mock_client: AsyncMock) -> None:
        record = TaskRecord(task_id="t1", model_id="mj-v7", status="QUEUED")
        mock_client.post.return_value = {"status": "FAILED", "errorMessage": "error"}
        result = await poll_and_update(mock_client, record)
        assert result.status == "FAILED"
        assert "error" in result.error_message

    @patch("styleclaw.providers.runninghub.tasks.TASK_TIMEOUT", 0.03)
    @patch("styleclaw.providers.runninghub.tasks.POLL_INTERVAL", 0.01)
    async def test_updates_on_timeout(self, mock_client: AsyncMock) -> None:
        record = TaskRecord(task_id="t1", model_id="mj-v7", status="QUEUED")
        mock_client.post.return_value = {"status": "RUNNING"}
        result = await poll_and_update(mock_client, record)
        assert result.status == "FAILED"
        assert "timed out" in result.error_message
