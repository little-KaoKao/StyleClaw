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

    async def test_captures_endpoint_in_record(self, mock_client: AsyncMock) -> None:
        """The endpoint must be persisted so resubmits target the same
        endpoint (e.g. an i2i task should not be silently retried via t2i)."""
        mock_client.post.return_value = {"taskId": "t1"}
        record = await submit_task(
            mock_client, "/openapi/v2/foo/image-to-image", {"prompt": "x"}, "nb2",
        )
        assert record.endpoint == "/openapi/v2/foo/image-to-image"


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


class TestPollJitter:
    """Verify the ±20% jitter we added to poll_task spreads the retry beats —
    without this, 100 tasks fan out in lockstep and the backend gets a
    thundering herd on every beat."""

    async def test_poll_wait_uses_jittered_backoff(
        self, mock_client: AsyncMock, monkeypatch,
    ) -> None:
        # Pin random.uniform so wait values are deterministic, then confirm
        # the loop actually multiplies by the jitter on every sleep — both
        # in the pre-exponential phase and after.
        mock_client.post.side_effect = [
            {"status": "RUNNING"},
            {"status": "RUNNING"},
            {"status": "RUNNING"},
            {"status": "RUNNING"},
            {"status": "SUCCESS", "results": []},
        ]
        sleeps: list[float] = []

        async def _capture_sleep(d):
            sleeps.append(d)

        monkeypatch.setattr(
            "styleclaw.providers.runninghub.tasks.asyncio.sleep", _capture_sleep,
        )
        from styleclaw.providers.runninghub import tasks as tasks_mod
        # Force the multiplier to its lower bound; without jitter, sleeps[i]
        # would equal `interval` (or post-3 base) exactly. With jitter * 0.8
        # they should be strictly smaller.
        monkeypatch.setattr(tasks_mod.random, "uniform", lambda a, b: 0.8)

        await poll_task(mock_client, "t-jitter", interval=2.0, timeout=60.0)

        # First 3 sleeps: base = 2.0, jittered = 1.6
        assert sleeps[0] == pytest.approx(1.6)
        assert sleeps[1] == pytest.approx(1.6)
        assert sleeps[2] == pytest.approx(1.6)
        # 4th sleep: base = 2.0 * 1.5^1 = 3.0, jittered = 2.4
        assert sleeps[3] == pytest.approx(2.4)
