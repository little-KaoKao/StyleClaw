from __future__ import annotations

import pytest
import respx
import httpx

from styleclaw.providers.runninghub.client import RunningHubClient


@pytest.fixture
def rh_client() -> RunningHubClient:
    return RunningHubClient(api_key="test-key")


class TestRunningHubClient:
    @respx.mock
    async def test_post_success(self, rh_client: RunningHubClient) -> None:
        respx.post("https://www.runninghub.cn/api/test").respond(
            json={"taskId": "abc123"}
        )
        result = await rh_client.post("/api/test", {"prompt": "hello"})
        assert result["taskId"] == "abc123"

    @respx.mock
    async def test_post_retries_on_failure(self, rh_client: RunningHubClient) -> None:
        route = respx.post("https://www.runninghub.cn/api/test")
        route.side_effect = [
            httpx.Response(500, text="error"),
            httpx.Response(500, text="error"),
            httpx.Response(200, json={"ok": True}),
        ]
        result = await rh_client.post("/api/test", {})
        assert result["ok"] is True

    @respx.mock
    async def test_post_raises_after_max_retries(self, rh_client: RunningHubClient) -> None:
        respx.post("https://www.runninghub.cn/api/test").respond(status_code=500)
        with pytest.raises(RuntimeError, match="failed after 3 retries"):
            await rh_client.post("/api/test", {})

    @respx.mock
    async def test_upload_success(self, rh_client: RunningHubClient, tmp_path) -> None:
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake image data")

        respx.post("https://www.runninghub.cn/api/upload").respond(
            json={"code": 0, "data": {"download_url": "https://example.com/f.png", "fileName": "f.png"}}
        )
        result = await rh_client.upload("/api/upload", str(test_file))
        assert result["code"] == 0

    @respx.mock
    async def test_upload_retries_on_failure(self, rh_client: RunningHubClient, tmp_path) -> None:
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake image data")

        route = respx.post("https://www.runninghub.cn/api/upload")
        route.side_effect = [
            httpx.Response(500, text="error"),
            httpx.Response(200, json={"code": 0}),
        ]
        result = await rh_client.upload("/api/upload", str(test_file))
        assert result["code"] == 0

    @respx.mock
    async def test_upload_raises_after_max_retries(self, rh_client: RunningHubClient, tmp_path) -> None:
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake image data")

        respx.post("https://www.runninghub.cn/api/upload").respond(status_code=500)
        with pytest.raises(RuntimeError, match="failed after 3 retries"):
            await rh_client.upload("/api/upload", str(test_file))

    async def test_close(self, rh_client: RunningHubClient) -> None:
        await rh_client.close()


class TestAsyncContextManager:
    async def test_aenter_returns_self(self) -> None:
        client = RunningHubClient(api_key="test-key")
        async with client as ctx:
            assert ctx is client

    async def test_aexit_closes_http_client(self) -> None:
        client = RunningHubClient(api_key="test-key")
        async with client:
            pass
        assert client._client.is_closed

    @respx.mock
    async def test_usable_inside_context(self) -> None:
        respx.post("https://www.runninghub.cn/api/test").respond(
            json={"ok": True}
        )
        async with RunningHubClient(api_key="test-key") as client:
            result = await client.post("/api/test", {})
            assert result["ok"] is True


class TestSemaphoreIsInstanceAttribute:
    async def test_each_client_has_own_semaphore(self) -> None:
        c1 = RunningHubClient(api_key="k1")
        c2 = RunningHubClient(api_key="k2")
        assert c1._semaphore is not c2._semaphore

    async def test_semaphore_has_correct_limit(self) -> None:
        from styleclaw.providers.runninghub.client import CONCURRENCY_LIMIT
        client = RunningHubClient(api_key="test-key")
        assert client._semaphore._value == CONCURRENCY_LIMIT

    async def test_no_module_level_semaphore_map(self) -> None:
        import styleclaw.providers.runninghub.client as mod
        assert not hasattr(mod, "_semaphore_map")
