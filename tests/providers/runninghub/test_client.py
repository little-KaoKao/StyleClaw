from __future__ import annotations

import pytest
import respx
import httpx

from styleclaw.providers.runninghub.client import RunningHubClient, _get_semaphore
import styleclaw.providers.runninghub.client as client_mod


@pytest.fixture(autouse=True)
def reset_semaphore():
    client_mod._semaphore_map.clear()
    yield
    client_mod._semaphore_map.clear()


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


class TestSemaphore:
    async def test_get_semaphore_creates_once(self) -> None:
        s1 = _get_semaphore()
        s2 = _get_semaphore()
        assert s1 is s2
