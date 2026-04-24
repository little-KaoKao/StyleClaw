from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from styleclaw.providers.runninghub.upload import upload_file


@pytest.fixture
def mock_client() -> AsyncMock:
    return AsyncMock()


class TestUploadFile:
    async def test_success(self, mock_client: AsyncMock, tmp_path) -> None:
        test_file = tmp_path / "ref.png"
        test_file.write_bytes(b"image data")

        mock_client.upload.return_value = {
            "code": 0,
            "data": {"download_url": "https://cdn.example.com/ref.png", "fileName": "ref.png"},
        }

        record = await upload_file(mock_client, test_file)
        assert record.url == "https://cdn.example.com/ref.png"
        assert record.file_name == "ref.png"
        assert str(test_file) in record.local_path

    async def test_raises_on_error_code(self, mock_client: AsyncMock, tmp_path) -> None:
        test_file = tmp_path / "ref.png"
        test_file.write_bytes(b"image data")

        mock_client.upload.return_value = {"code": 1, "message": "File too large"}
        with pytest.raises(RuntimeError, match="File too large"):
            await upload_file(mock_client, test_file)
