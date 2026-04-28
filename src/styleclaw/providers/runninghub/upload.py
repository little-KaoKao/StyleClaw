from __future__ import annotations

import logging
from pathlib import Path

from styleclaw.core.models import UploadRecord
from styleclaw.providers.runninghub.client import RunningHubClient

logger = logging.getLogger(__name__)

UPLOAD_PATH = "/openapi/v2/media/upload/binary"


async def upload_file(client: RunningHubClient, file_path: Path) -> UploadRecord:
    resp = await client.upload(UPLOAD_PATH, str(file_path))

    if resp.get("code") != 0:
        raise RuntimeError(f"Upload failed: {resp.get('message', resp)}")

    data = resp.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"Upload response missing 'data' field: {resp}")
    url = data.get("download_url")
    file_name = data.get("fileName")
    if not url or not file_name:
        raise RuntimeError(f"Upload response missing required fields in 'data': {data}")
    record = UploadRecord(
        local_path=str(file_path),
        url=url,
        file_name=file_name,
    )
    logger.info("Uploaded %s -> %s", file_path.name, record.url)
    return record
