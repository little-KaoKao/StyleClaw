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

    data = resp["data"]
    record = UploadRecord(
        local_path=str(file_path),
        url=data["download_url"],
        file_name=data["fileName"],
    )
    logger.info("Uploaded %s -> %s", file_path.name, record.url)
    return record
