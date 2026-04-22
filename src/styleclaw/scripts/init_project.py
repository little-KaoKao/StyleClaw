from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from styleclaw.core.models import ProjectConfig, UploadRecord
from styleclaw.providers.runninghub.client import RunningHubClient
from styleclaw.providers.runninghub.upload import upload_file
from styleclaw.storage import project_store

logger = logging.getLogger(__name__)


async def init_project(
    name: str,
    ref_images: list[Path],
    ip_info: str,
    description: str,
    client: RunningHubClient,
) -> Path:
    config = ProjectConfig(
        name=name,
        description=description,
        ip_info=ip_info,
    )

    root = project_store.create_project(config)
    refs_dir = root / "refs"

    upload_records: list[UploadRecord] = []
    ref_local_names: list[str] = []

    for i, img_path in enumerate(ref_images, 1):
        suffix = img_path.suffix or ".png"
        dest_name = f"ref-{i:03d}{suffix}"
        dest = refs_dir / dest_name
        shutil.copy2(img_path, dest)
        ref_local_names.append(f"refs/{dest_name}")

        record = await upload_file(client, dest)
        upload_records.append(record)
        logger.info("Uploaded ref %d/%d: %s", i, len(ref_images), dest_name)

    project_store.save_uploads(name, upload_records)

    updated_config = config.model_copy(update={"ref_images": ref_local_names})
    project_store._write_json(root / "config.json", updated_config.model_dump())

    logger.info("Project '%s' initialized with %d reference images.", name, len(ref_images))
    return root
