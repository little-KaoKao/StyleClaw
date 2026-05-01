from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from styleclaw.core.image_utils import verify_ref_image
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
    for img_path in ref_images:
        verify_ref_image(img_path)

    config = ProjectConfig(
        name=name,
        description=description,
        ip_info=ip_info,
    )

    root = project_store.create_project(config)
    refs_dir = root / "refs"

    ref_local_names: list[str] = []
    local_dests: list[Path] = []

    for i, img_path in enumerate(ref_images, 1):
        suffix = img_path.suffix or ".png"
        dest_name = f"ref-{i:03d}{suffix}"
        dest = refs_dir / dest_name
        shutil.copy2(img_path, dest)
        ref_local_names.append(f"refs/{dest_name}")
        local_dests.append(dest)

    results: dict[int, UploadRecord] = {}

    async def _upload(idx: int, dest: Path) -> None:
        results[idx] = await upload_file(client, dest)
        logger.info("Uploaded ref %d/%d: %s", idx + 1, len(local_dests), dest.name)

    async with asyncio.TaskGroup() as tg:
        for idx, dest in enumerate(local_dests):
            tg.create_task(_upload(idx, dest))

    upload_records = [results[i] for i in range(len(local_dests))]
    project_store.save_uploads(name, upload_records)

    updated_config = config.model_copy(update={"ref_images": ref_local_names})
    project_store.save_config(name, updated_config)

    logger.info("Project '%s' initialized with %d reference images.", name, len(ref_images))
    return root
