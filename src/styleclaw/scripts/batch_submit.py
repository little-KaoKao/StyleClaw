from __future__ import annotations

import asyncio
import logging
from typing import Any

from styleclaw.core.models import BatchCase, BatchConfig, TaskRecord
from styleclaw.core.prompt_builder import build_params
from styleclaw.providers.runninghub.client import RunningHubClient
from styleclaw.providers.runninghub.models import get_model
from styleclaw.providers.runninghub.tasks import submit_task
from styleclaw.storage import project_store

logger = logging.getLogger(__name__)


async def batch_submit_t2i(
    name: str,
    client: RunningHubClient,
    batch_num: int,
    model_id: str,
    sref_url: str = "",
) -> dict[str, TaskRecord]:
    config = project_store.load_batch_config(name, batch_num)
    model_config = get_model(model_id)
    pending = [c for c in config.cases if c.status == "pending"]

    tasks: dict[str, asyncio.Task] = {}

    async def _submit_one(case: BatchCase) -> TaskRecord:
        params = build_params(
            model_id=model_id,
            trigger_phrase=config.trigger_phrase,
            character_desc=case.description,
            aspect_ratio=case.aspect_ratio,
            sref_url=sref_url,
        )
        record = await submit_task(client, model_config.t2i_endpoint, params, model_id)
        project_store.save_batch_task_record(name, batch_num, case.id, record)
        return record

    async with asyncio.TaskGroup() as tg:
        for case in pending:
            tasks[case.id] = tg.create_task(_submit_one(case))

    records: dict[str, TaskRecord] = {}
    updated_cases: list[BatchCase] = []
    for case in config.cases:
        if case.id in tasks:
            records[case.id] = tasks[case.id].result()
            updated_cases.append(case.model_copy(update={"status": "submitted"}))
        else:
            updated_cases.append(case)

    updated_config = config.model_copy(update={"cases": updated_cases})
    project_store.save_batch_config(name, batch_num, updated_config)

    logger.info("Submitted %d batch-t2i tasks for batch %d.", len(records), batch_num)
    return records


async def batch_submit_i2i(
    name: str,
    client: RunningHubClient,
    batch_num: int,
    model_id: str,
    trigger_phrase: str,
) -> dict[str, TaskRecord]:
    uploads = project_store.load_i2i_uploads(name, batch_num)
    model_config = get_model(model_id)
    tasks: dict[str, asyncio.Task] = {}

    async def _submit_one(idx: int, image_url: str) -> TaskRecord:
        case_id = f"i2i-{idx:03d}"
        params: dict[str, Any] = {
            "prompt": trigger_phrase,
            "imageUrls": [image_url],
        }
        if model_config.model_id in ("mj-v7", "niji7"):
            params = {"prompt": trigger_phrase, "imageUrl": image_url, "iw": 0.5}

        record = await submit_task(client, model_config.i2i_endpoint, params, model_id)
        project_store.save_i2i_task_record(name, batch_num, case_id, record)
        return record

    async with asyncio.TaskGroup() as tg:
        for i, upload in enumerate(uploads, 1):
            tasks[f"i2i-{i:03d}"] = tg.create_task(_submit_one(i, upload.url))

    records: dict[str, TaskRecord] = {}
    cases: list[BatchCase] = []
    for case_id, task in tasks.items():
        records[case_id] = task.result()
        cases.append(BatchCase(
            id=case_id,
            category="i2i",
            description=f"Image-to-image from uploaded reference",
            status="submitted",
        ))

    i2i_config = BatchConfig(
        batch=batch_num,
        trigger_phrase=trigger_phrase,
        cases=cases,
    )
    project_store.save_i2i_batch_config(name, batch_num, i2i_config)

    logger.info("Submitted %d batch-i2i tasks for batch %d.", len(records), batch_num)
    return records
