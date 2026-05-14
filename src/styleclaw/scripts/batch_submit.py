from __future__ import annotations

import asyncio
import logging

from tqdm import tqdm

from styleclaw.core.checkpoint import Checkpoint
from styleclaw.core.models import BatchCase, BatchConfig, TaskRecord, TaskStatus
from styleclaw.core.prompt_builder import build_params
from styleclaw.providers.runninghub.client import RunningHubClient
from styleclaw.providers.runninghub.models import build_i2i_params, get_model
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
    if not config.cases:
        raise ValueError(
            f"No cases found in batch {batch_num}. Run 'design-cases' first."
        )
    model_config = get_model(model_id)

    checkpoint = Checkpoint(project_store.project_dir(name), f"batch-t2i-{batch_num:03d}")
    submitted_ids = set(checkpoint.get("submitted", []))

    pending = [
        c for c in config.cases
        if c.status == "pending" and c.id not in submitted_ids
    ]
    if submitted_ids:
        logger.info(
            "Resuming batch-t2i %d: skipping %d cases already in checkpoint.",
            batch_num, len(submitted_ids),
        )

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
        checkpoint.add_to_set("submitted", case.id)
        return record

    async with asyncio.TaskGroup() as tg:
        progress = tqdm(total=len(pending), desc=f"Submitting batch-t2i {batch_num}", unit="case") if pending else None
        for case in pending:
            t = tg.create_task(_submit_one(case))
            if progress is not None:
                t.add_done_callback(lambda _t, p=progress: p.update(1))
            tasks[case.id] = t
    if pending and progress is not None:
        progress.close()

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

    if all(c.status != "pending" for c in updated_cases):
        checkpoint.clear()

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

    existing = project_store.load_all_i2i_task_records(name, batch_num)
    submitted_case_ids = {
        cid for cid, rec in existing.items() if rec.status != TaskStatus.FAILED
    }

    tasks: dict[str, asyncio.Task] = {}

    async def _submit_one(idx: int, image_url: str) -> TaskRecord:
        case_id = f"i2i-{idx:03d}"
        params = build_i2i_params(model_config, trigger_phrase, image_url)

        record = await submit_task(client, model_config.i2i_endpoint, params, model_id)
        project_store.save_i2i_task_record(name, batch_num, case_id, record)
        return record

    async with asyncio.TaskGroup() as tg:
        to_submit = [
            (i, upload) for i, upload in enumerate(uploads, 1)
            if f"i2i-{i:03d}" not in submitted_case_ids
        ]
        progress = tqdm(total=len(to_submit), desc=f"Submitting batch-i2i {batch_num}", unit="case") if to_submit else None
        for i, upload in enumerate(uploads, 1):
            case_id = f"i2i-{i:03d}"
            if case_id in submitted_case_ids:
                logger.debug("Skipping already submitted case %s.", case_id)
                continue
            t = tg.create_task(_submit_one(i, upload.url))
            if progress is not None:
                t.add_done_callback(lambda _t, p=progress: p.update(1))
            tasks[case_id] = t
    if progress is not None:
        progress.close()

    records: dict[str, TaskRecord] = {}
    new_cases: dict[str, BatchCase] = {}
    for case_id, task in tasks.items():
        records[case_id] = task.result()
        new_cases[case_id] = BatchCase(
            id=case_id,
            category="i2i",
            description="Image-to-image from uploaded reference",
            status="submitted",
        )

    try:
        prev_config = project_store.load_i2i_batch_config(name, batch_num)
        existing_cases = {c.id: c for c in prev_config.cases}
    except FileNotFoundError:
        existing_cases = {}

    merged_cases = {**existing_cases, **new_cases}
    i2i_config = BatchConfig(
        batch=batch_num,
        trigger_phrase=trigger_phrase,
        cases=list(merged_cases.values()),
    )
    project_store.save_i2i_batch_config(name, batch_num, i2i_config)

    logger.info("Submitted %d batch-i2i tasks for batch %d.", len(records), batch_num)
    return records
