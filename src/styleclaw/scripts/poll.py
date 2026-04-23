from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from styleclaw.core.models import TaskRecord
from styleclaw.providers.runninghub.client import RunningHubClient
from styleclaw.providers.runninghub.tasks import poll_and_update
from styleclaw.storage import project_store
from styleclaw.storage.image_store import download_image

logger = logging.getLogger(__name__)


async def poll_model_select(
    name: str,
    client: RunningHubClient,
) -> dict[str, TaskRecord]:
    records = project_store.load_all_task_records(name)
    if not records:
        raise RuntimeError(f"No task records found for project '{name}'")

    updated: dict[str, TaskRecord] = {}

    for model_id, record in records.items():
        if record.status == "SUCCESS":
            logger.info("Task %s already completed, skipping.", record.task_id)
            updated[model_id] = record
            continue

        if not record.task_id:
            logger.warning("Skipping model %s: no task_id (submission may have failed).", model_id)
            updated[model_id] = record
            continue

        logger.info("Polling task %s for model %s...", record.task_id, model_id)
        new_record = await poll_and_update(client, record)
        project_store.save_task_record(name, model_id, new_record)

        results_dir = project_store.model_results_dir(name, model_id)
        for i, result in enumerate(new_record.results, 1):
            url = result.get("url", "")
            if url:
                dest = results_dir / f"output-{i:03d}.png"
                await download_image(url, dest)
                logger.info("Downloaded %s -> %s", url[:60], dest.name)

        updated[model_id] = new_record

    logger.info("Poll complete. %d models processed.", len(updated))
    return updated


async def poll_style_refine(
    name: str,
    client: RunningHubClient,
    round_num: int,
) -> dict[str, TaskRecord]:
    records = project_store.load_all_round_task_records(name, round_num)
    if not records:
        raise RuntimeError(f"No task records for round {round_num} in project '{name}'")

    updated: dict[str, TaskRecord] = {}

    for model_id, record in records.items():
        if record.status == "SUCCESS":
            logger.info("Task %s already completed, skipping.", record.task_id)
            updated[model_id] = record
            continue

        logger.info("Polling task %s for model %s (round %d)...", record.task_id, model_id, round_num)
        new_record = await poll_and_update(client, record)
        project_store.save_round_task_record(name, round_num, model_id, new_record)

        results_dir = project_store.round_results_dir(name, round_num, model_id)
        for i, result in enumerate(new_record.results, 1):
            url = result.get("url", "")
            if url:
                dest = results_dir / f"output-{i:03d}.png"
                await download_image(url, dest)
                logger.info("Downloaded %s -> %s", url[:60], dest.name)

        updated[model_id] = new_record

    logger.info("Round %d poll complete. %d models processed.", round_num, len(updated))
    return updated


async def poll_batch(
    name: str,
    client: RunningHubClient,
    batch_num: int,
    phase: str = "t2i",
) -> dict[str, TaskRecord]:
    if phase == "i2i":
        records = project_store.load_all_i2i_task_records(name, batch_num)
    else:
        records = project_store.load_all_batch_task_records(name, batch_num)

    if not records:
        raise RuntimeError(f"No task records for batch {batch_num} in project '{name}'")

    updated: dict[str, TaskRecord] = {}

    for case_id, record in records.items():
        if record.status in ("SUCCESS", "FAILED"):
            updated[case_id] = record
            continue

        if not record.task_id:
            logger.warning("Skipping case %s: no task_id.", case_id)
            updated[case_id] = record
            continue

        logger.info("Polling task %s for case %s...", record.task_id, case_id)
        new_record = await poll_and_update(client, record)

        if phase == "i2i":
            project_store.save_i2i_task_record(name, batch_num, case_id, new_record)
            case_dir = project_store.batch_i2i_case_dir(name, batch_num, case_id)
        else:
            project_store.save_batch_task_record(name, batch_num, case_id, new_record)
            case_dir = project_store.batch_t2i_case_dir(name, batch_num, case_id)

        for i, result in enumerate(new_record.results, 1):
            url = result.get("url", "")
            if url:
                dest = case_dir / f"output-{i:03d}.png"
                await download_image(url, dest)
                logger.info("Downloaded %s -> %s", url[:60], dest.name)

        updated[case_id] = new_record

    logger.info("Batch %d poll complete. %d cases processed.", batch_num, len(updated))
    return updated
