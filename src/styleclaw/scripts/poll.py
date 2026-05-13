from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from styleclaw.core.models import TaskRecord, TaskStatus
from styleclaw.providers.runninghub.client import RunningHubClient
from styleclaw.providers.runninghub.models import get_model
from styleclaw.providers.runninghub.tasks import poll_and_update, submit_task
from styleclaw.storage import project_store
from styleclaw.storage.image_store import download_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadStats:
    attempted: int = 0
    succeeded: int = 0
    failed_urls: tuple[str, ...] = field(default_factory=tuple)

    @property
    def failed(self) -> int:
        return self.attempted - self.succeeded

    def __add__(self, other: "DownloadStats") -> "DownloadStats":
        return DownloadStats(
            attempted=self.attempted + other.attempted,
            succeeded=self.succeeded + other.succeeded,
            failed_urls=self.failed_urls + other.failed_urls,
        )


async def _download_results(
    results: list[dict[str, Any]], dest_dir: Path,
) -> DownloadStats:
    attempted = 0
    succeeded = 0
    failed: list[str] = []
    for i, result in enumerate(results, 1):
        url = result.get("url", "")
        if not url:
            logger.warning("Result %d has no URL, skipping download.", i)
            continue
        attempted += 1
        dest = dest_dir / f"output-{i:03d}.png"
        try:
            actual = await download_image(url, dest)
            succeeded += 1
            logger.debug("Downloaded %s -> %s", url[:60], actual.name)
        except RuntimeError as exc:
            logger.error("Failed to download result %d from %s: %s", i, url[:80], exc)
            failed.append(url)

    if failed:
        record_path = dest_dir / "failed_downloads.json"
        try:
            record_path.write_text(
                json.dumps({"failed_urls": failed}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not write %s: %s", record_path, exc)

    return DownloadStats(
        attempted=attempted, succeeded=succeeded, failed_urls=tuple(failed),
    )


def _log_download_summary(scope: str, stats: DownloadStats, task_count: int) -> None:
    """Log poll completion. Download failures are surfaced but do not mark
    the task as failed — missing a few images does not invalidate the overall
    result.
    """
    summary = f"{scope} poll complete. {task_count} tasks processed."
    if stats.attempted:
        summary += f" Downloads: {stats.succeeded}/{stats.attempted}"
    if stats.failed:
        logger.warning("%s (%d image downloads failed — see earlier errors).", summary, stats.failed)
    else:
        logger.info(summary)


async def _poll_one_model_select(
    name: str,
    key: str,
    record: TaskRecord,
    client: RunningHubClient,
    pass_num: int,
) -> tuple[str, TaskRecord, DownloadStats]:
    if record.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
        logger.debug("Task %s already terminal (%s), skipping.", record.task_id, record.status)
        return key, record, DownloadStats()

    if not record.task_id:
        logger.warning("Skipping %s: no task_id (submission may have failed).", key)
        return key, record, DownloadStats()

    logger.debug("Polling task %s for %s (pass %d)...", record.task_id, key, pass_num)
    new_record = await poll_and_update(client, record)

    if "/" in key:
        model_id, variant = key.split("/", 1)
        project_store.save_task_record(
            name, model_id, new_record, variant=variant, pass_num=pass_num,
        )
        results_dir = project_store.model_results_dir(
            name, model_id, variant=variant, pass_num=pass_num,
        )
    else:
        project_store.save_task_record(name, key, new_record, pass_num=pass_num)
        results_dir = project_store.model_results_dir(name, key, pass_num=pass_num)

    stats = await _download_results(new_record.results, results_dir)
    return key, new_record, stats


async def poll_model_select(
    name: str,
    client: RunningHubClient,
    pass_num: int = 1,
) -> dict[str, TaskRecord]:
    records = project_store.load_all_task_records(name, pass_num=pass_num)
    if not records:
        raise RuntimeError(
            f"No task records found for project '{name}' pass {pass_num}"
        )

    updated: dict[str, TaskRecord] = {}
    total_stats = DownloadStats()

    async with asyncio.TaskGroup() as tg:
        tasks = {
            key: tg.create_task(
                _poll_one_model_select(name, key, record, client, pass_num)
            )
            for key, record in records.items()
        }

    for key, task in tasks.items():
        _, new_record, stats = task.result()
        updated[key] = new_record
        total_stats += stats

    _log_download_summary(f"Pass {pass_num}", total_stats, len(updated))
    return updated


async def _poll_one_style_refine(
    name: str,
    round_num: int,
    model_id: str,
    record: TaskRecord,
    client: RunningHubClient,
    pass_num: int,
) -> tuple[str, TaskRecord, DownloadStats]:
    if record.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
        logger.debug("Task %s already completed, skipping.", record.task_id)
        return model_id, record, DownloadStats()

    if not record.task_id:
        logger.warning("Skipping model %s (round %d): no task_id.", model_id, round_num)
        return model_id, record, DownloadStats()

    logger.debug("Polling task %s for model %s (round %d)...", record.task_id, model_id, round_num)
    new_record = await poll_and_update(client, record)
    project_store.save_round_task_record(
        name, round_num, model_id, new_record, pass_num=pass_num,
    )

    results_dir = project_store.round_results_dir(
        name, round_num, model_id, pass_num=pass_num,
    )
    stats = await _download_results(new_record.results, results_dir)

    return model_id, new_record, stats


async def poll_style_refine(
    name: str,
    client: RunningHubClient,
    round_num: int,
    pass_num: int = 1,
) -> dict[str, TaskRecord]:
    records = project_store.load_all_round_task_records(
        name, round_num, pass_num=pass_num,
    )
    if not records:
        raise RuntimeError(f"No task records for round {round_num} in project '{name}'")

    updated: dict[str, TaskRecord] = {}
    total_stats = DownloadStats()

    async with asyncio.TaskGroup() as tg:
        tasks = {
            model_id: tg.create_task(
                _poll_one_style_refine(name, round_num, model_id, record, client, pass_num)
            )
            for model_id, record in records.items()
        }

    for model_id, task in tasks.items():
        _, new_record, stats = task.result()
        updated[model_id] = new_record
        total_stats += stats

    _log_download_summary(f"Round {round_num}", total_stats, len(updated))
    return updated


async def _poll_one_batch(
    name: str,
    batch_num: int,
    case_id: str,
    record: TaskRecord,
    client: RunningHubClient,
    phase: str,
) -> tuple[str, TaskRecord, DownloadStats]:
    if record.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
        return case_id, record, DownloadStats()

    if not record.task_id:
        logger.warning("Skipping case %s: no task_id.", case_id)
        return case_id, record, DownloadStats()

    logger.debug("Polling task %s for case %s...", record.task_id, case_id)
    new_record = await poll_and_update(client, record)

    if phase == "i2i":
        project_store.save_i2i_task_record(name, batch_num, case_id, new_record)
        case_dir = project_store.batch_i2i_case_dir(name, batch_num, case_id)
    else:
        project_store.save_batch_task_record(name, batch_num, case_id, new_record)
        case_dir = project_store.batch_t2i_case_dir(name, batch_num, case_id)

    stats = await _download_results(new_record.results, case_dir)

    return case_id, new_record, stats


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
    total_stats = DownloadStats()

    async with asyncio.TaskGroup() as tg:
        tasks = {
            case_id: tg.create_task(
                _poll_one_batch(name, batch_num, case_id, record, client, phase)
            )
            for case_id, record in records.items()
        }

    for case_id, task in tasks.items():
        _, new_record, stats = task.result()
        updated[case_id] = new_record
        total_stats += stats

    _log_download_summary(f"Batch {batch_num} ({phase})", total_stats, len(updated))
    return updated


async def _resubmit_from_record(
    client: RunningHubClient, record: TaskRecord,
) -> TaskRecord | None:
    """Resubmit a failed task using its stored params/model_id. Returns the new
    TaskRecord on success, or None if resubmission could not be attempted."""
    if not record.model_id or not record.params:
        logger.warning(
            "Cannot resubmit task %s: missing model_id or params.", record.task_id,
        )
        return None
    try:
        config = get_model(record.model_id)
    except (KeyError, ValueError) as exc:
        logger.warning("Cannot resubmit task %s: %s", record.task_id, exc)
        return None
    try:
        return await submit_task(
            client, config.t2i_endpoint, record.params, record.model_id,
        )
    except (RuntimeError, httpx.HTTPError) as exc:
        logger.warning("Resubmit failed for %s: %s", record.task_id, exc)
        return None


async def retry_failed_model_select(
    name: str,
    client: RunningHubClient,
    pass_num: int = 1,
) -> dict[str, TaskRecord]:
    """Resubmit FAILED model-select tasks once, then poll the new submissions.
    Returns the final records dict (mixing previously-succeeded, newly-resubmitted,
    and any still-failed entries)."""
    records = project_store.load_all_task_records(name, pass_num=pass_num)
    failed_keys = [k for k, r in records.items() if r.status == TaskStatus.FAILED]
    if not failed_keys:
        return records

    logger.info("Retrying %d failed model-select task(s) once...", len(failed_keys))
    for key in failed_keys:
        new_record = await _resubmit_from_record(client, records[key])
        if new_record is None:
            continue
        if "/" in key:
            model_id, variant = key.split("/", 1)
            project_store.save_task_record(
                name, model_id, new_record, variant=variant, pass_num=pass_num,
            )
        else:
            project_store.save_task_record(name, key, new_record, pass_num=pass_num)

    return await poll_model_select(name, client, pass_num=pass_num)


async def retry_failed_style_refine(
    name: str,
    client: RunningHubClient,
    round_num: int,
    pass_num: int = 1,
) -> dict[str, TaskRecord]:
    """Resubmit FAILED style-refine tasks once, then poll. Same contract as
    retry_failed_model_select."""
    records = project_store.load_all_round_task_records(
        name, round_num, pass_num=pass_num,
    )
    failed_keys = [k for k, r in records.items() if r.status == TaskStatus.FAILED]
    if not failed_keys:
        return records

    logger.info(
        "Retrying %d failed style-refine task(s) (round %d) once...",
        len(failed_keys), round_num,
    )
    for model_id in failed_keys:
        new_record = await _resubmit_from_record(client, records[model_id])
        if new_record is None:
            continue
        project_store.save_round_task_record(
            name, round_num, model_id, new_record, pass_num=pass_num,
        )

    return await poll_style_refine(name, client, round_num, pass_num=pass_num)
