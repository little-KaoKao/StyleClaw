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

    progress = (
        tqdm(total=len(pending), desc=f"Submitting batch-t2i {batch_num}", unit="case")
        if pending else None
    )

    async def _wrapped(case: BatchCase) -> TaskRecord:
        try:
            return await _submit_one(case)
        finally:
            if progress is not None:
                progress.update(1)

    # Use gather(return_exceptions=True) so one failed submission doesn't
    # cancel the other 99. Each case's success/failure is recorded
    # independently — checkpoint updates only on success, so retrying picks
    # the failures back up.
    raw_results = await asyncio.gather(
        *[_wrapped(c) for c in pending],
        return_exceptions=True,
    )
    if pending and progress is not None:
        progress.close()

    records: dict[str, TaskRecord] = {}
    failures: list[tuple[str, str]] = []
    for case, outcome in zip(pending, raw_results):
        if isinstance(outcome, BaseException):
            failures.append((case.id, f"{type(outcome).__name__}: {outcome}"))
            logger.error("Submit failed for case %s: %s", case.id, outcome)
            continue
        records[case.id] = outcome

    updated_cases: list[BatchCase] = []
    for case in config.cases:
        if case.id in records:
            updated_cases.append(case.model_copy(update={"status": "submitted"}))
        else:
            updated_cases.append(case)

    updated_config = config.model_copy(update={"cases": updated_cases})
    project_store.save_batch_config(name, batch_num, updated_config)

    if all(c.status != "pending" for c in updated_cases):
        checkpoint.clear()

    if failures:
        logger.warning(
            "batch-t2i %d: submitted %d/%d cases; %d failed (rerun to retry): %s",
            batch_num, len(records), len(pending), len(failures),
            ", ".join(f"{cid} ({err[:60]})" for cid, err in failures[:5]),
        )
    else:
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

    async def _submit_one(idx: int, image_url: str) -> TaskRecord:
        case_id = f"i2i-{idx:03d}"
        params = build_i2i_params(model_config, trigger_phrase, image_url)

        record = await submit_task(client, model_config.i2i_endpoint, params, model_id)
        project_store.save_i2i_task_record(name, batch_num, case_id, record)
        return record

    to_submit = [
        (i, upload) for i, upload in enumerate(uploads, 1)
        if f"i2i-{i:03d}" not in submitted_case_ids
    ]
    progress = (
        tqdm(total=len(to_submit), desc=f"Submitting batch-i2i {batch_num}", unit="case")
        if to_submit else None
    )

    async def _wrapped(idx: int, url: str) -> TaskRecord:
        try:
            return await _submit_one(idx, url)
        finally:
            if progress is not None:
                progress.update(1)

    raw_results = await asyncio.gather(
        *[_wrapped(i, upload.url) for i, upload in to_submit],
        return_exceptions=True,
    )
    if progress is not None:
        progress.close()

    records: dict[str, TaskRecord] = {}
    failures: list[tuple[str, str]] = []
    for (i, _upload), outcome in zip(to_submit, raw_results):
        case_id = f"i2i-{i:03d}"
        if isinstance(outcome, BaseException):
            failures.append((case_id, f"{type(outcome).__name__}: {outcome}"))
            logger.error("Submit failed for case %s: %s", case_id, outcome)
            continue
        records[case_id] = outcome

    new_cases: dict[str, BatchCase] = {
        case_id: BatchCase(
            id=case_id,
            category="i2i",
            description="Image-to-image from uploaded reference",
            status="submitted",
        )
        for case_id in records
    }

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

    if failures:
        logger.warning(
            "batch-i2i %d: submitted %d/%d cases; %d failed (rerun to retry): %s",
            batch_num, len(records), len(to_submit), len(failures),
            ", ".join(f"{cid} ({err[:60]})" for cid, err in failures[:5]),
        )
    else:
        logger.info("Submitted %d batch-i2i tasks for batch %d.", len(records), batch_num)
    return records
