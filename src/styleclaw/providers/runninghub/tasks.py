from __future__ import annotations

import asyncio
import logging
from typing import Any

from styleclaw.core.models import TaskRecord, TaskStatus
from styleclaw.providers.runninghub.client import RunningHubClient

logger = logging.getLogger(__name__)

QUERY_PATH = "/openapi/v2/query"
POLL_INTERVAL = 3
TASK_TIMEOUT = 300


SUBMIT_RETRIES = 3
SUBMIT_RETRY_DELAY = 2


async def submit_task(
    client: RunningHubClient,
    endpoint: str,
    params: dict[str, Any],
    model_id: str,
) -> TaskRecord:
    resp: dict[str, Any] = {}
    for attempt in range(SUBMIT_RETRIES):
        resp = await client.post(endpoint, params)
        task_id = resp.get("taskId", "")
        error_code = resp.get("errorCode", "")
        if task_id:
            break
        logger.warning(
            "Submit to %s returned empty taskId (attempt %d/%d, errorCode=%s, errorMessage=%s). Retrying...",
            endpoint, attempt + 1, SUBMIT_RETRIES, error_code, resp.get("errorMessage", ""),
        )
        if attempt < SUBMIT_RETRIES - 1:
            await asyncio.sleep(SUBMIT_RETRY_DELAY * (attempt + 1))

    task_id = resp.get("taskId", "")
    status = resp.get("status", "QUEUED")
    results = resp.get("results") or []

    record = TaskRecord(
        task_id=task_id,
        model_id=model_id,
        status=status if task_id else TaskStatus.FAILED,
        prompt=params.get("prompt", ""),
        params=params,
        results=results,
        error_message=resp.get("errorMessage", "") if not task_id else "",
    )
    logger.info("Submitted task %s to %s (status=%s)", task_id or "(empty)", endpoint, record.status)
    return record


async def query_task(client: RunningHubClient, task_id: str) -> dict[str, Any]:
    return await client.post(QUERY_PATH, {"taskId": task_id})


async def poll_task(
    client: RunningHubClient,
    task_id: str,
    interval: float = POLL_INTERVAL,
    timeout: float = TASK_TIMEOUT,
) -> dict[str, Any]:
    elapsed = 0.0
    while elapsed < timeout:
        result = await query_task(client, task_id)
        status = result.get("status", "")
        if status == "SUCCESS":
            return result
        if status == "FAILED":
            raise RuntimeError(
                f"Task {task_id} failed: {result.get('errorMessage', 'unknown error')}"
            )
        logger.debug("Task %s status=%s, waiting %ss...", task_id, status, interval)
        await asyncio.sleep(interval)
        elapsed += interval

    raise TimeoutError(f"Task {task_id} timed out after {timeout}s")


async def poll_and_update(
    client: RunningHubClient, record: TaskRecord
) -> TaskRecord:
    if record.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
        return record

    from datetime import datetime, timezone

    try:
        result = await poll_task(client, record.task_id)
    except (RuntimeError, TimeoutError) as exc:
        logger.warning("Task %s failed: %s", record.task_id, exc)
        return record.model_copy(update={
            "status": TaskStatus.FAILED,
            "error_message": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

    return record.model_copy(update={
        "status": TaskStatus.SUCCESS,
        "results": result.get("results", []),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
