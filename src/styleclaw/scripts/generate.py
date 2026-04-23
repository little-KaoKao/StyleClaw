from __future__ import annotations

import asyncio
import logging
from typing import Any

from styleclaw.core.models import TaskRecord
from styleclaw.core.prompt_builder import build_params
from styleclaw.providers.runninghub.client import RunningHubClient
from styleclaw.providers.runninghub.models import MODEL_REGISTRY, get_model
from styleclaw.providers.runninghub.tasks import submit_task
from styleclaw.storage import project_store

logger = logging.getLogger(__name__)

IMAGES_PER_MODEL = 3


async def generate_model_select(
    name: str,
    client: RunningHubClient,
    trigger_phrase: str,
    sref_url: str = "",
    models: list[str] | None = None,
) -> dict[str, TaskRecord]:
    model_ids = models or list(MODEL_REGISTRY.keys())
    tasks: dict[str, asyncio.Task] = {}

    async def _submit_one(model_id: str) -> TaskRecord:
        config = get_model(model_id)
        params = build_params(
            model_id=model_id,
            trigger_phrase=trigger_phrase,
            aspect_ratio="9:16",
            sref_url=sref_url,
        )
        record = await submit_task(client, config.t2i_endpoint, params, model_id)
        project_store.save_task_record(name, model_id, record)
        return record

    async with asyncio.TaskGroup() as tg:
        for mid in model_ids:
            tasks[mid] = tg.create_task(_submit_one(mid))

    records: dict[str, TaskRecord] = {}
    for mid, task in tasks.items():
        records[mid] = task.result()

    logger.info("Submitted %d generation tasks for model-select.", len(records))
    return records


IMAGES_PER_MODEL_REFINE = 3


async def generate_style_refine(
    name: str,
    client: RunningHubClient,
    round_num: int,
    trigger_phrase: str,
    sref_url: str = "",
    extra_model_params: dict[str, dict[str, Any]] | None = None,
) -> dict[str, TaskRecord]:
    state = project_store.load_state(name)
    model_ids = state.selected_models
    tasks: dict[str, asyncio.Task] = {}

    async def _submit_one(model_id: str) -> TaskRecord:
        config = get_model(model_id)
        extra = (extra_model_params or {}).get(model_id, {})
        if not config.supports_sref:
            extra.setdefault("maxImages", IMAGES_PER_MODEL_REFINE)
        params = build_params(
            model_id=model_id,
            trigger_phrase=trigger_phrase,
            aspect_ratio="9:16",
            sref_url=sref_url,
            extra_params=extra,
        )
        record = await submit_task(client, config.t2i_endpoint, params, model_id)
        project_store.save_round_task_record(name, round_num, model_id, record)
        return record

    async with asyncio.TaskGroup() as tg:
        for mid in model_ids:
            tasks[mid] = tg.create_task(_submit_one(mid))

    records: dict[str, TaskRecord] = {}
    for mid, task in tasks.items():
        records[mid] = task.result()

    logger.info("Submitted %d generation tasks for style-refine round %d.", len(records), round_num)
    return records
