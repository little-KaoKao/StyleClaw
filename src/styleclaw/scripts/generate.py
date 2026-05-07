from __future__ import annotations

import asyncio
import logging
from typing import Any

from styleclaw.core.models import TaskRecord, TaskStatus
from styleclaw.core.prompt_builder import build_params
from styleclaw.providers.runninghub.client import RunningHubClient
from styleclaw.providers.runninghub.models import MODEL_REGISTRY, SrefMode, get_model
from styleclaw.providers.runninghub.tasks import submit_task
from styleclaw.storage import project_store

logger = logging.getLogger(__name__)


TEST_SUBJECTS = {
    "male": "a young man with short hair wearing casual streetwear",
    "female": "a young woman with long hair wearing casual streetwear",
}

VARIANT_PROMPT_ONLY = "prompt-only"
VARIANT_PROMPT_SREF = "prompt-sref"


async def generate_model_select(
    name: str,
    client: RunningHubClient,
    trigger_phrase: str,
    sref_url: str = "",
    models: list[str] | None = None,
    pass_num: int = 1,
) -> dict[str, TaskRecord]:
    model_ids = models or list(MODEL_REGISTRY.keys())

    existing = project_store.load_all_task_records(name, pass_num=pass_num)

    to_submit: list[tuple[str, str, str]] = []  # (model_id, variant, gender)
    skipped: dict[str, TaskRecord] = {}

    for mid in model_ids:
        for variant in (VARIANT_PROMPT_ONLY, VARIANT_PROMPT_SREF):
            for gender in TEST_SUBJECTS:
                key = f"{mid}/{variant}-{gender}"
                prev = existing.get(key)
                if prev and prev.status != TaskStatus.FAILED:
                    logger.info("Skipping %s: already has %s record.", key, prev.status)
                    skipped[key] = prev
                else:
                    to_submit.append((mid, variant, gender))

    tasks: dict[str, asyncio.Task[TaskRecord]] = {}

    async def _submit_one(model_id: str, variant: str, gender: str) -> TaskRecord:
        config = get_model(model_id)
        use_sref = sref_url if variant == VARIANT_PROMPT_SREF else ""
        params = build_params(
            model_id=model_id,
            trigger_phrase=trigger_phrase,
            character_desc=TEST_SUBJECTS[gender],
            aspect_ratio="9:16",
            sref_url=use_sref,
        )
        record = await submit_task(client, config.t2i_endpoint, params, model_id)
        project_store.save_task_record(
            name, model_id, record, variant=f"{variant}-{gender}", pass_num=pass_num,
        )
        return record

    async with asyncio.TaskGroup() as tg:
        for mid, variant, gender in to_submit:
            key = f"{mid}/{variant}-{gender}"
            tasks[key] = tg.create_task(_submit_one(mid, variant, gender))

    records: dict[str, TaskRecord] = {**skipped}
    for key, task in tasks.items():
        records[key] = task.result()

    logger.info(
        "Submitted %d generation tasks for model-select pass %d (%d skipped).",
        len(tasks), pass_num, len(skipped),
    )
    return records


IMAGES_PER_MODEL_REFINE = 3


async def generate_style_refine(
    name: str,
    client: RunningHubClient,
    round_num: int,
    trigger_phrase: str,
    sref_url: str = "",
    extra_model_params: dict[str, dict[str, Any]] | None = None,
    pass_num: int = 1,
) -> dict[str, TaskRecord]:
    state = project_store.load_state(name)
    model_ids = state.selected_models

    existing = project_store.load_all_round_task_records(
        name, round_num, pass_num=pass_num,
    )

    to_submit: list[str] = []
    skipped: dict[str, TaskRecord] = {}
    for mid in model_ids:
        prev = existing.get(mid)
        if prev and prev.status != TaskStatus.FAILED:
            logger.info("Skipping model %s round %d: already has %s record.", mid, round_num, prev.status)
            skipped[mid] = prev
        else:
            to_submit.append(mid)

    tasks: dict[str, asyncio.Task[TaskRecord]] = {}

    async def _submit_one(model_id: str) -> TaskRecord:
        config = get_model(model_id)
        extra = dict((extra_model_params or {}).get(model_id, {}))
        if config.sref_mode != SrefMode.PARAM:
            extra.setdefault("maxImages", IMAGES_PER_MODEL_REFINE)
        params = build_params(
            model_id=model_id,
            trigger_phrase=trigger_phrase,
            aspect_ratio="9:16",
            sref_url=sref_url,
            extra_params=extra,
        )
        record = await submit_task(client, config.t2i_endpoint, params, model_id)
        project_store.save_round_task_record(
            name, round_num, model_id, record, pass_num=pass_num,
        )
        return record

    async with asyncio.TaskGroup() as tg:
        for mid in to_submit:
            tasks[mid] = tg.create_task(_submit_one(mid))

    records: dict[str, TaskRecord] = {**skipped}
    for mid, task in tasks.items():
        records[mid] = task.result()

    logger.info(
        "Submitted %d generation tasks for style-refine round %d (%d skipped).",
        len(tasks), round_num, len(skipped),
    )
    return records
