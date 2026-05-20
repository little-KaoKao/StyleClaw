from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from styleclaw.core.case_generator import CASES_PER_CATEGORY, CATEGORIES, generate_case_skeleton
from styleclaw.core.config import DESIGN_CASES_SHARDS
from styleclaw.core.models import BatchCase, BatchConfig
from styleclaw.core.text_utils import clean_json, recover_truncated_json, sanitize_braces
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "design_cases_shard.md"
)


async def design_cases(
    llm: LLMProvider,
    ip_info: str,
    trigger_phrase: str,
    batch_num: int,
    feedback: str = "",
) -> BatchConfig:
    shards = DESIGN_CASES_SHARDS
    cat_ids = [c["id"] for c in CATEGORIES]
    cats_per_shard = len(CATEGORIES) // shards
    partitions: list[list[str]] = [
        cat_ids[i : i + cats_per_shard]
        for i in range(0, len(cat_ids), cats_per_shard)
    ]

    feedback_section = _build_feedback_section(feedback)
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    coros = [
        _design_one_shard(
            llm=llm,
            template=template,
            ip_info=ip_info,
            trigger_phrase=trigger_phrase,
            feedback_section=feedback_section,
            shard_index=i + 1,
            total_shards=shards,
            shard_categories=partition,
        )
        for i, partition in enumerate(partitions)
    ]
    shard_results: list[list[BatchCase]] = await asyncio.gather(*coros)

    merged: list[BatchCase] = []
    for cases in shard_results:
        merged.extend(cases)

    expected_total = len(CATEGORIES) * CASES_PER_CATEGORY
    if len(merged) != expected_total:
        raise ValueError(
            f"design_cases expected {expected_total} total cases across "
            f"{shards} shards, got {len(merged)}"
        )

    logger.info("Designed %d test cases for batch %d (%d shards).",
                len(merged), batch_num, shards)
    return BatchConfig(
        batch=batch_num,
        trigger_phrase=trigger_phrase,
        cases=merged,
    )


async def _design_one_shard(
    *,
    llm: LLMProvider,
    template: str,
    ip_info: str,
    trigger_phrase: str,
    feedback_section: str,
    shard_index: int,
    total_shards: int,
    shard_categories: list[str],
) -> list[BatchCase]:
    shard_cases = len(shard_categories) * CASES_PER_CATEGORY
    skeleton = [
        case
        for case in generate_case_skeleton()
        if case.category in shard_categories
    ]
    skeleton_text = _format_skeleton(skeleton)

    system_prompt = (
        template
        .replace("{ip_info}", sanitize_braces(ip_info))
        .replace("{trigger_phrase}", trigger_phrase)
        .replace("{case_skeleton}", skeleton_text)
        .replace("{feedback_section}", feedback_section)
        .replace("{shard_index}", str(shard_index))
        .replace("{total_shards}", str(total_shards))
        .replace("{shard_category_count}", str(len(shard_categories)))
        .replace("{shard_cases}", str(shard_cases))
    )

    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"Design {shard_cases} diverse test cases for this shard."},
    ]}]

    raw = await llm.invoke(system=system_prompt, messages=messages, max_tokens=4096)

    cleaned = clean_json(raw)
    recovered = recover_truncated_json(cleaned)
    data = json.loads(recovered)
    if "cases" not in data:
        raise ValueError(f"shard {shard_index} LLM response missing 'cases' key")
    cases = [BatchCase.model_validate(c) for c in data["cases"]]
    if not cases:
        raise ValueError(
            f"shard {shard_index} returned zero cases — response may have been truncated."
        )

    allowed = set(shard_categories)
    stray = [c for c in cases if c.category not in allowed]
    if stray:
        raise ValueError(
            f"shard {shard_index} returned cases outside its assigned categories "
            f"{sorted(allowed)}: got {sorted({c.category for c in stray})}"
        )

    return cases


def _build_feedback_section(feedback: str) -> str:
    if not feedback.strip():
        return ""
    return (
        f"\n\n## User feedback on previous batch\n\n{sanitize_braces(feedback)}\n\n"
        "Apply this feedback when designing the new batch — adjust subjects, "
        "scenes, or angles accordingly while keeping the generalization rule."
    )


def _format_skeleton(cases: list[BatchCase]) -> str:
    lines: list[str] = []
    current_cat = ""
    for c in cases:
        if c.category != current_cat:
            current_cat = c.category
            lines.append(f"\n### {current_cat} (aspect: {c.aspect_ratio})")
        lines.append(f"- {c.id}: (fill in description)")
    return "\n".join(lines)
