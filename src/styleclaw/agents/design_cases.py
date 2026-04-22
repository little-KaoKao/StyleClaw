from __future__ import annotations

import json
import logging
from pathlib import Path

from styleclaw.core.case_generator import generate_case_skeleton
from styleclaw.core.models import BatchCase, BatchConfig
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "design_cases.md"
)


async def design_cases(
    llm: LLMProvider,
    ip_info: str,
    trigger_phrase: str,
    batch_num: int,
) -> BatchConfig:
    skeleton = generate_case_skeleton()
    skeleton_text = _format_skeleton(skeleton)

    system_prompt = (
        PROMPT_TEMPLATE_PATH.read_text()
        .replace("{ip_info}", ip_info)
        .replace("{trigger_phrase}", trigger_phrase)
        .replace("{case_skeleton}", skeleton_text)
    )

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Design 100 diverse test cases for batch image generation."},
    ]}]

    raw = await llm.invoke(system=system_prompt, messages=messages, max_tokens=8192)

    cleaned = _clean_json(raw)
    data = json.loads(cleaned)

    cases = [BatchCase.model_validate(c) for c in data["cases"]]
    config = BatchConfig(
        batch=batch_num,
        trigger_phrase=trigger_phrase,
        cases=cases,
    )
    logger.info("Designed %d test cases for batch %d.", len(cases), batch_num)
    return config


def _format_skeleton(cases: list[BatchCase]) -> str:
    lines: list[str] = []
    current_cat = ""
    for c in cases:
        if c.category != current_cat:
            current_cat = c.category
            lines.append(f"\n### {current_cat} (aspect: {c.aspect_ratio})")
        lines.append(f"- {c.id}: (fill in description)")
    return "\n".join(lines)


def _clean_json(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()
