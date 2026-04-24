from __future__ import annotations

import json
import logging
from pathlib import Path

from styleclaw.core.case_generator import generate_case_skeleton
from styleclaw.core.models import BatchCase, BatchConfig
from styleclaw.core.text_utils import clean_json
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
        PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("{ip_info}", ip_info)
        .replace("{trigger_phrase}", trigger_phrase)
        .replace("{case_skeleton}", skeleton_text)
    )

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Design 100 diverse test cases for batch image generation."},
    ]}]

    raw = await llm.invoke(system=system_prompt, messages=messages, max_tokens=16384)

    cleaned = clean_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        last_brace = cleaned.rfind("}")
        if last_brace < 0:
            raise
        truncated = cleaned[: last_brace + 1]
        bracket = truncated.rfind("]")
        if bracket < 0:
            raise
        data = json.loads(truncated[: bracket + 1].rsplit(",", 1)[0] + "]}")

    cases = [BatchCase.model_validate(c) for c in data["cases"]]
    if not cases:
        raise ValueError("LLM returned zero cases — response may have been truncated.")
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
