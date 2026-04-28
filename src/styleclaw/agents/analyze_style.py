from __future__ import annotations

import logging
from pathlib import Path

from styleclaw.core.image_utils import build_image_block, encode_image_for_llm
from styleclaw.core.models import StyleAnalysis
from styleclaw.core.text_utils import parse_llm_response, sanitize_braces
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "analyze.md"


async def analyze_style(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    ip_info: str,
) -> StyleAnalysis:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    system_prompt = template.replace("{ip_info}", sanitize_braces(ip_info))

    content: list[dict] = [build_image_block(p) for p in ref_image_paths]
    content.append({
        "type": "text",
        "text": "Analyze these reference images and generate a style trigger phrase.",
    })

    messages = [{"role": "user", "content": content}]
    raw = await llm.invoke(system=system_prompt, messages=messages)

    analysis = parse_llm_response(raw, StyleAnalysis, "style analysis")
    logger.info("Style analysis complete. Trigger: %s", analysis.trigger_phrase[:80])
    return analysis
