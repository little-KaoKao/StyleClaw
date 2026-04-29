from __future__ import annotations

import logging
from pathlib import Path

from styleclaw.core.image_utils import build_image_block
from styleclaw.core.models import StyleAnalysis
from styleclaw.core.text_utils import parse_llm_response, sanitize_braces
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "analyze.md"


def _build_messages(ref_image_paths: list[Path]) -> list[dict]:
    content: list[dict] = [build_image_block(p) for p in ref_image_paths]
    content.append({
        "type": "text",
        "text": "Analyze these reference images and generate a style trigger phrase.",
    })
    return [{"role": "user", "content": content}]


def _build_system_prompt(ip_info: str) -> str:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("{ip_info}", sanitize_braces(ip_info))


async def analyze_style(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    ip_info: str,
) -> StyleAnalysis:
    raw = await llm.invoke(
        system=_build_system_prompt(ip_info),
        messages=_build_messages(ref_image_paths),
    )
    analysis = parse_llm_response(raw, StyleAnalysis, "style analysis")
    logger.info("Style analysis complete. Trigger: %s", analysis.trigger_phrase[:80])
    return analysis


async def analyze_style_with_thinking(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    ip_info: str,
    thinking_budget: int = 5000,
) -> tuple[StyleAnalysis, str]:
    response = await llm.invoke_with_thinking(
        system=_build_system_prompt(ip_info),
        messages=_build_messages(ref_image_paths),
        thinking_budget=thinking_budget,
    )
    analysis = parse_llm_response(response.text, StyleAnalysis, "style analysis")
    logger.info("Style analysis (with thinking) complete. Trigger: %s", analysis.trigger_phrase[:80])
    return analysis, response.thinking
