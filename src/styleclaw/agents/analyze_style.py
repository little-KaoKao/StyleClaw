from __future__ import annotations

import json
import logging
from pathlib import Path

from styleclaw.core.image_utils import image_to_base64, media_type_for
from styleclaw.core.models import StyleAnalysis
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "analyze.md"


async def analyze_style(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    ip_info: str,
) -> StyleAnalysis:
    system_prompt = PROMPT_TEMPLATE_PATH.read_text().replace("{ip_info}", ip_info)

    content: list[dict] = []
    for img_path in ref_image_paths:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type_for(img_path),
                "data": image_to_base64(img_path),
            },
        })
    content.append({
        "type": "text",
        "text": "Analyze these reference images and generate a style trigger phrase.",
    })

    messages = [{"role": "user", "content": content}]
    raw = await llm.invoke(system=system_prompt, messages=messages)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    analysis = StyleAnalysis.model_validate(data)
    logger.info("Style analysis complete. Trigger: %s", analysis.trigger_phrase[:80])
    return analysis
