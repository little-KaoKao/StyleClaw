from __future__ import annotations

import json
import logging
from pathlib import Path

from styleclaw.core.image_utils import encode_image_for_llm
from styleclaw.core.models import StyleAnalysis
from styleclaw.core.text_utils import clean_json
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "analyze.md"


async def analyze_style(
    llm: LLMProvider,
    ref_image_paths: list[Path],
    ip_info: str,
) -> StyleAnalysis:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    system_prompt = template.replace("{ip_info}", ip_info.replace("{", "{{").replace("}", "}}"))

    content: list[dict] = []
    for img_path in ref_image_paths:
        b64_data, media_type = encode_image_for_llm(img_path)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data,
            },
        })
    content.append({
        "type": "text",
        "text": "Analyze these reference images and generate a style trigger phrase.",
    })

    messages = [{"role": "user", "content": content}]
    raw = await llm.invoke(system=system_prompt, messages=messages)

    cleaned = clean_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON for style analysis: {exc}") from exc
    try:
        analysis = StyleAnalysis.model_validate(data)
    except Exception as exc:
        raise ValueError(f"LLM response failed validation for StyleAnalysis: {exc}") from exc
    logger.info("Style analysis complete. Trigger: %s", analysis.trigger_phrase[:80])
    return analysis
