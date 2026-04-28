from __future__ import annotations

import logging
from typing import Any

from styleclaw.providers.runninghub.models import ModelConfig, get_model

logger = logging.getLogger(__name__)

SEEDREAM_TRUNCATION_LIMIT = 2000

ASPECT_RATIO_TO_WH: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2560, 1440),
    "9:16": (1600, 2848),
    "4:3": (2048, 1536),
    "3:4": (1600, 2136),
    "3:2": (2048, 1368),
    "2:3": (1600, 2400),
}


def build_params(
    model_id: str,
    trigger_phrase: str,
    character_desc: str = "",
    aspect_ratio: str = "9:16",
    sref_url: str = "",
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = get_model(model_id)
    prompt = _build_prompt(trigger_phrase, character_desc)
    prompt = _truncate_prompt(prompt, config)

    params: dict[str, Any] = {"prompt": prompt}
    params.update(config.default_params)

    if config.uses_width_height:
        w, h = ASPECT_RATIO_TO_WH.get(aspect_ratio, (2048, 2048))
        params["width"] = w
        params["height"] = h
    else:
        params[config.aspect_ratio_key] = aspect_ratio

    if sref_url and config.supports_sref:
        params["sref"] = sref_url
        params["sw"] = 100

    if extra_params:
        reserved = {"prompt", "width", "height", "sref", "sw", config.aspect_ratio_key}
        safe = {k: v for k, v in extra_params.items() if k not in reserved}
        params.update(safe)

    return params


def _build_prompt(trigger_phrase: str, character_desc: str) -> str:
    parts = [p for p in (trigger_phrase, character_desc) if p.strip()]
    return ", ".join(parts)


def _truncate_prompt(prompt: str, config: ModelConfig) -> str:
    if len(prompt) <= config.max_prompt_length:
        return prompt

    logger.warning(
        "Prompt for %s exceeds %d chars (%d), truncating.",
        config.model_id, config.max_prompt_length, len(prompt),
    )
    return prompt[: config.max_prompt_length]
