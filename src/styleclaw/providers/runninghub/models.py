from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    name: str
    t2i_endpoint: str
    i2i_endpoint: str
    max_prompt_length: int
    aspect_ratio_key: str = "aspectRatio"
    aspect_ratio_values: tuple[str, ...] = ("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3")
    default_params: dict[str, Any] = field(default_factory=dict)
    supports_sref: bool = False
    uses_width_height: bool = False


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "mj-v7": ModelConfig(
        model_id="mj-v7",
        name="悠船文生图 v7 (Midjourney)",
        t2i_endpoint="/openapi/v2/youchuan/text-to-image-v7",
        i2i_endpoint="/openapi/v2/youchuan/text-to-image-v7",
        max_prompt_length=8192,
        supports_sref=True,
        default_params={"stylize": 200},
    ),
    "niji7": ModelConfig(
        model_id="niji7",
        name="悠船文生图 niji7 (Midjourney)",
        t2i_endpoint="/openapi/v2/youchuan/text-to-image-niji7",
        i2i_endpoint="/openapi/v2/youchuan/text-to-image-niji7",
        max_prompt_length=8192,
        supports_sref=True,
        default_params={"stylize": 200},
    ),
    "nb2": ModelConfig(
        model_id="nb2",
        name="全能图片V2 (NanoBanana2)",
        t2i_endpoint="/openapi/v2/rhart-image-n-g31-flash-official/text-to-image",
        i2i_endpoint="/openapi/v2/rhart-image-n-g31-flash-official/image-to-image",
        max_prompt_length=20000,
        default_params={"resolution": "2k"},
    ),
    "seedream": ModelConfig(
        model_id="seedream",
        name="Seedream v5-lite (即梦)",
        t2i_endpoint="/openapi/v2/seedream-v5-lite/text-to-image",
        i2i_endpoint="/openapi/v2/seedream-v5-lite/image-to-image",
        max_prompt_length=2000,
        uses_width_height=True,
        default_params={"resolution": "2k"},
    ),
}


def get_model(model_id: str) -> ModelConfig:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_id}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_id]
