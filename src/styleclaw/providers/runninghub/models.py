from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SrefMode(StrEnum):
    PARAM = "param"
    PROMPT = "prompt"


class I2IParamStyle(StrEnum):
    """How a model's i2i endpoint expects the input image to be passed."""
    SINGLE_URL_IW = "single_url_iw"  # MJ family: imageUrl=<url>, iw=0.5
    MULTI_URLS = "multi_urls"        # everyone else: imageUrls=[<url>, ...]


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    name: str
    t2i_endpoint: str
    i2i_endpoint: str
    max_prompt_length: int
    sref_mode: SrefMode = SrefMode.PROMPT
    aspect_ratio_key: str = "aspectRatio"
    aspect_ratio_values: tuple[str, ...] = ("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3")
    default_params: dict[str, Any] = field(default_factory=dict)
    uses_width_height: bool = False
    i2i_param_style: I2IParamStyle = I2IParamStyle.MULTI_URLS
    # When False, the t2i endpoint is pure text-to-image and rejects imageUrls.
    # Callers that need to pass a style reference (prompt-sref variant in
    # model-select / style-refine, sref-aware batch-t2i) must route to the
    # i2i endpoint instead. Use resolve_submit_endpoint() to do this.
    t2i_accepts_sref: bool = True


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "mj-v7": ModelConfig(
        model_id="mj-v7",
        name="悠船文生图 v7 (Midjourney)",
        t2i_endpoint="/openapi/v2/youchuan/text-to-image-v7",
        i2i_endpoint="/openapi/v2/youchuan/text-to-image-v7",
        max_prompt_length=8192,
        sref_mode=SrefMode.PARAM,
        default_params={"stylize": 200},
        i2i_param_style=I2IParamStyle.SINGLE_URL_IW,
    ),
    "niji7": ModelConfig(
        model_id="niji7",
        name="悠船文生图 niji7 (Midjourney)",
        t2i_endpoint="/openapi/v2/youchuan/text-to-image-niji7",
        i2i_endpoint="/openapi/v2/youchuan/text-to-image-niji7",
        max_prompt_length=8192,
        sref_mode=SrefMode.PARAM,
        default_params={"stylize": 200},
        i2i_param_style=I2IParamStyle.SINGLE_URL_IW,
    ),
    "nb2": ModelConfig(
        model_id="nb2",
        name="全能图片V2 (NanoBanana2)",
        t2i_endpoint="/openapi/v2/rhart-image-n-g31-flash-official/text-to-image",
        i2i_endpoint="/openapi/v2/rhart-image-n-g31-flash-official/image-to-image",
        max_prompt_length=20000,
        sref_mode=SrefMode.PROMPT,
        default_params={"resolution": "2k"},
    ),
    "seedream": ModelConfig(
        model_id="seedream",
        name="Seedream v5-lite (即梦)",
        t2i_endpoint="/openapi/v2/seedream-v5-lite/text-to-image",
        i2i_endpoint="/openapi/v2/seedream-v5-lite/image-to-image",
        max_prompt_length=2000,
        sref_mode=SrefMode.PROMPT,
        uses_width_height=True,
    ),
    "gpt-image-2": ModelConfig(
        model_id="gpt-image-2",
        name="全能图片G-2 (GPT-Image-2)",
        t2i_endpoint="/openapi/v2/rhart-image-g-2-official/text-to-image",
        i2i_endpoint="/openapi/v2/rhart-image-g-2-official/image-to-image",
        max_prompt_length=20000,
        sref_mode=SrefMode.PROMPT,
        aspect_ratio_values=("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"),
        default_params={"resolution": "2k", "quality": "medium"},
    ),
    "n-pro": ModelConfig(
        model_id="n-pro",
        name="全能图片PRO 官方稳定版",
        # text-to-image is pure t2i (no imageUrls); /edit is the i2i endpoint
        # that takes up to 10 imageUrls. prompt-sref variant routes to /edit
        # via resolve_submit_endpoint() because t2i_accepts_sref=False.
        t2i_endpoint="/openapi/v2/rhart-image-n-pro-official/text-to-image",
        i2i_endpoint="/openapi/v2/rhart-image-n-pro-official/edit",
        max_prompt_length=20000,
        sref_mode=SrefMode.PROMPT,
        aspect_ratio_values=("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"),
        default_params={"resolution": "2k"},
        t2i_accepts_sref=False,
    ),
}


def get_model(model_id: str) -> ModelConfig:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_id}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_id]


def build_i2i_params(model: ModelConfig, trigger_phrase: str, image_url: str) -> dict[str, Any]:
    """Build the request body for an i2i submission, routing based on the
    model's declared i2i_param_style. Adding a new model only requires
    setting its i2i_param_style — no caller code changes.

    ``default_params`` are folded in so endpoint-required fields like
    ``resolution`` (n-pro /edit) and ``stylize`` (MJ) reach the i2i path
    too, not just t2i.
    """
    if model.i2i_param_style == I2IParamStyle.SINGLE_URL_IW:
        base: dict[str, Any] = {"prompt": trigger_phrase, "imageUrl": image_url, "iw": 0.5}
    else:
        base = {"prompt": trigger_phrase, "imageUrls": [image_url]}
    return {**model.default_params, **base}


def resolve_submit_endpoint(model: ModelConfig, sref_url: str) -> str:
    """Choose the submission endpoint for a t2i-style call given the sref state.

    Most PROMPT-mode models (nb2, gpt-image-2) accept ``imageUrls`` on the
    t2i endpoint, so prompt-sref stays on t2i. n-pro's t2i is pure
    text-to-image — when sref is present we route to its /edit endpoint,
    which accepts ``imageUrls``. PARAM-mode models (mj-v7, niji7) pass
    sref via ``sref`` parameter, no rerouting needed.
    """
    if sref_url and not model.t2i_accepts_sref:
        return model.i2i_endpoint
    return model.t2i_endpoint

