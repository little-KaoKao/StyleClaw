from __future__ import annotations

import importlib

from styleclaw.providers.llm.base import LLMProvider
from styleclaw.providers.llm.openai_compat import OpenAICompatProvider


def build_panel_providers() -> list[tuple[LLMProvider, str]]:
    """Instantiate three OpenAI-compat providers, one per panel model id.

    Reloads ``styleclaw.core.config`` so callers that flip env vars at runtime
    (tests, repls) see updated values without managing module state by hand.
    All providers share the same base URL + API key; only ``model_id`` differs.
    Caller is responsible for closing the returned providers (httpx clients).
    """
    config_mod = importlib.reload(importlib.import_module("styleclaw.core.config"))

    if not (config_mod.PANEL_REFINE_ENABLED or config_mod.PANEL_MODEL_SELECT_ENABLED):
        raise RuntimeError(
            "build_panel_providers() called but no panel toggle is enabled "
            "(set STYLECLAW_PANEL_REFINE=1 or STYLECLAW_PANEL_MODEL_SELECT=1)."
        )

    errors = config_mod.validate_panel_config()
    if errors:
        raise ValueError("; ".join(errors))

    pairs: list[tuple[LLMProvider, str]] = []
    for model_id, label in zip(config_mod.PANEL_MODELS, config_mod.PANEL_LABELS):
        provider = OpenAICompatProvider(model_id=model_id)
        pairs.append((provider, label))
    return pairs


async def close_panel_providers(pairs: list[tuple[LLMProvider, str]]) -> None:
    """Best-effort close of httpx clients held by the providers."""
    for provider, _ in pairs:
        close = getattr(provider, "close", None)
        if close is not None:
            await close()
