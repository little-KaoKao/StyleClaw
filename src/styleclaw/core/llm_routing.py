from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    """LLM call-site roles. Each agent function maps to exactly one role.

    Mirrors the str-Enum pattern used by Phase: JSON-serializable for free,
    and IDE autocomplete prevents typos at call sites.
    """
    VISION_CRITIC = "vision_critic"
    VISION_ANALYST = "vision_analyst"
    WRITER = "writer"
    PLANNER = "planner"


@dataclass(frozen=True)
class RoleConfig:
    """Resolved config for one role.

    base_url / api_key are extension hooks for future cross-provider routing —
    they are always None today. Plumbing them through here means a future
    change to support per-role gateways only touches this module, not any
    call site.
    """
    model_id: str
    base_url: str | None = None
    api_key: str | None = None
