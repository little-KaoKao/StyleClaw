from __future__ import annotations

import os
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


class RoleRouter:
    """Lazy-build LLMProvider instances, scoped by Role.

    Lifecycle owned by ExecutionContext (built in cli._build_context, closed
    in cli._close_resource). Construction is lazy so commands that don't need
    LLM access don't open HTTP clients.
    """

    def __init__(
        self,
        role_configs: dict[Role, RoleConfig],
        panel_pools: dict[Role, list[str]],
    ) -> None:
        self._role_configs = role_configs
        self._panel_pools = panel_pools

    @classmethod
    def from_env(cls) -> "RoleRouter":
        """Read env vars; do NOT construct any provider yet."""
        role_configs = {role: cls._resolve_role(role) for role in Role}
        # Panel pools added in Task 3; default to empty for now.
        panel_pools: dict[Role, list[str]] = {}
        return cls(role_configs, panel_pools)

    @staticmethod
    def _resolve_role(role: Role) -> RoleConfig:
        env_name = f"STYLECLAW_MODEL_{role.value.upper()}"
        model_id = os.getenv(env_name) or os.getenv("LLM_MODEL", "")
        return RoleConfig(model_id=model_id)

    def config_for(self, role: Role) -> RoleConfig:
        return self._role_configs[role]

