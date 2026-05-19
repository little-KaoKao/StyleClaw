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


# Roles that participate in panel mode.
_PANEL_ROLES: frozenset[Role] = frozenset({Role.VISION_CRITIC, Role.VISION_ANALYST})


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
        self._cached_single: dict[Role, "LLMProvider"] = {}
        self._cached_panel: dict[Role, tuple[list, list[str]]] = {}

    @classmethod
    def from_env(cls) -> "RoleRouter":
        """Read env vars; do NOT construct any provider yet."""
        role_configs = {role: cls._resolve_role(role) for role in Role}
        panel_pools = {role: cls._resolve_panel_pool(role) for role in Role}
        return cls(role_configs, panel_pools)

    @staticmethod
    def _resolve_role(role: Role) -> RoleConfig:
        env_name = f"STYLECLAW_MODEL_{role.value.upper()}"
        model_id = os.getenv(env_name) or os.getenv("LLM_MODEL", "")
        return RoleConfig(model_id=model_id)

    @staticmethod
    def _resolve_panel_pool(role: Role) -> list[str]:
        if role not in _PANEL_ROLES:
            return []
        # Role-specific env wins over the global pool.
        role_env = f"STYLECLAW_PANEL_MODELS_{role.value.upper()}"
        raw = os.getenv(role_env, "").strip()
        if raw:
            return [m.strip() for m in raw.split(",") if m.strip()]
        # Read the live config module so test monkeypatching + reload propagates.
        import importlib
        cfg = importlib.import_module("styleclaw.core.config")
        return list(cfg.PANEL_MODELS)

    def config_for(self, role: Role) -> RoleConfig:
        return self._role_configs[role]

    def panel_pool_for(self, role: Role) -> list[str]:
        return list(self._panel_pools[role])

    def get(self, role: Role):
        """Return a cached single-model provider for the role.

        Constructs on first call using the existing provider-class precedence
        (OpenAI-compat > RunningHub LLM > Bedrock).
        """
        if role not in self._cached_single:
            self._cached_single[role] = _build_provider_for_role(
                self._role_configs[role]
            )
        return self._cached_single[role]

    async def close(self) -> None:
        """Placeholder; Task 6 replaces this with idempotent teardown."""
        return None


def _build_provider_for_role(cfg: RoleConfig):
    """Pick a provider class via the existing precedence rule and pass model_id.

    OpenAI-compat > RunningHub LLM > Bedrock. Duplicates the logic in
    cli._build_llm_provider on purpose — Part 3 will delete the cli copy and
    route everything through here.
    """
    from styleclaw.core.config import env_truthy

    model_id = cfg.model_id or None  # provider classes accept None as "use env default"

    if os.getenv("OPENAI_COMPAT_API_KEY"):
        from styleclaw.providers.llm.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(model_id=model_id)
    if env_truthy("RUNNINGHUB_LLM"):
        from styleclaw.providers.llm.runninghub_llm import RunningHubLLMProvider
        return RunningHubLLMProvider(model_id=model_id)
    from styleclaw.providers.llm.bedrock import BedrockProvider
    return BedrockProvider(model_id=model_id)


