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

        Constructs on first call using the OpenAI-compatible provider.
        """
        if role not in self._cached_single:
            self._cached_single[role] = _build_provider_for_role(
                self._role_configs[role]
            )
        return self._cached_single[role]

    def get_panel(self, role: Role) -> tuple[list, list[str]]:
        """Return (providers, labels) for the role's panel pool.

        Constructs all providers on first call (one per model_id in the pool).
        Labels default to model_id — per-role labels are deliberately not
        supported in this iteration (see spec § Environment Variables).
        """
        if role not in self._cached_panel:
            pool = self._panel_pools.get(role, [])
            providers = [
                _build_provider_for_role(RoleConfig(model_id=mid)) for mid in pool
            ]
            labels = list(pool)
            self._cached_panel[role] = (providers, labels)
        providers, labels = self._cached_panel[role]
        # Return copies of the lists so callers can't mutate the cache, but the
        # provider instances themselves are shared (that's the whole point).
        return list(providers), list(labels)

    async def close(self) -> None:
        """Best-effort close every provider built so far. Idempotent.

        Exceptions from individual closes are swallowed (logged would be nicer
        but the module deliberately avoids a logger dependency at the top
        level — callers see them via _close_resource in cli.py if they care).
        """
        for provider in self._cached_single.values():
            await self._safe_close(provider)
        for providers, _ in self._cached_panel.values():
            for provider in providers:
                await self._safe_close(provider)
        self._cached_single.clear()
        self._cached_panel.clear()

    @staticmethod
    async def _safe_close(provider) -> None:
        close = getattr(provider, "close", None)
        if close is None:
            return
        try:
            result = close()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            # Swallow — close failures shouldn't propagate during teardown.
            pass


def validate_routing_env() -> list[str]:
    """Return human-readable error strings for misconfigured routing envs.

    Empty list when everything is fine. Called by ``config.validate_env()`` at
    CLI startup. Never raises — aggregates errors so the user sees them all
    in one pass.
    """
    errors: list[str] = []

    # 1. Per-role single-model resolvability.
    for role in Role:
        env_name = f"STYLECLAW_MODEL_{role.value.upper()}"
        if not os.getenv(env_name) and not os.getenv("LLM_MODEL"):
            errors.append(
                f"no model resolvable for role '{role.value}': "
                f"set {env_name} or LLM_MODEL"
            )

    # 2. Per-role panel pool length when any of its toggles are on.
    # Duplicates resolution logic from _resolve_panel_pool on purpose — the
    # validator runs without instantiating the router. Read all envs through
    # os.getenv directly so callers don't need to reload core.config first.
    from styleclaw.core.config import env_truthy
    for role, toggle_envs in _PANEL_TOGGLES_FOR_ROLE.items():
        active_toggles = [t for t in toggle_envs if env_truthy(t)]
        if not active_toggles:
            continue
        role_env = f"STYLECLAW_PANEL_MODELS_{role.value.upper()}"
        role_raw = os.getenv(role_env, "").strip()
        if role_raw:
            pool = [m.strip() for m in role_raw.split(",") if m.strip()]
            source = role_env
        else:
            global_raw = os.getenv("STYLECLAW_PANEL_MODELS", "").strip()
            pool = [m.strip() for m in global_raw.split(",") if m.strip()]
            source = "STYLECLAW_PANEL_MODELS"
        toggle_label = " / ".join(active_toggles)
        if not pool:
            errors.append(
                f"panel for '{role.value}' is enabled ({toggle_label}=1) "
                f"but no pool is configured: set {role_env} or {source}"
            )
        elif len(pool) != 3:
            errors.append(
                f"panel pool for '{role.value}' must have exactly 3 models "
                f"(got {len(pool)} from {source})"
            )

    return errors


# Which env vars turn on the panel for each role. A role can be activated by
# any of the listed toggles (e.g. vision_analyst is reused by both `refine`
# and `analyze`).
_PANEL_TOGGLES_FOR_ROLE: dict[Role, tuple[str, ...]] = {
    Role.VISION_CRITIC: ("STYLECLAW_PANEL_MODEL_SELECT",),
    Role.VISION_ANALYST: ("STYLECLAW_PANEL_REFINE", "STYLECLAW_PANEL_ANALYZE"),
}


def _build_provider_for_role(cfg: RoleConfig):
    """Construct an OpenAI-compatible provider for the given role.

    This helper is the single entry point for constructing per-role providers —
    cli no longer has a parallel _build_llm_provider.
    """
    model_id = cfg.model_id or None  # provider class accepts None as "use env default"

    from styleclaw.providers.llm.openai_compat import OpenAICompatProvider
    return OpenAICompatProvider(model_id=model_id)


