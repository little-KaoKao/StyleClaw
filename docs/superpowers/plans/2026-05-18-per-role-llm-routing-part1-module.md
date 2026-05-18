# Per-Role LLM Routing — Part 1: New `llm_routing` Module

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the new `core/llm_routing.py` module (Role enum, RoleConfig dataclass, RoleRouter class) in complete isolation from the rest of the codebase. After this part, the module exists with full unit-test coverage but no other file imports it yet.

**Architecture:** Pure addition. New module under `src/styleclaw/core/llm_routing.py` plus tests at `tests/core/test_llm_routing.py`. Provider-class selection (`OpenAICompat > RunningHubLLM > Bedrock`) is duplicated here in a helper rather than imported from `cli.py` — `cli.py` will switch to using this helper in Part 3.

**Tech Stack:** Python 3.11+, Pydantic-free (plain `@dataclass` + `enum.Enum`), pytest + `monkeypatch` for env-var tests, no httpx mocks needed (provider classes are constructed but never `invoke`d in these tests).

**Reference:** [spec](../specs/2026-05-18-per-role-llm-routing-design.md) — sections "Roles", "Environment Variables", "Architecture / New module".

**Dependencies:** None. Run before Part 2.

---

## File Structure

- **Create:** `src/styleclaw/core/llm_routing.py` — `Role`, `RoleConfig`, `RoleRouter`, `_build_provider_for_role()`
- **Create:** `tests/core/test_llm_routing.py` — unit tests covering parsing, caching, panel pools, close lifecycle

No existing files are modified in this part.

---

## Task 1: Skeleton — `Role` enum + `RoleConfig` dataclass

**Files:**
- Create: `src/styleclaw/core/llm_routing.py`
- Create: `tests/core/test_llm_routing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_llm_routing.py`:

```python
from __future__ import annotations

from styleclaw.core.llm_routing import Role, RoleConfig


class TestRoleEnum:
    def test_four_roles_with_str_values(self) -> None:
        assert Role.VISION_CRITIC.value == "vision_critic"
        assert Role.VISION_ANALYST.value == "vision_analyst"
        assert Role.WRITER.value == "writer"
        assert Role.PLANNER.value == "planner"

    def test_role_is_str_subclass(self) -> None:
        # Matches Phase pattern: JSON-serializable, IDE autocomplete.
        assert isinstance(Role.WRITER, str)


class TestRoleConfig:
    def test_minimal_construction(self) -> None:
        cfg = RoleConfig(model_id="gemini-2.5-pro")
        assert cfg.model_id == "gemini-2.5-pro"
        assert cfg.base_url is None
        assert cfg.api_key is None

    def test_is_frozen(self) -> None:
        import dataclasses
        cfg = RoleConfig(model_id="x")
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            cfg.model_id = "y"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/core/test_llm_routing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'styleclaw.core.llm_routing'`

- [ ] **Step 3: Write minimal implementation**

Create `src/styleclaw/core/llm_routing.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/core/test_llm_routing.py -v`
Expected: PASS — 3 tests in `TestRoleEnum` + `TestRoleConfig`

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/llm_routing.py tests/core/test_llm_routing.py
git commit -m "$(cat <<'EOF'
feat(llm_routing): Role enum + RoleConfig skeleton

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `RoleRouter.from_env()` — parse single-model role envs

**Files:**
- Modify: `src/styleclaw/core/llm_routing.py` (add `RoleRouter` class with `from_env` + `_role_config_for`)
- Modify: `tests/core/test_llm_routing.py` (add `TestFromEnvSingle` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_llm_routing.py`:

```python
import pytest


class TestFromEnvSingle:
    """Test single-model role resolution: per-role env > LLM_MODEL > empty."""

    @pytest.fixture(autouse=True)
    def _clear_envs(self, monkeypatch):
        # Strip all role + global env vars before each test.
        for role in Role:
            monkeypatch.delenv(f"STYLECLAW_MODEL_{role.value.upper()}", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        yield

    def test_role_env_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_MODEL_VISION_CRITIC", "claude-sonnet-4-6")
        monkeypatch.setenv("LLM_MODEL", "fallback-model")

        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()

        assert router.config_for(Role.VISION_CRITIC).model_id == "claude-sonnet-4-6"

    def test_fallback_to_llm_model(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "global-default")

        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()

        for role in Role:
            assert router.config_for(role).model_id == "global-default", role

    def test_all_unset_yields_empty_model_id(self):
        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()

        # Empty string is fine here — validate_env() (Part 2) is responsible
        # for refusing to start the CLI when no model is resolvable.
        for role in Role:
            assert router.config_for(role).model_id == "", role

    def test_each_role_env_is_independent(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_MODEL_VISION_CRITIC", "critic-model")
        monkeypatch.setenv("STYLECLAW_MODEL_WRITER", "writer-model")

        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()

        assert router.config_for(Role.VISION_CRITIC).model_id == "critic-model"
        assert router.config_for(Role.VISION_ANALYST).model_id == ""
        assert router.config_for(Role.WRITER).model_id == "writer-model"
        assert router.config_for(Role.PLANNER).model_id == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/core/test_llm_routing.py::TestFromEnvSingle -v`
Expected: FAIL with `AttributeError: module 'styleclaw.core.llm_routing' has no attribute 'RoleRouter'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/styleclaw/core/llm_routing.py`:

```python
import os


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
        """Test-friendly accessor for the resolved config. Used by tests; the
        production code path goes through ``get()`` / ``get_panel()``."""
        return self._role_configs[role]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/core/test_llm_routing.py -v`
Expected: PASS — all tests in `TestRoleEnum`, `TestRoleConfig`, `TestFromEnvSingle`

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/llm_routing.py tests/core/test_llm_routing.py
git commit -m "$(cat <<'EOF'
feat(llm_routing): RoleRouter.from_env single-model resolution

Per-role STYLECLAW_MODEL_<ROLE> env var, falling back to LLM_MODEL.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `RoleRouter.from_env()` — parse panel-pool envs

**Files:**
- Modify: `src/styleclaw/core/llm_routing.py` (extend `from_env`, add `_resolve_panel_pool`, add `panel_pool_for`)
- Modify: `tests/core/test_llm_routing.py` (add `TestFromEnvPanel` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_llm_routing.py`:

```python
class TestFromEnvPanel:
    """Panel pool resolution: per-role env > global STYLECLAW_PANEL_MODELS > empty.

    Only VISION_CRITIC and VISION_ANALYST get panel pools — WRITER and PLANNER
    are never paneled.
    """

    @pytest.fixture(autouse=True)
    def _clear_envs(self, monkeypatch):
        for role in Role:
            monkeypatch.delenv(f"STYLECLAW_MODEL_{role.value.upper()}", raising=False)
            monkeypatch.delenv(f"STYLECLAW_PANEL_MODELS_{role.value.upper()}", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODELS", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        # Reload config_mod so PANEL_MODELS reflects the cleared state.
        import importlib, styleclaw.core.config as cfg
        importlib.reload(cfg)
        yield

    def _reload_config(self):
        import importlib, styleclaw.core.config as cfg
        importlib.reload(cfg)

    def test_role_specific_pool(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS_VISION_CRITIC", "a,b,c")
        self._reload_config()

        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()

        assert router.panel_pool_for(Role.VISION_CRITIC) == ["a", "b", "c"]
        assert router.panel_pool_for(Role.VISION_ANALYST) == []

    def test_fallback_to_global(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "x,y,z")
        self._reload_config()

        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()

        assert router.panel_pool_for(Role.VISION_CRITIC) == ["x", "y", "z"]
        assert router.panel_pool_for(Role.VISION_ANALYST) == ["x", "y", "z"]

    def test_role_overrides_global(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "g1,g2,g3")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS_VISION_CRITIC", "c1,c2,c3")
        self._reload_config()

        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()

        assert router.panel_pool_for(Role.VISION_CRITIC) == ["c1", "c2", "c3"]
        assert router.panel_pool_for(Role.VISION_ANALYST) == ["g1", "g2", "g3"]

    def test_only_panel_roles_have_pools(self, monkeypatch):
        # WRITER and PLANNER never panel — pool is always empty for them.
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "g1,g2,g3")
        self._reload_config()

        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()

        assert router.panel_pool_for(Role.WRITER) == []
        assert router.panel_pool_for(Role.PLANNER) == []

    def test_empty_when_no_envs(self):
        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()

        for role in Role:
            assert router.panel_pool_for(role) == [], role
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/core/test_llm_routing.py::TestFromEnvPanel -v`
Expected: FAIL with `AttributeError: 'RoleRouter' object has no attribute 'panel_pool_for'`

- [ ] **Step 3: Write minimal implementation**

Modify `src/styleclaw/core/llm_routing.py` — replace the `from_env` classmethod and add `_resolve_panel_pool` + `panel_pool_for`:

```python
# Roles that participate in panel mode.
_PANEL_ROLES: frozenset[Role] = frozenset({Role.VISION_CRITIC, Role.VISION_ANALYST})


class RoleRouter:
    # ... __init__ unchanged ...

    @classmethod
    def from_env(cls) -> "RoleRouter":
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/core/test_llm_routing.py -v`
Expected: PASS — all tests so far (3 classes)

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/llm_routing.py tests/core/test_llm_routing.py
git commit -m "$(cat <<'EOF'
feat(llm_routing): RoleRouter.from_env panel-pool resolution

Per-role STYLECLAW_PANEL_MODELS_<ROLE> env var, falling back to global
STYLECLAW_PANEL_MODELS. Only VISION_CRITIC and VISION_ANALYST get pools.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `RoleRouter.get()` — lazy single-model provider construction + caching

**Files:**
- Modify: `src/styleclaw/core/llm_routing.py` (add `_build_provider_for_role`, `get`)
- Modify: `tests/core/test_llm_routing.py` (add `TestGetSingle` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_llm_routing.py`:

```python
class TestGetSingle:
    """Test lazy construction + caching of single-model providers.

    Uses OPENAI_COMPAT_* envs so the router picks OpenAICompatProvider — the
    same path real users hit when they configure gptproto.com.
    """

    @pytest.fixture(autouse=True)
    def _setup_openai_compat(self, monkeypatch):
        for role in Role:
            monkeypatch.delenv(f"STYLECLAW_MODEL_{role.value.upper()}", raising=False)
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://test.local/v1")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "test-key")
        monkeypatch.delenv("RUNNINGHUB_LLM", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.setenv("STYLECLAW_MODEL_VISION_CRITIC", "critic-model")
        monkeypatch.setenv("STYLECLAW_MODEL_WRITER", "writer-model")
        yield

    def test_get_returns_openai_compat_provider(self, monkeypatch):
        from styleclaw.core.llm_routing import RoleRouter
        from styleclaw.providers.llm.openai_compat import OpenAICompatProvider

        router = RoleRouter.from_env()
        try:
            provider = router.get(Role.VISION_CRITIC)
            assert isinstance(provider, OpenAICompatProvider)
            assert provider._model_id == "critic-model"
        finally:
            import asyncio
            asyncio.run(router.close())

    def test_get_caches_provider_instance(self):
        from styleclaw.core.llm_routing import RoleRouter

        router = RoleRouter.from_env()
        try:
            a = router.get(Role.VISION_CRITIC)
            b = router.get(Role.VISION_CRITIC)
            assert a is b  # Same instance reused.
        finally:
            import asyncio
            asyncio.run(router.close())

    def test_get_different_roles_get_different_instances(self):
        from styleclaw.core.llm_routing import RoleRouter

        router = RoleRouter.from_env()
        try:
            critic = router.get(Role.VISION_CRITIC)
            writer = router.get(Role.WRITER)
            assert critic is not writer
            assert critic._model_id == "critic-model"
            assert writer._model_id == "writer-model"
        finally:
            import asyncio
            asyncio.run(router.close())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/core/test_llm_routing.py::TestGetSingle -v`
Expected: FAIL with `AttributeError: 'RoleRouter' object has no attribute 'get'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/styleclaw/core/llm_routing.py` — add provider selection helper + `get` method + `__init__` cache field:

```python
# Top-level helper (used by `get` and `get_panel`).
def _build_provider_for_role(cfg: RoleConfig):
    """Pick a provider class via the existing precedence rule and pass model_id.

    OpenAI-compat > RunningHub LLM > Bedrock. Duplicates the logic in
    cli._build_llm_provider on purpose — Part 3 will delete the cli copy and
    route everything through here.
    """
    import os
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
```

In the `RoleRouter` class, update `__init__` and add `get`:

```python
def __init__(
    self,
    role_configs: dict[Role, RoleConfig],
    panel_pools: dict[Role, list[str]],
) -> None:
    self._role_configs = role_configs
    self._panel_pools = panel_pools
    self._cached_single: dict[Role, "LLMProvider"] = {}
    self._cached_panel: dict[Role, tuple[list, list[str]]] = {}

def get(self, role: Role):
    """Return a cached single-model provider for the role.

    Constructs on first call using the existing provider-class precedence
    (OpenAI-compat > RunningHub LLM > Bedrock).
    """
    if role not in self._cached_single:
        self._cached_single[role] = _build_provider_for_role(self._role_configs[role])
    return self._cached_single[role]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/core/test_llm_routing.py -v`
Expected: PASS — `TestGetSingle` adds 3 new tests, all green.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/llm_routing.py tests/core/test_llm_routing.py
git commit -m "$(cat <<'EOF'
feat(llm_routing): RoleRouter.get with lazy construction + caching

Provider-class precedence (OpenAI-compat > RunningHub LLM > Bedrock) lives
in _build_provider_for_role; will replace cli._build_llm_provider in Part 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `RoleRouter.get_panel()` — lazy panel-pool construction

**Files:**
- Modify: `src/styleclaw/core/llm_routing.py` (add `get_panel`)
- Modify: `tests/core/test_llm_routing.py` (add `TestGetPanel` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_llm_routing.py`:

```python
class TestGetPanel:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        for role in Role:
            monkeypatch.delenv(f"STYLECLAW_MODEL_{role.value.upper()}", raising=False)
            monkeypatch.delenv(f"STYLECLAW_PANEL_MODELS_{role.value.upper()}", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODELS", raising=False)
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://test.local/v1")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "test-key")
        monkeypatch.delenv("RUNNINGHUB_LLM", raising=False)
        import importlib, styleclaw.core.config as cfg
        importlib.reload(cfg)
        yield

    def test_get_panel_returns_three_providers_and_labels(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS_VISION_CRITIC", "m1,m2,m3")
        import importlib, styleclaw.core.config as cfg
        importlib.reload(cfg)

        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()
        try:
            providers, labels = router.get_panel(Role.VISION_CRITIC)
            assert [p._model_id for p in providers] == ["m1", "m2", "m3"]
            assert labels == ["m1", "m2", "m3"]
        finally:
            import asyncio
            asyncio.run(router.close())

    def test_get_panel_caches(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS_VISION_ANALYST", "a,b,c")
        import importlib, styleclaw.core.config as cfg
        importlib.reload(cfg)

        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()
        try:
            first_providers, _ = router.get_panel(Role.VISION_ANALYST)
            second_providers, _ = router.get_panel(Role.VISION_ANALYST)
            # Same instances both times.
            assert all(a is b for a, b in zip(first_providers, second_providers))
        finally:
            import asyncio
            asyncio.run(router.close())

    def test_get_panel_empty_pool_returns_empty_lists(self):
        from styleclaw.core.llm_routing import RoleRouter
        router = RoleRouter.from_env()
        try:
            providers, labels = router.get_panel(Role.VISION_CRITIC)
            assert providers == []
            assert labels == []
        finally:
            import asyncio
            asyncio.run(router.close())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/core/test_llm_routing.py::TestGetPanel -v`
Expected: FAIL with `AttributeError: 'RoleRouter' object has no attribute 'get_panel'`

- [ ] **Step 3: Write minimal implementation**

Add `get_panel` to `RoleRouter` in `src/styleclaw/core/llm_routing.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/core/test_llm_routing.py -v`
Expected: PASS — `TestGetPanel` adds 3 new tests, all green.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/llm_routing.py tests/core/test_llm_routing.py
git commit -m "$(cat <<'EOF'
feat(llm_routing): RoleRouter.get_panel for role-scoped panel pools

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `RoleRouter.close()` — idempotent close of all built providers

**Files:**
- Modify: `src/styleclaw/core/llm_routing.py` (add `close`)
- Modify: `tests/core/test_llm_routing.py` (add `TestClose` class)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_llm_routing.py`:

```python
class TestClose:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        for role in Role:
            monkeypatch.delenv(f"STYLECLAW_MODEL_{role.value.upper()}", raising=False)
            monkeypatch.delenv(f"STYLECLAW_PANEL_MODELS_{role.value.upper()}", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODELS", raising=False)
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://test.local/v1")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "test-key")
        monkeypatch.delenv("RUNNINGHUB_LLM", raising=False)
        monkeypatch.setenv("STYLECLAW_MODEL_VISION_CRITIC", "c")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS_VISION_ANALYST", "a1,a2,a3")
        import importlib, styleclaw.core.config as cfg
        importlib.reload(cfg)
        yield

    def test_close_is_idempotent(self):
        import asyncio
        from styleclaw.core.llm_routing import RoleRouter

        router = RoleRouter.from_env()
        router.get(Role.VISION_CRITIC)  # Force a build.
        router.get_panel(Role.VISION_ANALYST)  # Force panel builds too.

        asyncio.run(router.close())
        asyncio.run(router.close())  # Must not raise.

    def test_close_with_no_builds_is_noop(self):
        import asyncio
        from styleclaw.core.llm_routing import RoleRouter

        router = RoleRouter.from_env()
        asyncio.run(router.close())  # No providers ever built; must not raise.

    def test_close_clears_caches(self):
        import asyncio
        from styleclaw.core.llm_routing import RoleRouter

        router = RoleRouter.from_env()
        router.get(Role.VISION_CRITIC)
        router.get_panel(Role.VISION_ANALYST)
        assert router._cached_single
        assert router._cached_panel

        asyncio.run(router.close())
        assert not router._cached_single
        assert not router._cached_panel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/core/test_llm_routing.py::TestClose -v`
Expected: FAIL with `AttributeError: 'RoleRouter' object has no attribute 'close'`

- [ ] **Step 3: Write minimal implementation**

Add `close` to `RoleRouter` in `src/styleclaw/core/llm_routing.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/core/test_llm_routing.py -v`
Expected: PASS — `TestClose` adds 3 new tests.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/llm_routing.py tests/core/test_llm_routing.py
git commit -m "$(cat <<'EOF'
feat(llm_routing): RoleRouter.close — idempotent provider teardown

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Wrap-up — full test sweep + sanity import

**Files:**
- No code changes; this is a verification + final commit checkpoint.

- [ ] **Step 1: Run the full new test file**

Run: `uv run python -m pytest tests/core/test_llm_routing.py -v`
Expected: PASS — every test added in Tasks 1–6. Approximate count: 3 + 4 + 5 + 3 + 3 + 3 = 21 tests.

- [ ] **Step 2: Run the full test suite to confirm zero regressions**

Run: `uv run python -m pytest tests/ -x -q`
Expected: PASS — the new module is not yet imported by any production code, so this part can only break things via the new test file itself.

- [ ] **Step 3: Sanity check — import the module from a Python shell**

Run:
```bash
uv run python -c "from styleclaw.core.llm_routing import Role, RoleConfig, RoleRouter; print(list(Role), RoleRouter.from_env())"
```
Expected output (env may vary; the key part is no exceptions and four Role entries):
```
[<Role.VISION_CRITIC: 'vision_critic'>, <Role.VISION_ANALYST: 'vision_analyst'>, <Role.WRITER: 'writer'>, <Role.PLANNER: 'planner'>] <styleclaw.core.llm_routing.RoleRouter object at 0x...>
```

- [ ] **Step 4: Done — no further commit needed**

Part 1 is complete when Task 7 passes. The module is fully built and tested but not yet wired into any caller. Part 2 (validation + Pydantic field) and Part 3 (integration into ExecutionContext / cli / actions) can begin.
