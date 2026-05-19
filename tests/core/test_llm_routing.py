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
