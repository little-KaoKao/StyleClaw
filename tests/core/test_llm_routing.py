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
