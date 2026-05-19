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
