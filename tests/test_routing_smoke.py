"""End-to-end smoke for per-role LLM routing.

Exercises the full env-var → RoleRouter → provider pipeline without making
any network call. Provider construction is real (creates an httpx client),
but `close()` tears it down promptly via the existing teardown path.
"""
from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip all routing envs before each test so leaks from other tests
    don't poison the smoke."""
    from styleclaw.core.llm_routing import Role

    for role in Role:
        monkeypatch.delenv(f"STYLECLAW_MODEL_{role.value.upper()}", raising=False)
        monkeypatch.delenv(f"STYLECLAW_PANEL_MODELS_{role.value.upper()}", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("STYLECLAW_PANEL_MODELS", raising=False)
    monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
    monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
    # Force OpenAI-compat path; no real request will be made.
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:9/v1")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "smoke-test-key")
    monkeypatch.delenv("RUNNINGHUB_LLM", raising=False)
    yield


def test_smoke_each_role_resolves_to_its_env_model(monkeypatch):
    """Set a distinct model per role; verify provider._model_id matches."""
    monkeypatch.setenv("STYLECLAW_MODEL_VISION_CRITIC", "model-critic")
    monkeypatch.setenv("STYLECLAW_MODEL_VISION_ANALYST", "model-analyst")
    monkeypatch.setenv("STYLECLAW_MODEL_WRITER", "model-writer")
    monkeypatch.setenv("STYLECLAW_MODEL_PLANNER", "model-planner")

    from styleclaw.core.llm_routing import Role, RoleRouter
    router = RoleRouter.from_env()
    try:
        assert router.get(Role.VISION_CRITIC)._model_id == "model-critic"
        assert router.get(Role.VISION_ANALYST)._model_id == "model-analyst"
        assert router.get(Role.WRITER)._model_id == "model-writer"
        assert router.get(Role.PLANNER)._model_id == "model-planner"
    finally:
        asyncio.run(router.close())


def test_smoke_fallback_to_llm_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "global-default")

    from styleclaw.core.llm_routing import Role, RoleRouter
    router = RoleRouter.from_env()
    try:
        for role in Role:
            assert router.get(role)._model_id == "global-default", role
    finally:
        asyncio.run(router.close())


def test_smoke_panel_pool_resolves_per_role(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "default")  # silences single-model validation
    monkeypatch.setenv("STYLECLAW_PANEL_MODELS_VISION_CRITIC", "c1,c2,c3")
    monkeypatch.setenv("STYLECLAW_PANEL_MODELS_VISION_ANALYST", "a1,a2,a3")

    from styleclaw.core.llm_routing import Role, RoleRouter
    router = RoleRouter.from_env()
    try:
        critic_llms, _ = router.get_panel(Role.VISION_CRITIC)
        analyst_llms, _ = router.get_panel(Role.VISION_ANALYST)
        assert [p._model_id for p in critic_llms] == ["c1", "c2", "c3"]
        assert [p._model_id for p in analyst_llms] == ["a1", "a2", "a3"]
    finally:
        asyncio.run(router.close())


def test_smoke_validate_env_passes_with_all_role_envs(monkeypatch):
    """A realistic happy-path config: role envs set, no panel mode, all
    provider creds present. validate_env should return [] (no errors)."""
    monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
    monkeypatch.setenv("STYLECLAW_MODEL_VISION_CRITIC", "x")
    monkeypatch.setenv("STYLECLAW_MODEL_VISION_ANALYST", "y")
    monkeypatch.setenv("STYLECLAW_MODEL_WRITER", "z")
    monkeypatch.setenv("STYLECLAW_MODEL_PLANNER", "w")

    from styleclaw.core.config import validate_env
    assert validate_env() == []
