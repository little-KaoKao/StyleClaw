from __future__ import annotations

import pytest


def _reload_config():
    import importlib
    import styleclaw.core.config as config_mod
    importlib.reload(config_mod)
    return config_mod


class TestBuildPanelProviders:
    def test_raises_when_neither_toggle_on(self, monkeypatch):
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        _reload_config()
        from styleclaw.providers.llm.panel_factory import build_panel_providers
        with pytest.raises(RuntimeError, match="no panel toggle is enabled"):
            build_panel_providers()

    def test_raises_when_validation_fails(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "a,b")  # only 2
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        _reload_config()
        from styleclaw.providers.llm.panel_factory import build_panel_providers
        with pytest.raises(ValueError, match="STYLECLAW_PANEL_MODELS"):
            build_panel_providers()

    def test_returns_three_providers_with_distinct_model_ids(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.setenv("STYLECLAW_PANEL_LABELS", "One,Two,Three")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        _reload_config()
        from styleclaw.providers.llm.panel_factory import build_panel_providers
        pairs = build_panel_providers()
        assert [label for _, label in pairs] == ["One", "Two", "Three"]
        assert [p._model_id for p, _ in pairs] == ["m1", "m2", "m3"]
        # All share the same base URL.
        assert all(p._base_url == "http://x" for p, _ in pairs)

    def test_labels_fall_back_to_model_ids(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "m1,m2,m3")
        monkeypatch.delenv("STYLECLAW_PANEL_LABELS", raising=False)
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        _reload_config()
        from styleclaw.providers.llm.panel_factory import build_panel_providers
        pairs = build_panel_providers()
        assert [label for _, label in pairs] == ["m1", "m2", "m3"]
