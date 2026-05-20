from styleclaw.core.config import validate_env


class TestValidateEnv:
    def test_openai_compat_satisfies_llm_requirement(self, monkeypatch) -> None:
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "dummy-model")  # satisfy per-role routing check
        assert validate_env() == []

    def test_no_llm_when_only_image_key(self, monkeypatch) -> None:
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
        errs = validate_env()
        assert any("OPENAI_COMPAT_API_KEY" in e for e in errs)

    def test_validate_env_reports_missing_role_models(self, monkeypatch):
        # All provider creds set so the existing checks pass, but no LLM_MODEL
        # and no role envs — validate_env should surface the routing errors.
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        for role_name in ("VISION_CRITIC", "VISION_ANALYST", "WRITER", "PLANNER"):
            monkeypatch.delenv(f"STYLECLAW_MODEL_{role_name}", raising=False)
        # validate_routing_env reads os.getenv directly — no config reload required.
        from styleclaw.core.config import validate_env
        errs = validate_env()
        assert any("vision_critic" in e for e in errs)


class TestConfigDefaults:
    def test_max_auto_rounds_default(self):
        from styleclaw.core.config import MAX_AUTO_ROUNDS
        assert MAX_AUTO_ROUNDS == 5

    def test_concurrency_limit_default(self):
        from styleclaw.core.config import CONCURRENCY_LIMIT
        assert CONCURRENCY_LIMIT == 10

    def test_task_timeout_default(self):
        from styleclaw.core.config import TASK_TIMEOUT
        assert TASK_TIMEOUT == 300.0

    def test_poll_interval_default(self):
        from styleclaw.core.config import POLL_INTERVAL
        assert POLL_INTERVAL == 3.0

    def test_orchestrator_poll_interval_default(self):
        from styleclaw.core.config import ORCHESTRATOR_POLL_INTERVAL
        assert ORCHESTRATOR_POLL_INTERVAL == 30.0

    def test_max_poll_cycles_default(self):
        from styleclaw.core.config import MAX_POLL_CYCLES
        assert MAX_POLL_CYCLES == 60

    def test_design_cases_shards_default(self):
        from styleclaw.core.config import DESIGN_CASES_SHARDS
        assert DESIGN_CASES_SHARDS == 5


class TestConfigEnvOverrides:
    def test_max_auto_rounds_from_env(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_MAX_ROUNDS", "10")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        assert config_mod.MAX_AUTO_ROUNDS == 10

    def test_concurrency_limit_from_env(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_CONCURRENCY", "20")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        assert config_mod.CONCURRENCY_LIMIT == 20

    def test_task_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_TASK_TIMEOUT", "600")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        assert config_mod.TASK_TIMEOUT == 600.0

    def test_poll_interval_from_env(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_POLL_INTERVAL", "5")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        assert config_mod.POLL_INTERVAL == 5.0

    def test_orchestrator_poll_interval_from_env(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_ORCH_POLL_INTERVAL", "60")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        assert config_mod.ORCHESTRATOR_POLL_INTERVAL == 60.0

    def test_max_poll_cycles_from_env(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_MAX_POLL_CYCLES", "120")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        assert config_mod.MAX_POLL_CYCLES == 120

    def test_design_cases_shards_from_env(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_DESIGN_CASES_SHARDS", "2")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        assert config_mod.DESIGN_CASES_SHARDS == 2


class TestStreamDisplayDefault:
    """STREAM_DISPLAY should default to True only when stdout is a TTY, so
    `print(delta, end='', flush=True)` doesn't blast token bytes into piped
    output, CI logs, or interleave across parallel async LLM calls."""

    def test_default_false_when_stdout_not_tty(self, monkeypatch) -> None:
        monkeypatch.delenv("STYLECLAW_STREAM_DISPLAY", raising=False)
        import sys
        # pytest captures stdout, so isatty() is already False here.
        assert sys.stdout.isatty() is False
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        assert config_mod.STREAM_DISPLAY is False

    def test_explicit_env_overrides_isatty(self, monkeypatch) -> None:
        monkeypatch.setenv("STYLECLAW_STREAM_DISPLAY", "1")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        assert config_mod.STREAM_DISPLAY is True

    def test_env_zero_disables(self, monkeypatch) -> None:
        monkeypatch.setenv("STYLECLAW_STREAM_DISPLAY", "0")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        assert config_mod.STREAM_DISPLAY is False


class TestPanelConfig:
    """STYLECLAW_PANEL_* validation.

    Both toggles default off. Either toggle on requires exactly 3 model ids;
    labels (if given) must match length. Both off → panel env is fully ignored.
    """

    def _reload(self):
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        return config_mod

    def test_both_off_ignores_panel_envs(self, monkeypatch):
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "only-one")
        monkeypatch.setenv("STYLECLAW_PANEL_LABELS", "A,B")  # mismatched len, ignored
        cfg = self._reload()
        assert cfg.PANEL_REFINE_ENABLED is False
        assert cfg.PANEL_MODEL_SELECT_ENABLED is False
        # Models list is parsed but not validated when both toggles off.
        # validate_panel_config() returns no errors.
        assert cfg.validate_panel_config() == []

    def test_refine_on_requires_three_models(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "a,b")
        monkeypatch.delenv("STYLECLAW_PANEL_LABELS", raising=False)
        cfg = self._reload()
        errs = cfg.validate_panel_config()
        assert any("STYLECLAW_PANEL_MODELS" in e and "exactly 3" in e for e in errs)

    def test_select_on_with_three_models_ok(self, monkeypatch):
        monkeypatch.delenv("STYLECLAW_PANEL_REFINE", raising=False)
        monkeypatch.setenv("STYLECLAW_PANEL_MODEL_SELECT", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "a, b ,c")
        monkeypatch.delenv("STYLECLAW_PANEL_LABELS", raising=False)
        cfg = self._reload()
        assert cfg.PANEL_MODELS == ["a", "b", "c"]
        assert cfg.validate_panel_config() == []

    def test_labels_must_match_length(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "a,b,c")
        monkeypatch.setenv("STYLECLAW_PANEL_LABELS", "Opus,GPT")
        cfg = self._reload()
        errs = cfg.validate_panel_config()
        assert any("STYLECLAW_PANEL_LABELS" in e for e in errs)

    def test_labels_default_to_model_ids(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "a,b,c")
        monkeypatch.delenv("STYLECLAW_PANEL_LABELS", raising=False)
        cfg = self._reload()
        assert cfg.PANEL_LABELS == ["a", "b", "c"]

    def test_validate_env_calls_panel_validator(self, monkeypatch):
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.setenv("STYLECLAW_PANEL_REFINE", "1")
        monkeypatch.setenv("STYLECLAW_PANEL_MODELS", "only-one")
        cfg = self._reload()
        errs = cfg.validate_env()
        assert any("STYLECLAW_PANEL_MODELS" in e for e in errs)


class TestValidateDesignCasesShards:
    def test_valid_values_accepted(self, monkeypatch):
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "dummy")
        for value in ("1", "2", "5", "10"):
            monkeypatch.setenv("STYLECLAW_DESIGN_CASES_SHARDS", value)
            import importlib
            import styleclaw.core.config as config_mod
            importlib.reload(config_mod)
            errs = config_mod.validate_env()
            assert not any("DESIGN_CASES_SHARDS" in e for e in errs), (
                f"value {value} should pass: {errs}"
            )

    def test_invalid_value_3_rejected(self, monkeypatch):
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "dummy")
        monkeypatch.setenv("STYLECLAW_DESIGN_CASES_SHARDS", "3")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        errs = config_mod.validate_env()
        assert any("DESIGN_CASES_SHARDS" in e and "3" in e for e in errs)

    def test_invalid_value_0_rejected(self, monkeypatch):
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "dummy")
        monkeypatch.setenv("STYLECLAW_DESIGN_CASES_SHARDS", "0")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        errs = config_mod.validate_env()
        assert any("DESIGN_CASES_SHARDS" in e for e in errs)
