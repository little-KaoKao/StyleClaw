import os

import pytest


class TestConfigDefaults:
    def test_max_auto_rounds_default(self):
        from styleclaw.core.config import MAX_AUTO_ROUNDS
        assert MAX_AUTO_ROUNDS == 5

    def test_concurrency_limit_default(self):
        from styleclaw.core.config import CONCURRENCY_LIMIT
        assert CONCURRENCY_LIMIT == 5

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
