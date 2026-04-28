from __future__ import annotations

import os


def _int_env(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Invalid value for {name}: '{raw}'. Expected an integer.") from None


def _float_env(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Invalid value for {name}: '{raw}'. Expected a number.") from None


MAX_AUTO_ROUNDS: int = _int_env("STYLECLAW_MAX_ROUNDS", "5")
CONCURRENCY_LIMIT: int = _int_env("STYLECLAW_CONCURRENCY", "5")
TASK_TIMEOUT: float = _float_env("STYLECLAW_TASK_TIMEOUT", "300")
POLL_INTERVAL: float = _float_env("STYLECLAW_POLL_INTERVAL", "3")
ORCHESTRATOR_POLL_INTERVAL: float = _float_env("STYLECLAW_ORCH_POLL_INTERVAL", "30")
MAX_POLL_CYCLES: int = _int_env("STYLECLAW_MAX_POLL_CYCLES", "60")
