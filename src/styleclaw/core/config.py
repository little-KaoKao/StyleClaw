from __future__ import annotations

import os

MAX_AUTO_ROUNDS: int = int(os.getenv("STYLECLAW_MAX_ROUNDS", "5"))
CONCURRENCY_LIMIT: int = int(os.getenv("STYLECLAW_CONCURRENCY", "5"))
TASK_TIMEOUT: float = float(os.getenv("STYLECLAW_TASK_TIMEOUT", "300"))
POLL_INTERVAL: float = float(os.getenv("STYLECLAW_POLL_INTERVAL", "3"))
ORCHESTRATOR_POLL_INTERVAL: float = float(os.getenv("STYLECLAW_ORCH_POLL_INTERVAL", "30"))
MAX_POLL_CYCLES: int = int(os.getenv("STYLECLAW_MAX_POLL_CYCLES", "60"))
