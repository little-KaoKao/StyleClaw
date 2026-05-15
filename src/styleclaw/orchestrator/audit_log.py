"""Per-run audit log: persist plan + execution outcome under
``data/projects/<name>/runs/<timestamp>/`` so we can later answer
"why did the orchestrator do X" without re-running the LLM.

Two files per run:

- ``plan.json``: the user intent + LLM-generated ActionPlan, captured
  before execution starts.
- ``execution-log.json``: per-step results (ok, message, elapsed),
  written incrementally so a crashed run still leaves partial trace.

Secrets are not written: the only inputs are the user's intent string,
the planner's structured output, and step results.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from styleclaw.core.models import ActionPlan
from styleclaw.storage import project_store

logger = logging.getLogger(__name__)


def _now_stamp() -> str:
    """UTC timestamp suitable for use as a directory name."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


@dataclass
class AuditLogger:
    """One instance per ``styleclaw run`` invocation.

    The directory is created lazily on first write so dry-runs / planning
    failures don't leave empty stub directories.
    """

    project: str
    intent: str
    started_at: str
    run_dir: Path
    _entries: list[dict[str, Any]]
    _step_started_at: dict[int, float]

    @classmethod
    def create(cls, project: str, intent: str) -> "AuditLogger":
        stamp = _now_stamp()
        return cls(
            project=project,
            intent=intent,
            started_at=stamp,
            run_dir=project_store.project_dir(project) / "runs" / stamp,
            _entries=[],
            _step_started_at={},
        )

    def record_plan(self, plan: ActionPlan) -> None:
        payload = {
            "project": self.project,
            "intent": self.intent,
            "started_at": self.started_at,
            "plan": plan.model_dump(),
        }
        _atomic_write(self.run_dir / "plan.json", payload)

    def step_started(self, index: int) -> None:
        self._step_started_at[index] = time.monotonic()

    def step_finished(
        self, index: int, name: str, ok: bool, message: str,
    ) -> None:
        elapsed = (
            time.monotonic() - self._step_started_at.pop(index, time.monotonic())
        )
        entry = {
            "index": index,
            "name": name,
            "ok": ok,
            "message": message,
            "elapsed_seconds": round(elapsed, 3),
        }
        self._entries.append(entry)
        # Flush after every step so a crash mid-run still leaves a trace.
        try:
            _atomic_write(
                self.run_dir / "execution-log.json",
                {
                    "project": self.project,
                    "started_at": self.started_at,
                    "steps": list(self._entries),
                },
            )
        except OSError as exc:
            # Don't let audit-log failures break execution — the user's
            # work is more important than the log.
            logger.warning("Failed to write audit log: %s", exc)

    def cancelled(self) -> None:
        """Record that the user cancelled at the confirmation prompt."""
        entry = {
            "index": -1,
            "name": "<cancelled>",
            "ok": False,
            "message": "User cancelled before execution.",
            "elapsed_seconds": 0.0,
        }
        try:
            _atomic_write(
                self.run_dir / "execution-log.json",
                {
                    "project": self.project,
                    "started_at": self.started_at,
                    "steps": [entry],
                },
            )
        except OSError as exc:
            logger.warning("Failed to write audit log: %s", exc)
