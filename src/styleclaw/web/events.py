from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class _Event(BaseModel):
    model_config = {"frozen": True}


class RunStartedEvent(_Event):
    type: Literal["run_started"] = "run_started"
    run_id: str
    project: str
    kind: str  # "plan" | "phase" | "action"
    steps: list[str]


class StepStartEvent(_Event):
    type: Literal["step_start"] = "step_start"
    index: int
    name: str
    description: str


class LlmDeltaEvent(_Event):
    type: Literal["llm_delta"] = "llm_delta"
    step_index: int
    role: str
    text: str


class StepDoneEvent(_Event):
    type: Literal["step_done"] = "step_done"
    index: int
    name: str
    status: str  # "ok" | "fail"
    summary: str


class NeedsHumanEvent(_Event):
    type: Literal["needs_human"] = "needs_human"
    round: int
    weakest_dim: str
    score: float
    suggestion: str


class PhasePausedEvent(_Event):
    type: Literal["phase_paused"] = "phase_paused"
    phase: str
    next_phase: str


class DoneEvent(_Event):
    type: Literal["done"] = "done"
    run_id: str


class ErrorEvent(_Event):
    type: Literal["error"] = "error"
    message: str
    detail: str = ""
