from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Phase(StrEnum):
    INIT = "INIT"
    MODEL_SELECT = "MODEL_SELECT"
    STYLE_REFINE = "STYLE_REFINE"
    BATCH_T2I = "BATCH_T2I"
    BATCH_I2I = "BATCH_I2I"
    COMPLETED = "COMPLETED"


class TaskStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class HistoryEntry(_FrozenModel):
    phase: Phase
    completed_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectConfig(_FrozenModel):
    name: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""
    ip_info: str = ""
    ref_images: list[str] = Field(default_factory=list)


class UploadRecord(_FrozenModel):
    local_path: str
    url: str
    file_name: str


class TaskRecord(_FrozenModel):
    task_id: str
    model_id: str
    status: str = TaskStatus.QUEUED
    prompt: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    results: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str = ""


class StyleAnalysis(_FrozenModel):
    color_palette: str = ""
    line_style: str = ""
    lighting: str = ""
    texture: str = ""
    composition: str = ""
    mood: str = ""
    trigger_phrase: str = ""
    trigger_variants: list[str] = Field(default_factory=list)
    model_suggestions: list[str] = Field(default_factory=list)


class DimensionScores(_FrozenModel):
    color_palette: float = 0.0
    line_style: float = 0.0
    lighting: float = 0.0
    texture: float = 0.0
    overall_mood: float = 0.0

    def _all_scores(self) -> list[float]:
        return [self.color_palette, self.line_style, self.lighting, self.texture, self.overall_mood]

    def average(self) -> float:
        vals = self._all_scores()
        return sum(vals) / len(vals)

    def min_score(self) -> float:
        return min(self._all_scores())

    def all_above(self, threshold: float) -> bool:
        return self.min_score() >= threshold


class ModelScore(_FrozenModel):
    model: str
    variant: str = ""
    image: str = ""
    scores: DimensionScores = Field(default_factory=DimensionScores)
    total: float = 0.0
    analysis: str = ""
    suggestions: str = ""


class ModelEvaluation(_FrozenModel):
    evaluations: list[ModelScore] = Field(default_factory=list)
    recommendation: str = ""
    recommended_variant: str = ""
    next_direction: str = ""


class PromptConfig(_FrozenModel):
    round: int = 0
    trigger_phrase: str = ""
    model_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    derived_from: str = ""
    adjustment_note: str = ""


class RoundScore(_FrozenModel):
    model: str
    image: str = ""
    scores: DimensionScores = Field(default_factory=DimensionScores)
    total: float = 0.0
    analysis: str = ""
    suggestions: str = ""


class RoundEvaluation(_FrozenModel):
    round: int = 0
    evaluations: list[RoundScore] = Field(default_factory=list)
    recommendation: str = ""
    next_direction: str = ""

    def should_approve(self) -> bool:
        if not self.evaluations:
            return False
        return all(
            e.scores.all_above(7.0) and e.total >= 7.5
            for e in self.evaluations
        )

    def needs_human(self) -> bool:
        return any(
            e.scores.min_score() < 5.0
            for e in self.evaluations
        )


class BatchCase(_FrozenModel):
    id: str
    category: str
    description: str
    aspect_ratio: str = "9:16"
    status: str = "pending"


class BatchConfig(_FrozenModel):
    batch: int = 0
    trigger_phrase: str = ""
    cases: list[BatchCase] = Field(default_factory=list)


class ProjectState(_FrozenModel):
    phase: Phase = Phase.INIT
    selected_models: list[str] = Field(default_factory=list)
    current_round: int = 0
    current_batch: int = 0
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    history: list[HistoryEntry] = Field(default_factory=list)

    def with_phase(self, phase: Phase, metadata: dict[str, Any] | None = None) -> ProjectState:
        now = datetime.now(timezone.utc).isoformat()
        return self.model_copy(update={
            "phase": phase,
            "last_updated": now,
            "history": [
                *self.history,
                HistoryEntry(
                    phase=self.phase,
                    completed_at=now,
                    metadata=metadata or {},
                ),
            ],
        })

    def with_selected_models(self, models: list[str]) -> ProjectState:
        return self.model_copy(update={
            "selected_models": models,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })

    def with_round(self, round_num: int) -> ProjectState:
        return self.model_copy(update={
            "current_round": round_num,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })

    def with_batch(self, batch_num: int) -> ProjectState:
        return self.model_copy(update={
            "current_batch": batch_num,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })


class Action(BaseModel):
    name: str
    description: str
    args: dict[str, Any] = Field(default_factory=dict)


class LoopConfig(BaseModel):
    start_step: int
    end_step: int
    max_iterations: int = 5


class ActionPlan(BaseModel):
    summary: str
    steps: list[Action]
    loop: LoopConfig | None = None
