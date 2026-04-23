from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Phase(StrEnum):
    INIT = "INIT"
    MODEL_SELECT = "MODEL_SELECT"
    STYLE_REFINE = "STYLE_REFINE"
    BATCH_T2I = "BATCH_T2I"
    BATCH_I2I = "BATCH_I2I"
    COMPLETED = "COMPLETED"


class HistoryEntry(BaseModel):
    phase: Phase
    completed_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectConfig(BaseModel):
    name: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""
    ip_info: str = ""
    ref_images: list[str] = Field(default_factory=list)


class UploadRecord(BaseModel):
    local_path: str
    url: str
    file_name: str


class TaskRecord(BaseModel):
    task_id: str
    model_id: str
    status: str = "QUEUED"
    prompt: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    results: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str = ""


class StyleAnalysis(BaseModel):
    color_palette: str = ""
    line_style: str = ""
    lighting: str = ""
    texture: str = ""
    composition: str = ""
    mood: str = ""
    trigger_phrase: str = ""
    trigger_variants: list[str] = Field(default_factory=list)
    model_suggestions: list[str] = Field(default_factory=list)


class ModelScore(BaseModel):
    model: str
    image: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    total: float = 0.0
    analysis: str = ""
    suggestions: str = ""


class ModelEvaluation(BaseModel):
    evaluations: list[ModelScore] = Field(default_factory=list)
    recommendation: str = ""
    next_direction: str = ""


class PromptConfig(BaseModel):
    round: int = 0
    trigger_phrase: str = ""
    model_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    derived_from: str = ""
    adjustment_note: str = ""


class DimensionScores(BaseModel):
    color_palette: float = 0.0
    line_style: float = 0.0
    lighting: float = 0.0
    texture: float = 0.0
    overall_mood: float = 0.0

    def average(self) -> float:
        vals = [self.color_palette, self.line_style, self.lighting, self.texture, self.overall_mood]
        return sum(vals) / len(vals)

    def min_score(self) -> float:
        return min(
            self.color_palette, self.line_style, self.lighting, self.texture, self.overall_mood
        )

    def all_above(self, threshold: float) -> bool:
        return self.min_score() >= threshold


class RoundScore(BaseModel):
    model: str
    image: str = ""
    scores: DimensionScores = Field(default_factory=DimensionScores)
    total: float = 0.0
    analysis: str = ""
    suggestions: str = ""


class RoundEvaluation(BaseModel):
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


class BatchCase(BaseModel):
    id: str
    category: str
    description: str
    aspect_ratio: str = "9:16"
    status: str = "pending"


class BatchConfig(BaseModel):
    batch: int = 0
    trigger_phrase: str = ""
    cases: list[BatchCase] = Field(default_factory=list)


class ProjectState(BaseModel):
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
