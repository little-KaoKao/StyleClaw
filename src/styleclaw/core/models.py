from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from styleclaw.core.time_utils import utcnow_iso as _utcnow_iso


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
    created_at: str = Field(default_factory=_utcnow_iso)
    description: str = ""
    ip_info: str = ""
    ref_images: list[str] = Field(default_factory=list)
    sref_index: int = 0  # index into ref_images / uploads used as style reference


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
    created_at: str = Field(default_factory=_utcnow_iso)
    completed_at: str = ""
    # Endpoint used for the original submission. Empty string for older task
    # records that pre-date this field; callers that need to resubmit should
    # fall back to the model's t2i_endpoint when this is empty.
    endpoint: str = ""


class StyleAnalysis(_FrozenModel):
    # 7个核心维度
    visual_style: str = ""
    color_science: str = ""
    lighting_quality: str = ""
    material_texture: str = ""
    post_processing: str = ""
    spatial_perspective: str = ""
    dynamic_state: str = ""
    # 输出
    trigger_phrase: str = ""
    trigger_variants: list[str] = Field(default_factory=list)
    # De-IP'd character descriptions extracted from the refs, keyed by gender
    # ("male" / "female"). Used as character_desc during MODEL_SELECT generate.
    # Empty dict / missing key falls back to the hardcoded TEST_SUBJECTS in
    # scripts/generate.py.
    test_subjects: dict[str, str] = Field(default_factory=dict)
    model_id: str = ""


class DimensionScores(_FrozenModel):
    """7 scoring dimensions, aligned 1:1 with the analyze.md analysis schema.

    Old projects saved with the legacy 5-dim schema (color_palette / line_style
    / lighting / texture / overall_mood) will load with the legacy fields
    dropped and the new fields defaulting to 0.0 — the resulting averages will
    fail the approval threshold, so stale evaluations must be re-run.
    """
    visual_style: float = 0.0
    color_science: float = 0.0
    lighting_quality: float = 0.0
    material_texture: float = 0.0
    post_processing: float = 0.0
    spatial_perspective: float = 0.0
    dynamic_state: float = 0.0

    def _all_scores(self) -> list[float]:
        return [
            self.visual_style,
            self.color_science,
            self.lighting_quality,
            self.material_texture,
            self.post_processing,
            self.spatial_perspective,
            self.dynamic_state,
        ]

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
    model_id: str = ""


class PromptConfig(_FrozenModel):
    round: int = 0
    trigger_phrase: str = ""
    model_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    derived_from: str = ""
    adjustment_note: str = ""
    model_id: str = ""


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
    model_id: str = ""

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
    model_id: str = ""


class ProjectState(_FrozenModel):
    phase: Phase = Phase.INIT
    selected_models: list[str] = Field(default_factory=list)
    selected_variant: str = "prompt-sref"  # variant chosen at select-model: "prompt-sref" or "prompt-only"
    current_round: int = 0
    current_batch: int = 0
    current_model_select_pass: int = 0
    last_updated: str = Field(default_factory=_utcnow_iso)
    history: list[HistoryEntry] = Field(default_factory=list)

    def with_phase(self, phase: Phase, metadata: dict[str, Any] | None = None) -> ProjectState:
        now = _utcnow_iso()
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

    def with_selected_models(self, models: list[str], variant: str = "") -> ProjectState:
        update: dict[str, Any] = {
            "selected_models": models,
            "last_updated": _utcnow_iso(),
        }
        if variant:
            update["selected_variant"] = variant
        return self.model_copy(update=update)

    def with_round(self, round_num: int) -> ProjectState:
        return self.model_copy(update={
            "current_round": round_num,
            "last_updated": _utcnow_iso(),
        })

    def with_batch(self, batch_num: int) -> ProjectState:
        return self.model_copy(update={
            "current_batch": batch_num,
            "last_updated": _utcnow_iso(),
        })

    def with_model_select_pass(self, pass_num: int) -> ProjectState:
        return self.model_copy(update={
            "current_model_select_pass": pass_num,
            "last_updated": _utcnow_iso(),
        })


class Action(_FrozenModel):
    name: str
    description: str
    args: dict[str, Any] = Field(default_factory=dict)


class LoopConfig(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    start_step: int = Field(alias="from", validation_alias=AliasChoices("start_step", "from"))
    end_step: int = Field(alias="to", validation_alias=AliasChoices("end_step", "to"))
    max_iterations: int = 5
    condition: str = ""


class ActionPlan(_FrozenModel):
    summary: str
    steps: list[Action]
    loop: LoopConfig | None = None
    stop_summary: str = ""


class PanelProposal(_FrozenModel):
    """One participant's submitted artifact in a panel round."""
    model_id: str
    label: str = ""
    payload: dict[str, Any]
    thinking: str = ""


class PanelScore(_FrozenModel):
    """One evaluator's score for one proposal (never the evaluator's own)."""
    evaluator_model_id: str
    target_model_id: str
    score: float
    rationale: str = ""


class PanelResult(_FrozenModel):
    """Aggregate outcome of a three-model panel round."""
    proposals: list[PanelProposal] = Field(default_factory=list)
    scores: list[PanelScore] = Field(default_factory=list)
    winner_model_id: str = ""
    averages: dict[str, float] = Field(default_factory=dict)
    degraded: bool = False
    error_log: list[str] = Field(default_factory=list)
