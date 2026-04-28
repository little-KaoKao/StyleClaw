from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from styleclaw.core.models import (
    BatchConfig,
    ModelEvaluation,
    ProjectConfig,
    ProjectState,
    PromptConfig,
    RoundEvaluation,
    StyleAnalysis,
    TaskRecord,
    UploadRecord,
)

DATA_ROOT = Path(os.getenv("STYLECLAW_DATA_ROOT", "data/projects"))

_PROJECT_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

T = TypeVar("T", bound=BaseModel)


def _validate_project_name(name: str) -> None:
    if not _PROJECT_NAME_RE.match(name):
        raise ValueError(
            f"Invalid project name '{name}'. "
            "Use only letters, digits, hyphens, and underscores."
        )


def _load_model(model_cls: type[T], path: Path) -> T:
    return model_cls.model_validate(_read_json(path))


def _save_model(model: BaseModel, path: Path) -> None:
    _write_json(path, model.model_dump())


def _load_all_records(results_dir: Path) -> dict[str, TaskRecord]:
    records: dict[str, TaskRecord] = {}
    if not results_dir.exists():
        return records
    for d in results_dir.iterdir():
        task_file = d / "task.json"
        if d.is_dir() and task_file.exists():
            records[d.name] = _load_model(TaskRecord, task_file)
    return records


def project_dir(name: str) -> Path:
    _validate_project_name(name)
    return DATA_ROOT / name


def create_project(config: ProjectConfig) -> Path:
    root = project_dir(config.name)
    if root.exists():
        raise FileExistsError(f"Project '{config.name}' already exists at {root}")

    root.mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "model-select" / "results").mkdir(parents=True)
    (root / "style-refine").mkdir()
    (root / "batch-t2i").mkdir()
    (root / "batch-i2i").mkdir()

    _write_json(root / "config.json", config.model_dump())
    state = ProjectState()
    _write_json(root / "state.json", state.model_dump())
    return root


def list_projects() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted(
        d.name for d in DATA_ROOT.iterdir() if d.is_dir() and (d / "config.json").exists()
    )


def save_config(name: str, config: ProjectConfig) -> None:
    _save_model(config, project_dir(name) / "config.json")


def load_config(name: str) -> ProjectConfig:
    return _load_model(ProjectConfig, project_dir(name) / "config.json")


def load_state(name: str) -> ProjectState:
    return _load_model(ProjectState, project_dir(name) / "state.json")


def save_state(name: str, state: ProjectState) -> None:
    _save_model(state, project_dir(name) / "state.json")


def load_uploads(name: str) -> list[UploadRecord]:
    path = project_dir(name) / "refs" / "uploads.json"
    if not path.exists():
        return []
    return [UploadRecord.model_validate(r) for r in _read_json(path)]


def save_uploads(name: str, records: list[UploadRecord]) -> None:
    _write_json(
        project_dir(name) / "refs" / "uploads.json",
        [r.model_dump() for r in records],
    )


def save_analysis(name: str, analysis: StyleAnalysis) -> None:
    _save_model(analysis, project_dir(name) / "model-select" / "initial-analysis.json")


def load_analysis(name: str) -> StyleAnalysis:
    return _load_model(StyleAnalysis, project_dir(name) / "model-select" / "initial-analysis.json")


def model_results_dir(name: str, model_id: str, variant: str = "") -> Path:
    if variant:
        d = project_dir(name) / "model-select" / "results" / model_id / variant
    else:
        d = project_dir(name) / "model-select" / "results" / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_task_record(name: str, model_id: str, record: TaskRecord, variant: str = "") -> None:
    _save_model(record, model_results_dir(name, model_id, variant) / "task.json")


def load_task_record(name: str, model_id: str, variant: str = "") -> TaskRecord:
    return _load_model(TaskRecord, model_results_dir(name, model_id, variant) / "task.json")


def _load_variant_records(results_dir: Path) -> dict[str, TaskRecord]:
    """Load task records from variant sub-dirs: results/<model>/<variant>/task.json"""
    records: dict[str, TaskRecord] = {}
    if not results_dir.exists():
        return records
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for variant_dir in model_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            task_file = variant_dir / "task.json"
            if task_file.exists():
                key = f"{model_dir.name}/{variant_dir.name}"
                records[key] = _load_model(TaskRecord, task_file)
    return records


def load_all_task_records(name: str) -> dict[str, TaskRecord]:
    results_dir = project_dir(name) / "model-select" / "results"
    records = _load_variant_records(results_dir)
    if records:
        return records
    return _load_all_records(results_dir)


def save_evaluation(name: str, evaluation: ModelEvaluation) -> None:
    _save_model(evaluation, project_dir(name) / "model-select" / "evaluation.json")


def load_evaluation(name: str) -> ModelEvaluation:
    return _load_model(ModelEvaluation, project_dir(name) / "model-select" / "evaluation.json")


# --- Phase 3: style-refine round-level storage ---


def round_dir(name: str, round_num: int) -> Path:
    d = project_dir(name) / "style-refine" / f"round-{round_num:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def round_results_dir(name: str, round_num: int, model_id: str) -> Path:
    d = round_dir(name, round_num) / "results" / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_prompt_config(name: str, round_num: int, config: PromptConfig) -> None:
    _save_model(config, round_dir(name, round_num) / "prompt.json")


def load_prompt_config(name: str, round_num: int) -> PromptConfig:
    return _load_model(PromptConfig, round_dir(name, round_num) / "prompt.json")


def save_round_task_record(
    name: str, round_num: int, model_id: str, record: TaskRecord,
) -> None:
    _save_model(record, round_results_dir(name, round_num, model_id) / "task.json")


def load_round_task_record(
    name: str, round_num: int, model_id: str,
) -> TaskRecord:
    return _load_model(TaskRecord, round_results_dir(name, round_num, model_id) / "task.json")


def load_all_round_task_records(
    name: str, round_num: int,
) -> dict[str, TaskRecord]:
    return _load_all_records(round_dir(name, round_num) / "results")


def save_round_evaluation(name: str, round_num: int, evaluation: RoundEvaluation) -> None:
    _save_model(evaluation, round_dir(name, round_num) / "evaluation.json")


def load_round_evaluation(name: str, round_num: int) -> RoundEvaluation:
    return _load_model(RoundEvaluation, round_dir(name, round_num) / "evaluation.json")


# --- Phase 4: batch-t2i storage ---


def batch_t2i_dir(name: str, batch_num: int) -> Path:
    d = project_dir(name) / "batch-t2i" / f"batch-{batch_num:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def batch_t2i_case_dir(name: str, batch_num: int, case_id: str) -> Path:
    d = batch_t2i_dir(name, batch_num) / "results" / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_batch_config(name: str, batch_num: int, config: BatchConfig) -> None:
    _save_model(config, batch_t2i_dir(name, batch_num) / "cases.json")


def load_batch_config(name: str, batch_num: int) -> BatchConfig:
    return _load_model(BatchConfig, batch_t2i_dir(name, batch_num) / "cases.json")


def save_batch_task_record(
    name: str, batch_num: int, case_id: str, record: TaskRecord,
) -> None:
    _save_model(record, batch_t2i_case_dir(name, batch_num, case_id) / "task.json")


def load_batch_task_record(
    name: str, batch_num: int, case_id: str,
) -> TaskRecord:
    return _load_model(TaskRecord, batch_t2i_case_dir(name, batch_num, case_id) / "task.json")


def load_all_batch_task_records(
    name: str, batch_num: int,
) -> dict[str, TaskRecord]:
    return _load_all_records(batch_t2i_dir(name, batch_num) / "results")


# --- Phase 5: batch-i2i storage ---


def batch_i2i_dir(name: str, batch_num: int) -> Path:
    d = project_dir(name) / "batch-i2i" / f"batch-{batch_num:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def batch_i2i_case_dir(name: str, batch_num: int, case_id: str) -> Path:
    d = batch_i2i_dir(name, batch_num) / "results" / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_i2i_uploads(name: str, batch_num: int, records: list[UploadRecord]) -> None:
    _write_json(
        batch_i2i_dir(name, batch_num) / "uploads.json",
        [r.model_dump() for r in records],
    )


def load_i2i_uploads(name: str, batch_num: int) -> list[UploadRecord]:
    path = batch_i2i_dir(name, batch_num) / "uploads.json"
    if not path.exists():
        return []
    return [UploadRecord.model_validate(r) for r in _read_json(path)]


def save_i2i_batch_config(name: str, batch_num: int, config: BatchConfig) -> None:
    _save_model(config, batch_i2i_dir(name, batch_num) / "cases.json")


def load_i2i_batch_config(name: str, batch_num: int) -> BatchConfig:
    return _load_model(BatchConfig, batch_i2i_dir(name, batch_num) / "cases.json")


def save_i2i_task_record(
    name: str, batch_num: int, case_id: str, record: TaskRecord,
) -> None:
    _save_model(record, batch_i2i_case_dir(name, batch_num, case_id) / "task.json")


def load_i2i_task_record(
    name: str, batch_num: int, case_id: str,
) -> TaskRecord:
    return _load_model(TaskRecord, batch_i2i_case_dir(name, batch_num, case_id) / "task.json")


def load_all_i2i_task_records(
    name: str, batch_num: int,
) -> dict[str, TaskRecord]:
    return _load_all_records(batch_i2i_dir(name, batch_num) / "results")


def _read_json(path: Path) -> dict | list:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupted JSON file: {path}: {exc}") from exc


def _write_json(path: Path, data: dict | list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
    except OSError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
