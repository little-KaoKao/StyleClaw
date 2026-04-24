from __future__ import annotations

import json
import os
from pathlib import Path

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


def project_dir(name: str) -> Path:
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
    _write_json(project_dir(name) / "config.json", config.model_dump())


def load_config(name: str) -> ProjectConfig:
    path = project_dir(name) / "config.json"
    return ProjectConfig.model_validate(_read_json(path))


def load_state(name: str) -> ProjectState:
    path = project_dir(name) / "state.json"
    return ProjectState.model_validate(_read_json(path))


def save_state(name: str, state: ProjectState) -> None:
    _write_json(project_dir(name) / "state.json", state.model_dump())


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
    _write_json(
        project_dir(name) / "model-select" / "initial-analysis.json",
        analysis.model_dump(),
    )


def load_analysis(name: str) -> StyleAnalysis:
    path = project_dir(name) / "model-select" / "initial-analysis.json"
    return StyleAnalysis.model_validate(_read_json(path))


def model_results_dir(name: str, model_id: str) -> Path:
    d = project_dir(name) / "model-select" / "results" / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_task_record(name: str, model_id: str, record: TaskRecord) -> None:
    d = model_results_dir(name, model_id)
    _write_json(d / "task.json", record.model_dump())


def load_task_record(name: str, model_id: str) -> TaskRecord:
    d = model_results_dir(name, model_id)
    return TaskRecord.model_validate(_read_json(d / "task.json"))


def load_all_task_records(name: str) -> dict[str, TaskRecord]:
    results_dir = project_dir(name) / "model-select" / "results"
    records: dict[str, TaskRecord] = {}
    if not results_dir.exists():
        return records
    for d in results_dir.iterdir():
        task_file = d / "task.json"
        if d.is_dir() and task_file.exists():
            records[d.name] = TaskRecord.model_validate(_read_json(task_file))
    return records


def save_evaluation(name: str, evaluation: ModelEvaluation) -> None:
    _write_json(
        project_dir(name) / "model-select" / "evaluation.json",
        evaluation.model_dump(),
    )


def load_evaluation(name: str) -> ModelEvaluation:
    path = project_dir(name) / "model-select" / "evaluation.json"
    return ModelEvaluation.model_validate(_read_json(path))


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
    _write_json(round_dir(name, round_num) / "prompt.json", config.model_dump())


def load_prompt_config(name: str, round_num: int) -> PromptConfig:
    path = round_dir(name, round_num) / "prompt.json"
    return PromptConfig.model_validate(_read_json(path))


def save_round_task_record(
    name: str, round_num: int, model_id: str, record: TaskRecord,
) -> None:
    d = round_results_dir(name, round_num, model_id)
    _write_json(d / "task.json", record.model_dump())


def load_round_task_record(
    name: str, round_num: int, model_id: str,
) -> TaskRecord:
    d = round_results_dir(name, round_num, model_id)
    return TaskRecord.model_validate(_read_json(d / "task.json"))


def load_all_round_task_records(
    name: str, round_num: int,
) -> dict[str, TaskRecord]:
    results = round_dir(name, round_num) / "results"
    records: dict[str, TaskRecord] = {}
    if not results.exists():
        return records
    for d in results.iterdir():
        task_file = d / "task.json"
        if d.is_dir() and task_file.exists():
            records[d.name] = TaskRecord.model_validate(_read_json(task_file))
    return records


def save_round_evaluation(name: str, round_num: int, evaluation: RoundEvaluation) -> None:
    _write_json(round_dir(name, round_num) / "evaluation.json", evaluation.model_dump())


def load_round_evaluation(name: str, round_num: int) -> RoundEvaluation:
    path = round_dir(name, round_num) / "evaluation.json"
    return RoundEvaluation.model_validate(_read_json(path))


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
    _write_json(batch_t2i_dir(name, batch_num) / "cases.json", config.model_dump())


def load_batch_config(name: str, batch_num: int) -> BatchConfig:
    path = batch_t2i_dir(name, batch_num) / "cases.json"
    return BatchConfig.model_validate(_read_json(path))


def save_batch_task_record(
    name: str, batch_num: int, case_id: str, record: TaskRecord,
) -> None:
    d = batch_t2i_case_dir(name, batch_num, case_id)
    _write_json(d / "task.json", record.model_dump())


def load_batch_task_record(
    name: str, batch_num: int, case_id: str,
) -> TaskRecord:
    d = batch_t2i_case_dir(name, batch_num, case_id)
    return TaskRecord.model_validate(_read_json(d / "task.json"))


def load_all_batch_task_records(
    name: str, batch_num: int,
) -> dict[str, TaskRecord]:
    results = batch_t2i_dir(name, batch_num) / "results"
    records: dict[str, TaskRecord] = {}
    if not results.exists():
        return records
    for d in results.iterdir():
        task_file = d / "task.json"
        if d.is_dir() and task_file.exists():
            records[d.name] = TaskRecord.model_validate(_read_json(task_file))
    return records


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
    _write_json(batch_i2i_dir(name, batch_num) / "cases.json", config.model_dump())


def load_i2i_batch_config(name: str, batch_num: int) -> BatchConfig:
    path = batch_i2i_dir(name, batch_num) / "cases.json"
    return BatchConfig.model_validate(_read_json(path))


def save_i2i_task_record(
    name: str, batch_num: int, case_id: str, record: TaskRecord,
) -> None:
    d = batch_i2i_case_dir(name, batch_num, case_id)
    _write_json(d / "task.json", record.model_dump())


def load_i2i_task_record(
    name: str, batch_num: int, case_id: str,
) -> TaskRecord:
    d = batch_i2i_case_dir(name, batch_num, case_id)
    return TaskRecord.model_validate(_read_json(d / "task.json"))


def load_all_i2i_task_records(
    name: str, batch_num: int,
) -> dict[str, TaskRecord]:
    results = batch_i2i_dir(name, batch_num) / "results"
    records: dict[str, TaskRecord] = {}
    if not results.exists():
        return records
    for d in results.iterdir():
        task_file = d / "task.json"
        if d.is_dir() and task_file.exists():
            records[d.name] = TaskRecord.model_validate(_read_json(task_file))
    return records


def _read_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
