from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar

from filelock import FileLock
from pydantic import BaseModel

from styleclaw.core.models import (
    BatchConfig,
    ModelEvaluation,
    PanelResult,
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
_PROJECT_LOCK_TIMEOUT = 30  # seconds; a stuck process beyond this surfaces, not waits forever

T = TypeVar("T", bound=BaseModel)


def _validate_project_name(name: str) -> None:
    if not _PROJECT_NAME_RE.match(name):
        raise ValueError(
            f"Invalid project name '{name}'. "
            "Use only letters, digits, hyphens, and underscores."
        )


# --- Naming helpers ---
#
# These render the canonical zero-padded format we use for passes, rounds,
# and batches: ``pass-007`` / ``round-003`` / ``batch-012``. Centralizing
# them means a single edit changes the format everywhere — paths, log
# lines, user-facing messages, glob patterns. Use them everywhere instead
# of inlining ``f"pass-{n:03d}"``.


def pass_label(pass_num: int) -> str:
    """Canonical display label for a model-select pass, e.g. ``pass-007``."""
    return f"pass-{pass_num:03d}"


def round_label(round_num: int) -> str:
    """Canonical display label for a style-refine round, e.g. ``round-003``."""
    return f"round-{round_num:03d}"


def batch_label(batch_num: int) -> str:
    """Canonical display label for a batch, e.g. ``batch-012``."""
    return f"batch-{batch_num:03d}"


def round_glob_across_passes(round_num: int) -> str:
    """Glob pattern matching this round under any pass — useful for the
    rollback command's existence check that scans ``style-refine/pass-*/``."""
    return f"pass-*/{round_label(round_num)}"


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
    candidate = (DATA_ROOT / name).resolve()
    root = DATA_ROOT.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(
            f"Project path escapes data root: {candidate} is not under {root}"
        )
    return candidate


def create_project(config: ProjectConfig, force: bool = False) -> Path:
    root = project_dir(config.name)
    if root.exists():
        if not force:
            raise FileExistsError(f"Project '{config.name}' already exists at {root}")
        # Back up the existing project before recreating, so manual recovery
        # remains possible if the new run fails.
        from datetime import datetime
        from shutil import move
        backup = root.with_name(f"{root.name}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        move(str(root), str(backup))

    root.mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "model-select").mkdir()
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


@contextmanager
def project_lock(name: str, timeout: float = _PROJECT_LOCK_TIMEOUT) -> Iterator[None]:
    """Inter-process advisory lock scoped to a single project.

    Wraps any read-modify-write sequence that touches ``state.json``,
    ``config.json``, or other per-project files. Uses ``filelock`` so the
    same lock semantics hold on POSIX and Windows.

    Locks per-project on disk at ``<project>/.lock`` — concurrent invocations
    of ``styleclaw`` on different projects don't block each other.

    Times out after ``timeout`` seconds rather than blocking forever; that
    surfaces a stuck process as an error instead of hanging the CLI.
    """
    root = project_dir(name)
    # The project dir is created by ``create_project``; for the rare case
    # where a caller locks before init (defensive), create it first.
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".lock"
    lock = FileLock(str(lock_path), timeout=timeout)
    with lock:
        yield


def update_state(
    name: str,
    mutator: Callable[[ProjectState], ProjectState],
) -> ProjectState:
    """Atomic read-modify-write of ``state.json`` under the project lock.

    ``mutator`` receives the current state and must return a new state
    (typically via ``state.with_X(...)`` / ``model_copy(update=...)``).
    The lock ensures two concurrent ``styleclaw`` invocations cannot
    silently lose updates by both loading the same baseline.
    """
    with project_lock(name):
        state = load_state(name)
        new_state = mutator(state)
        save_state(name, new_state)
        return new_state


def update_config(
    name: str,
    mutator: Callable[[ProjectConfig], ProjectConfig],
) -> ProjectConfig:
    """Atomic read-modify-write of ``config.json`` under the project lock."""
    with project_lock(name):
        config = load_config(name)
        new_config = mutator(config)
        save_config(name, new_config)
        return new_config


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


def model_select_dir(name: str, pass_num: int) -> Path:
    if pass_num < 1:
        raise ValueError(f"pass_num must be >= 1, got {pass_num}")
    d = project_dir(name) / "model-select" / pass_label(pass_num)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_analysis(name: str, analysis: StyleAnalysis, pass_num: int = 1) -> None:
    _save_model(analysis, model_select_dir(name, pass_num) / "initial-analysis.json")


def load_analysis(name: str, pass_num: int = 1) -> StyleAnalysis:
    # pass_num >= 2 falls back to pass-1 so that a fresh retest-models pass
    # can still see the analysis that F2 seeded into the target pass directly.
    pass_path = model_select_dir(name, pass_num) / "initial-analysis.json"
    if pass_path.exists() or pass_num == 1:
        return _load_model(StyleAnalysis, pass_path)
    fallback = model_select_dir(name, 1) / "initial-analysis.json"
    if fallback.exists():
        return _load_model(StyleAnalysis, fallback)
    return _load_model(StyleAnalysis, pass_path)


def model_results_dir(
    name: str, model_id: str, variant: str = "", pass_num: int = 1,
) -> Path:
    base = model_select_dir(name, pass_num) / "results" / model_id
    d = base / variant if variant else base
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_task_record(
    name: str, model_id: str, record: TaskRecord,
    variant: str = "", pass_num: int = 1,
) -> None:
    _save_model(
        record,
        model_results_dir(name, model_id, variant, pass_num) / "task.json",
    )


def load_task_record(
    name: str, model_id: str, variant: str = "", pass_num: int = 1,
) -> TaskRecord:
    return _load_model(
        TaskRecord,
        model_results_dir(name, model_id, variant, pass_num) / "task.json",
    )


def _load_variant_records(results_dir: Path) -> dict[str, TaskRecord]:
    records: dict[str, TaskRecord] = {}
    if not results_dir.exists():
        return records
    # Compute fingerprint first, then check the cache: makes the intent
    # explicit ("here's the state we observed; do we already have records
    # matching it?") rather than the original "look up first, then ask
    # whether the lookup matches reality". Equivalent on a local FS where
    # mtime updates are atomic; clearer to read.
    fingerprint = _results_dir_fingerprint(results_dir)
    cached = _VARIANT_RECORD_CACHE.get(str(results_dir))
    if cached is not None and cached[0] == fingerprint:
        return dict(cached[1])
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
    _VARIANT_RECORD_CACHE[str(results_dir)] = (fingerprint, dict(records))
    return records


# Module-level memo. Cache entry = (fingerprint, dict). The fingerprint is a
# tuple of (results_dir_mtime_ns, max(task.json mtime_ns across all task files))
# so any change — adding a model, finishing a poll, retrying a failed task —
# invalidates the entry on the next call. Capped at 32 entries to keep memory
# bounded in long-running sessions; the LRU here is dead simple (oldest wins).
_VARIANT_RECORD_CACHE: dict[str, tuple[tuple[int, int], dict[str, TaskRecord]]] = {}
_VARIANT_RECORD_CACHE_MAX = 32


def _results_dir_fingerprint(results_dir: Path) -> tuple[int, int]:
    try:
        dir_mtime = results_dir.stat().st_mtime_ns
    except OSError:
        return (0, 0)
    max_task_mtime = 0
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for variant_dir in model_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            task_file = variant_dir / "task.json"
            try:
                m = task_file.stat().st_mtime_ns
            except OSError:
                continue
            if m > max_task_mtime:
                max_task_mtime = m
    # Evict the oldest entries if the cache grew unbounded.
    if len(_VARIANT_RECORD_CACHE) > _VARIANT_RECORD_CACHE_MAX:
        for old_key in list(_VARIANT_RECORD_CACHE)[:-_VARIANT_RECORD_CACHE_MAX]:
            _VARIANT_RECORD_CACHE.pop(old_key, None)
    return (dir_mtime, max_task_mtime)


def load_all_task_records(name: str, pass_num: int = 1) -> dict[str, TaskRecord]:
    results_dir = model_select_dir(name, pass_num) / "results"
    records = _load_variant_records(results_dir)
    if records:
        return records
    return _load_all_records(results_dir)


def save_evaluation(name: str, evaluation: ModelEvaluation, pass_num: int = 1) -> None:
    _save_model(evaluation, model_select_dir(name, pass_num) / "evaluation.json")


def load_evaluation(name: str, pass_num: int = 1) -> ModelEvaluation:
    return _load_model(
        ModelEvaluation,
        model_select_dir(name, pass_num) / "evaluation.json",
    )


# --- Phase 3: style-refine round-level storage ---
#
# Rounds are pass-scoped: writes always go to pass-{current}/round-{r}/. Reads
# start at the current pass and walk down toward pass-1 so that a round
# created in an earlier pass is still visible after `retest-models` advances
# the project to a new pass (current_round is shared across passes).


def round_dir(name: str, round_num: int, pass_num: int = 1) -> Path:
    if pass_num < 1:
        raise ValueError(f"pass_num must be >= 1, got {pass_num}")
    d = (
        project_dir(name) / "style-refine"
        / pass_label(pass_num) / round_label(round_num)
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def round_results_dir(
    name: str, round_num: int, model_id: str, pass_num: int = 1,
) -> Path:
    d = round_dir(name, round_num, pass_num) / "results" / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_round_file(
    name: str, round_num: int, pass_num: int, filename: str,
) -> Path:
    """Find a round-scoped file by walking passes from `pass_num` down to 1.
    Returns the pass-`pass_num` path (not yet existing) when nothing is found,
    so callers get a sensible error location.
    """
    for p in range(pass_num, 0, -1):
        candidate = round_dir(name, round_num, p) / filename
        if candidate.exists():
            return candidate
    return round_dir(name, round_num, pass_num) / filename


def _resolve_round_subdir(
    name: str, round_num: int, pass_num: int, subdir: str,
) -> Path:
    for p in range(pass_num, 0, -1):
        candidate = round_dir(name, round_num, p) / subdir
        if candidate.exists():
            return candidate
    return round_dir(name, round_num, pass_num) / subdir


def save_prompt_config(
    name: str, round_num: int, config: PromptConfig, pass_num: int = 1,
) -> None:
    _save_model(config, round_dir(name, round_num, pass_num) / "prompt.json")


def load_prompt_config(
    name: str, round_num: int, pass_num: int = 1,
) -> PromptConfig:
    return _load_model(
        PromptConfig, _resolve_round_file(name, round_num, pass_num, "prompt.json"),
    )


def save_round_task_record(
    name: str, round_num: int, model_id: str, record: TaskRecord,
    pass_num: int = 1,
) -> None:
    _save_model(
        record,
        round_results_dir(name, round_num, model_id, pass_num) / "task.json",
    )


def load_round_task_record(
    name: str, round_num: int, model_id: str, pass_num: int = 1,
) -> TaskRecord:
    pass_path = (
        round_results_dir(name, round_num, model_id, pass_num) / "task.json"
    )
    if pass_path.exists():
        return _load_model(TaskRecord, pass_path)
    for p in range(pass_num - 1, 0, -1):
        candidate = round_dir(name, round_num, p) / "results" / model_id / "task.json"
        if candidate.exists():
            return _load_model(TaskRecord, candidate)
    return _load_model(TaskRecord, pass_path)


def load_all_round_task_records(
    name: str, round_num: int, pass_num: int = 1,
) -> dict[str, TaskRecord]:
    results_dir = _resolve_round_subdir(name, round_num, pass_num, "results")
    return _load_all_records(results_dir)


def save_round_evaluation(
    name: str, round_num: int, evaluation: RoundEvaluation, pass_num: int = 1,
) -> None:
    _save_model(
        evaluation, round_dir(name, round_num, pass_num) / "evaluation.json",
    )


def load_round_evaluation(
    name: str, round_num: int, pass_num: int = 1,
) -> RoundEvaluation:
    return _load_model(
        RoundEvaluation,
        _resolve_round_file(name, round_num, pass_num, "evaluation.json"),
    )


# --- Panel sidecar storage (model-select and style-refine rounds) ---


def save_round_panel_result(
    name: str, round_num: int, result: PanelResult, pass_num: int = 1,
) -> None:
    _save_model(result, round_dir(name, round_num, pass_num) / "panel.json")


def load_round_panel_result(
    name: str, round_num: int, pass_num: int = 1,
) -> PanelResult | None:
    path = round_dir(name, round_num, pass_num) / "panel.json"
    if not path.exists():
        return None
    return _load_model(PanelResult, path)


def save_model_select_panel_result(
    name: str, result: PanelResult, pass_num: int = 1,
) -> None:
    _save_model(result, model_select_dir(name, pass_num) / "panel.json")


def load_model_select_panel_result(
    name: str, pass_num: int = 1,
) -> PanelResult | None:
    path = model_select_dir(name, pass_num) / "panel.json"
    if not path.exists():
        return None
    return _load_model(PanelResult, path)


# --- Phase 4: batch-t2i storage ---


def batch_t2i_dir(name: str, batch_num: int) -> Path:
    d = project_dir(name) / "batch-t2i" / batch_label(batch_num)
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
    d = project_dir(name) / "batch-i2i" / batch_label(batch_num)
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


def save_thinking(json_path: Path, thinking: str) -> None:
    """Write thinking text to a sibling .thinking.md file.

    No-op when thinking is empty. Called by agents after saving their JSON output.
    """
    if not thinking:
        return
    md_path = json_path.with_suffix(".thinking.md")
    md_path.write_text(thinking, encoding="utf-8")


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
