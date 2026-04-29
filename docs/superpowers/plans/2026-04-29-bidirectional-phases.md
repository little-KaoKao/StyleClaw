# StyleClaw: Bidirectional Phase Transitions + MODEL_SELECT Re-entry

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Loosen the phase state machine so the user can re-test all models after refining the trigger phrase, and roll back forward (BATCH_I2I → BATCH_T2I) when batch results disappoint. MODEL_SELECT becomes a re-entrant phase with versioned "pass" folders (`pass-001/`, `pass-002/`, …) that mirror how `style-refine/round-NNN/` already works.

**Architecture:** (1) Add three new transitions to `TRANSITIONS`: `STYLE_REFINE → MODEL_SELECT`, `BATCH_T2I → MODEL_SELECT`, `BATCH_I2I → BATCH_T2I`. (2) `ProjectState` gains `current_model_select_pass: int`, auto-incremented every time we enter MODEL_SELECT. (3) `project_store` introduces `model_select_dir(name, pass_num)`; all model-select reads/writes become pass-scoped. Legacy flat layout (`model-select/results/…` without pass folder) is migrated on first read via a one-shot helper. (4) `do_analyze` runs the LLM only on pass 1; later passes reuse the original `StyleAnalysis` but use the current round's refined trigger phrase when generating. (5) New orchestrator action `retest-models` drives STYLE_REFINE → MODEL_SELECT. (6) CLI gets `retest-models` and an extended `rollback` that accepts forward-style fallback edges.

**Tech Stack:** Python 3.11+, Pydantic v2 (frozen models + `model_copy(update=...)`), Typer, pytest + respx. Follow existing frozen-model and `with_*` builder conventions.

**Assumption:** Part A (`2026-04-29-llm-thinking.md`) has already been merged. This plan builds on top of the `ExecutionContext` / `do_analyze` / `do_evaluate` shapes established there. If A is still in flight, hold this plan until A lands.

---

## File Structure

**Modify:**
- [src/styleclaw/core/state_machine.py](src/styleclaw/core/state_machine.py) — extend TRANSITIONS
- [src/styleclaw/core/models.py](src/styleclaw/core/models.py) — add `current_model_select_pass` + builder
- [src/styleclaw/storage/project_store.py](src/styleclaw/storage/project_store.py) — pass-scoped dirs, legacy fallback
- [src/styleclaw/scripts/generate.py](src/styleclaw/scripts/generate.py) — pick trigger from current round when pass > 1
- [src/styleclaw/scripts/poll.py](src/styleclaw/scripts/poll.py) — poll the current pass
- [src/styleclaw/scripts/report.py](src/styleclaw/scripts/report.py) — scope reports to pass
- [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py) — `do_analyze` skip on pass > 1; new `do_retest_models` and `do_back_to_t2i`; expanded `PHASE_ACTIONS`
- [src/styleclaw/cli.py](src/styleclaw/cli.py) — `retest-models`, `back-to-t2i` commands, status shows pass

**Create:**
- `tests/core/test_state_machine_bidirectional.py`
- `tests/core/test_model_select_pass.py`
- `tests/storage/test_pass_dirs.py`
- `tests/orchestrator/test_retest_models.py`

---

## Part B: Bidirectional Transitions + MODEL_SELECT Passes

### Task B1: Extend TRANSITIONS with three new edges

**Files:**
- Modify: [src/styleclaw/core/state_machine.py](src/styleclaw/core/state_machine.py)
- Create: `tests/core/test_state_machine_bidirectional.py`

Pure state machine change. No storage or agent changes. Three new edges:
- `STYLE_REFINE → MODEL_SELECT` (refined, re-test models)
- `BATCH_T2I → MODEL_SELECT` (batch disappointed, try different model)
- `BATCH_I2I → BATCH_T2I` (i2i underperformed, return to t2i for more trigger work)

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_state_machine_bidirectional.py`:

```python
import pytest

from styleclaw.core.models import Phase, ProjectState
from styleclaw.core.state_machine import advance, can_advance


class TestBidirectionalTransitions:
    def test_style_refine_to_model_select_allowed(self):
        assert can_advance(Phase.STYLE_REFINE, Phase.MODEL_SELECT) is True

    def test_batch_t2i_to_model_select_allowed(self):
        assert can_advance(Phase.BATCH_T2I, Phase.MODEL_SELECT) is True

    def test_batch_i2i_to_batch_t2i_allowed(self):
        assert can_advance(Phase.BATCH_I2I, Phase.BATCH_T2I) is True

    def test_advance_style_refine_to_model_select(self):
        state = ProjectState(phase=Phase.STYLE_REFINE, current_round=2)
        new_state = advance(state, Phase.MODEL_SELECT)
        assert new_state.phase == Phase.MODEL_SELECT
        # Existing fields preserved
        assert new_state.current_round == 2

    def test_advance_batch_t2i_to_model_select(self):
        state = ProjectState(phase=Phase.BATCH_T2I, current_batch=1)
        new_state = advance(state, Phase.MODEL_SELECT)
        assert new_state.phase == Phase.MODEL_SELECT
        assert new_state.current_batch == 1

    def test_advance_batch_i2i_to_batch_t2i(self):
        state = ProjectState(phase=Phase.BATCH_I2I, current_batch=1)
        new_state = advance(state, Phase.BATCH_T2I)
        assert new_state.phase == Phase.BATCH_T2I

    def test_existing_transitions_not_broken(self):
        """The three new edges must not accidentally block old ones."""
        assert can_advance(Phase.INIT, Phase.MODEL_SELECT) is True
        assert can_advance(Phase.MODEL_SELECT, Phase.STYLE_REFINE) is True
        assert can_advance(Phase.STYLE_REFINE, Phase.BATCH_T2I) is True
        assert can_advance(Phase.STYLE_REFINE, Phase.STYLE_REFINE) is True
        assert can_advance(Phase.BATCH_T2I, Phase.BATCH_I2I) is True
        assert can_advance(Phase.BATCH_I2I, Phase.COMPLETED) is True

    def test_invalid_transitions_still_blocked(self):
        assert can_advance(Phase.INIT, Phase.BATCH_T2I) is False
        assert can_advance(Phase.INIT, Phase.COMPLETED) is False
        assert can_advance(Phase.COMPLETED, Phase.INIT) is False
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/core/test_state_machine_bidirectional.py -v`
Expected: FAIL — e.g. `can_advance(STYLE_REFINE, MODEL_SELECT)` returns False.

- [ ] **Step 3: Edit TRANSITIONS**

In [src/styleclaw/core/state_machine.py](src/styleclaw/core/state_machine.py), replace the `TRANSITIONS` dict (lines 5–11) with:

```python
TRANSITIONS: dict[Phase, list[Phase]] = {
    Phase.INIT: [Phase.MODEL_SELECT],
    Phase.MODEL_SELECT: [Phase.STYLE_REFINE],
    Phase.STYLE_REFINE: [Phase.BATCH_T2I, Phase.STYLE_REFINE, Phase.MODEL_SELECT],
    Phase.BATCH_T2I: [Phase.BATCH_I2I, Phase.STYLE_REFINE, Phase.MODEL_SELECT],
    Phase.BATCH_I2I: [Phase.STYLE_REFINE, Phase.BATCH_T2I, Phase.COMPLETED],
}
```

- [ ] **Step 4: Run all state machine tests**

Run: `uv run python -m pytest tests/core/test_state_machine.py tests/core/test_state_machine_bidirectional.py -v`
Expected: PASS — 7 new + all existing.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/state_machine.py tests/core/test_state_machine_bidirectional.py
git commit -m "feat(state): allow STYLE_REFINE/BATCH_T2I->MODEL_SELECT and BATCH_I2I->BATCH_T2I"
```

---

### Task B2: ProjectState gains `current_model_select_pass`

**Files:**
- Modify: [src/styleclaw/core/models.py](src/styleclaw/core/models.py)
- Create: `tests/core/test_model_select_pass.py`

The field tracks which pass we're currently in. Pass 0 = "never entered"; pass 1 = first model-select (from INIT); pass 2+ = re-entries. A new builder `with_model_select_pass(n)` mirrors `with_round(n)` / `with_batch(n)`.

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_model_select_pass.py`:

```python
from styleclaw.core.models import Phase, ProjectState


class TestModelSelectPass:
    def test_default_is_zero(self):
        state = ProjectState()
        assert state.current_model_select_pass == 0

    def test_with_pass_returns_new_instance(self):
        state = ProjectState()
        new_state = state.with_model_select_pass(1)
        assert new_state.current_model_select_pass == 1
        # Immutability: old instance unchanged
        assert state.current_model_select_pass == 0

    def test_with_pass_updates_timestamp(self):
        state = ProjectState()
        original_ts = state.last_updated
        new_state = state.with_model_select_pass(2)
        assert new_state.last_updated != original_ts

    def test_with_pass_preserves_other_fields(self):
        state = ProjectState(
            phase=Phase.MODEL_SELECT,
            selected_models=["mj-v7"],
            current_round=3,
            current_batch=1,
        )
        new_state = state.with_model_select_pass(2)
        assert new_state.phase == Phase.MODEL_SELECT
        assert new_state.selected_models == ["mj-v7"]
        assert new_state.current_round == 3
        assert new_state.current_batch == 1
        assert new_state.current_model_select_pass == 2

    def test_pass_persists_through_with_phase(self):
        state = ProjectState(current_model_select_pass=2)
        new_state = state.with_phase(Phase.MODEL_SELECT)
        assert new_state.current_model_select_pass == 2
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/core/test_model_select_pass.py -v`
Expected: FAIL — `ProjectState` has no `current_model_select_pass` field and no `with_model_select_pass` method.

- [ ] **Step 3: Extend ProjectState**

In [src/styleclaw/core/models.py](src/styleclaw/core/models.py), in the `ProjectState` class (around line 165), add the field and builder. The full class becomes:

```python
class ProjectState(_FrozenModel):
    phase: Phase = Phase.INIT
    selected_models: list[str] = Field(default_factory=list)
    current_round: int = 0
    current_batch: int = 0
    current_model_select_pass: int = 0
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

    def with_model_select_pass(self, pass_num: int) -> ProjectState:
        return self.model_copy(update={
            "current_model_select_pass": pass_num,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        })
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/core/test_model_select_pass.py tests/core/test_models.py -v`
Expected: PASS — new tests plus existing model tests still green.

- [ ] **Step 5: Verify state file backward compat**

Pydantic v2 with `extra="ignore"` default plus a `= 0` default means old `state.json` files (without the field) still load. Quick sanity check:

Run: `uv run python -c "from styleclaw.core.models import ProjectState; s = ProjectState.model_validate({'phase': 'INIT'}); print(s.current_model_select_pass)"`
Expected output: `0`

- [ ] **Step 6: Commit**

```bash
git add src/styleclaw/core/models.py tests/core/test_model_select_pass.py
git commit -m "feat(models): ProjectState.current_model_select_pass + builder"
```

---

### Task B3: Pass-scoped project_store helpers

**Files:**
- Modify: [src/styleclaw/storage/project_store.py](src/styleclaw/storage/project_store.py)
- Create: `tests/storage/test_pass_dirs.py`

Introduce `model_select_dir(name, pass_num)` and route all model-select storage through it. Legacy layout (`model-select/initial-analysis.json` and `model-select/results/...`) must still readable — we treat it as "pass 0" and transparently fall back when the pass-scoped path doesn't exist. `create_project` stops pre-creating the `model-select/results` path because that's now a per-pass concern.

Pass 0 semantics (legacy fallback only): when `pass_num == 0`, paths resolve to the old `model-select/` root without a `pass-NNN` subdir. All **writes** must use `pass_num >= 1`. Reads prefer the explicit pass; if it doesn't exist and `pass_num == 1`, also try legacy.

- [ ] **Step 1: Write the failing test**

Create `tests/storage/test_pass_dirs.py`:

```python
from __future__ import annotations

import pytest
from PIL import Image

from styleclaw.core.models import (
    ModelEvaluation,
    ModelScore,
    ProjectConfig,
    StyleAnalysis,
    TaskRecord,
    TaskStatus,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def proj():
    cfg = ProjectConfig(name="p", ip_info="x", ref_images=[])
    project_store.create_project(cfg)
    return "p"


class TestModelSelectDir:
    def test_pass_dir_created(self, proj):
        d = project_store.model_select_dir(proj, pass_num=1)
        assert d.exists()
        assert d.name == "pass-001"
        assert d.parent.name == "model-select"

    def test_pass_dir_separates_passes(self, proj):
        d1 = project_store.model_select_dir(proj, pass_num=1)
        d2 = project_store.model_select_dir(proj, pass_num=2)
        assert d1 != d2
        assert d2.name == "pass-002"


class TestPassScopedAnalysis:
    def test_save_and_load_analysis_uses_pass(self, proj):
        analysis = StyleAnalysis(trigger_phrase="bold ink")
        project_store.save_analysis(proj, analysis, pass_num=1)
        loaded = project_store.load_analysis(proj, pass_num=1)
        assert loaded.trigger_phrase == "bold ink"

    def test_legacy_analysis_readable_at_pass_1(self, proj):
        """A pre-existing `model-select/initial-analysis.json` is readable as pass 1."""
        legacy_dir = project_store.project_dir(proj) / "model-select"
        legacy_dir.mkdir(exist_ok=True)
        legacy_path = legacy_dir / "initial-analysis.json"
        legacy_path.write_text('{"trigger_phrase": "legacy"}', encoding="utf-8")

        loaded = project_store.load_analysis(proj, pass_num=1)
        assert loaded.trigger_phrase == "legacy"

    def test_pass_specific_shadows_legacy(self, proj):
        """If pass-001 has its own analysis, prefer it over legacy."""
        legacy_dir = project_store.project_dir(proj) / "model-select"
        legacy_dir.mkdir(exist_ok=True)
        (legacy_dir / "initial-analysis.json").write_text(
            '{"trigger_phrase": "legacy"}', encoding="utf-8",
        )
        new = StyleAnalysis(trigger_phrase="pass1 override")
        project_store.save_analysis(proj, new, pass_num=1)

        loaded = project_store.load_analysis(proj, pass_num=1)
        assert loaded.trigger_phrase == "pass1 override"


class TestPassScopedTaskRecords:
    def test_save_and_load_task_record_uses_pass(self, proj):
        record = TaskRecord(task_id="t1", model_id="mj-v7", status=TaskStatus.SUCCESS)
        project_store.save_task_record(
            proj, "mj-v7", record, variant="prompt-only", pass_num=2,
        )
        loaded = project_store.load_task_record(
            proj, "mj-v7", variant="prompt-only", pass_num=2,
        )
        assert loaded.task_id == "t1"

    def test_load_all_pass_records_are_isolated(self, proj):
        r1 = TaskRecord(task_id="t-pass1", model_id="mj-v7", status=TaskStatus.SUCCESS)
        r2 = TaskRecord(task_id="t-pass2", model_id="mj-v7", status=TaskStatus.SUCCESS)
        project_store.save_task_record(proj, "mj-v7", r1, variant="prompt-only", pass_num=1)
        project_store.save_task_record(proj, "mj-v7", r2, variant="prompt-only", pass_num=2)

        pass1 = project_store.load_all_task_records(proj, pass_num=1)
        pass2 = project_store.load_all_task_records(proj, pass_num=2)
        assert pass1["mj-v7/prompt-only"].task_id == "t-pass1"
        assert pass2["mj-v7/prompt-only"].task_id == "t-pass2"

    def test_legacy_flat_records_readable_at_pass_1(self, proj):
        legacy = (
            project_store.project_dir(proj)
            / "model-select" / "results" / "mj-v7" / "prompt-only"
        )
        legacy.mkdir(parents=True)
        (legacy / "task.json").write_text(
            '{"task_id": "legacy-t", "model_id": "mj-v7", "status": "SUCCESS"}',
            encoding="utf-8",
        )
        records = project_store.load_all_task_records(proj, pass_num=1)
        assert records["mj-v7/prompt-only"].task_id == "legacy-t"


class TestPassScopedEvaluation:
    def test_save_and_load_evaluation_uses_pass(self, proj):
        ev = ModelEvaluation(
            evaluations=[ModelScore(model="mj-v7", total=8.5)],
            recommendation="mj-v7",
        )
        project_store.save_evaluation(proj, ev, pass_num=1)
        loaded = project_store.load_evaluation(proj, pass_num=1)
        assert loaded.recommendation == "mj-v7"

    def test_evaluations_isolated_between_passes(self, proj):
        ev1 = ModelEvaluation(recommendation="mj-v7")
        ev2 = ModelEvaluation(recommendation="niji7")
        project_store.save_evaluation(proj, ev1, pass_num=1)
        project_store.save_evaluation(proj, ev2, pass_num=2)
        assert project_store.load_evaluation(proj, pass_num=1).recommendation == "mj-v7"
        assert project_store.load_evaluation(proj, pass_num=2).recommendation == "niji7"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/storage/test_pass_dirs.py -v`
Expected: FAIL — `model_select_dir` / pass-scoped arg missing, current signatures don't accept `pass_num`.

- [ ] **Step 3: Implement pass-scoped helpers**

In [src/styleclaw/storage/project_store.py](src/styleclaw/storage/project_store.py), replace the model-select section (the six functions from `save_analysis` on line 118 through `load_evaluation` on line 174) with:

```python
# --- Phase 2: model-select with pass versioning ---
#
# Passes live under model-select/pass-NNN/. Pass 1 may transparently fall back
# to the legacy flat layout (model-select/initial-analysis.json,
# model-select/results/<model>/<variant>/) when a given pass file is absent.
# This keeps projects created before pass-versioning readable.


def model_select_dir(name: str, pass_num: int) -> Path:
    if pass_num < 1:
        raise ValueError(f"pass_num must be >= 1, got {pass_num}")
    d = project_dir(name) / "model-select" / f"pass-{pass_num:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _legacy_model_select_dir(name: str) -> Path:
    return project_dir(name) / "model-select"


def _resolve_analysis_path(name: str, pass_num: int) -> Path:
    pass_path = model_select_dir(name, pass_num) / "initial-analysis.json"
    if pass_path.exists():
        return pass_path
    if pass_num == 1:
        legacy = _legacy_model_select_dir(name) / "initial-analysis.json"
        if legacy.exists():
            return legacy
    return pass_path


def save_analysis(name: str, analysis: StyleAnalysis, pass_num: int = 1) -> None:
    _save_model(analysis, model_select_dir(name, pass_num) / "initial-analysis.json")


def load_analysis(name: str, pass_num: int = 1) -> StyleAnalysis:
    return _load_model(StyleAnalysis, _resolve_analysis_path(name, pass_num))


def model_results_dir(
    name: str, model_id: str, variant: str = "", pass_num: int = 1,
) -> Path:
    base = model_select_dir(name, pass_num) / "results" / model_id
    d = base / variant if variant else base
    d.mkdir(parents=True, exist_ok=True)
    return d


def _legacy_model_results_dir(
    name: str, model_id: str, variant: str = "",
) -> Path | None:
    base = _legacy_model_select_dir(name) / "results" / model_id
    candidate = base / variant if variant else base
    return candidate if candidate.exists() else None


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
    pass_path = model_results_dir(name, model_id, variant, pass_num) / "task.json"
    if pass_path.exists():
        return _load_model(TaskRecord, pass_path)
    if pass_num == 1:
        legacy = _legacy_model_results_dir(name, model_id, variant)
        if legacy is not None:
            legacy_file = legacy / "task.json"
            if legacy_file.exists():
                return _load_model(TaskRecord, legacy_file)
    return _load_model(TaskRecord, pass_path)


def load_all_task_records(name: str, pass_num: int = 1) -> dict[str, TaskRecord]:
    results_dir = model_select_dir(name, pass_num) / "results"
    records = _load_variant_records(results_dir)
    if not records and results_dir.exists():
        records = _load_all_records(results_dir)
    if records:
        return records
    if pass_num == 1:
        legacy_results = _legacy_model_select_dir(name) / "results"
        records = _load_variant_records(legacy_results)
        if records:
            return records
        return _load_all_records(legacy_results)
    return records


def _resolve_evaluation_path(name: str, pass_num: int) -> Path:
    pass_path = model_select_dir(name, pass_num) / "evaluation.json"
    if pass_path.exists():
        return pass_path
    if pass_num == 1:
        legacy = _legacy_model_select_dir(name) / "evaluation.json"
        if legacy.exists():
            return legacy
    return pass_path


def save_evaluation(name: str, evaluation: ModelEvaluation, pass_num: int = 1) -> None:
    _save_model(evaluation, model_select_dir(name, pass_num) / "evaluation.json")


def load_evaluation(name: str, pass_num: int = 1) -> ModelEvaluation:
    return _load_model(ModelEvaluation, _resolve_evaluation_path(name, pass_num))
```

- [ ] **Step 4: Stop pre-creating `model-select/results` in `create_project`**

In [src/styleclaw/storage/project_store.py](src/styleclaw/storage/project_store.py), inside `create_project` (around line 67) change:

```python
(root / "model-select" / "results").mkdir(parents=True)
```

to:

```python
(root / "model-select").mkdir()
```

(Per-pass result dirs are created on first write by `model_select_dir`.)

- [ ] **Step 5: Run new and existing storage tests**

Run: `uv run python -m pytest tests/storage/ -v`
Expected: Some existing callers of `save_analysis` / `load_evaluation` / etc. may still pass without `pass_num` — defaults to 1, so they keep working. The new tests all PASS. If an existing test checks the exact directory, update it to the pass-001 path.

Specifically look at [tests/storage/test_project_store.py](tests/storage/test_project_store.py) for any test that asserts on `model-select/results/` directly; if found, update expectation to `model-select/pass-001/results/` or add a note that the test exercises legacy-fallback.

- [ ] **Step 6: Commit**

```bash
git add src/styleclaw/storage/project_store.py tests/storage/test_pass_dirs.py tests/storage/test_project_store.py
git commit -m "feat(storage): pass-scoped model-select dirs with legacy fallback"
```

---

### Task B4: generate_model_select is pass-aware and picks the right trigger

**Files:**
- Modify: [src/styleclaw/scripts/generate.py](src/styleclaw/scripts/generate.py)

Pass 1: use `analysis.trigger_phrase` (current behaviour). Pass 2+: use the latest refined trigger from the current round's `PromptConfig`. The function now accepts `pass_num` and forwards it to `save_task_record`.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_generate.py`:

```python
from unittest.mock import AsyncMock, patch

import pytest

from styleclaw.core.models import (
    PromptConfig,
    ProjectConfig,
    ProjectState,
    Phase,
    TaskRecord,
    TaskStatus,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


class TestGenerateModelSelectPass:
    async def test_pass_1_uses_initial_trigger_and_saves_to_pass_001(self, tmp_path):
        project_store.create_project(ProjectConfig(name="p", ip_info="x"))

        from styleclaw.scripts.generate import generate_model_select

        async def fake_submit(client, endpoint, params, model_id):
            # Spot-check the prompt contains the initial trigger
            assert "initial-trigger" in params.get("prompt", "")
            return TaskRecord(task_id=f"t-{model_id}", model_id=model_id, status=TaskStatus.QUEUED)

        with patch("styleclaw.scripts.generate.submit_task", side_effect=fake_submit):
            client = AsyncMock()
            await generate_model_select(
                "p", client, trigger_phrase="initial-trigger",
                sref_url="", pass_num=1,
            )

        # Verify records live under pass-001
        pass1_dir = project_store.project_dir("p") / "model-select" / "pass-001" / "results"
        assert any(pass1_dir.rglob("task.json"))

    async def test_pass_2_saves_to_pass_002(self, tmp_path):
        project_store.create_project(ProjectConfig(name="p", ip_info="x"))

        from styleclaw.scripts.generate import generate_model_select

        async def fake_submit(client, endpoint, params, model_id):
            return TaskRecord(task_id=f"t-{model_id}", model_id=model_id, status=TaskStatus.QUEUED)

        with patch("styleclaw.scripts.generate.submit_task", side_effect=fake_submit):
            client = AsyncMock()
            await generate_model_select(
                "p", client, trigger_phrase="refined-trigger",
                sref_url="", pass_num=2, models=["mj-v7"],
            )

        pass2_dir = project_store.project_dir("p") / "model-select" / "pass-002" / "results"
        assert any(pass2_dir.rglob("task.json"))
        # Pass 1 directory must stay empty
        pass1_dir = project_store.project_dir("p") / "model-select" / "pass-001" / "results"
        assert not any(pass1_dir.rglob("task.json")) if pass1_dir.exists() else True
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/scripts/test_generate.py::TestGenerateModelSelectPass -v`
Expected: FAIL — `generate_model_select` doesn't accept `pass_num`.

- [ ] **Step 3: Modify generate_model_select**

In [src/styleclaw/scripts/generate.py](src/styleclaw/scripts/generate.py), replace `generate_model_select` (lines 21–73) with:

```python
async def generate_model_select(
    name: str,
    client: RunningHubClient,
    trigger_phrase: str,
    sref_url: str = "",
    models: list[str] | None = None,
    pass_num: int = 1,
) -> dict[str, TaskRecord]:
    model_ids = models or list(MODEL_REGISTRY.keys())

    existing = project_store.load_all_task_records(name, pass_num=pass_num)

    to_submit: list[tuple[str, str]] = []
    skipped: dict[str, TaskRecord] = {}

    for mid in model_ids:
        for variant in (VARIANT_PROMPT_ONLY, VARIANT_PROMPT_SREF):
            key = f"{mid}/{variant}"
            prev = existing.get(key)
            if prev and prev.status != TaskStatus.FAILED:
                logger.info("Skipping %s: already has %s record.", key, prev.status)
                skipped[key] = prev
            else:
                to_submit.append((mid, variant))

    tasks: dict[str, asyncio.Task[TaskRecord]] = {}

    async def _submit_one(model_id: str, variant: str) -> TaskRecord:
        config = get_model(model_id)
        use_sref = sref_url if variant == VARIANT_PROMPT_SREF else ""
        params = build_params(
            model_id=model_id,
            trigger_phrase=trigger_phrase,
            aspect_ratio="9:16",
            sref_url=use_sref,
        )
        record = await submit_task(client, config.t2i_endpoint, params, model_id)
        project_store.save_task_record(
            name, model_id, record, variant=variant, pass_num=pass_num,
        )
        return record

    async with asyncio.TaskGroup() as tg:
        for mid, variant in to_submit:
            key = f"{mid}/{variant}"
            tasks[key] = tg.create_task(_submit_one(mid, variant))

    records: dict[str, TaskRecord] = {**skipped}
    for key, task in tasks.items():
        records[key] = task.result()

    logger.info(
        "Submitted %d generation tasks for model-select pass %d (%d skipped).",
        len(tasks), pass_num, len(skipped),
    )
    return records
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/scripts/test_generate.py -v`
Expected: PASS — new tests pass; existing tests that called `generate_model_select` without `pass_num` still work (default=1).

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/scripts/generate.py tests/scripts/test_generate.py
git commit -m "feat(generate): generate_model_select accepts pass_num"
```

---

### Task B5: poll_model_select is pass-aware

**Files:**
- Modify: [src/styleclaw/scripts/poll.py](src/styleclaw/scripts/poll.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_poll.py`:

```python
class TestPollModelSelectPass:
    async def test_polls_records_in_specified_pass(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch
        from styleclaw.core.models import ProjectConfig, TaskRecord, TaskStatus
        from styleclaw.scripts.poll import poll_model_select
        from styleclaw.storage import project_store

        monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")
        project_store.create_project(ProjectConfig(name="p", ip_info="x"))

        # Seed a task record in pass 2
        r = TaskRecord(task_id="t-p2", model_id="mj-v7", status=TaskStatus.QUEUED)
        project_store.save_task_record(
            "p", "mj-v7", r, variant="prompt-only", pass_num=2,
        )

        async def fake_poll(client, record):
            return record.model_copy(update={"status": TaskStatus.SUCCESS, "results": []})

        with patch("styleclaw.scripts.poll.poll_and_update", side_effect=fake_poll):
            client = AsyncMock()
            updated = await poll_model_select("p", client, pass_num=2)

        assert "mj-v7/prompt-only" in updated
        assert updated["mj-v7/prompt-only"].status == TaskStatus.SUCCESS
        # Pass-1 dir must still be empty
        pass1_records = project_store.load_all_task_records("p", pass_num=1)
        assert not pass1_records or all(
            r.task_id != "t-p2" for r in pass1_records.values()
        )
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/scripts/test_poll.py::TestPollModelSelectPass -v`
Expected: FAIL — `poll_model_select` doesn't accept `pass_num`.

- [ ] **Step 3: Modify poll_model_select**

In [src/styleclaw/scripts/poll.py](src/styleclaw/scripts/poll.py), replace `_poll_one_model_select` (lines 31–57) and `poll_model_select` (lines 60–83) with:

```python
async def _poll_one_model_select(
    name: str,
    key: str,
    record: TaskRecord,
    client: RunningHubClient,
    pass_num: int,
) -> tuple[str, TaskRecord]:
    if record.status in (TaskStatus.SUCCESS, TaskStatus.FAILED):
        logger.info("Task %s already terminal (%s), skipping.", record.task_id, record.status)
        return key, record

    if not record.task_id:
        logger.warning("Skipping %s: no task_id (submission may have failed).", key)
        return key, record

    logger.info("Polling task %s for %s (pass %d)...", record.task_id, key, pass_num)
    new_record = await poll_and_update(client, record)

    if "/" in key:
        model_id, variant = key.split("/", 1)
        project_store.save_task_record(
            name, model_id, new_record, variant=variant, pass_num=pass_num,
        )
        results_dir = project_store.model_results_dir(
            name, model_id, variant=variant, pass_num=pass_num,
        )
    else:
        project_store.save_task_record(name, key, new_record, pass_num=pass_num)
        results_dir = project_store.model_results_dir(name, key, pass_num=pass_num)

    await _download_results(new_record.results, results_dir)
    return key, new_record


async def poll_model_select(
    name: str,
    client: RunningHubClient,
    pass_num: int = 1,
) -> dict[str, TaskRecord]:
    records = project_store.load_all_task_records(name, pass_num=pass_num)
    if not records:
        raise RuntimeError(
            f"No task records found for project '{name}' pass {pass_num}"
        )

    updated: dict[str, TaskRecord] = {}

    async with asyncio.TaskGroup() as tg:
        tasks = {
            key: tg.create_task(
                _poll_one_model_select(name, key, record, client, pass_num)
            )
            for key, record in records.items()
        }

    for key, task in tasks.items():
        _, new_record = task.result()
        updated[key] = new_record

    logger.info("Pass %d poll complete. %d tasks processed.", pass_num, len(updated))
    return updated
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/scripts/test_poll.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/scripts/poll.py tests/scripts/test_poll.py
git commit -m "feat(poll): poll_model_select accepts pass_num"
```

---

### Task B6: generate_model_select_report is pass-aware

**Files:**
- Modify: [src/styleclaw/scripts/report.py](src/styleclaw/scripts/report.py)

The report lives beside the pass's data: `model-select/pass-NNN/report.html`.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_report.py`:

```python
class TestModelSelectReportPass:
    def test_report_written_into_pass_dir(self, tmp_path, monkeypatch):
        from styleclaw.core.models import (
            ModelEvaluation, ModelScore, ProjectConfig, StyleAnalysis,
        )
        from styleclaw.scripts.report import generate_model_select_report
        from styleclaw.storage import project_store

        monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")
        project_store.create_project(
            ProjectConfig(name="p", ip_info="x", ref_images=[]),
        )
        project_store.save_analysis(
            "p", StyleAnalysis(trigger_phrase="t"), pass_num=2,
        )
        project_store.save_evaluation(
            "p",
            ModelEvaluation(
                evaluations=[ModelScore(model="mj-v7", total=8.0)],
                recommendation="mj-v7",
            ),
            pass_num=2,
        )

        path = generate_model_select_report("p", pass_num=2)
        assert path.parent.name == "pass-002"
        assert path.exists()
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/scripts/test_report.py::TestModelSelectReportPass -v`
Expected: FAIL — function doesn't take `pass_num`.

- [ ] **Step 3: Modify `generate_model_select_report`**

In [src/styleclaw/scripts/report.py](src/styleclaw/scripts/report.py), replace `generate_model_select_report` (lines 28–69) with:

```python
def generate_model_select_report(name: str, pass_num: int = 1) -> Path:
    config = project_store.load_config(name)
    analysis = project_store.load_analysis(name, pass_num=pass_num)
    evaluation = project_store.load_evaluation(name, pass_num=pass_num)

    root = project_store.project_dir(name)
    ref_images = [_img_to_data_uri(root / r) for r in config.ref_images]

    model_data: list[dict] = []
    for ev in evaluation.evaluations:
        if ev.variant:
            results_dir = project_store.model_results_dir(
                name, ev.model, variant=ev.variant, pass_num=pass_num,
            )
        else:
            results_dir = project_store.model_results_dir(
                name, ev.model, pass_num=pass_num,
            )
        images = sorted(results_dir.glob("output-*.png"))
        model_data.append({
            "model": ev.model,
            "variant": ev.variant,
            "scores": ev.scores.model_dump(),
            "total": ev.total,
            "analysis": ev.analysis,
            "suggestions": ev.suggestions,
            "images": [_img_to_data_uri(p) for p in images],
        })

    template = _env.get_template("model_select.html")
    html = template.render(
        project_name=name,
        pass_num=pass_num,
        trigger_phrase=analysis.trigger_phrase,
        recommendation=evaluation.recommendation,
        recommended_variant=evaluation.recommended_variant,
        ref_images=ref_images,
        models=model_data,
    )

    dest = project_store.model_select_dir(name, pass_num) / "report.html"
    dest.write_text(html, encoding="utf-8")
    logger.info("Model-select report saved: %s", dest)
    return dest
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/scripts/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/scripts/report.py tests/scripts/test_report.py
git commit -m "feat(report): generate_model_select_report scoped to pass"
```

---

### Task B7: orchestrator — do_analyze only runs on pass 1, other actions read current pass

**Files:**
- Modify: [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py)

Behaviour changes:
1. `do_analyze` is now called only during the INIT → MODEL_SELECT transition. It sets `pass_num = 1` and saves analysis under pass-001.
2. `do_generate`, `do_poll`, `do_evaluate` in MODEL_SELECT branch read `state.current_model_select_pass` and forward it to storage/scripts.
3. `do_select_model` stays unchanged shape-wise; the advance MODEL_SELECT → STYLE_REFINE does not bump the pass counter (we only bump *on entry* to MODEL_SELECT).

- [ ] **Step 1: Write the failing test**

Create `tests/orchestrator/test_retest_models.py` (note: this file will also cover Task B8; start with the do_analyze test here):

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from styleclaw.core.models import (
    Phase,
    ProjectConfig,
    ProjectState,
    StyleAnalysis,
    TaskRecord,
    TaskStatus,
)
from styleclaw.orchestrator.actions import ExecutionContext, do_analyze, do_generate
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def project_with_ref(tmp_path):
    name = "p"
    project_store.create_project(
        ProjectConfig(name=name, ip_info="ip", ref_images=["refs/ref.png"]),
    )
    ref = project_store.project_dir(name) / "refs" / "ref.png"
    Image.new("RGB", (16, 16), "red").save(ref)
    return name


class TestDoAnalyzeSetsPass1:
    async def test_analyze_sets_current_pass_to_1(self, project_with_ref):
        fake_llm = AsyncMock()
        fake_llm.invoke = AsyncMock(return_value='{"trigger_phrase": "t"}')

        ctx = ExecutionContext(project=project_with_ref, llm=fake_llm)
        result = await do_analyze(ctx, {})
        assert result.ok

        state = project_store.load_state(project_with_ref)
        assert state.phase == Phase.MODEL_SELECT
        assert state.current_model_select_pass == 1

        # Analysis file lives under pass-001
        pass1 = (
            project_store.project_dir(project_with_ref)
            / "model-select" / "pass-001" / "initial-analysis.json"
        )
        assert pass1.exists()


class TestDoGenerateUsesCurrentPass:
    async def test_pass_2_picks_refined_trigger(self, project_with_ref):
        # Seed state: we're in MODEL_SELECT pass 2, with a round-1 refined trigger
        from styleclaw.core.models import PromptConfig

        project_store.save_analysis(
            project_with_ref, StyleAnalysis(trigger_phrase="initial"), pass_num=1,
        )
        project_store.save_prompt_config(
            project_with_ref, 1,
            PromptConfig(round=1, trigger_phrase="refined-after-round-1"),
        )
        state = ProjectState(
            phase=Phase.MODEL_SELECT,
            current_round=1,
            current_model_select_pass=2,
            selected_models=["mj-v7"],
        )
        project_store.save_state(project_with_ref, state)

        captured = []

        async def fake_submit(client, endpoint, params, model_id):
            captured.append(params.get("prompt", ""))
            return TaskRecord(task_id=f"t-{model_id}", model_id=model_id, status=TaskStatus.QUEUED)

        with patch("styleclaw.scripts.generate.submit_task", side_effect=fake_submit):
            ctx = ExecutionContext(project=project_with_ref, client=AsyncMock())
            result = await do_generate(ctx, {})

        assert result.ok
        assert any("refined-after-round-1" in p for p in captured)
        assert not any("initial" in p for p in captured)
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/orchestrator/test_retest_models.py -v`
Expected: FAIL — `do_analyze` doesn't set pass=1; `do_generate` MODEL_SELECT branch uses the initial analysis trigger not the refined one.

- [ ] **Step 3: Modify do_analyze, do_generate, do_poll, do_evaluate**

In [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py):

**do_analyze** — replace the save + advance block so it sets pass=1:

```python
async def do_analyze(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.agents.analyze_style import analyze_style, analyze_style_with_thinking
    from styleclaw.core.state_machine import advance

    config = project_store.load_config(ctx.project)
    root = project_store.project_dir(ctx.project)
    ref_paths = [root / r for r in config.ref_images]

    thinking = ""
    if ctx.show_thinking:
        analysis, thinking = await analyze_style_with_thinking(
            ctx.llm, ref_paths, config.ip_info, thinking_budget=ctx.thinking_budget,
        )
    else:
        analysis = await analyze_style(ctx.llm, ref_paths, config.ip_info)

    pass_num = 1
    project_store.save_analysis(ctx.project, analysis, pass_num=pass_num)
    if thinking:
        project_store.save_thinking(
            project_store.model_select_dir(ctx.project, pass_num) / "initial-analysis.json",
            thinking,
        )

    state = project_store.load_state(ctx.project)
    new_state = advance(state, Phase.MODEL_SELECT).with_model_select_pass(pass_num)
    project_store.save_state(ctx.project, new_state)

    msg = f"Trigger: {analysis.trigger_phrase}"
    if thinking:
        msg += f" | thinking saved ({len(thinking)} chars)"
    return StepResult(ok=True, message=msg)
```

**do_generate** (MODEL_SELECT branch) — replace lines 64–72 with:

```python
    if state.phase == Phase.MODEL_SELECT:
        pass_num = state.current_model_select_pass or 1
        if pass_num > 1 and state.current_round >= 1:
            # Re-entry: use the most recent refined trigger phrase.
            prompt_cfg = project_store.load_prompt_config(ctx.project, state.current_round)
            trigger = prompt_cfg.trigger_phrase
        else:
            analysis = project_store.load_analysis(ctx.project, pass_num=pass_num)
            trigger = analysis.trigger_phrase
        uploads = project_store.load_uploads(ctx.project)
        sref_url = uploads[0].url if uploads else ""
        records = await generate_model_select(
            ctx.project, ctx.client, trigger,
            sref_url=sref_url, pass_num=pass_num,
        )
        return StepResult(ok=True, message=f"Submitted {len(records)} model tasks (pass {pass_num})")
```

**do_poll** (MODEL_SELECT branch) — inside the `for cycle in range(max_cycles):` loop, replace the MODEL_SELECT arm (line 95–96) with:

```python
        if state.phase == Phase.MODEL_SELECT:
            pass_num = state.current_model_select_pass or 1
            records = await poll_model_select(ctx.project, ctx.client, pass_num=pass_num)
```

**do_evaluate** (MODEL_SELECT branch) — prepend the pass lookup and forward to storage + report:

```python
    if state.phase == Phase.MODEL_SELECT:
        from styleclaw.agents.select_model import (
            evaluate_models,
            evaluate_models_with_thinking,
        )
        from styleclaw.scripts.report import generate_model_select_report

        pass_num = state.current_model_select_pass or 1

        model_images: dict[str, list[Path]] = {}
        records = project_store.load_all_task_records(ctx.project, pass_num=pass_num)
        for key in records:
            if "/" in key:
                model_id, variant = key.split("/", 1)
                results_dir = project_store.model_results_dir(
                    ctx.project, model_id, variant=variant, pass_num=pass_num,
                )
            else:
                results_dir = project_store.model_results_dir(
                    ctx.project, key, pass_num=pass_num,
                )
            images = sorted(results_dir.glob("output-*.png"))
            if images:
                model_images[key] = images

        if not model_images:
            return StepResult(ok=False, message="No generated images found")

        thinking = ""
        if ctx.show_thinking:
            evaluation, thinking = await evaluate_models_with_thinking(
                ctx.llm, ref_paths, model_images, thinking_budget=ctx.thinking_budget,
            )
        else:
            evaluation = await evaluate_models(ctx.llm, ref_paths, model_images)
        project_store.save_evaluation(ctx.project, evaluation, pass_num=pass_num)
        if thinking:
            project_store.save_thinking(
                project_store.model_select_dir(ctx.project, pass_num) / "evaluation.json",
                thinking,
            )
        generate_model_select_report(ctx.project, pass_num=pass_num)

        msg = f"Recommendation: {evaluation.recommendation} (pass {pass_num})"
        if thinking:
            msg += f" | thinking saved ({len(thinking)} chars)"
        return StepResult(
            ok=True, message=msg,
            data={"recommendation": evaluation.recommendation, "pass_num": pass_num},
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/orchestrator/ tests/scripts/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/orchestrator/actions.py tests/orchestrator/test_retest_models.py
git commit -m "feat(orchestrator): actions read/write current model-select pass"
```

---

### Task B8: New actions — `retest-models` and `back-to-t2i`

**Files:**
- Modify: [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py)

`do_retest_models` is the "go back and test models again" action:
- Called from STYLE_REFINE or BATCH_T2I.
- Advances phase to MODEL_SELECT.
- Bumps `current_model_select_pass` by 1.
- Does NOT call the analyze LLM (we reuse pass-1 analysis).

`do_back_to_t2i` handles BATCH_I2I → BATCH_T2I:
- Called from BATCH_I2I.
- Advances phase back to BATCH_T2I (uses the existing transition we added in B1).
- Leaves `current_batch` pointing at the current t2i batch (don't reset it).

- [ ] **Step 1: Write the failing test**

Append to `tests/orchestrator/test_retest_models.py`:

```python
class TestDoRetestModels:
    async def test_from_style_refine_bumps_pass(self, project_with_ref):
        from styleclaw.orchestrator.actions import do_retest_models

        state = ProjectState(
            phase=Phase.STYLE_REFINE, current_round=2, current_model_select_pass=1,
        )
        project_store.save_state(project_with_ref, state)

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_retest_models(ctx, {})
        assert result.ok
        new_state = project_store.load_state(project_with_ref)
        assert new_state.phase == Phase.MODEL_SELECT
        assert new_state.current_model_select_pass == 2
        # Round preserved so later actions can pick the refined trigger
        assert new_state.current_round == 2

    async def test_from_batch_t2i_bumps_pass(self, project_with_ref):
        from styleclaw.orchestrator.actions import do_retest_models

        state = ProjectState(
            phase=Phase.BATCH_T2I,
            current_batch=1,
            current_round=3,
            current_model_select_pass=2,
        )
        project_store.save_state(project_with_ref, state)

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_retest_models(ctx, {})
        assert result.ok
        new_state = project_store.load_state(project_with_ref)
        assert new_state.phase == Phase.MODEL_SELECT
        assert new_state.current_model_select_pass == 3
        assert new_state.current_batch == 1

    async def test_retest_not_allowed_from_init(self, project_with_ref):
        from styleclaw.orchestrator.actions import do_retest_models

        ctx = ExecutionContext(project=project_with_ref)  # default phase INIT
        result = await do_retest_models(ctx, {})
        assert result.ok is False
        assert "INIT" in result.message


class TestDoBackToT2i:
    async def test_from_batch_i2i(self, project_with_ref):
        from styleclaw.orchestrator.actions import do_back_to_t2i

        state = ProjectState(phase=Phase.BATCH_I2I, current_batch=1)
        project_store.save_state(project_with_ref, state)

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_back_to_t2i(ctx, {})
        assert result.ok
        new_state = project_store.load_state(project_with_ref)
        assert new_state.phase == Phase.BATCH_T2I
        assert new_state.current_batch == 1

    async def test_not_allowed_from_other_phases(self, project_with_ref):
        from styleclaw.orchestrator.actions import do_back_to_t2i

        state = ProjectState(phase=Phase.STYLE_REFINE)
        project_store.save_state(project_with_ref, state)

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_back_to_t2i(ctx, {})
        assert result.ok is False
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run python -m pytest tests/orchestrator/test_retest_models.py -v`
Expected: FAIL — `do_retest_models`, `do_back_to_t2i` not defined.

- [ ] **Step 3: Implement the two actions and register them**

In [src/styleclaw/orchestrator/actions.py](src/styleclaw/orchestrator/actions.py), add these two functions after `do_report`:

```python
async def do_retest_models(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.core.state_machine import advance

    state = project_store.load_state(ctx.project)
    if state.phase not in (Phase.STYLE_REFINE, Phase.BATCH_T2I):
        return StepResult(
            ok=False,
            message=f"retest-models requires STYLE_REFINE or BATCH_T2I (current: {state.phase})",
        )

    new_pass = (state.current_model_select_pass or 0) + 1
    new_state = (
        advance(state, Phase.MODEL_SELECT)
        .with_model_select_pass(new_pass)
    )
    project_store.save_state(ctx.project, new_state)
    return StepResult(
        ok=True,
        message=f"Entered MODEL_SELECT pass {new_pass} for re-test",
        data={"pass_num": new_pass},
    )


async def do_back_to_t2i(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.core.state_machine import advance

    state = project_store.load_state(ctx.project)
    if state.phase != Phase.BATCH_I2I:
        return StepResult(
            ok=False,
            message=f"back-to-t2i requires BATCH_I2I (current: {state.phase})",
        )
    new_state = advance(state, Phase.BATCH_T2I)
    project_store.save_state(ctx.project, new_state)
    return StepResult(ok=True, message="Returned to BATCH_T2I")
```

And register them in `ACTION_REGISTRY` (around line 336):

```python
ACTION_REGISTRY: dict[str, ActionDef] = {
    "analyze":        ActionDef(fn=do_analyze,       needs_client=False, needs_llm=True),
    "generate":       ActionDef(fn=do_generate,      needs_client=True,  needs_llm=False),
    "poll":           ActionDef(fn=do_poll,          needs_client=True,  needs_llm=False),
    "evaluate":       ActionDef(fn=do_evaluate,      needs_client=False, needs_llm=True),
    "select-model":   ActionDef(fn=do_select_model,  needs_client=False, needs_llm=False, requires_confirmation=True),
    "refine":         ActionDef(fn=do_refine,        needs_client=False, needs_llm=True),
    "approve":        ActionDef(fn=do_approve,       needs_client=False, needs_llm=False),
    "design-cases":   ActionDef(fn=do_design_cases,  needs_client=False, needs_llm=True),
    "batch-submit":   ActionDef(fn=do_batch_submit,  needs_client=True,  needs_llm=False),
    "report":         ActionDef(fn=do_report,        needs_client=False, needs_llm=False),
    "retest-models":  ActionDef(fn=do_retest_models, needs_client=False, needs_llm=False),
    "back-to-t2i":    ActionDef(fn=do_back_to_t2i,   needs_client=False, needs_llm=False),
}
```

And extend `PHASE_ACTIONS` (around line 350):

```python
PHASE_ACTIONS: dict[Phase, list[str]] = {
    Phase.INIT:         ["analyze"],
    Phase.MODEL_SELECT: ["generate", "poll", "evaluate", "select-model"],
    Phase.STYLE_REFINE: ["refine", "generate", "poll", "evaluate", "approve", "select-model", "retest-models"],
    Phase.BATCH_T2I:    ["design-cases", "batch-submit", "poll", "report", "approve", "retest-models"],
    Phase.BATCH_I2I:    ["batch-submit", "poll", "report", "approve", "back-to-t2i"],
    Phase.COMPLETED:    [],
}
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/orchestrator/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/orchestrator/actions.py tests/orchestrator/test_retest_models.py
git commit -m "feat(orchestrator): retest-models and back-to-t2i actions"
```

---

### Task B9: CLI commands `retest-models` and `back-to-t2i`; status shows pass

**Files:**
- Modify: [src/styleclaw/cli.py](src/styleclaw/cli.py)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
class TestRetestModelsCommand:
    def test_retest_models_advances_to_model_select(self, setup_project):
        from styleclaw.core.models import Phase, ProjectState
        from styleclaw.storage import project_store

        # Fast-forward the fixture project to STYLE_REFINE
        state = ProjectState(
            phase=Phase.STYLE_REFINE,
            current_round=1,
            current_model_select_pass=1,
        )
        project_store.save_state("test-project", state)

        result = CliRunner().invoke(app, ["retest-models", "test-project"])
        assert result.exit_code == 0
        new_state = project_store.load_state("test-project")
        assert new_state.phase == Phase.MODEL_SELECT
        assert new_state.current_model_select_pass == 2


class TestBackToT2iCommand:
    def test_back_to_t2i_from_batch_i2i(self, setup_project):
        from styleclaw.core.models import Phase, ProjectState
        from styleclaw.storage import project_store

        state = ProjectState(phase=Phase.BATCH_I2I, current_batch=1)
        project_store.save_state("test-project", state)

        result = CliRunner().invoke(app, ["back-to-t2i", "test-project"])
        assert result.exit_code == 0
        assert project_store.load_state("test-project").phase == Phase.BATCH_T2I


class TestStatusShowsPass:
    def test_status_includes_pass_number(self, setup_project):
        from styleclaw.core.models import Phase, ProjectState
        from styleclaw.storage import project_store

        state = ProjectState(
            phase=Phase.MODEL_SELECT,
            current_model_select_pass=2,
        )
        project_store.save_state("test-project", state)

        result = CliRunner().invoke(app, ["status", "test-project"])
        assert result.exit_code == 0
        assert "Pass:" in result.stdout
        assert "2" in result.stdout
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run python -m pytest tests/test_cli.py::TestRetestModelsCommand tests/test_cli.py::TestBackToT2iCommand tests/test_cli.py::TestStatusShowsPass -v`
Expected: FAIL — commands don't exist, status doesn't show pass.

- [ ] **Step 3: Add CLI commands and update status**

In [src/styleclaw/cli.py](src/styleclaw/cli.py), inside the `status` command (after the "Round:" line near line 135), add:

```python
    typer.echo(f"Pass:    {state.current_model_select_pass}")
```

Then add two new commands (e.g. after the `rollback` command near line 366):

```python
@app.command(name="retest-models")
def retest_models_cmd(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Re-enter MODEL_SELECT to re-test all models with the current trigger."""
    state = project_store.load_state(name)
    if state.phase not in (Phase.STYLE_REFINE, Phase.BATCH_T2I):
        typer.echo(
            f"Error: retest-models requires STYLE_REFINE or BATCH_T2I "
            f"(current: {state.phase})",
            err=True,
        )
        raise typer.Exit(1)

    result = _run_action(name, "retest-models")
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)
    typer.echo(
        "Next: run 'generate', 'poll', 'evaluate', then 'select-model' to pick a model."
    )


@app.command(name="back-to-t2i")
def back_to_t2i_cmd(
    name: str = typer.Argument(..., help="Project name"),
) -> None:
    """Return from BATCH_I2I to BATCH_T2I when i2i results are unsatisfying."""
    state = project_store.load_state(name)
    if state.phase != Phase.BATCH_I2I:
        typer.echo(
            f"Error: back-to-t2i requires BATCH_I2I phase (current: {state.phase})",
            err=True,
        )
        raise typer.Exit(1)

    result = _run_action(name, "back-to-t2i")
    if not result.ok:
        typer.echo(f"Error: {result.message}", err=True)
        raise typer.Exit(1)
    typer.echo(result.message)
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/test_cli.py -v`
Expected: PASS — three new tests green, existing CLI tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/cli.py tests/test_cli.py
git commit -m "feat(cli): retest-models and back-to-t2i commands; status shows pass"
```

---

### Task B10: End-to-end integration test — full re-entry workflow

**Files:**
- Create: `tests/orchestrator/test_pass2_e2e.py`

Simulate: analyze → generate → poll → evaluate → select-model (pass 1) → refine → generate → poll → evaluate → retest-models → generate → poll → evaluate (pass 2). Verify storage isolation between passes and correct trigger use.

- [ ] **Step 1: Write the test**

Create `tests/orchestrator/test_pass2_e2e.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from styleclaw.core.models import (
    ModelEvaluation,
    ModelScore,
    Phase,
    ProjectConfig,
    PromptConfig,
    ProjectState,
    RoundEvaluation,
    StyleAnalysis,
    TaskRecord,
    TaskStatus,
)
from styleclaw.orchestrator.actions import (
    ExecutionContext,
    do_evaluate,
    do_generate,
    do_retest_models,
    do_select_model,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def seeded_project(tmp_path):
    name = "e2e"
    project_store.create_project(
        ProjectConfig(name=name, ip_info="ip", ref_images=["refs/r.png"]),
    )
    ref = project_store.project_dir(name) / "refs" / "r.png"
    Image.new("RGB", (16, 16), "red").save(ref)

    # Pass 1 analysis + state fast-forwarded to STYLE_REFINE round 1
    project_store.save_analysis(
        name, StyleAnalysis(trigger_phrase="initial-trigger"), pass_num=1,
    )
    project_store.save_prompt_config(
        name, 1, PromptConfig(round=1, trigger_phrase="refined-trigger-r1"),
    )
    project_store.save_state(
        name,
        ProjectState(
            phase=Phase.STYLE_REFINE,
            current_round=1,
            current_model_select_pass=1,
            selected_models=["mj-v7"],
        ),
    )
    return name


async def test_pass2_flow_isolates_storage(seeded_project):
    # Step 1: retest-models from STYLE_REFINE → MODEL_SELECT pass 2
    ctx = ExecutionContext(project=seeded_project)
    result = await do_retest_models(ctx, {})
    assert result.ok

    state = project_store.load_state(seeded_project)
    assert state.phase == Phase.MODEL_SELECT
    assert state.current_model_select_pass == 2

    # Step 2: generate in MODEL_SELECT — must use the round-1 refined trigger
    captured: list[str] = []

    async def fake_submit(client, endpoint, params, model_id):
        captured.append(params.get("prompt", ""))
        return TaskRecord(task_id=f"t-{model_id}", model_id=model_id, status=TaskStatus.QUEUED)

    with patch("styleclaw.scripts.generate.submit_task", side_effect=fake_submit):
        ctx = ExecutionContext(project=seeded_project, client=AsyncMock())
        result = await do_generate(ctx, {})

    assert result.ok
    assert all("refined-trigger-r1" in p for p in captured)
    assert not any("initial-trigger" in p for p in captured)

    # Pass-2 task records must live under pass-002 only
    pass1_records = project_store.load_all_task_records(seeded_project, pass_num=1)
    pass2_records = project_store.load_all_task_records(seeded_project, pass_num=2)
    assert pass2_records
    assert not pass1_records  # pass 1 was never written in this test

    # Step 3: simulate poll completion — set statuses to SUCCESS directly
    for key, rec in pass2_records.items():
        model_id, variant = key.split("/", 1)
        project_store.save_task_record(
            seeded_project, model_id,
            rec.model_copy(update={"status": TaskStatus.SUCCESS}),
            variant=variant, pass_num=2,
        )
        # Seed one fake output image so evaluate() has something to grab
        results_dir = project_store.model_results_dir(
            seeded_project, model_id, variant=variant, pass_num=2,
        )
        img = results_dir / "output-001.png"
        Image.new("RGB", (16, 16), "green").save(img)

    # Step 4: evaluate pass 2
    fake_llm = AsyncMock()
    fake_llm.invoke = AsyncMock(
        return_value='{"evaluations": [{"model": "mj-v7", "variant": "prompt-only", "total": 8.2}], "recommendation": "mj-v7"}'
    )
    ctx = ExecutionContext(project=seeded_project, llm=fake_llm)
    result = await do_evaluate(ctx, {})
    assert result.ok
    assert "mj-v7" in result.message
    assert "pass 2" in result.message.lower()

    # Pass-2 evaluation file exists; pass-1 is untouched (or absent)
    pass2_eval = project_store.load_evaluation(seeded_project, pass_num=2)
    assert pass2_eval.recommendation == "mj-v7"
    with pytest.raises(FileNotFoundError):
        project_store.load_evaluation(seeded_project, pass_num=1)

    # Step 5: select-model pass 2 → back to STYLE_REFINE, round preserved
    ctx = ExecutionContext(project=seeded_project)
    result = await do_select_model(ctx, {"models": "mj-v7"})
    assert result.ok

    final_state = project_store.load_state(seeded_project)
    assert final_state.phase == Phase.STYLE_REFINE
    assert final_state.current_round == 1
    assert final_state.current_model_select_pass == 2
```

- [ ] **Step 2: Run the end-to-end test**

Run: `uv run python -m pytest tests/orchestrator/test_pass2_e2e.py -v`
Expected: PASS.

- [ ] **Step 3: Run the full test suite with coverage**

Run: `uv run python -m pytest tests/ --cov=src`
Expected: PASS, coverage ≥ 80%.

- [ ] **Step 4: Commit**

```bash
git add tests/orchestrator/test_pass2_e2e.py
git commit -m "test: end-to-end pass-2 re-entry flow with storage isolation"
```

---

### Part B Self-Review Checklist

- [ ] `uv run python -m pytest tests/ --cov=src` fully green, coverage ≥ 80%
- [ ] New CLI commands `retest-models`, `back-to-t2i` appear in `uv run styleclaw --help`
- [ ] `uv run styleclaw status <project>` prints `Pass: N`
- [ ] A legacy project (no pass folders) still loads via `load_analysis` / `load_all_task_records` / `load_evaluation`
- [ ] From STYLE_REFINE, running `retest-models → generate → poll → evaluate → select-model` produces files under `model-select/pass-002/` and leaves `pass-001/` intact
- [ ] From BATCH_I2I, running `back-to-t2i` flips the phase without resetting `current_batch`
- [ ] `PHASE_ACTIONS` enumerates the new actions so the LLM planner can pick them in `run "..."` intents
- [ ] `do_analyze` is NOT called on pass ≥ 2 (analyze only runs from INIT)

---

## Summary of New User Workflow

After this plan lands, a user can:

```bash
# Standard forward path (unchanged)
styleclaw init my-proj --ref a.png --ref b.png --info "…"
styleclaw analyze my-proj           # pass 1 analysis
styleclaw generate my-proj          # pass 1 model-select generation
styleclaw poll my-proj
styleclaw evaluate my-proj          # pass 1 evaluation
styleclaw select-model my-proj --models mj-v7
styleclaw refine my-proj            # round 1
styleclaw generate my-proj
styleclaw poll my-proj
styleclaw evaluate my-proj

# NEW: re-test all models with the refined trigger
styleclaw retest-models my-proj     # → MODEL_SELECT, pass 2
styleclaw generate my-proj          # uses round-1 refined trigger
styleclaw poll my-proj
styleclaw evaluate my-proj          # pass 2 evaluation
styleclaw select-model my-proj --models niji7   # maybe a different model now wins

# Continue to batch testing
styleclaw approve my-proj
styleclaw design-cases my-proj
styleclaw batch-submit my-proj
styleclaw poll my-proj

# Batch disappointing? Go back and try yet another model
styleclaw retest-models my-proj     # → MODEL_SELECT, pass 3

# i2i not as good as t2i? Drop back to t2i and iterate further
styleclaw back-to-t2i my-proj       # → BATCH_T2I
```

All historic data is preserved per-pass and visible in reports.
