# design-cases Parallel Sharding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `design_cases()` from one large LLM request into N parallel shard requests (default N=5), each generating 20 of the 100 batch cases.

**Architecture:** Internal refactor of `agents/design_cases.py`. Public signature unchanged. Partition the 10 fixed categories into N contiguous slices, fire `asyncio.gather(*shard_coros)`, validate each shard's response stays inside its assigned categories, concatenate results. Provider's existing 3× retry covers transient failures; no extra retry layer added.

**Tech Stack:** Python 3.11, asyncio, Pydantic v2, pytest (with `pytest-asyncio` and `monkeypatch`).

**Reference spec:** [docs/superpowers/specs/2026-05-20-design-cases-parallel-sharding-design.md](../specs/2026-05-20-design-cases-parallel-sharding-design.md)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/styleclaw/core/config.py` | Modify | Add `DESIGN_CASES_SHARDS` constant + `validate_design_cases_config()`; wire into `validate_env()`. |
| `src/styleclaw/providers/llm/prompts/design_cases_shard.md` | Create | Per-shard prompt template (20-case scoped). |
| `src/styleclaw/providers/llm/prompts/design_cases.md` | Delete | Old single-call template — no fallback path. |
| `src/styleclaw/agents/design_cases.py` | Rewrite | Partition + fan-out + merge logic. |
| `tests/core/test_config.py` | Modify | Default value + env override + invalid-value validation tests. |
| `tests/agents/test_design_cases.py` | Modify | Add sharding tests; force SHARDS=1 in existing tests via autouse fixture. |
| `tests/agents/test_design_cases_error.py` | Modify | Force SHARDS=1 in existing tests via autouse fixture. |
| `CLAUDE.md` | Modify | Document new env var in Runtime Tunables table. |
| `.env.example` | Modify | Add `STYLECLAW_DESIGN_CASES_SHARDS` row. |

---

### Task 1: Add `DESIGN_CASES_SHARDS` config constant

**Files:**
- Modify: `src/styleclaw/core/config.py:30-42` (add constant near other `_int_env` lines)
- Test: `tests/core/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_config.py` inside `class TestConfigDefaults`:

```python
    def test_design_cases_shards_default(self):
        from styleclaw.core.config import DESIGN_CASES_SHARDS
        assert DESIGN_CASES_SHARDS == 5
```

And inside `class TestConfigEnvOverrides`:

```python
    def test_design_cases_shards_from_env(self, monkeypatch):
        monkeypatch.setenv("STYLECLAW_DESIGN_CASES_SHARDS", "2")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        assert config_mod.DESIGN_CASES_SHARDS == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/core/test_config.py::TestConfigDefaults::test_design_cases_shards_default tests/core/test_config.py::TestConfigEnvOverrides::test_design_cases_shards_from_env -v`

Expected: 2 failures with `ImportError: cannot import name 'DESIGN_CASES_SHARDS'`

- [ ] **Step 3: Add the constant**

In `src/styleclaw/core/config.py`, add immediately after the `MAX_POLL_CYCLES` line (around line 42):

```python
# Number of parallel LLM shards used by design_cases. Must divide 10 (the
# fixed category count) evenly — allowed values are 1, 2, 5, 10. Smaller
# shards = smaller per-request token budgets, lower 429/500 risk, but more
# total system-prompt overhead.
DESIGN_CASES_SHARDS: int = _int_env("STYLECLAW_DESIGN_CASES_SHARDS", "5")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/core/test_config.py -v`

Expected: PASS (both new tests + all existing tests still green)

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/config.py tests/core/test_config.py
git commit -m "feat(config): add DESIGN_CASES_SHARDS constant (default 5)"
```

---

### Task 2: Validate `DESIGN_CASES_SHARDS` via `validate_env()`

**Files:**
- Modify: `src/styleclaw/core/config.py` (add `validate_design_cases_config()`, wire into `validate_env()`)
- Test: `tests/core/test_config.py`

- [ ] **Step 1: Write the failing test**

Append a new test class to `tests/core/test_config.py`:

```python
class TestValidateDesignCasesShards:
    def test_valid_values_accepted(self, monkeypatch):
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "dummy")
        for value in ("1", "2", "5", "10"):
            monkeypatch.setenv("STYLECLAW_DESIGN_CASES_SHARDS", value)
            import importlib
            import styleclaw.core.config as config_mod
            importlib.reload(config_mod)
            errs = config_mod.validate_env()
            assert not any("DESIGN_CASES_SHARDS" in e for e in errs), (
                f"value {value} should pass: {errs}"
            )

    def test_invalid_value_3_rejected(self, monkeypatch):
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "dummy")
        monkeypatch.setenv("STYLECLAW_DESIGN_CASES_SHARDS", "3")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        errs = config_mod.validate_env()
        assert any("DESIGN_CASES_SHARDS" in e and "3" in e for e in errs)

    def test_invalid_value_0_rejected(self, monkeypatch):
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "dummy")
        monkeypatch.setenv("STYLECLAW_DESIGN_CASES_SHARDS", "0")
        import importlib
        import styleclaw.core.config as config_mod
        importlib.reload(config_mod)
        errs = config_mod.validate_env()
        assert any("DESIGN_CASES_SHARDS" in e for e in errs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/core/test_config.py::TestValidateDesignCasesShards -v`

Expected: 3 failures — `validate_env()` doesn't yet report DESIGN_CASES_SHARDS errors.

- [ ] **Step 3: Implement the validator and wire it in**

Edit `src/styleclaw/core/config.py`. Add after `validate_panel_config()` (around line 119):

```python
_ALLOWED_DESIGN_CASES_SHARDS = (1, 2, 5, 10)


def validate_design_cases_config() -> list[str]:
    """Return error strings if DESIGN_CASES_SHARDS is not a value that
    evenly partitions the 10 fixed categories."""
    errors: list[str] = []
    if DESIGN_CASES_SHARDS not in _ALLOWED_DESIGN_CASES_SHARDS:
        errors.append(
            f"STYLECLAW_DESIGN_CASES_SHARDS={DESIGN_CASES_SHARDS} must be one of "
            f"{_ALLOWED_DESIGN_CASES_SHARDS} (each evenly partitions the 10 fixed "
            f"categories)."
        )
    return errors
```

In `validate_env()`, add a call after `validate_panel_config()`. Find the existing line:

```python
    errors.extend(validate_panel_config())
```

Add immediately after it:

```python
    errors.extend(validate_design_cases_config())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/core/test_config.py -v`

Expected: PASS — 3 new tests + all existing.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/core/config.py tests/core/test_config.py
git commit -m "feat(config): validate DESIGN_CASES_SHARDS via validate_env"
```

---

### Task 3: Create the shard prompt template

**Files:**
- Create: `src/styleclaw/providers/llm/prompts/design_cases_shard.md`

No automated test for this task — content is exercised by Task 5+ integration tests.

- [ ] **Step 1: Create the template file**

Write `src/styleclaw/providers/llm/prompts/design_cases_shard.md` with this exact content:

````markdown
You are an expert at designing diverse test cases for AI image generation. Given the IP information and a style trigger phrase, design character and scene descriptions for batch testing.

## Shard Context

You are worker {shard_index}/{total_shards} designing cases for {shard_category_count} of the 10 total categories. The remaining categories are handled by other workers running in parallel. Do NOT generate cases for categories outside your assigned set listed below.

Other workers have no visibility into your output and vice versa. Lean toward unusual or less-obvious subjects within your assigned categories so global diversity across the full batch is preserved. Avoid clichés (e.g. "red-clothed swordswoman in bamboo forest", "old monk meditating under a waterfall") that other workers are statistically likely to also pick.

## Task

Fill in the `description` field for each test case below. Each description should be:
- 50-150 characters in English
- Specific enough to test style consistency across diverse subjects
- Varied WITHIN each category (different poses, expressions, settings, etc.)

## IP Information

{ip_info}

## Trigger Phrase (will be prepended automatically, do NOT include it)

{trigger_phrase}

## Categories and Cases

{case_skeleton}
{feedback_section}
## Rules

1. Descriptions should describe the CHARACTER or SCENE only, not the style.
2. Within each category, ensure variety (different ages, body types, clothing, actions, environments).
3. For character categories: describe appearance, pose, clothing, action.
4. For scene categories: describe setting, time of day, weather, objects, mood.
5. For group: describe number of characters, relationships, interaction, setting.
6. **CRITICAL — Generalization testing**: Within this shard's {shard_cases} cases, AT MOST 1 may reference IP-specific elements (e.g., costumes, props, or settings directly tied to the IP). The rest MUST describe completely original, diverse characters and scenes with NO connection to the IP. This tests whether the style trigger generalizes beyond the source material.

## Output Format

Return ONLY valid JSON (no markdown fences):

```
{
  "cases": [
    {
      "id": "case-adult_male-01",
      "category": "adult_male",
      "description": "A tall man in a dark suit standing on a rainy street corner, holding an umbrella",
      "aspect_ratio": "9:16"
    }
  ]
}
```

Return ALL {shard_cases} cases for the categories listed in the "Categories and Cases" section above. Do not include cases from any other category.
````

- [ ] **Step 2: Verify the file is parseable**

Run: `uv run python -c "from pathlib import Path; t = Path('src/styleclaw/providers/llm/prompts/design_cases_shard.md').read_text(encoding='utf-8'); assert '{shard_index}' in t and '{total_shards}' in t and '{case_skeleton}' in t and '{ip_info}' in t and '{trigger_phrase}' in t and '{feedback_section}' in t and '{shard_cases}' in t and '{shard_category_count}' in t; print('placeholders ok')"`

Expected output: `placeholders ok`

- [ ] **Step 3: Commit**

```bash
git add src/styleclaw/providers/llm/prompts/design_cases_shard.md
git commit -m "feat(prompts): add per-shard design_cases template"
```

---

### Task 4: Refactor `design_cases.py` to fan out into shards

**Files:**
- Rewrite: `src/styleclaw/agents/design_cases.py`
- Test (new tests): `tests/agents/test_design_cases.py`

This is the largest task. It writes the shard-aware tests first, then implements.

- [ ] **Step 1: Write the failing tests**

Append a new test class to `tests/agents/test_design_cases.py`:

```python
import asyncio
from unittest.mock import AsyncMock

import pytest

from styleclaw.agents.design_cases import design_cases
from styleclaw.core.case_generator import CATEGORIES


def _shard_response(category_ids: list[str], cases_per_category: int = 10) -> str:
    """Build a JSON response covering exactly the given categories."""
    cases = []
    for cat_id in category_ids:
        aspect = next(c["aspect"] for c in CATEGORIES if c["id"] == cat_id)
        for i in range(1, cases_per_category + 1):
            cases.append({
                "id": f"case-{cat_id}-{i:02d}",
                "category": cat_id,
                "description": f"placeholder description for {cat_id} #{i:02d}",
                "aspect_ratio": aspect,
            })
    return json.dumps({"cases": cases})


class TestShardedDesignCases:
    """Default config (SHARDS=5) — verify fan-out, partitioning, merging."""

    @pytest.fixture
    def mock_llm_5shards(self) -> AsyncMock:
        """Returns 5 distinct shard responses, one per call."""
        llm = AsyncMock()
        # Partition CATEGORIES into 5 contiguous slices of 2.
        cat_ids = [c["id"] for c in CATEGORIES]
        partitions = [cat_ids[i:i+2] for i in range(0, 10, 2)]
        llm.invoke.side_effect = [_shard_response(p) for p in partitions]
        return llm

    async def test_default_shards_makes_5_calls(self, mock_llm_5shards) -> None:
        await design_cases(mock_llm_5shards, "anime", "trigger", batch_num=1)
        assert mock_llm_5shards.invoke.call_count == 5

    async def test_combines_to_100_cases(self, mock_llm_5shards) -> None:
        result = await design_cases(mock_llm_5shards, "anime", "trigger", batch_num=1)
        assert len(result.cases) == 100
        # Every category present with exactly 10 cases.
        from collections import Counter
        per_cat = Counter(c.category for c in result.cases)
        assert all(per_cat[c["id"]] == 10 for c in CATEGORIES), per_cat

    async def test_each_shard_prompt_has_only_its_categories(
        self, mock_llm_5shards
    ) -> None:
        await design_cases(mock_llm_5shards, "anime", "trigger", batch_num=1)
        cat_ids = [c["id"] for c in CATEGORIES]
        expected_partitions = [cat_ids[i:i+2] for i in range(0, 10, 2)]
        for call, expected_cats in zip(
            mock_llm_5shards.invoke.call_args_list, expected_partitions
        ):
            system = call.kwargs["system"]
            # The two assigned categories appear in the prompt skeleton.
            for cat in expected_cats:
                assert cat in system, f"expected {cat} in shard system prompt"
            # No other category appears as an "### <name>" header.
            others = [c for c in cat_ids if c not in expected_cats]
            for other in others:
                assert f"### {other} " not in system, (
                    f"unexpected category {other} in shard prompt for {expected_cats}"
                )

    async def test_each_shard_prompt_has_worker_marker(
        self, mock_llm_5shards
    ) -> None:
        await design_cases(mock_llm_5shards, "anime", "trigger", batch_num=1)
        markers = ["worker 1/5", "worker 2/5", "worker 3/5", "worker 4/5", "worker 5/5"]
        seen = {m: False for m in markers}
        for call in mock_llm_5shards.invoke.call_args_list:
            system = call.kwargs["system"]
            for m in markers:
                if m in system:
                    seen[m] = True
        assert all(seen.values()), f"missing worker markers: {seen}"

    async def test_single_shard_failure_propagates(self) -> None:
        llm = AsyncMock()
        cat_ids = [c["id"] for c in CATEGORIES]
        partitions = [cat_ids[i:i+2] for i in range(0, 10, 2)]
        # Third shard raises.
        side = [_shard_response(partitions[0]), _shard_response(partitions[1])]
        side.append(RuntimeError("provider exhausted retries"))
        side.extend([_shard_response(partitions[3]), _shard_response(partitions[4])])
        llm.invoke.side_effect = side
        with pytest.raises(RuntimeError, match="provider exhausted retries"):
            await design_cases(llm, "anime", "trigger", batch_num=1)

    async def test_wrong_category_in_shard_rejected(self) -> None:
        llm = AsyncMock()
        cat_ids = [c["id"] for c in CATEGORIES]
        partitions = [cat_ids[i:i+2] for i in range(0, 10, 2)]
        # First shard returns cases from a category it doesn't own (e.g. group).
        bad_first = _shard_response(["adult_male", "group"])
        side = [bad_first] + [_shard_response(p) for p in partitions[1:]]
        llm.invoke.side_effect = side
        with pytest.raises(ValueError, match="outside its assigned categories"):
            await design_cases(llm, "anime", "trigger", batch_num=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/agents/test_design_cases.py::TestShardedDesignCases -v`

Expected: 6 failures (current implementation makes 1 call, not 5).

- [ ] **Step 3: Rewrite `agents/design_cases.py`**

Replace the full content of `src/styleclaw/agents/design_cases.py` with:

```python
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from styleclaw.core.case_generator import CASES_PER_CATEGORY, CATEGORIES, generate_case_skeleton
from styleclaw.core.config import DESIGN_CASES_SHARDS
from styleclaw.core.models import BatchCase, BatchConfig
from styleclaw.core.text_utils import clean_json, recover_truncated_json, sanitize_braces
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "design_cases_shard.md"
)


async def design_cases(
    llm: LLMProvider,
    ip_info: str,
    trigger_phrase: str,
    batch_num: int,
    feedback: str = "",
) -> BatchConfig:
    shards = DESIGN_CASES_SHARDS
    cat_ids = [c["id"] for c in CATEGORIES]
    cats_per_shard = len(CATEGORIES) // shards
    partitions: list[list[str]] = [
        cat_ids[i : i + cats_per_shard]
        for i in range(0, len(cat_ids), cats_per_shard)
    ]

    feedback_section = _build_feedback_section(feedback)
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    coros = [
        _design_one_shard(
            llm=llm,
            template=template,
            ip_info=ip_info,
            trigger_phrase=trigger_phrase,
            feedback_section=feedback_section,
            shard_index=i + 1,
            total_shards=shards,
            shard_categories=partition,
        )
        for i, partition in enumerate(partitions)
    ]
    shard_results: list[list[BatchCase]] = await asyncio.gather(*coros)

    merged: list[BatchCase] = []
    for cases in shard_results:
        merged.extend(cases)

    expected_total = len(CATEGORIES) * CASES_PER_CATEGORY
    if len(merged) != expected_total:
        raise ValueError(
            f"design_cases expected {expected_total} total cases across "
            f"{shards} shards, got {len(merged)}"
        )

    logger.info("Designed %d test cases for batch %d (%d shards).",
                len(merged), batch_num, shards)
    return BatchConfig(
        batch=batch_num,
        trigger_phrase=trigger_phrase,
        cases=merged,
    )


async def _design_one_shard(
    *,
    llm: LLMProvider,
    template: str,
    ip_info: str,
    trigger_phrase: str,
    feedback_section: str,
    shard_index: int,
    total_shards: int,
    shard_categories: list[str],
) -> list[BatchCase]:
    shard_cases = len(shard_categories) * CASES_PER_CATEGORY
    skeleton = [
        case
        for case in generate_case_skeleton()
        if case.category in shard_categories
    ]
    skeleton_text = _format_skeleton(skeleton)

    system_prompt = (
        template
        .replace("{ip_info}", sanitize_braces(ip_info))
        .replace("{trigger_phrase}", trigger_phrase)
        .replace("{case_skeleton}", skeleton_text)
        .replace("{feedback_section}", feedback_section)
        .replace("{shard_index}", str(shard_index))
        .replace("{total_shards}", str(total_shards))
        .replace("{shard_category_count}", str(len(shard_categories)))
        .replace("{shard_cases}", str(shard_cases))
    )

    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"Design {shard_cases} diverse test cases for this shard."},
    ]}]

    raw = await llm.invoke(system=system_prompt, messages=messages, max_tokens=4096)

    cleaned = clean_json(raw)
    recovered = recover_truncated_json(cleaned)
    data = json.loads(recovered)
    if "cases" not in data:
        raise ValueError(f"shard {shard_index} LLM response missing 'cases' key")
    cases = [BatchCase.model_validate(c) for c in data["cases"]]
    if not cases:
        raise ValueError(
            f"shard {shard_index} returned zero cases — response may have been truncated."
        )

    allowed = set(shard_categories)
    stray = [c for c in cases if c.category not in allowed]
    if stray:
        raise ValueError(
            f"shard {shard_index} returned cases outside its assigned categories "
            f"{sorted(allowed)}: got {sorted({c.category for c in stray})}"
        )

    return cases


def _build_feedback_section(feedback: str) -> str:
    if not feedback.strip():
        return ""
    return (
        f"\n\n## User feedback on previous batch\n\n{sanitize_braces(feedback)}\n\n"
        "Apply this feedback when designing the new batch — adjust subjects, "
        "scenes, or angles accordingly while keeping the generalization rule."
    )


def _format_skeleton(cases: list[BatchCase]) -> str:
    lines: list[str] = []
    current_cat = ""
    for c in cases:
        if c.category != current_cat:
            current_cat = c.category
            lines.append(f"\n### {current_cat} (aspect: {c.aspect_ratio})")
        lines.append(f"- {c.id}: (fill in description)")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run python -m pytest tests/agents/test_design_cases.py::TestShardedDesignCases -v`

Expected: 6 PASS.

Note: the existing `TestDesignCases` and `TestFormatSkeleton` classes will still fail at this point because their mocks only return 10 cases — Task 5 fixes them.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/agents/design_cases.py tests/agents/test_design_cases.py
git commit -m "feat(agents): shard design_cases across N parallel LLM calls"
```

---

### Task 5: Restore existing tests via SHARDS=1 fixture

**Files:**
- Modify: `tests/agents/test_design_cases.py` (existing `TestDesignCases` class — add autouse fixture, update mock)
- Modify: `tests/agents/test_design_cases_error.py` (add autouse fixture)

The existing tests treat `design_cases` as a single LLM call. Force `DESIGN_CASES_SHARDS=1` for those tests so one mock invocation drives the full result, then update the mock to return the full 100-case skeleton (otherwise the new "total != 100" validation will reject).

- [ ] **Step 1: Update `tests/agents/test_design_cases.py` existing fixture and add autouse fixture**

Replace the existing `mock_llm` fixture and add an autouse fixture for `TestDesignCases`:

```python
@pytest.fixture(autouse=False)
def force_single_shard(monkeypatch):
    monkeypatch.setattr(
        "styleclaw.agents.design_cases.DESIGN_CASES_SHARDS", 1
    )


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Mock that returns a complete 100-case response covering all 10 categories."""
    from styleclaw.core.case_generator import CATEGORIES
    cases = []
    for cat in CATEGORIES:
        for i in range(1, 11):
            cases.append({
                "id": f"case-{cat['id']}-{i:02d}",
                "category": cat["id"],
                "description": f"placeholder {cat['id']} #{i:02d}",
                "aspect_ratio": cat["aspect"],
            })
    llm = AsyncMock()
    llm.invoke.return_value = json.dumps({"cases": cases})
    return llm


class TestDesignCases:
    @pytest.fixture(autouse=True)
    def _force_single_shard(self, force_single_shard):
        return force_single_shard

    async def test_returns_batch_config(self, mock_llm) -> None:
        result = await design_cases(mock_llm, "anime", "bold style", batch_num=1)
        assert result.batch == 1
        assert result.trigger_phrase == "bold style"
        assert len(result.cases) == 100

    async def test_ip_info_in_system_prompt(self, mock_llm) -> None:
        await design_cases(mock_llm, "Spider-Verse", "trigger", batch_num=1)
        call_args = mock_llm.invoke.call_args
        assert "Spider-Verse" in call_args.kwargs["system"]

    async def test_no_feedback_section_when_empty(self, mock_llm) -> None:
        await design_cases(mock_llm, "anime", "x", batch_num=1)
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        assert "{feedback_section}" not in sys_prompt
        assert "User feedback on previous batch" not in sys_prompt

    async def test_feedback_appended_to_prompt(self, mock_llm) -> None:
        feedback = "上一批室内场景太少，多加一些咖啡馆和书房"
        await design_cases(mock_llm, "anime", "x", batch_num=2, feedback=feedback)
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        assert "User feedback on previous batch" in sys_prompt
        assert feedback in sys_prompt

    async def test_whitespace_only_feedback_treated_as_empty(self, mock_llm) -> None:
        await design_cases(mock_llm, "anime", "x", batch_num=2, feedback="   \n  ")
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        assert "User feedback on previous batch" not in sys_prompt
```

The `_format_skeleton` import in `TestFormatSkeleton` still works — the function name is preserved in the rewrite. No change needed there.

- [ ] **Step 2: Update `tests/agents/test_design_cases_error.py` to force SHARDS=1**

At the top of the file, add the autouse fixture inside `TestDesignCasesErrorRecovery`:

```python
class TestDesignCasesErrorRecovery:
    @pytest.fixture(autouse=True)
    def _force_single_shard(self, monkeypatch):
        monkeypatch.setattr(
            "styleclaw.agents.design_cases.DESIGN_CASES_SHARDS", 1
        )

    # ... existing tests unchanged ...
```

The existing 6 tests in this class mock arbitrary LLM responses (truncated JSON, garbage, empty cases). With SHARDS=1, exactly one shard runs. The shard validation now wraps these errors in additional context, but pytest's `match=` regex remains satisfied because the underlying error messages still contain "zero cases", `JSONDecodeError`, etc.

**One adjustment is required**: `test_recovers_from_truncated_json` returns one valid case under category `adult_male`. With SHARDS=1, the merged-total check expects 100 cases. To keep this test passing as-written, change the assertion at the end of the test from:

```python
        config = await design_cases(mock_llm, "anime", "bold style", 1)
        assert len(config.cases) == 1
        assert config.cases[0].id == "am-01"
```

to assert that the call raises a `ValueError` mentioning the total mismatch:

```python
        with pytest.raises(ValueError, match="expected 100 total cases"):
            await design_cases(mock_llm, "anime", "bold style", 1)
```

Apply the same change to `test_valid_response_parses_normally` (currently asserts 1 case; change to assert that a `ValueError` is raised mentioning the total mismatch), since the new behavior is to reject anything under 100. Update the test name to `test_valid_short_response_rejected_by_total_check`.

- [ ] **Step 3: Run all existing design_cases tests**

Run: `uv run python -m pytest tests/agents/test_design_cases.py tests/agents/test_design_cases_error.py -v`

Expected: all PASS — the original 5 `TestDesignCases` tests, the 6 new `TestShardedDesignCases` tests, the 2 `TestFormatSkeleton` tests, and the 6 `TestDesignCasesErrorRecovery` tests (with 2 of them now asserting `ValueError` instead of success).

- [ ] **Step 4: Commit**

```bash
git add tests/agents/test_design_cases.py tests/agents/test_design_cases_error.py
git commit -m "test(design_cases): align legacy tests with sharding refactor"
```

---

### Task 6: Run full suite + remove unused old template

**Files:**
- Delete: `src/styleclaw/providers/llm/prompts/design_cases.md`

- [ ] **Step 1: Confirm no remaining references to the old template**

Run: `grep -rn "design_cases.md" src/ tests/ 2>&1`

Expected: zero matches (`agents/design_cases.py` now uses `design_cases_shard.md`).

If any reference exists, fix it before deleting.

- [ ] **Step 2: Delete the old template**

```bash
git rm src/styleclaw/providers/llm/prompts/design_cases.md
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run python -m pytest tests/ -v`

Expected: full suite green. If anything outside `tests/agents/` or `tests/core/` fails, investigate before committing — likely an import-time side effect from the renamed prompt file.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(prompts): delete obsolete single-call design_cases template"
```

---

### Task 7: Update CLAUDE.md and .env.example

**Files:**
- Modify: `CLAUDE.md` (Runtime Tunables table)
- Modify: `.env.example`

- [ ] **Step 1: Add row to CLAUDE.md Runtime Tunables table**

In `CLAUDE.md`, locate the Runtime Tunables markdown table. Add this row after `STYLECLAW_LLM_IMAGE_CACHE`:

```
| `STYLECLAW_DESIGN_CASES_SHARDS` | `5` | Number of parallel LLM shards for `design_cases`. Must evenly divide the 10 fixed categories — allowed values: 1, 2, 5, 10. Lower = simpler/cheaper but larger per-request token budgets; higher = more parallelism. |
```

- [ ] **Step 2: Add entry to .env.example**

Append to `.env.example`:

```
# Number of parallel LLM shards used by design_cases.
# Each shard generates 100/N cases. Allowed values: 1, 2, 5, 10. Default 5.
# STYLECLAW_DESIGN_CASES_SHARDS=5
```

- [ ] **Step 3: Verify CLAUDE.md still renders correctly (no broken table)**

Run: `uv run python -c "import re; t = open('CLAUDE.md', encoding='utf-8').read(); rows = re.findall(r'^\\| \`STYLECLAW_', t, re.M); print(f'{len(rows)} STYLECLAW_ env rows in table')"`

Expected: 18+ rows (was 17, +1 for our new entry).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md .env.example
git commit -m "docs: document STYLECLAW_DESIGN_CASES_SHARDS"
```

---

## Self-Review Notes

Cross-checked against the spec:
- ✅ Public API unchanged (Task 4 keeps signature)
- ✅ Default N=5 (Task 1)
- ✅ Validate 1/2/5/10 via validate_env (Task 2)
- ✅ Shard prompt template with worker context (Task 3)
- ✅ asyncio.gather with provider semaphore (Task 4 — no extra Semaphore)
- ✅ Per-shard category validation (Task 4 — "stray" check)
- ✅ Total count validation (Task 4 — `expected_total` check)
- ✅ Failure propagation, no fallback (Task 4 — gather raises; Test in Task 4 Step 1)
- ✅ Existing tests adjusted via monkeypatch SHARDS=1 (Task 5)
- ✅ Old template deleted (Task 6)
- ✅ Docs updated (Task 7)

Type/name consistency checked:
- `DESIGN_CASES_SHARDS` referenced identically across config, agent, tests
- `_design_one_shard` is internal helper, not exposed in any test import
- `CATEGORIES` and `CASES_PER_CATEGORY` imported from `core.case_generator` (already exported there)
- Prompt placeholders `{shard_index}`, `{total_shards}`, `{shard_category_count}`, `{shard_cases}` consistently used in template (Task 3) and replaced (Task 4)
