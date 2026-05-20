# design-cases parallel sharding

**Date:** 2026-05-20
**Status:** Approved, pending implementation
**Author:** Claude (with user)

## Problem

`agents/design_cases.py` issues a single LLM request with `max_tokens=16384`, asking the model to produce all 100 `BatchCase` descriptions at once. This is fragile against the OpenAI-compatible provider's transient pressure: a single 429 / 500 / "Overloaded" response burns the entire batch. Observed in the wushan run (2026-05-20): `design-cases` succeeded but the same gateway returned 500 on an adjacent call.

A single large request also has high tail latency — the model must produce ~12 KB of JSON before the first byte of the *next* shard's content is available.

## Goal

Reduce per-request size and overall wall-clock by splitting the work across N parallel LLM calls (default N=5), each producing 20 cases. Keep the public function signature and downstream contract identical so `orchestrator/actions.py` and `cli.py` are untouched.

## Non-goals

- No cross-shard deduplication (LLM call or embedding). Diversity is a prompt-level concern only.
- No fallback to single-shard path. Provider retry (already 3x) is the only recovery mechanism.
- No CLI flag. Config is environment-only.

## Design

### Public API (unchanged)

```python
async def design_cases(
    llm: LLMProvider,
    ip_info: str,
    trigger_phrase: str,
    batch_num: int,
    feedback: str = "",
) -> BatchConfig: ...
```

Returned `BatchConfig.cases` length is 100 with 10 per category, same as today.

### Internal flow

1. Read `STYLECLAW_DESIGN_CASES_SHARDS` (default 5) from `core/config.py`.
2. Validate shard count divides 10 evenly (allowed: 1, 2, 5, 10). Invalid → `ValueError` at startup (consistent with how `_int_env` rejects malformed values).
3. Partition the 10 categories into N contiguous slices of size `10 // N`. Stable ordering by `CATEGORIES` list index.
4. For each partition, build a system prompt from a new template `design_cases_shard.md` parameterized by:
   - `ip_info`, `trigger_phrase`, `feedback_section` (same as today)
   - `case_skeleton` (only this shard's categories)
   - `total_shards`, `shard_index` (1-based, for diversity hint)
   - `shard_cases` (20 by default)
5. Fire all N requests via `asyncio.gather(*coros)`. The provider's existing `LLM_CONCURRENCY_LIMIT` semaphore (currently 6 in user's env, default 4) handles outbound limiting — no extra semaphore in `design_cases`.
6. Validate each shard's output: parse JSON, recover truncation, assert each shard returned its expected cases (count + categories).
7. Concatenate results in partition order. Final list is exactly 100 cases, 10 per category.
8. Construct and return `BatchConfig`.

### Prompt template (`design_cases_shard.md`)

New file under `src/styleclaw/providers/llm/prompts/`. Differs from `design_cases.md`:

- Replace "ALL 100 cases" → "ALL {shard_cases} cases in your shard".
- Add a "## Shard Context" section before "## Rules":
  > You are worker {shard_index}/{total_shards} designing cases for {shard_category_count} of 10 categories. The remaining categories are handled by other workers in parallel. Do NOT generate cases for categories outside your assigned set.
  >
  > Other workers have no visibility into your output and vice versa. Lean toward unusual or less-obvious subjects within your assigned categories so global diversity is preserved across the full batch. Avoid clichés (e.g. "red-clothed swordswoman in bamboo forest", "old monk meditating under a waterfall") that other workers are statistically likely to also pick.
- Adjust the IP-generalization rule from "1-2 of 100" to "at most 1 of your {shard_cases} may reference IP-specific elements; the rest must be original." This loosens the global cap from 1-2/100 to ≤N/100 in worst case, but in practice each shard rarely fills its IP slot, so the global count stays around 2-5 — acceptable.

Old `design_cases.md` is deleted after migration (no backward-compat path).

### Configuration

Add to `core/config.py`:

```python
DESIGN_CASES_SHARDS: int = _int_env("STYLECLAW_DESIGN_CASES_SHARDS", "5")
if DESIGN_CASES_SHARDS not in (1, 2, 5, 10):
    raise RuntimeError(
        f"STYLECLAW_DESIGN_CASES_SHARDS={DESIGN_CASES_SHARDS} must be one of "
        f"1, 2, 5, or 10 (each evenly partitions the 10 fixed categories)."
    )
```

Whitelist comparison avoids a `ZeroDivisionError` if someone sets the env to `0`, and is more readable than `10 % N != 0`.

Update CLAUDE.md and .env.example with the new variable. README updates are not required for this change (internal optimization, no user-facing flow change).

### Failure handling

| Failure | Behavior |
|---|---|
| Single shard returns 429 / 500 / network error | Provider's 3x retry absorbs it. No additional retry logic in `design_cases`. |
| Shard exhausts provider retries | `asyncio.gather` propagates exception. Caller sees `RuntimeError`. User reruns `design-cases --feedback ...` which creates batch-NNN+1 (existing behavior). |
| LLM returns wrong category in a shard | Post-shard validation rejects: raise `ValueError("shard N returned cases outside its assigned categories")`. |
| Shard returns fewer cases than expected | Existing `recover_truncated_json` attempts repair; if still short, `ValueError`. |
| Total cases ≠ 100 after concat | `ValueError("expected 100 cases, got N")`. |

### Concurrency

No new primitives. `OpenAICompatProvider._semaphore` already serializes outbound calls to `LLM_CONCURRENCY_LIMIT`. With 5 shards and limit ≥5, all fire in parallel. With limit <5, the semaphore queues them — still cheaper than the current single huge request because each smaller request finishes faster.

## Testing

Update `tests/agents/test_design_cases.py`:

- Existing tests (5 total): Default config has SHARDS=5, so the existing `mock_llm` (returns 10 cases) becomes inadequate. Two paths:
  1. Set `monkeypatch.setattr("styleclaw.core.config.DESIGN_CASES_SHARDS", 1)` in the fixture → existing assertions still pass against the 10-case mock.
  2. Or: update the mock to return 20 cases per call across 5 calls (more faithful). Prefer (1) for minimal churn to existing assertions.

- New tests:
  - `test_default_shards_makes_5_calls`: `mock_llm.invoke.call_count == 5`.
  - `test_shard_prompt_contains_only_assigned_categories`: inspect each `invoke.call_args.kwargs["system"]`, assert exactly 2 category labels appear in `## Categories and Cases` section.
  - `test_shard_prompt_contains_shard_context`: each call's system prompt includes "worker N/5".
  - `test_combines_to_100_cases`: with 5 calls each returning 20 cases, final `BatchConfig.cases` length == 100, with 10 per category.
  - `test_invalid_shard_count_rejected_at_config_load`: setting `STYLECLAW_DESIGN_CASES_SHARDS=3` raises before any LLM call (verify by reloading `core.config`).
  - `test_single_shard_failure_propagates`: mock one of 5 `invoke` calls to raise `RuntimeError`, assert `design_cases` raises (do not swallow).
  - `test_wrong_category_in_shard_rejected`: mock a shard returning cases from a category not in its partition, assert `ValueError`.

`test_design_cases_error.py` continues to cover truncation/recovery — that logic is unchanged per-shard.

No changes to `tests/orchestrator/test_actions_do.py` (it mocks `design_cases` itself).

## Trade-offs accepted

- **Token cost ↑**: 5 system prompts instead of 1. Input tokens roughly 4× the single-call baseline because the IP info / trigger / shard rules repeat. Output tokens unchanged (still 100 cases total). Acceptable: provider pricing is dominated by output tokens in this workload.
- **Diversity floor**: pure prompt-level control. If LLM ignores the "avoid clichés" hint, two shards can produce similar subjects. We do not detect this. User can reroll via `design-cases --feedback "more variety please"` if needed.
- **IP-reference cap loosens**: from "1-2/100" to "≤N/100". In practice the LLM rarely consumes its IP slot, so the effective rate stays around 2-5/100. Tightening this would require a global coordinator (or post-merge fix-up), which we explicitly chose not to build.

## Migration

1. Add `DESIGN_CASES_SHARDS` to config.
2. Add `design_cases_shard.md` prompt template.
3. Refactor `agents/design_cases.py` per "Internal flow" above.
4. Delete `design_cases.md` (old template).
5. Update tests.
6. Update CLAUDE.md table + `.env.example`.
7. Run `uv run python -m pytest tests/ -v` — all green.

After merge, existing wushan project rerun via `design-cases --feedback "..."` will use new path automatically. No data migration needed.
