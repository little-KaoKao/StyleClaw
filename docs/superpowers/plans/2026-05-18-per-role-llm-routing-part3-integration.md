# Per-Role LLM Routing — Part 3: ExecutionContext + cli + Actions Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the `RoleRouter` from Part 1 into the running system. After this part, every LLM call in StyleClaw resolves through `ctx.llm_router.get(Role.X)` (or `get_panel`), and each persisted artifact records the actual model used.

**Architecture:** Two-phase integration. First (Tasks 1-2) we add `llm_router` to `ExecutionContext` *alongside* the legacy `llm` field so existing actions keep working — the test suite stays green between commits. Then (Tasks 3-6) each `do_*` action switches to the router individually. Finally (Tasks 7-9) we drop the legacy `llm` field and delete the now-orphan `panel_factory.py`.

**Tech Stack:** Same as parts 1-2. Most touchpoints are mechanical refactors; the only new behavior is `model_id` recording on artifacts.

**Reference:** [spec](../specs/2026-05-18-per-role-llm-routing-design.md) — sections "Wiring: ExecutionContext", "Per-action injection", "Artifact Recording".

**Dependencies:** Parts 1 and 2 must be complete. `RoleRouter` exists with `get` / `get_panel` / `close`, and the 5 Pydantic models have `model_id` fields.

---

## File Structure

- **Modify:** `src/styleclaw/orchestrator/actions.py` — `ExecutionContext` gains `llm_router`; 4 `do_*` actions switch to router; `model_id` populated post-parse
- **Modify:** `src/styleclaw/orchestrator/executor.py` — accept either `ctx.llm_router` or `ctx.llm` during transition
- **Modify:** `src/styleclaw/cli.py` — `_build_context` builds `RoleRouter`; `cli.run` plans via `router.get(Role.PLANNER)`; eventually remove `_build_llm_provider`
- **Delete:** `src/styleclaw/providers/llm/panel_factory.py` (orphaned by router.get_panel)
- **Delete:** `tests/providers/llm/test_panel_factory.py`
- **Modify:** Tests for the 4 actions (use `MockRouter` from `tests/orchestrator/_routing_helpers.py`) and any tests stubbing `ExecutionContext.llm`

---

## Task 1: Add `llm_router` field to `ExecutionContext` (transitional)

**Files:**
- Modify: `src/styleclaw/orchestrator/actions.py` (extend dataclass)
- Modify: `src/styleclaw/orchestrator/executor.py` (relax `needs_llm` check)
- Modify: `tests/orchestrator/test_actions.py` (add field to defaults test)

- [ ] **Step 1: Write the failing test**

Append to `tests/orchestrator/test_actions.py` (inside the existing `TestExecutionContextThinking` class — rename it `TestExecutionContextFields` if you want, but keeping the class lets you add tests next to existing ones):

```python
    def test_llm_router_field_defaults_none(self):
        from styleclaw.orchestrator.actions import ExecutionContext
        ctx = ExecutionContext(project="p")
        assert ctx.llm_router is None

    def test_llm_router_can_be_set(self):
        from styleclaw.core.llm_routing import RoleRouter
        from styleclaw.orchestrator.actions import ExecutionContext

        router = RoleRouter.from_env()
        ctx = ExecutionContext(project="p", llm_router=router)
        assert ctx.llm_router is router
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/orchestrator/test_actions.py::TestExecutionContextThinking -v`
Expected: FAIL — `TypeError: ExecutionContext.__init__() got an unexpected keyword argument 'llm_router'`

- [ ] **Step 3: Write the implementation**

Modify `src/styleclaw/orchestrator/actions.py::ExecutionContext` (around line 28). Add the `llm_router` field while keeping the existing `llm` field for now:

```python
@dataclass(frozen=True)
class ExecutionContext:
    project: str
    client: RunningHubClient | None = None
    llm: LLMProvider | None = None  # legacy; will be removed in Task 8
    llm_router: "RoleRouter | None" = None
    poll_interval: float = ORCHESTRATOR_POLL_INTERVAL
    show_thinking: bool = False
    thinking_budget: int = 5000
```

At the top of `actions.py`, add the `RoleRouter` import inside a `TYPE_CHECKING` block so we don't trip the existing circular-import shape:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from styleclaw.core.llm_routing import RoleRouter
```

Then update `src/styleclaw/orchestrator/executor.py` (around line 184) to accept either source:

```python
        if action_def.needs_llm and ctx.llm is None and ctx.llm_router is None:
            result = StepResult(
                ok=False,
                message=(
                    f"Action '{step.name}' requires an LLM provider but "
                    f"neither ctx.llm nor ctx.llm_router was supplied"
                ),
            )
            results.append(result)
            if on_step_done:
                on_step_done(i, step.name, result)
            return result
```

(Keep the old `ctx.llm is None` arm so existing tests that set only `llm=` still work.)

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run python -m pytest tests/orchestrator/test_actions.py tests/orchestrator/test_executor.py -v`
Expected: PASS — new tests green; the executor check still permits old `ctx.llm`-only callers.

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/orchestrator/actions.py src/styleclaw/orchestrator/executor.py \
        tests/orchestrator/test_actions.py
git commit -m "$(cat <<'EOF'
refactor(orchestrator): add llm_router field to ExecutionContext

Transitional shape: both ExecutionContext.llm and ExecutionContext.llm_router
exist. Executor's needs_llm guard accepts either. Per-action switch follows
in Tasks 3-6; legacy llm field is removed in Task 8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `cli._build_context` constructs `RoleRouter`

**Files:**
- Modify: `src/styleclaw/cli.py` (build router, pass to context, close on teardown)
- No test changes — existing CLI tests stub at the action layer, not the router.

- [ ] **Step 1: Update `_build_context`**

Replace `src/styleclaw/cli.py::_build_context` (around line 103) with a version that always builds a router and exposes it on the context:

```python
@asynccontextmanager
async def _build_context(
    project: str,
    needs_client: bool = False,
    needs_llm: bool = False,
    show_thinking: bool = False,
    thinking_budget: int = 5000,
    existing_router: "RoleRouter | None" = None,
) -> AsyncIterator[ExecutionContext]:
    from styleclaw.core.llm_routing import RoleRouter
    from styleclaw.providers.runninghub.client import RunningHubClient

    client = None
    router = existing_router
    owns_router = False
    try:
        if needs_client:
            client = RunningHubClient(api_key=_get_api_key())
        if needs_llm and router is None:
            router = RoleRouter.from_env()
            owns_router = True
        yield ExecutionContext(
            project=project,
            client=client,
            llm_router=router,
            show_thinking=show_thinking,
            thinking_budget=thinking_budget,
        )
    finally:
        if client is not None:
            await _close_resource(client, "client")
        if router is not None and owns_router:
            await _close_resource(router, "llm_router")
```

Note: `_close_resource` already handles "has async close method", so router's `close()` works through the existing helper without changes.

Add at the top of `cli.py` (with other TYPE_CHECKING imports — there aren't any today, so just import `RoleRouter` lazily inside `_build_context`, which is what the snippet above does already).

- [ ] **Step 2: Update `_run_action` to use the new context shape**

`_run_action` (around line 170) passes `show_thinking` / `thinking_budget` already and doesn't reference llm directly; it just needs to not pass `existing_llm` anymore. Replace its `_exec` body's `_build_context` call (around line 202) — drop the `existing_llm=` argument; no other changes:

```python
    async def _exec() -> StepResult:
        async with _build_context(
            project,
            needs_client=action_def.needs_client,
            needs_llm=action_def.needs_llm,
            show_thinking=show_thinking,
            thinking_budget=thinking_budget,
        ) as ctx:
            results = await execute(plan, ctx)
            return results[-1] if results else StepResult(
                ok=False, message="executor returned no result",
            )
```

- [ ] **Step 3: Update `cli.run` planner path**

This is where `_build_llm_provider()` is currently called outside `_build_context` to build an llm for `plan()`. Replace the planner-aware portion of `cli.run::_plan_and_execute` (around lines 1368-1426) — the diff is the llm build at the top and the `existing_llm` argument at the bottom:

```python
    async def _plan_and_execute() -> None:
        from styleclaw.core.llm_routing import Role, RoleRouter

        router = RoleRouter.from_env()
        try:
            action_plan = await plan(router.get(Role.PLANNER), project, intent)

            display_plan(action_plan, project)

            if dry_run:
                typer.echo("(dry-run) 未执行；去掉 --dry-run 后再跑即可")
                return

            if audit is not None:
                audit.record_plan(action_plan)

            if not yes and not typer.confirm("Execute?"):
                typer.echo("Cancelled.")
                if audit is not None:
                    audit.cancelled()
                raise typer.Exit(0)

            needs_client = any(
                ACTION_REGISTRY.get(s.name) and ACTION_REGISTRY[s.name].needs_client
                for s in action_plan.steps
            )
            needs_llm = any(
                ACTION_REGISTRY.get(s.name) and ACTION_REGISTRY[s.name].needs_llm
                for s in action_plan.steps
            )

            def _on_start(i: int, name: str, desc: str) -> None:
                typer.echo(f"\n  [{i + 1}/{len(action_plan.steps)}] {name} — {desc}")
                if audit is not None:
                    audit.step_started(i)

            def _on_done(i: int, name: str, result: StepResult) -> None:
                if result.ok:
                    typer.echo(f"  -> {result.message}")
                else:
                    typer.echo(f"  x  {result.message}", err=True)
                if audit is not None:
                    audit.step_finished(i, name, result.ok, result.message)

            confirm_fn = None if yes else _confirm_dispatch

            async with _build_context(
                project, needs_client, needs_llm,
                show_thinking=show_thinking, thinking_budget=thinking_budget,
                existing_router=router if needs_llm else None,
            ) as ctx:
                results = await execute(
                    action_plan, ctx,
                    on_step_start=_on_start,
                    on_step_done=_on_done,
                    on_confirm=confirm_fn,
                )
                if results and not results[-1].ok:
                    raise typer.Exit(1)
        finally:
            await _close_resource(router, "llm_router")
```

- [ ] **Step 4: Run the full test suite**

Run: `uv run python -m pytest tests/ -x -q`
Expected: PASS — actions still use the legacy `ctx.llm` (which is now always `None`), so any test calling an LLM-requiring action would fail. **But** none of the existing CLI/orchestrator tests actually invoke a full do_* with an LLM — they stub the agent functions. Re-check fail mode:

If some tests fail with `"requires an LLM provider but neither ctx.llm nor ctx.llm_router was supplied"`, that means those tests construct an `ExecutionContext` without `llm_router` and run an action that calls `ctx.llm.invoke(...)`. The action will then crash on `ctx.llm.invoke` because `ctx.llm` is now always None.

In that case: pause Task 2, jump ahead to identify which action it is, and complete its Task 3-6 first. Most likely the analyzer/evaluator tests are the ones; they should be fixed in their respective tasks.

For a clean Part 2 → Part 3 transition: if Step 4 here is fully green (because the affected actions are still using `ctx.llm` which is now None), the actions will only fail when ACTUALLY called with LLM input. The existing tests are mocked at the agent function level, so they don't hit `ctx.llm`. Verify by running:

Run: `uv run python -m pytest tests/orchestrator/ -v 2>&1 | grep -E "(PASS|FAIL)" | head -50`
Expected: All PASS, or only the specific `do_*` tests that fully exercise the chain fail (which is normal — Tasks 3-6 fix them).

- [ ] **Step 5: Commit**

```bash
git add src/styleclaw/cli.py
git commit -m "$(cat <<'EOF'
refactor(cli): wire RoleRouter through _build_context and cli.run

ExecutionContext now receives a populated llm_router; existing actions still
read from the legacy ctx.llm (now always None) until Tasks 3-6 switch them
one by one. cli.run plans via router.get(Role.PLANNER).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `do_analyze` uses router + records `model_id`

**Files:**
- Modify: `src/styleclaw/orchestrator/actions.py::do_analyze`
- Modify: `tests/orchestrator/test_actions_do.py` and/or `tests/orchestrator/test_actions_thinking.py`

- [ ] **Step 1: Create the shared `MockRouter` helper module**

Create `tests/orchestrator/_routing_helpers.py` (note the leading underscore — this is a private test helper, not a fixture file). Putting it here from the start avoids the "extract later" churn we'd otherwise hit in Task 7:

```python
"""Shared test helpers for routing-aware action tests."""
from __future__ import annotations


class MockRouter:
    """Minimal router stub that returns a fixed provider for every role.

    Real action code only calls .get(role) and .get_panel(role) — never
    inspects which role it received — so a single shared provider is enough
    for action-level tests. Tagging the provider with `_model_id` lets the
    action's `getattr(llm, "_model_id", "")` recording path pick it up.
    """

    def __init__(self, llm, model_id: str = "test-model") -> None:
        self._llm = llm
        self._model_id = model_id
        setattr(llm, "_model_id", model_id)

    def get(self, role):
        return self._llm

    def get_panel(self, role):
        return ([self._llm, self._llm, self._llm], ["m1", "m2", "m3"])

    async def close(self) -> None:
        return None
```

In tests, import with `from tests.orchestrator._routing_helpers import MockRouter`.

- [ ] **Step 2: Write the failing test**

Append to `tests/orchestrator/test_actions_do.py` (create the file if it doesn't already exist; if there's already a `do_analyze` test, supplement it with the model_id assertion rather than duplicating):

```python
import pytest

from tests.orchestrator._routing_helpers import MockRouter


@pytest.mark.asyncio
async def test_do_analyze_records_model_id(tmp_path, monkeypatch):
    # Set up a minimal project on disk.
    monkeypatch.setenv("STYLECLAW_DATA_ROOT", str(tmp_path))
    import importlib, styleclaw.storage.project_store as ps
    importlib.reload(ps)

    from styleclaw.core.models import ProjectConfig, ProjectState
    project = "p"
    ps.save_config(project, ProjectConfig(name=project, ref_images=[]))
    ps.save_state(project, ProjectState())

    # Stub the analyze_style agent so we don't hit any real LLM.
    from styleclaw.core.models import StyleAnalysis
    fake_analysis = StyleAnalysis(trigger_phrase="t")

    async def fake_analyze(llm, ref_paths, ip_info):
        return fake_analysis

    async def fake_analyze_with_thinking(llm, ref_paths, ip_info, thinking_budget=0):
        return fake_analysis, ""

    monkeypatch.setattr(
        "styleclaw.agents.analyze_style.analyze_style", fake_analyze
    )
    monkeypatch.setattr(
        "styleclaw.agents.analyze_style.analyze_style_with_thinking",
        fake_analyze_with_thinking,
    )

    # Build context with a mock router.
    from styleclaw.orchestrator.actions import ExecutionContext, do_analyze
    fake_llm = type("FakeLLM", (), {})()
    ctx = ExecutionContext(
        project=project, llm_router=MockRouter(fake_llm, "gemini-2.5-pro"),
    )

    result = await do_analyze(ctx, {})
    assert result.ok

    # Saved analysis must carry the model_id from the router.
    saved = ps.load_analysis(project, pass_num=1)
    assert saved.trigger_phrase == "t"
    assert saved.model_id == "gemini-2.5-pro"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/orchestrator/test_actions_do.py::test_do_analyze_records_model_id -v`
Expected: FAIL — either `AttributeError: 'NoneType' has no attribute 'invoke'` (action still uses `ctx.llm`) or `assert saved.model_id == "gemini-2.5-pro"` (action doesn't fill model_id yet).

- [ ] **Step 4: Update `do_analyze`**

Replace the body of `do_analyze` in `src/styleclaw/orchestrator/actions.py` (around line 223):

```python
async def do_analyze(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.agents.analyze_style import analyze_style, analyze_style_with_thinking
    from styleclaw.core.llm_routing import Role
    from styleclaw.core.state_machine import advance

    config = project_store.load_config(ctx.project)
    root = project_store.project_dir(ctx.project)
    ref_paths = [root / r for r in config.ref_images]

    llm = ctx.llm_router.get(Role.VISION_ANALYST)
    model_id = getattr(llm, "_model_id", "")

    thinking = ""
    if ctx.show_thinking:
        analysis, thinking = await analyze_style_with_thinking(
            llm, ref_paths, config.ip_info, thinking_budget=ctx.thinking_budget,
        )
    else:
        analysis = await analyze_style(llm, ref_paths, config.ip_info)

    analysis = analysis.model_copy(update={"model_id": model_id})

    pass_num = 1
    project_store.save_analysis(ctx.project, analysis, pass_num=pass_num)
    if thinking:
        project_store.save_thinking(
            project_store.model_select_dir(ctx.project, pass_num) / "initial-analysis.json",
            thinking,
        )

    project_store.update_state(
        ctx.project,
        lambda s: advance(s, Phase.MODEL_SELECT).with_model_select_pass(pass_num),
    )

    msg = f"Trigger: {analysis.trigger_phrase}"
    if thinking:
        msg += f" | thinking saved ({len(thinking)} chars)"
    return StepResult(ok=True, message=msg)
```

The two changes from the original:
1. `llm = ctx.llm_router.get(Role.VISION_ANALYST)` replaces direct `ctx.llm` use.
2. `analysis = analysis.model_copy(update={"model_id": model_id})` adds the recording step.

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run python -m pytest tests/orchestrator/test_actions_do.py tests/agents/test_analyze_style.py -v`
Expected: PASS — new test green; existing agent-level tests unaffected (they test `analyze_style`, not `do_analyze`).

- [ ] **Step 6: Commit**

```bash
git add src/styleclaw/orchestrator/actions.py tests/orchestrator/test_actions_do.py
git commit -m "$(cat <<'EOF'
refactor(actions): do_analyze uses RoleRouter + records model_id

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `do_evaluate` uses router + records `model_id` (both phases + panel)

**Files:**
- Modify: `src/styleclaw/orchestrator/actions.py::do_evaluate` (lines ~485-659)
- Modify: `tests/orchestrator/test_actions_do.py` (test for each branch)

This is the most complex action — it has MODEL_SELECT (single + panel) and STYLE_REFINE branches. Refactor in three sub-steps, one branch each, but commit together.

- [ ] **Step 1: Update `do_evaluate` MODEL_SELECT single-model branch**

In `do_evaluate`, the MODEL_SELECT path begins around line 491. Replace the single-model arm (the `else` to `if _cfg.PANEL_MODEL_SELECT_ENABLED`, around lines 586-607) with:

```python
        # Single-model path.
        from styleclaw.core.llm_routing import Role
        llm = ctx.llm_router.get(Role.VISION_CRITIC)
        model_id = getattr(llm, "_model_id", "")

        thinking = ""
        if ctx.show_thinking:
            evaluation, thinking = await evaluate_models_with_thinking(
                llm, ref_paths, model_images, thinking_budget=ctx.thinking_budget,
            )
        else:
            evaluation = await evaluate_models(llm, ref_paths, model_images)
        evaluation = evaluation.model_copy(update={"model_id": model_id})
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

- [ ] **Step 2: Update `do_evaluate` MODEL_SELECT panel branch**

In the same MODEL_SELECT path, replace the panel branch (around lines 521-583) with the router-based version. The panel result already records `model_id` on each `PanelProposal`; what we add is `model_id` on the winning `ModelEvaluation` itself:

```python
        if _cfg.PANEL_MODEL_SELECT_ENABLED:
            from styleclaw.agents.select_model_panel import select_models_with_panel
            from styleclaw.core.llm_routing import Role

            llms, labels = ctx.llm_router.get_panel(Role.VISION_CRITIC)
            try:
                evaluation, panel_result = await select_models_with_panel(
                    llms, labels, ref_paths, model_images,
                )
            except RuntimeError as exc:
                return StepResult(ok=False, message=f"model-select panel failed: {exc}")

            project_store.save_model_select_panel_result(
                ctx.project, panel_result, pass_num=pass_num,
            )

            if panel_result.degraded and not _cfg.ALLOW_DEGRADED_PANEL:
                return StepResult(
                    ok=False,
                    message=(
                        f"model-select panel returned degraded "
                        f"({len(panel_result.error_log)} issue(s), winner='{panel_result.winner_model_id}'). "
                        f"Refusing to save evaluation.json or advance — a blind winner "
                        f"would propagate into refinement. "
                        f"See panel.json for details; re-run `evaluate` once the issue clears, "
                        f"or set STYLECLAW_ALLOW_DEGRADED_PANEL=1 to accept this run as-is."
                    ),
                    data={
                        "panel": True, "degraded": True,
                        "pass_num": pass_num,
                        "error_log": panel_result.error_log,
                    },
                )

            evaluation = evaluation.model_copy(
                update={"model_id": panel_result.winner_model_id},
            )
            project_store.save_evaluation(ctx.project, evaluation, pass_num=pass_num)
            generate_model_select_report(ctx.project, pass_num=pass_num)

            msg = (
                f"Recommendation: {evaluation.recommendation} "
                f"[panel:{panel_result.winner_model_id}] (pass {pass_num})"
            )
            if panel_result.degraded:
                msg += f" (degraded; accepted via STYLECLAW_ALLOW_DEGRADED_PANEL — {len(panel_result.error_log)} issue(s))"
            return StepResult(
                ok=True, message=msg,
                data={
                    "recommendation": evaluation.recommendation,
                    "pass_num": pass_num,
                    "panel": True,
                    "degraded": panel_result.degraded,
                },
            )
```

The key removals: no more `build_panel_providers() / close_panel_providers(pairs)` calls — the router owns lifecycle. No more `try / finally` around the panel call (router.close handles it at end of context).

- [ ] **Step 3: Update `do_evaluate` STYLE_REFINE branch**

Around line 609. Replace the single-model arm:

```python
    if state.phase == Phase.STYLE_REFINE:
        from styleclaw.agents.evaluate_result import (
            evaluate_round,
            evaluate_round_with_thinking,
        )
        from styleclaw.core.llm_routing import Role
        from styleclaw.scripts.report import generate_style_refine_report
        from styleclaw.storage.image_store import list_output_images

        pass_num = state.current_model_select_pass or 1
        round_num = state.current_round
        model_images = {}
        records = project_store.load_all_round_task_records(
            ctx.project, round_num, pass_num=pass_num,
        )
        for mid in records:
            results_dir = project_store.round_results_dir(
                ctx.project, round_num, mid, pass_num=pass_num,
            )
            images = list_output_images(results_dir)
            if images:
                model_images[mid] = images

        if not model_images:
            return StepResult(ok=False, message="No generated images for this round")

        llm = ctx.llm_router.get(Role.VISION_CRITIC)
        model_id = getattr(llm, "_model_id", "")

        thinking = ""
        if ctx.show_thinking:
            evaluation, thinking = await evaluate_round_with_thinking(
                llm, ref_paths, model_images, round_num,
                thinking_budget=ctx.thinking_budget,
            )
        else:
            evaluation = await evaluate_round(llm, ref_paths, model_images, round_num)
        evaluation = evaluation.model_copy(update={"model_id": model_id})
        project_store.save_round_evaluation(
            ctx.project, round_num, evaluation, pass_num=pass_num,
        )
        if thinking:
            round_d = project_store.round_dir(ctx.project, round_num, pass_num=pass_num)
            project_store.save_thinking(round_d / "evaluation.json", thinking)
        generate_style_refine_report(ctx.project, round_num, pass_num=pass_num)

        passed = evaluation.should_approve()
        scores_msg = ", ".join(
            f"{e.model}={e.total:.1f}" for e in evaluation.evaluations
        )
        msg = f"Scores: [{scores_msg}] {'PASS' if passed else 'needs refinement'}"
        if thinking:
            msg += f" | thinking saved ({len(thinking)} chars)"
        return StepResult(ok=True, message=msg, data={"passed": passed})
```

- [ ] **Step 4: Write/update tests**

Append to `tests/orchestrator/test_actions_do.py` (`MockRouter` already exists in `tests/orchestrator/_routing_helpers.py` from Task 3):

```python
@pytest.mark.asyncio
async def test_do_evaluate_model_select_records_model_id(tmp_path, monkeypatch):
    monkeypatch.setenv("STYLECLAW_DATA_ROOT", str(tmp_path))
    import importlib, styleclaw.storage.project_store as ps
    importlib.reload(ps)
    # Disable panel mode for this test.
    monkeypatch.delenv("STYLECLAW_PANEL_MODEL_SELECT", raising=False)
    import styleclaw.core.config as cfg_mod
    importlib.reload(cfg_mod)

    from styleclaw.core.models import (
        ModelEvaluation, ProjectConfig, ProjectState, Phase, TaskRecord, TaskStatus,
    )
    project = "p"
    ps.save_config(project, ProjectConfig(name=project, ref_images=[]))
    ps.save_state(project, ProjectState(phase=Phase.MODEL_SELECT, current_model_select_pass=1))
    # save a fake task record + image so model_images is non-empty
    rec = TaskRecord(task_id="t", model_id="mj-v7", status=TaskStatus.SUCCESS)
    ps.save_task_record(project, rec, pass_num=1)
    results_dir = ps.model_results_dir(project, "mj-v7", pass_num=1)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "out.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    fake_evaluation = ModelEvaluation(recommendation="mj-v7")

    async def fake_evaluate(llm, ref_paths, model_images):
        return fake_evaluation

    async def fake_evaluate_with_thinking(llm, ref_paths, model_images, thinking_budget=0):
        return fake_evaluation, ""

    monkeypatch.setattr(
        "styleclaw.agents.select_model.evaluate_models", fake_evaluate
    )
    monkeypatch.setattr(
        "styleclaw.agents.select_model.evaluate_models_with_thinking",
        fake_evaluate_with_thinking,
    )
    # Stub the report generator so we don't need the full HTML pipeline.
    monkeypatch.setattr(
        "styleclaw.scripts.report.generate_model_select_report",
        lambda *a, **kw: tmp_path / "report.html",
    )

    from styleclaw.orchestrator.actions import ExecutionContext, do_evaluate
    fake_llm = type("FakeLLM", (), {})()
    ctx = ExecutionContext(
        project=project, llm_router=MockRouter(fake_llm, "claude-sonnet-4-6"),
    )

    result = await do_evaluate(ctx, {})
    assert result.ok, result.message

    saved = ps.load_evaluation(project, pass_num=1)
    assert saved.model_id == "claude-sonnet-4-6"
    assert saved.recommendation == "mj-v7"


@pytest.mark.asyncio
async def test_do_evaluate_style_refine_records_model_id(tmp_path, monkeypatch):
    """STYLE_REFINE branch: evaluate_round → ctx.llm_router.get(VISION_CRITIC)."""
    monkeypatch.setenv("STYLECLAW_DATA_ROOT", str(tmp_path))
    import importlib, styleclaw.storage.project_store as ps
    importlib.reload(ps)

    from styleclaw.core.models import (
        Phase, ProjectConfig, ProjectState, RoundEvaluation, TaskRecord, TaskStatus,
    )
    project = "p"
    ps.save_config(project, ProjectConfig(name=project, ref_images=[]))
    ps.save_state(project, ProjectState(
        phase=Phase.STYLE_REFINE,
        current_round=1,
        current_model_select_pass=1,
        selected_models=["mj-v7"],
    ))
    # Save a fake round task record + image.
    rec = TaskRecord(task_id="t", model_id="mj-v7", status=TaskStatus.SUCCESS)
    ps.save_round_task_record(project, round_num=1, record=rec, pass_num=1)
    results_dir = ps.round_results_dir(project, 1, "mj-v7", pass_num=1)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "out.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    fake_evaluation = RoundEvaluation(round=1, recommendation="keep")

    async def fake_evaluate_round(llm, ref_paths, model_images, round_num):
        return fake_evaluation

    async def fake_evaluate_round_with_thinking(llm, ref_paths, model_images, round_num, thinking_budget=0):
        return fake_evaluation, ""

    monkeypatch.setattr(
        "styleclaw.agents.evaluate_result.evaluate_round", fake_evaluate_round
    )
    monkeypatch.setattr(
        "styleclaw.agents.evaluate_result.evaluate_round_with_thinking",
        fake_evaluate_round_with_thinking,
    )
    monkeypatch.setattr(
        "styleclaw.scripts.report.generate_style_refine_report",
        lambda *a, **kw: tmp_path / "report.html",
    )

    from styleclaw.orchestrator.actions import ExecutionContext, do_evaluate
    fake_llm = type("FakeLLM", (), {})()
    ctx = ExecutionContext(
        project=project, llm_router=MockRouter(fake_llm, "gemini-2.5-pro"),
    )

    result = await do_evaluate(ctx, {})
    assert result.ok, result.message

    saved = ps.load_round_evaluation(project, round_num=1, pass_num=1)
    assert saved.model_id == "gemini-2.5-pro"
```

Note: the exact `project_store` API names (`save_round_task_record`, `round_results_dir`, `load_round_evaluation`) match what's already in [src/styleclaw/storage/project_store.py](../../../src/styleclaw/storage/project_store.py) — verify them when implementing if the signatures have drifted.

- [ ] **Step 5: Run tests + commit**

Run: `uv run python -m pytest tests/orchestrator/test_actions_do.py tests/agents/test_select_model.py tests/agents/test_evaluate_result.py -v`
Expected: PASS

```bash
git add src/styleclaw/orchestrator/actions.py tests/orchestrator/test_actions_do.py
git commit -m "$(cat <<'EOF'
refactor(actions): do_evaluate uses RoleRouter for both phases + panel

VISION_CRITIC role drives select_model + evaluate_round; panel branch
uses router.get_panel. Winning panel proposal's model_id propagates to
the saved evaluation. RoleRouter owns provider lifecycle — no more
build_panel_providers / close_panel_providers pairs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `do_refine` uses router + records `model_id`

**Files:**
- Modify: `src/styleclaw/orchestrator/actions.py::do_refine` (lines ~696-839)
- Modify: `tests/orchestrator/test_actions_do.py`

Same shape as Task 4 but for refine. Two branches: panel (lines ~756-811) and single-model (lines ~813-839).

- [ ] **Step 1: Update panel branch**

Replace the panel arm with router-based wiring:

```python
    if _cfg.PANEL_REFINE_ENABLED:
        from styleclaw.agents.refine_panel import refine_with_panel
        from styleclaw.core.llm_routing import Role

        llms, labels = ctx.llm_router.get_panel(Role.VISION_ANALYST)
        try:
            prompt_config, panel_result = await refine_with_panel(
                llms, labels, ref_paths, current_trigger, round_num,
                config.ip_info, evaluations, direction,
            )
        except RuntimeError as exc:
            return StepResult(ok=False, message=f"refine panel failed: {exc}")

        project_store.save_round_panel_result(
            ctx.project, round_num, panel_result, pass_num=pass_num,
        )

        if panel_result.degraded and not _cfg.ALLOW_DEGRADED_PANEL:
            return StepResult(
                ok=False,
                message=(
                    f"refine panel for round {round_num} returned degraded "
                    f"({len(panel_result.error_log)} issue(s), winner='{panel_result.winner_model_id}'). "
                    f"Refusing to save prompt.json or advance — a half-validated "
                    f"trigger would taint downstream rounds. "
                    f"See panel.json for details; re-run `refine` once the issue clears, "
                    f"or set STYLECLAW_ALLOW_DEGRADED_PANEL=1 to accept this run as-is."
                ),
                data={
                    "panel": True, "degraded": True,
                    "round": round_num,
                    "error_log": panel_result.error_log,
                },
            )

        prompt_config = prompt_config.model_copy(
            update={"model_id": panel_result.winner_model_id},
        )
        project_store.save_prompt_config(
            ctx.project, round_num, prompt_config, pass_num=pass_num,
        )

        project_store.update_state(ctx.project, lambda s: s.with_round(round_num))

        msg = f"Round {round_num} [panel:{panel_result.winner_model_id}]: {prompt_config.trigger_phrase}"
        if panel_result.degraded:
            msg += f" (degraded; accepted via STYLECLAW_ALLOW_DEGRADED_PANEL — {len(panel_result.error_log)} issue(s))"
        return StepResult(ok=True, message=msg, data={"panel": True, "degraded": panel_result.degraded})
```

- [ ] **Step 2: Update single-model branch**

```python
    # Single-model path.
    from styleclaw.core.llm_routing import Role
    llm = ctx.llm_router.get(Role.VISION_ANALYST)
    model_id = getattr(llm, "_model_id", "")

    thinking = ""
    if ctx.show_thinking:
        prompt_config, thinking = await refine_prompt_with_thinking(
            llm, ref_paths, current_trigger, round_num,
            config.ip_info, evaluations, direction,
            thinking_budget=ctx.thinking_budget,
        )
    else:
        prompt_config = await refine_prompt(
            llm, ref_paths, current_trigger, round_num,
            config.ip_info, evaluations, direction,
        )
    prompt_config = prompt_config.model_copy(update={"model_id": model_id})
    project_store.save_prompt_config(
        ctx.project, round_num, prompt_config, pass_num=pass_num,
    )

    if thinking:
        round_d = project_store.round_dir(ctx.project, round_num, pass_num=pass_num)
        project_store.save_thinking(round_d / "prompt.json", thinking)

    project_store.update_state(ctx.project, lambda s: s.with_round(round_num))

    msg = f"Round {round_num}: {prompt_config.trigger_phrase}"
    if thinking:
        msg += f" | thinking saved ({len(thinking)} chars)"
    return StepResult(ok=True, message=msg)
```

- [ ] **Step 3: Test**

Add a single-model do_refine test in `tests/orchestrator/test_actions_do.py` following the same shape as the do_analyze test. Confirm `saved_prompt_config.model_id == "claude-sonnet-4-6"`.

- [ ] **Step 4: Run + commit**

```bash
uv run python -m pytest tests/orchestrator/test_actions_do.py tests/agents/test_refine_prompt.py tests/agents/test_refine_panel.py -v
```

```bash
git add src/styleclaw/orchestrator/actions.py tests/orchestrator/test_actions_do.py
git commit -m "$(cat <<'EOF'
refactor(actions): do_refine uses RoleRouter for single + panel

VISION_ANALYST role drives refine_prompt + refine_panel. Winning panel
proposal's model_id propagates to saved prompt.json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `do_design_cases` uses router + records `model_id`

**Files:**
- Modify: `src/styleclaw/orchestrator/actions.py::do_design_cases` (around line 862)
- Modify: `tests/orchestrator/test_actions_do.py`

- [ ] **Step 1: Update `do_design_cases`**

Replace the LLM call and add `model_id` recording:

```python
async def do_design_cases(ctx: ExecutionContext, args: dict[str, Any]) -> StepResult:
    from styleclaw.agents.design_cases import design_cases
    from styleclaw.core.llm_routing import Role

    state = project_store.load_state(ctx.project)

    if state.phase != Phase.BATCH_T2I:
        return StepResult(
            ok=False,
            message=f"design-cases requires BATCH_T2I phase (current: {state.phase}). "
                    f"Run 'approve' from STYLE_REFINE first.",
        )
    if state.current_round < 1:
        return StepResult(
            ok=False,
            message="design-cases requires a refined trigger phrase — "
                    "no STYLE_REFINE round has been completed (current_round=0).",
        )

    config = project_store.load_config(ctx.project)
    batch_num = state.current_batch + 1

    pass_num = state.current_model_select_pass or 1
    prompt_config = project_store.load_prompt_config(
        ctx.project, state.current_round, pass_num=pass_num,
    )
    feedback = str(args.get("feedback", "") or "").strip()

    llm = ctx.llm_router.get(Role.WRITER)
    model_id = getattr(llm, "_model_id", "")
    batch_config = await design_cases(
        llm, config.ip_info, prompt_config.trigger_phrase, batch_num,
        feedback=feedback,
    )
    batch_config = batch_config.model_copy(update={"model_id": model_id})
    project_store.save_batch_config(ctx.project, batch_num, batch_config)

    project_store.update_state(ctx.project, lambda s: s.with_batch(batch_num))

    msg = f"Designed {len(batch_config.cases)} cases (batch {batch_num})"
    if feedback:
        msg += " [applied feedback]"
    return StepResult(ok=True, message=msg)
```

- [ ] **Step 2: Test**

Add a `test_do_design_cases_records_model_id` to `tests/orchestrator/test_actions_do.py` following the do_analyze pattern; stub `design_cases` to return a fixed `BatchConfig`, verify `saved_cases.model_id == expected`.

- [ ] **Step 3: Run + commit**

```bash
uv run python -m pytest tests/orchestrator/test_actions_do.py tests/agents/test_design_cases.py -v
```

```bash
git add src/styleclaw/orchestrator/actions.py tests/orchestrator/test_actions_do.py
git commit -m "$(cat <<'EOF'
refactor(actions): do_design_cases uses RoleRouter (WRITER) + records model_id

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Remove legacy `ExecutionContext.llm` field + `_build_llm_provider`

**Files:**
- Modify: `src/styleclaw/orchestrator/actions.py` (drop `llm` field)
- Modify: `src/styleclaw/orchestrator/executor.py` (simplify needs_llm guard)
- Modify: `src/styleclaw/cli.py` (delete `_build_llm_provider`)
- Modify: any tests that still set `ctx.llm=...`

- [ ] **Step 1: Grep for remaining usages**

Run: `uv run python -m grep -rn 'ctx\.llm[^_]' src/ tests/ 2>&1 | grep -v 'ctx.llm_router' | head -40`
Expected: zero results, or a small number that are easy to fix. If anything in `src/` still uses `ctx.llm`, finish that action's Task 3-6 work first — don't skip ahead.

- [ ] **Step 2: Drop the field**

In `src/styleclaw/orchestrator/actions.py::ExecutionContext`:

```python
@dataclass(frozen=True)
class ExecutionContext:
    project: str
    client: RunningHubClient | None = None
    llm_router: "RoleRouter | None" = None
    poll_interval: float = ORCHESTRATOR_POLL_INTERVAL
    show_thinking: bool = False
    thinking_budget: int = 5000
```

In `src/styleclaw/orchestrator/executor.py`:

```python
        if action_def.needs_llm and ctx.llm_router is None:
            result = StepResult(
                ok=False,
                message=f"Action '{step.name}' requires an LLM router but none was provided",
            )
            results.append(result)
            if on_step_done:
                on_step_done(i, step.name, result)
            return results
```

- [ ] **Step 3: Delete `_build_llm_provider` from cli.py**

Remove the `_build_llm_provider` function entirely (lines ~74-82). Any remaining references should be gone — grep first:

Run: `uv run python -m grep -rn '_build_llm_provider' src/ tests/`
Expected: no matches.

- [ ] **Step 4: Update tests still referencing `ctx.llm=`**

Search:
Run: `uv run python -m grep -rn 'llm=' tests/ | grep -v 'llm_router'`
Expected: a handful of tests. For each one, replace `ExecutionContext(... llm=fake_llm)` with `ExecutionContext(... llm_router=MockRouter(fake_llm))` and add `from tests.orchestrator._routing_helpers import MockRouter` at the top.

- [ ] **Step 5: Run full suite**

Run: `uv run python -m pytest tests/ -x -q`
Expected: PASS — full green.

- [ ] **Step 6: Commit**

```bash
git add src/styleclaw/orchestrator/actions.py src/styleclaw/orchestrator/executor.py \
        src/styleclaw/cli.py tests/
git commit -m "$(cat <<'EOF'
refactor: remove legacy ExecutionContext.llm + cli._build_llm_provider

Every action now reads through ctx.llm_router. Tests share MockRouter
via conftest.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Delete `panel_factory.py` and its test

**Files:**
- Delete: `src/styleclaw/providers/llm/panel_factory.py`
- Delete: `tests/providers/llm/test_panel_factory.py`

- [ ] **Step 1: Confirm no remaining importers**

Run: `uv run python -m grep -rn 'panel_factory' src/ tests/`
Expected: no matches (Tasks 4 and 5 removed the two production imports).

- [ ] **Step 2: Delete**

```bash
rm src/styleclaw/providers/llm/panel_factory.py
rm tests/providers/llm/test_panel_factory.py
```

- [ ] **Step 3: Run full suite**

Run: `uv run python -m pytest tests/ -x -q`
Expected: PASS — full green.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: drop orphan panel_factory module — RoleRouter.get_panel replaces it

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Wrap-up — full integration sweep

**Files:** None modified. Verification only.

- [ ] **Step 1: Full test suite, verbose**

Run: `uv run python -m pytest tests/ -v --tb=short 2>&1 | tail -40`
Expected: every test passes; output ends with a green summary.

- [ ] **Step 2: Static-style check**

Run: `uv run python -m grep -rn '\.llm\b' src/styleclaw/ 2>&1 | grep -v 'llm_router' | grep -v '\.llm_compat' | grep -v 'self\._' | head -20`
Expected: zero results — every production reference uses `llm_router`. (Some matches inside provider modules' own internal `self._...` attributes are fine; the grep filters those.)

- [ ] **Step 3: CLI smoke (no LLM call)**

Run: `STYLECLAW_SKIP_ENV_CHECK=1 uv run styleclaw --help`
Expected: the help text renders without import errors. (`STYLECLAW_SKIP_ENV_CHECK` lets us load the CLI without setting any provider creds.)

- [ ] **Step 4: Done**

Part 3 is complete. Every LLM call goes through `RoleRouter`; every persisted artifact carries the `model_id` that produced it. The system runs end-to-end against the same env config as before (single `LLM_MODEL` works) but now also honors any of the new `STYLECLAW_MODEL_<ROLE>` envs. Part 4 (docs + smoke test) finishes the feature.
