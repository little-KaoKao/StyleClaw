from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from styleclaw.core.models import (
    Phase,
    ProjectConfig,
    ProjectState,
    PromptConfig,
    StyleAnalysis,
    TaskRecord,
    TaskStatus,
)
from styleclaw.orchestrator.actions import ExecutionContext, StepResult, do_analyze, do_generate
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

        from tests.orchestrator._routing_helpers import MockRouter
        ctx = ExecutionContext(
            project=project_with_ref, llm_router=MockRouter(fake_llm),
        )
        result = await do_analyze(ctx, {})
        assert result.ok

        state = project_store.load_state(project_with_ref)
        assert state.phase == Phase.MODEL_SELECT
        assert state.current_model_select_pass == 1

        pass1 = (
            project_store.project_dir(project_with_ref)
            / "model-select" / "pass-001" / "initial-analysis.json"
        )
        assert pass1.exists()


class TestDoGenerateUsesCurrentPass:
    async def test_pass_2_picks_refined_trigger(self, project_with_ref):
        """After retest-models, do_generate should see the current trigger
        that retest-models persisted into pass-N/initial-analysis.json."""
        project_store.save_analysis(
            project_with_ref, StyleAnalysis(trigger_phrase="refined-after-round-1"),
            pass_num=2,
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

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_retest_models(ctx, {})
        assert result.ok is False
        assert "INIT" in result.message

    async def test_retest_writes_pass_n_analysis_from_prompt(self, project_with_ref):
        """F2: retest-models seeds pass-N/initial-analysis.json with the
        current trigger, so do_generate in pass-N can read from analysis
        uniformly regardless of pass."""
        from styleclaw.orchestrator.actions import do_retest_models

        project_store.save_analysis(
            project_with_ref, StyleAnalysis(trigger_phrase="old-trigger"), pass_num=1,
        )
        project_store.save_prompt_config(
            project_with_ref, 2,
            PromptConfig(round=2, trigger_phrase="refined-after-round-2"),
            pass_num=1,
        )
        state = ProjectState(
            phase=Phase.STYLE_REFINE, current_round=2, current_model_select_pass=1,
        )
        project_store.save_state(project_with_ref, state)

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_retest_models(ctx, {})
        assert result.ok

        pass2_analysis = project_store.load_analysis(project_with_ref, pass_num=2)
        assert pass2_analysis.trigger_phrase == "refined-after-round-2"

    async def test_retest_falls_back_to_analysis_when_no_rounds(self, project_with_ref):
        """If a user somehow retests without any refine rounds, fall back to
        the pass-N-1 analysis trigger."""
        from styleclaw.orchestrator.actions import do_retest_models

        project_store.save_analysis(
            project_with_ref, StyleAnalysis(trigger_phrase="only-analysis"), pass_num=1,
        )
        state = ProjectState(
            phase=Phase.STYLE_REFINE, current_round=0, current_model_select_pass=1,
        )
        project_store.save_state(project_with_ref, state)

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_retest_models(ctx, {})
        assert result.ok

        pass2_analysis = project_store.load_analysis(project_with_ref, pass_num=2)
        assert pass2_analysis.trigger_phrase == "only-analysis"

    async def test_trigger_override_replaces_carried_phrase(self, project_with_ref):
        """When the user supplies a new trigger phrase, retest-models writes
        the override into pass-N analysis instead of carrying the existing
        trigger forward. This is the orchestrator-level fix for "用这个触发词
        重测" — the planner can route the user's phrase through args.trigger."""
        from styleclaw.orchestrator.actions import do_retest_models

        project_store.save_analysis(
            project_with_ref, StyleAnalysis(trigger_phrase="old-trigger"), pass_num=1,
        )
        project_store.save_prompt_config(
            project_with_ref, 1,
            PromptConfig(round=1, trigger_phrase="refined-but-stale"),
            pass_num=1,
        )
        state = ProjectState(
            phase=Phase.MODEL_SELECT, current_round=1, current_model_select_pass=1,
        )
        project_store.save_state(project_with_ref, state)

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_retest_models(ctx, {"trigger": "brand new trigger phrase"})
        assert result.ok
        assert result.data["trigger_overridden"] is True

        pass2_analysis = project_store.load_analysis(project_with_ref, pass_num=2)
        assert pass2_analysis.trigger_phrase == "brand new trigger phrase"

    async def test_trigger_override_strips_surrounding_whitespace(
        self, project_with_ref,
    ) -> None:
        """LLM-supplied args often pick up trailing newlines from prompt
        templates. The action should normalize, not pass them through to the
        on-disk trigger phrase (which then leaks into the rendered prompt)."""
        from styleclaw.orchestrator.actions import do_retest_models

        project_store.save_analysis(
            project_with_ref, StyleAnalysis(trigger_phrase="old"), pass_num=1,
        )
        state = ProjectState(
            phase=Phase.MODEL_SELECT, current_model_select_pass=1,
        )
        project_store.save_state(project_with_ref, state)

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_retest_models(ctx, {"trigger": "  spaced trigger  \n"})
        assert result.ok

        pass2 = project_store.load_analysis(project_with_ref, pass_num=2)
        assert pass2.trigger_phrase == "spaced trigger"

    async def test_empty_trigger_arg_does_not_override(self, project_with_ref):
        """`args.trigger=""` is the default and should be indistinguishable
        from "no arg passed" — the carry-forward path must still run."""
        from styleclaw.orchestrator.actions import do_retest_models

        project_store.save_analysis(
            project_with_ref, StyleAnalysis(trigger_phrase="existing"), pass_num=1,
        )
        state = ProjectState(
            phase=Phase.MODEL_SELECT, current_model_select_pass=1,
        )
        project_store.save_state(project_with_ref, state)

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_retest_models(ctx, {"trigger": ""})
        assert result.ok
        assert result.data.get("trigger_overridden") is False

        pass2 = project_store.load_analysis(project_with_ref, pass_num=2)
        assert pass2.trigger_phrase == "existing"

    async def test_trigger_override_arg_rejected_via_schema(
        self, project_with_ref,
    ) -> None:
        """Schema layer must accept `trigger` and reject anything else, so
        the planner cannot smuggle in arbitrary keys (e.g. `trigger_override`,
        which is what the LLM tried before this fix)."""
        from styleclaw.orchestrator.actions import _validate_action_args

        validated, err = _validate_action_args(
            "retest-models", {"trigger": "ok"},
        )
        assert err is None
        assert validated == {"trigger": "ok"}

        _, err = _validate_action_args(
            "retest-models", {"trigger_override": "no"},
        )
        assert err is not None
        assert "trigger_override" in err.message

    async def test_test_subjects_carried_to_new_pass(self, project_with_ref):
        """De-IP'd character descriptions extracted during analyze are part
        of the ref's signal, not the trigger's. They must survive a pass
        bump so downstream `generate` keeps testing against the same
        characters."""
        from styleclaw.orchestrator.actions import do_retest_models

        project_store.save_analysis(
            project_with_ref,
            StyleAnalysis(
                trigger_phrase="old-trigger",
                test_subjects={"male": "M-DESC", "female": "F-DESC"},
            ),
            pass_num=1,
        )
        state = ProjectState(
            phase=Phase.MODEL_SELECT, current_model_select_pass=1,
        )
        project_store.save_state(project_with_ref, state)

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_retest_models(ctx, {})
        assert result.ok

        pass2 = project_store.load_analysis(project_with_ref, pass_num=2)
        assert pass2.test_subjects == {"male": "M-DESC", "female": "F-DESC"}

    async def test_test_subjects_carry_forward_with_override(self, project_with_ref):
        """Even when the trigger is overridden, the ref-derived subjects
        should still carry — the refs haven't changed."""
        from styleclaw.orchestrator.actions import do_retest_models

        project_store.save_analysis(
            project_with_ref,
            StyleAnalysis(
                trigger_phrase="old",
                test_subjects={"male": "M-DESC"},
            ),
            pass_num=1,
        )
        state = ProjectState(
            phase=Phase.MODEL_SELECT, current_model_select_pass=1,
        )
        project_store.save_state(project_with_ref, state)

        ctx = ExecutionContext(project=project_with_ref)
        result = await do_retest_models(ctx, {"trigger": "totally new"})
        assert result.ok

        pass2 = project_store.load_analysis(project_with_ref, pass_num=2)
        assert pass2.trigger_phrase == "totally new"
        assert pass2.test_subjects == {"male": "M-DESC"}


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
