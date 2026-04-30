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

        ctx = ExecutionContext(project=project_with_ref, llm=fake_llm)
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
