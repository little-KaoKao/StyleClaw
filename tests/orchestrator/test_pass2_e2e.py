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
    ctx = ExecutionContext(project=seeded_project)
    result = await do_retest_models(ctx, {})
    assert result.ok

    state = project_store.load_state(seeded_project)
    assert state.phase == Phase.MODEL_SELECT
    assert state.current_model_select_pass == 2

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

    pass1_records = project_store.load_all_task_records(seeded_project, pass_num=1)
    pass2_records = project_store.load_all_task_records(seeded_project, pass_num=2)
    assert pass2_records
    assert not pass1_records

    for key, rec in pass2_records.items():
        model_id, variant = key.split("/", 1)
        project_store.save_task_record(
            seeded_project, model_id,
            rec.model_copy(update={"status": TaskStatus.SUCCESS}),
            variant=variant, pass_num=2,
        )
        results_dir = project_store.model_results_dir(
            seeded_project, model_id, variant=variant, pass_num=2,
        )
        img = results_dir / "output-001.png"
        Image.new("RGB", (16, 16), "green").save(img)

    fake_llm = AsyncMock()
    fake_llm.invoke = AsyncMock(
        return_value='{"evaluations": [{"model": "mj-v7", "variant": "prompt-only", "total": 8.2}], "recommendation": "mj-v7"}'
    )
    ctx = ExecutionContext(project=seeded_project, llm=fake_llm)
    result = await do_evaluate(ctx, {})
    assert result.ok
    assert "mj-v7" in result.message
    assert "pass 2" in result.message.lower()

    pass2_eval = project_store.load_evaluation(seeded_project, pass_num=2)
    assert pass2_eval.recommendation == "mj-v7"
    with pytest.raises(FileNotFoundError):
        project_store.load_evaluation(seeded_project, pass_num=1)

    ctx = ExecutionContext(project=seeded_project)
    result = await do_select_model(ctx, {"models": "mj-v7"})
    assert result.ok

    final_state = project_store.load_state(seeded_project)
    assert final_state.phase == Phase.STYLE_REFINE
    assert final_state.current_round == 1
    assert final_state.current_model_select_pass == 2
