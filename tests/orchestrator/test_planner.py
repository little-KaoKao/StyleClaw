from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from styleclaw.core.models import (
    Phase,
    ProjectConfig,
    ProjectState,
)
from styleclaw.orchestrator.planner import plan
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def setup_project():
    config = ProjectConfig(name="test-proj", ip_info="anime style")
    root = project_store.create_project(config)
    (root / "refs" / "ref-001.png").write_bytes(b"fake image")
    return root


class TestPlanner:
    async def test_plan_returns_action_plan(self, setup_project) -> None:
        state = ProjectState(phase=Phase.INIT)
        project_store.save_state("test-proj", state)

        llm = AsyncMock()
        llm.invoke = AsyncMock(return_value=json.dumps({
            "summary": "Analyze references",
            "steps": [
                {"name": "analyze", "description": "分析参考图片", "args": {}},
            ],
            "loop": None,
        }))

        result = await plan(llm, "test-proj", "analyze my references")
        assert result.summary == "Analyze references"
        assert len(result.steps) == 1
        assert result.steps[0].name == "analyze"

    async def test_plan_with_loop(self, setup_project) -> None:
        state = ProjectState(phase=Phase.STYLE_REFINE, current_round=1, selected_models=["mj-v7"])
        project_store.save_state("test-proj", state)

        llm = AsyncMock()
        llm.invoke = AsyncMock(return_value=json.dumps({
            "summary": "Refine until pass",
            "steps": [
                {"name": "refine", "description": "精炼触发词", "args": {}},
                {"name": "generate", "description": "生成测试图", "args": {}},
                {"name": "poll", "description": "等待完成", "args": {}},
                {"name": "evaluate", "description": "评估结果", "args": {}},
            ],
            "loop": {"start_step": 0, "end_step": 3, "max_iterations": 3},
        }))

        result = await plan(llm, "test-proj", "refine until satisfied")
        assert result.loop is not None
        assert result.loop.max_iterations == 3

    async def test_plan_passes_state_to_llm(self, setup_project) -> None:
        state = ProjectState(phase=Phase.MODEL_SELECT, selected_models=["mj-v7"])
        project_store.save_state("test-proj", state)

        llm = AsyncMock()
        llm.invoke = AsyncMock(return_value=json.dumps({
            "summary": "Generate",
            "steps": [{"name": "generate", "description": "生成", "args": {}}],
            "loop": None,
        }))

        await plan(llm, "test-proj", "generate images")

        call_args = llm.invoke.call_args
        system_prompt = call_args.kwargs.get("system") or call_args[1].get("system") or call_args[0][0]
        assert "MODEL_SELECT" in system_prompt
        assert "anime style" in system_prompt

    async def test_plan_handles_markdown_fences(self, setup_project) -> None:
        state = ProjectState(phase=Phase.INIT)
        project_store.save_state("test-proj", state)

        llm = AsyncMock()
        llm.invoke = AsyncMock(return_value="```json\n" + json.dumps({
            "summary": "Test",
            "steps": [{"name": "analyze", "description": "分析", "args": {}}],
            "loop": None,
        }) + "\n```")

        result = await plan(llm, "test-proj", "analyze")
        assert result.steps[0].name == "analyze"


class TestPlannerValidation:
    async def test_retries_on_unknown_action(self, setup_project) -> None:
        state = ProjectState(phase=Phase.INIT)
        project_store.save_state("test-proj", state)

        llm = AsyncMock()
        llm.invoke = AsyncMock(side_effect=[
            json.dumps({
                "summary": "bad",
                "steps": [{"name": "fabricated-action", "description": "x", "args": {}}],
                "loop": None,
            }),
            json.dumps({
                "summary": "fixed",
                "steps": [{"name": "analyze", "description": "分析", "args": {}}],
                "loop": None,
            }),
        ])

        result = await plan(llm, "test-proj", "analyze")
        assert result.steps[0].name == "analyze"
        assert llm.invoke.call_count == 2

    async def test_retries_on_disallowed_for_phase(self, setup_project) -> None:
        """INIT phase can transition into MODEL_SELECT so its allowed list
        includes MODEL_SELECT actions, but not STYLE_REFINE / BATCH_T2I ones."""
        state = ProjectState(phase=Phase.INIT)
        project_store.save_state("test-proj", state)

        llm = AsyncMock()
        llm.invoke = AsyncMock(side_effect=[
            json.dumps({
                "summary": "wrong phase",
                "steps": [{"name": "batch-submit", "description": "?", "args": {}}],
                "loop": None,
            }),
            json.dumps({
                "summary": "ok",
                "steps": [{"name": "analyze", "description": "分析", "args": {}}],
                "loop": None,
            }),
        ])

        result = await plan(llm, "test-proj", "do stuff")
        assert result.steps[0].name == "analyze"

    async def test_raises_when_retry_also_invalid(self, setup_project) -> None:
        state = ProjectState(phase=Phase.INIT)
        project_store.save_state("test-proj", state)

        llm = AsyncMock()
        llm.invoke = AsyncMock(return_value=json.dumps({
            "summary": "still bad",
            "steps": [{"name": "hallucination", "description": "x", "args": {}}],
            "loop": None,
        }))

        with pytest.raises(ValueError, match="still produced unknown"):
            await plan(llm, "test-proj", "analyze")

    async def test_passes_feedback_to_retry_message(self, setup_project) -> None:
        state = ProjectState(phase=Phase.INIT)
        project_store.save_state("test-proj", state)

        llm = AsyncMock()
        llm.invoke = AsyncMock(side_effect=[
            json.dumps({
                "summary": "bad",
                "steps": [{"name": "fake-name", "description": "x", "args": {}}],
                "loop": None,
            }),
            json.dumps({
                "summary": "ok",
                "steps": [{"name": "analyze", "description": "分析", "args": {}}],
                "loop": None,
            }),
        ])

        await plan(llm, "test-proj", "analyze")

        retry_call = llm.invoke.call_args_list[1]
        retry_messages = retry_call.kwargs.get("messages", [])
        assert any("fake-name" in str(m.get("content", "")) for m in retry_messages)
