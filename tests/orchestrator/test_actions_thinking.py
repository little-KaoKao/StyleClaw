from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from PIL import Image

from styleclaw.core.models import (
    Phase,
    ProjectConfig,
    ProjectState,
)
from styleclaw.orchestrator.actions import ExecutionContext, do_analyze
from styleclaw.providers.llm.base import LLMResponse
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def project_with_ref(tmp_path):
    name = "proj"
    root = tmp_path / "projects" / name
    root.mkdir(parents=True)
    (root / "refs").mkdir()
    ref = root / "refs" / "ref.png"
    Image.new("RGB", (32, 32), "red").save(ref)
    config = ProjectConfig(name=name, ip_info="test", ref_images=["refs/ref.png"])
    state = ProjectState(phase=Phase.INIT)
    (root / "model-select" / "results").mkdir(parents=True)
    (root / "style-refine").mkdir()
    (root / "batch-t2i").mkdir()
    (root / "batch-i2i").mkdir()
    project_store.save_config(name, config)
    project_store.save_state(name, state)
    return name


class TestDoAnalyzeThinking:
    async def test_saves_thinking_file_when_enabled(self, project_with_ref):
        fake_llm = AsyncMock()
        fake_llm.invoke_with_thinking = AsyncMock(
            return_value=LLMResponse(
                text='{"trigger_phrase": "bold ink"}',
                thinking="Reasoning text here.",
            )
        )
        ctx = ExecutionContext(
            project=project_with_ref, llm=fake_llm,
            show_thinking=True, thinking_budget=3000,
        )
        result = await do_analyze(ctx, {})
        assert result.ok

        thinking_md = (
            project_store.project_dir(project_with_ref)
            / "model-select" / "pass-001" / "initial-analysis.thinking.md"
        )
        assert thinking_md.exists()
        assert "Reasoning text here." in thinking_md.read_text(encoding="utf-8")

    async def test_no_thinking_file_when_disabled(self, project_with_ref):
        fake_llm = AsyncMock()
        fake_llm.invoke = AsyncMock(return_value='{"trigger_phrase": "bold"}')
        ctx = ExecutionContext(project=project_with_ref, llm=fake_llm, show_thinking=False)
        result = await do_analyze(ctx, {})
        assert result.ok

        thinking_md = (
            project_store.project_dir(project_with_ref)
            / "model-select" / "pass-001" / "initial-analysis.thinking.md"
        )
        assert not thinking_md.exists()
