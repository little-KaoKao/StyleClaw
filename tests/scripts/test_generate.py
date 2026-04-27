from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from styleclaw.core.models import Phase, ProjectConfig, ProjectState, TaskRecord
from styleclaw.scripts.generate import generate_model_select, generate_style_refine
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def setup_project():
    config = ProjectConfig(name="test-proj", ip_info="anime")
    project_store.create_project(config)
    state = ProjectState(phase=Phase.STYLE_REFINE, selected_models=["mj-v7"])
    project_store.save_state("test-proj", state)


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.post.return_value = {"taskId": "t1", "status": "QUEUED"}
    return client


class TestGenerateModelSelect:
    async def test_submits_for_all_models(self, setup_project, mock_client) -> None:
        records = await generate_model_select(
            "test-proj", mock_client, "bold anime style",
        )
        assert len(records) == 4  # all 4 models in registry

    async def test_submits_for_specific_models(self, setup_project, mock_client) -> None:
        records = await generate_model_select(
            "test-proj", mock_client, "bold anime style", models=["mj-v7"],
        )
        assert len(records) == 1
        assert "mj-v7" in records

    async def test_saves_task_records(self, setup_project, mock_client) -> None:
        await generate_model_select(
            "test-proj", mock_client, "bold anime style", models=["mj-v7"],
        )
        record = project_store.load_task_record("test-proj", "mj-v7")
        assert record.task_id == "t1"


class TestGenerateStyleRefine:
    async def test_submits_for_selected_models(self, setup_project, mock_client) -> None:
        records = await generate_style_refine(
            "test-proj", mock_client, 1, "bold anime style",
        )
        assert len(records) == 1
        assert "mj-v7" in records

    async def test_saves_round_task_records(self, setup_project, mock_client) -> None:
        await generate_style_refine(
            "test-proj", mock_client, 1, "bold anime style",
        )
        record = project_store.load_round_task_record("test-proj", 1, "mj-v7")
        assert record.task_id == "t1"


class TestIdempotency:
    async def test_model_select_skips_existing_success(self, setup_project, mock_client) -> None:
        existing = TaskRecord(task_id="old-1", model_id="mj-v7", status="SUCCESS")
        project_store.save_task_record("test-proj", "mj-v7", existing)

        records = await generate_model_select(
            "test-proj", mock_client, "bold anime style", models=["mj-v7"],
        )
        assert records["mj-v7"].task_id == "old-1"
        mock_client.post.assert_not_called()

    async def test_model_select_resubmits_failed(self, setup_project, mock_client) -> None:
        existing = TaskRecord(task_id="old-1", model_id="mj-v7", status="FAILED")
        project_store.save_task_record("test-proj", "mj-v7", existing)

        records = await generate_model_select(
            "test-proj", mock_client, "bold anime style", models=["mj-v7"],
        )
        assert records["mj-v7"].task_id == "t1"
        mock_client.post.assert_called_once()

    async def test_model_select_skips_queued(self, setup_project, mock_client) -> None:
        existing = TaskRecord(task_id="old-1", model_id="mj-v7", status="QUEUED")
        project_store.save_task_record("test-proj", "mj-v7", existing)

        records = await generate_model_select(
            "test-proj", mock_client, "bold anime style", models=["mj-v7"],
        )
        assert records["mj-v7"].task_id == "old-1"
        mock_client.post.assert_not_called()

    async def test_style_refine_skips_existing_success(self, setup_project, mock_client) -> None:
        existing = TaskRecord(task_id="old-1", model_id="mj-v7", status="SUCCESS")
        project_store.save_round_task_record("test-proj", 1, "mj-v7", existing)

        records = await generate_style_refine(
            "test-proj", mock_client, 1, "bold anime style",
        )
        assert records["mj-v7"].task_id == "old-1"
        mock_client.post.assert_not_called()

    async def test_style_refine_resubmits_failed(self, setup_project, mock_client) -> None:
        existing = TaskRecord(task_id="old-1", model_id="mj-v7", status="FAILED")
        project_store.save_round_task_record("test-proj", 1, "mj-v7", existing)

        records = await generate_style_refine(
            "test-proj", mock_client, 1, "bold anime style",
        )
        assert records["mj-v7"].task_id == "t1"
        mock_client.post.assert_called_once()
