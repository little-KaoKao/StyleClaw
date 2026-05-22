from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from styleclaw.core.models import Phase, ProjectConfig, ProjectState, TaskRecord
from styleclaw.providers.runninghub.models import MODEL_REGISTRY
from styleclaw.scripts.generate import (
    TEST_SUBJECTS,
    generate_model_select,
    generate_style_refine,
    resolve_test_subjects,
)
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
        assert len(records) == len(MODEL_REGISTRY) * 2 * len(TEST_SUBJECTS)

    async def test_submits_for_specific_models(self, setup_project, mock_client) -> None:
        records = await generate_model_select(
            "test-proj", mock_client, "bold anime style", models=["mj-v7"],
        )
        assert len(records) == 2 * len(TEST_SUBJECTS)
        assert "mj-v7/prompt-only-male" in records
        assert "mj-v7/prompt-only-female" in records
        assert "mj-v7/prompt-sref-male" in records
        assert "mj-v7/prompt-sref-female" in records

    async def test_saves_task_records(self, setup_project, mock_client) -> None:
        await generate_model_select(
            "test-proj", mock_client, "bold anime style", models=["mj-v7"],
        )
        record = project_store.load_task_record(
            "test-proj", "mj-v7", variant="prompt-only-male",
        )
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
        existing_po = TaskRecord(task_id="old-po", model_id="mj-v7", status="SUCCESS")
        existing_ps = TaskRecord(task_id="old-ps", model_id="mj-v7", status="SUCCESS")
        for gender in TEST_SUBJECTS:
            project_store.save_task_record(
                "test-proj", "mj-v7", existing_po, variant=f"prompt-only-{gender}",
            )
            project_store.save_task_record(
                "test-proj", "mj-v7", existing_ps, variant=f"prompt-sref-{gender}",
            )

        records = await generate_model_select(
            "test-proj", mock_client, "bold anime style", models=["mj-v7"],
        )
        assert records["mj-v7/prompt-only-male"].task_id == "old-po"
        assert records["mj-v7/prompt-only-female"].task_id == "old-po"
        assert records["mj-v7/prompt-sref-male"].task_id == "old-ps"
        assert records["mj-v7/prompt-sref-female"].task_id == "old-ps"
        mock_client.post.assert_not_called()

    async def test_model_select_resubmits_failed(self, setup_project, mock_client) -> None:
        existing = TaskRecord(task_id="old-1", model_id="mj-v7", status="FAILED")
        project_store.save_task_record(
            "test-proj", "mj-v7", existing, variant="prompt-only-male",
        )

        records = await generate_model_select(
            "test-proj", mock_client, "bold anime style", models=["mj-v7"],
        )
        assert records["mj-v7/prompt-only-male"].task_id == "t1"

    async def test_model_select_skips_queued(self, setup_project, mock_client) -> None:
        existing = TaskRecord(task_id="old-1", model_id="mj-v7", status="QUEUED")
        project_store.save_task_record(
            "test-proj", "mj-v7", existing, variant="prompt-only-male",
        )

        records = await generate_model_select(
            "test-proj", mock_client, "bold anime style", models=["mj-v7"],
        )
        assert records["mj-v7/prompt-only-male"].task_id == "old-1"

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


class TestResolveTestSubjects:
    def test_none_returns_fallback(self) -> None:
        result = resolve_test_subjects(None)
        assert result == TEST_SUBJECTS

    def test_empty_dict_returns_fallback(self) -> None:
        result = resolve_test_subjects({})
        assert result == TEST_SUBJECTS

    def test_full_override(self) -> None:
        result = resolve_test_subjects({"male": "M", "female": "F"})
        assert result == {"male": "M", "female": "F"}

    def test_partial_override_male_only(self) -> None:
        result = resolve_test_subjects({"male": "M"})
        assert result["male"] == "M"
        assert result["female"] == TEST_SUBJECTS["female"]

    def test_partial_override_female_only(self) -> None:
        result = resolve_test_subjects({"female": "F"})
        assert result["male"] == TEST_SUBJECTS["male"]
        assert result["female"] == "F"

    def test_whitespace_falls_back(self) -> None:
        result = resolve_test_subjects({"male": "   ", "female": ""})
        assert result == TEST_SUBJECTS

    def test_unknown_keys_ignored(self) -> None:
        result = resolve_test_subjects({"male": "M", "nonbinary": "X"})
        assert result == {"male": "M", "female": TEST_SUBJECTS["female"]}


class TestGenerateModelSelectTestSubjects:
    async def test_uses_provided_subjects(self, setup_project, mock_client) -> None:
        await generate_model_select(
            "test-proj", mock_client, "bold anime style",
            models=["mj-v7"],
            test_subjects={"male": "BOY-SENTINEL", "female": "GIRL-SENTINEL"},
        )
        male_rec = project_store.load_task_record(
            "test-proj", "mj-v7", variant="prompt-only-male",
        )
        female_rec = project_store.load_task_record(
            "test-proj", "mj-v7", variant="prompt-only-female",
        )
        assert "BOY-SENTINEL" in male_rec.prompt
        assert "GIRL-SENTINEL" in female_rec.prompt
        # Sanity: each task's prompt must not bleed across genders.
        assert "GIRL-SENTINEL" not in male_rec.prompt
        assert "BOY-SENTINEL" not in female_rec.prompt

    async def test_partial_subjects_fall_back(self, setup_project, mock_client) -> None:
        await generate_model_select(
            "test-proj", mock_client, "bold anime style",
            models=["mj-v7"],
            test_subjects={"male": "BOY-ONLY-SENTINEL"},
        )
        male_rec = project_store.load_task_record(
            "test-proj", "mj-v7", variant="prompt-only-male",
        )
        female_rec = project_store.load_task_record(
            "test-proj", "mj-v7", variant="prompt-only-female",
        )
        assert "BOY-ONLY-SENTINEL" in male_rec.prompt
        assert TEST_SUBJECTS["female"] in female_rec.prompt

    async def test_default_uses_fallback(self, setup_project, mock_client) -> None:
        await generate_model_select(
            "test-proj", mock_client, "bold anime style", models=["mj-v7"],
        )
        male_rec = project_store.load_task_record(
            "test-proj", "mj-v7", variant="prompt-only-male",
        )
        assert TEST_SUBJECTS["male"] in male_rec.prompt
