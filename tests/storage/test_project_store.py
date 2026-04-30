import pytest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from styleclaw.core.models import (
    Phase,
    ProjectConfig,
    ProjectState,
    StyleAnalysis,
    TaskRecord,
    UploadRecord,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def sample_config():
    return ProjectConfig(
        name="test-project",
        description="Test description",
        ip_info="anime style",
        ref_images=["refs/ref-001.png"],
    )


class TestValidateProjectName:
    @pytest.mark.parametrize("name", [
        "my-project",
        "test_project",
        "Project123",
        "a",
        "x-y-z",
    ])
    def test_valid_names_accepted(self, name):
        project_store._validate_project_name(name)

    @pytest.mark.parametrize("name,reason", [
        ("../../etc", "path traversal with .."),
        ("../passwd", "path traversal with .."),
        (".hidden", "starts with dot"),
        ("foo/bar", "contains forward slash"),
        ("foo\\bar", "contains backslash"),
        ("", "empty string"),
        ("hello world", "contains space"),
        ("-starts-dash", "starts with non-alphanumeric"),
        ("_starts-underscore", "starts with non-alphanumeric"),
    ])
    def test_invalid_names_rejected(self, name, reason):
        with pytest.raises(ValueError, match="Invalid project name"):
            project_store._validate_project_name(name)

    def test_project_dir_rejects_traversal(self):
        with pytest.raises(ValueError):
            project_store.project_dir("../../etc")

    def test_create_project_rejects_traversal(self):
        config = ProjectConfig(
            name="../evil",
            ip_info="test",
        )
        with pytest.raises(ValueError):
            project_store.create_project(config)

    def test_project_dir_accepts_valid_name(self):
        path = project_store.project_dir("my-project")
        assert path.name == "my-project"


class TestCreateProject:
    def test_creates_directory_structure(self, sample_config):
        root = project_store.create_project(sample_config)
        assert root.exists()
        assert (root / "config.json").exists()
        assert (root / "state.json").exists()
        assert (root / "refs").is_dir()
        assert (root / "model-select").is_dir()

    def test_duplicate_project_raises(self, sample_config):
        project_store.create_project(sample_config)
        with pytest.raises(FileExistsError):
            project_store.create_project(sample_config)


class TestListProjects:
    def test_empty_when_no_projects(self):
        assert project_store.list_projects() == []

    def test_lists_created_projects(self, sample_config):
        project_store.create_project(sample_config)
        projects = project_store.list_projects()
        assert projects == ["test-project"]


class TestLoadSave:
    def test_round_trip_config(self, sample_config):
        project_store.create_project(sample_config)
        loaded = project_store.load_config("test-project")
        assert loaded.name == sample_config.name
        assert loaded.ip_info == sample_config.ip_info

    def test_round_trip_state(self, sample_config):
        project_store.create_project(sample_config)
        state = project_store.load_state("test-project")
        assert state.phase == Phase.INIT

        new_state = state.with_phase(Phase.MODEL_SELECT)
        project_store.save_state("test-project", new_state)

        reloaded = project_store.load_state("test-project")
        assert reloaded.phase == Phase.MODEL_SELECT

    def test_round_trip_uploads(self, sample_config):
        project_store.create_project(sample_config)
        records = [
            UploadRecord(local_path="refs/ref-001.png", url="https://example.com/1.png", file_name="1.png"),
        ]
        project_store.save_uploads("test-project", records)
        loaded = project_store.load_uploads("test-project")
        assert len(loaded) == 1
        assert loaded[0].url == "https://example.com/1.png"

    def test_round_trip_analysis(self, sample_config):
        project_store.create_project(sample_config)
        analysis = StyleAnalysis(
            trigger_phrase="watercolor soft lighting",
            color_palette="pastel tones",
        )
        project_store.save_analysis("test-project", analysis)
        loaded = project_store.load_analysis("test-project")
        assert loaded.trigger_phrase == "watercolor soft lighting"


class TestGenericHelpers:
    def test_load_model_returns_typed_instance(self, sample_config):
        project_store.create_project(sample_config)
        path = project_store.project_dir("test-project") / "config.json"
        result = project_store._load_model(ProjectConfig, path)
        assert isinstance(result, ProjectConfig)
        assert result.name == "test-project"

    def test_save_model_writes_valid_json(self, sample_config, tmp_path):
        dest = tmp_path / "out.json"
        project_store._save_model(sample_config, dest)
        loaded = project_store._load_model(ProjectConfig, dest)
        assert loaded.name == sample_config.name

    def test_load_all_records_returns_task_records(self, sample_config):
        project_store.create_project(sample_config)
        r1 = TaskRecord(task_id="t-1", model_id="mj-v7")
        r2 = TaskRecord(task_id="t-2", model_id="niji7")
        project_store.save_task_record("test-project", "mj-v7", r1, pass_num=1)
        project_store.save_task_record("test-project", "niji7", r2, pass_num=1)
        results_dir = project_store.model_select_dir("test-project", 1) / "results"
        records = project_store._load_all_records(results_dir)
        assert len(records) == 2
        assert records["mj-v7"].task_id == "t-1"
        assert records["niji7"].task_id == "t-2"

    def test_load_all_records_empty_dir(self, tmp_path):
        records = project_store._load_all_records(tmp_path / "nonexistent")
        assert records == {}

    def test_all_load_all_functions_delegate_to_generic(self, sample_config):
        """Verify the three load_all_* variants all use _load_all_records."""
        project_store.create_project(sample_config)
        with patch.object(
            project_store, "_load_all_records", return_value={}
        ) as mock:
            project_store.load_all_task_records("test-project")
            assert mock.call_count == 1

        state = project_store.load_state("test-project")
        new_state = state.with_phase(Phase.STYLE_REFINE).with_round(1)
        project_store.save_state("test-project", new_state)

        with patch.object(
            project_store, "_load_all_records", return_value={}
        ) as mock:
            project_store.load_all_round_task_records("test-project", 1)
            assert mock.call_count == 1

        new_state2 = new_state.with_phase(Phase.BATCH_T2I)
        project_store.save_state("test-project", new_state2)

        with patch.object(
            project_store, "_load_all_records", return_value={}
        ) as mock:
            project_store.load_all_batch_task_records("test-project", 1)
            assert mock.call_count == 1

        with patch.object(
            project_store, "_load_all_records", return_value={}
        ) as mock:
            project_store.load_all_i2i_task_records("test-project", 1)
            assert mock.call_count == 1


class TestSaveThinking:
    def test_writes_thinking_md_next_to_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")
        target = tmp_path / "projects" / "p" / "analysis.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}")

        project_store.save_thinking(target, "I reasoned step-by-step.")

        md = target.with_suffix(".thinking.md")
        assert md.exists()
        assert "I reasoned step-by-step." in md.read_text(encoding="utf-8")

    def test_empty_thinking_does_not_write_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")
        target = tmp_path / "projects" / "p" / "analysis.json"
        target.parent.mkdir(parents=True)

        project_store.save_thinking(target, "")

        md = target.with_suffix(".thinking.md")
        assert not md.exists()
