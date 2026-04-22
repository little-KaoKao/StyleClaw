import pytest
from pathlib import Path

from styleclaw.core.models import (
    Phase,
    ProjectConfig,
    ProjectState,
    StyleAnalysis,
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


class TestCreateProject:
    def test_creates_directory_structure(self, sample_config):
        root = project_store.create_project(sample_config)
        assert root.exists()
        assert (root / "config.json").exists()
        assert (root / "state.json").exists()
        assert (root / "refs").is_dir()
        assert (root / "model-select" / "results").is_dir()

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
