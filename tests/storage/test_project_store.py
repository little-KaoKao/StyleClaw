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


class TestUpdateState:
    """``update_state`` is the atomic load-modify-save wrapper. It guards
    against the classic lost-update pattern where two CLI invocations both
    load the same baseline and one's write clobbers the other's."""

    def test_basic_increment(self, sample_config):
        project_store.create_project(sample_config)
        new_state = project_store.update_state(
            sample_config.name, lambda s: s.with_round(s.current_round + 1),
        )
        assert new_state.current_round == 1
        assert project_store.load_state(sample_config.name).current_round == 1

    def test_concurrent_increments_no_lost_update(self, sample_config):
        # Spawn N threads that each increment current_round under the lock.
        # If lost-update protection works, the final value equals N.
        import threading

        project_store.create_project(sample_config)
        N = 8
        errors: list[BaseException] = []

        def bump() -> None:
            try:
                project_store.update_state(
                    sample_config.name,
                    lambda s: s.with_round(s.current_round + 1),
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=bump) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"thread failures: {errors}"
        final = project_store.load_state(sample_config.name)
        assert final.current_round == N, (
            f"expected {N} increments to all stick, got {final.current_round} "
            f"— lost-update protection is broken"
        )

    def test_project_lock_releases_on_exception(self, sample_config):
        # If the mutator raises, the lock must still be released so the next
        # caller doesn't deadlock.
        project_store.create_project(sample_config)
        with pytest.raises(RuntimeError, match="boom"):
            def _bad(_s):
                raise RuntimeError("boom")
            project_store.update_state(sample_config.name, _bad)

        # This must complete promptly, not block forever on a stuck lock.
        new_state = project_store.update_state(
            sample_config.name, lambda s: s.with_round(42),
        )
        assert new_state.current_round == 42


class TestLabelHelpers:
    """The single source of truth for ``pass-NNN`` / ``round-NNN`` /
    ``batch-NNN`` formatting. If the on-disk layout ever changes (UUIDs,
    date-prefixed dirs, etc.), it changes here and nowhere else."""

    def test_pass_label_zero_padded(self):
        assert project_store.pass_label(1) == "pass-001"
        assert project_store.pass_label(42) == "pass-042"
        assert project_store.pass_label(999) == "pass-999"

    def test_round_label_zero_padded(self):
        assert project_store.round_label(1) == "round-001"
        assert project_store.round_label(7) == "round-007"

    def test_batch_label_zero_padded(self):
        assert project_store.batch_label(1) == "batch-001"
        assert project_store.batch_label(12) == "batch-012"

    def test_round_glob_across_passes(self):
        # Used by the rollback "does this round exist?" check.
        assert project_store.round_glob_across_passes(3) == "pass-*/round-003"

    def test_label_helpers_used_by_dir_helpers(self, sample_config):
        # Verify the directory helpers route through the labels — change
        # the label format and the directory should follow.
        project_store.create_project(sample_config)
        msd = project_store.model_select_dir(sample_config.name, 5)
        assert msd.name == project_store.pass_label(5)
        rd = project_store.round_dir(sample_config.name, 3, pass_num=2)
        assert rd.name == project_store.round_label(3)
        assert rd.parent.name == project_store.pass_label(2)
        bd = project_store.batch_t2i_dir(sample_config.name, 7)
        assert bd.name == project_store.batch_label(7)


class TestPathTraversalGuard:
    """LLM-produced or hand-edited identifiers feeding subpath construction
    must not be able to escape the project directory via ``..`` or path
    separators."""

    @pytest.mark.parametrize("evil", [
        "../../etc/passwd",
        "..",
        "foo/bar",
        "foo\\bar",
        "/absolute",
        "",
        "-leading-dash",
    ])
    def test_model_id_rejected(self, sample_config, evil):
        project_store.create_project(sample_config)
        with pytest.raises(ValueError, match="model_id"):
            project_store.model_results_dir(sample_config.name, evil)

    @pytest.mark.parametrize("evil", [
        "../../etc/passwd",
        "foo/bar",
        "..",
        "/abs",
    ])
    def test_variant_rejected(self, sample_config, evil):
        project_store.create_project(sample_config)
        with pytest.raises(ValueError, match="variant"):
            project_store.model_results_dir(sample_config.name, "mj-v7", evil)

    @pytest.mark.parametrize("evil", [
        "../../config",
        "..",
        "case/sub",
    ])
    def test_case_id_rejected_t2i(self, sample_config, evil):
        project_store.create_project(sample_config)
        with pytest.raises(ValueError, match="case_id"):
            project_store.batch_t2i_case_dir(sample_config.name, 1, evil)

    @pytest.mark.parametrize("evil", [
        "../../config",
        "..",
        "case/sub",
    ])
    def test_case_id_rejected_i2i(self, sample_config, evil):
        project_store.create_project(sample_config)
        with pytest.raises(ValueError, match="case_id"):
            project_store.batch_i2i_case_dir(sample_config.name, 1, evil)

    @pytest.mark.parametrize("good", [
        "mj-v7",
        "niji7",
        "gpt-image-2",
        "model_v1",
        "case-001",
        "case.id.dotted",
    ])
    def test_well_formed_ids_accepted(self, sample_config, good):
        project_store.create_project(sample_config)
        # These should not raise.
        project_store.model_results_dir(sample_config.name, good)
        project_store.batch_t2i_case_dir(sample_config.name, 1, good)
        project_store.batch_i2i_case_dir(sample_config.name, 1, good)
