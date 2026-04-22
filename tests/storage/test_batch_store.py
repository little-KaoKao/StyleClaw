import pytest

from styleclaw.core.models import (
    BatchCase,
    BatchConfig,
    ProjectConfig,
    TaskRecord,
    UploadRecord,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def project_name():
    config = ProjectConfig(name="batch-test")
    project_store.create_project(config)
    return "batch-test"


class TestBatchT2IStorage:
    def test_round_trip_batch_config(self, project_name):
        cases = [
            BatchCase(id="c-01", category="adult_male", description="tall man"),
            BatchCase(id="c-02", category="outdoor_scene", description="park", aspect_ratio="16:9"),
        ]
        bc = BatchConfig(batch=1, trigger_phrase="watercolor", cases=cases)
        project_store.save_batch_config(project_name, 1, bc)
        loaded = project_store.load_batch_config(project_name, 1)
        assert loaded.trigger_phrase == "watercolor"
        assert len(loaded.cases) == 2
        assert loaded.cases[1].aspect_ratio == "16:9"

    def test_round_trip_batch_task_record(self, project_name):
        record = TaskRecord(task_id="bt-001", model_id="mj-v7", status="QUEUED")
        project_store.save_batch_task_record(project_name, 1, "c-01", record)
        loaded = project_store.load_batch_task_record(project_name, 1, "c-01")
        assert loaded.task_id == "bt-001"

    def test_load_all_batch_task_records(self, project_name):
        r1 = TaskRecord(task_id="bt-001", model_id="mj-v7")
        r2 = TaskRecord(task_id="bt-002", model_id="mj-v7")
        project_store.save_batch_task_record(project_name, 1, "c-01", r1)
        project_store.save_batch_task_record(project_name, 1, "c-02", r2)
        records = project_store.load_all_batch_task_records(project_name, 1)
        assert len(records) == 2

    def test_empty_batch_records(self, project_name):
        records = project_store.load_all_batch_task_records(project_name, 99)
        assert records == {}


class TestBatchI2IStorage:
    def test_round_trip_i2i_uploads(self, project_name):
        uploads = [
            UploadRecord(local_path="img1.png", url="https://example.com/1.png", file_name="1.png"),
            UploadRecord(local_path="img2.png", url="https://example.com/2.png", file_name="2.png"),
        ]
        project_store.save_i2i_uploads(project_name, 1, uploads)
        loaded = project_store.load_i2i_uploads(project_name, 1)
        assert len(loaded) == 2
        assert loaded[0].url == "https://example.com/1.png"

    def test_empty_i2i_uploads(self, project_name):
        loaded = project_store.load_i2i_uploads(project_name, 99)
        assert loaded == []

    def test_round_trip_i2i_task_record(self, project_name):
        record = TaskRecord(task_id="i2i-001", model_id="nb2")
        project_store.save_i2i_task_record(project_name, 1, "i2i-001", record)
        loaded = project_store.load_i2i_task_record(project_name, 1, "i2i-001")
        assert loaded.task_id == "i2i-001"

    def test_load_all_i2i_task_records(self, project_name):
        r1 = TaskRecord(task_id="i2i-001", model_id="nb2")
        r2 = TaskRecord(task_id="i2i-002", model_id="nb2")
        project_store.save_i2i_task_record(project_name, 1, "i2i-001", r1)
        project_store.save_i2i_task_record(project_name, 1, "i2i-002", r2)
        records = project_store.load_all_i2i_task_records(project_name, 1)
        assert len(records) == 2
