import pytest

from styleclaw.core.models import BatchCase, BatchConfig, ProjectConfig
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def project_name():
    config = ProjectConfig(name="i2i-cfg-test")
    project_store.create_project(config)
    return "i2i-cfg-test"


class TestI2IBatchConfig:
    def test_round_trip_i2i_batch_config(self, project_name):
        cases = [
            BatchCase(id="i2i-001", category="i2i", description="test ref 1", status="submitted"),
            BatchCase(id="i2i-002", category="i2i", description="test ref 2", status="submitted"),
        ]
        bc = BatchConfig(batch=1, trigger_phrase="watercolor", cases=cases)
        project_store.save_i2i_batch_config(project_name, 1, bc)
        loaded = project_store.load_i2i_batch_config(project_name, 1)
        assert loaded.trigger_phrase == "watercolor"
        assert len(loaded.cases) == 2
        assert loaded.cases[0].id == "i2i-001"
