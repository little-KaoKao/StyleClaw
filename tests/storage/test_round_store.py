import pytest

from styleclaw.core.models import (
    DimensionScores,
    Phase,
    ProjectConfig,
    PromptConfig,
    RoundEvaluation,
    RoundScore,
    TaskRecord,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def project_with_state():
    config = ProjectConfig(name="test-proj", ip_info="anime style")
    project_store.create_project(config)
    state = project_store.load_state("test-proj")
    new_state = state.with_phase(Phase.STYLE_REFINE).with_round(1)
    project_store.save_state("test-proj", new_state)
    return "test-proj"


class TestRoundStorage:
    def test_round_dir_created(self, project_with_state):
        d = project_store.round_dir("test-proj", 1)
        assert d.exists()
        assert "round-001" in str(d)

    def test_round_trip_prompt_config(self, project_with_state):
        pc = PromptConfig(
            round=1,
            trigger_phrase="watercolor soft lighting",
            derived_from="initial-analysis",
            adjustment_note="increased color saturation",
        )
        project_store.save_prompt_config("test-proj", 1, pc)
        loaded = project_store.load_prompt_config("test-proj", 1)
        assert loaded.trigger_phrase == "watercolor soft lighting"
        assert loaded.round == 1

    def test_round_trip_task_record(self, project_with_state):
        record = TaskRecord(task_id="t-123", model_id="mj-v7", status="SUCCESS")
        project_store.save_round_task_record("test-proj", 1, "mj-v7", record)
        loaded = project_store.load_round_task_record("test-proj", 1, "mj-v7")
        assert loaded.task_id == "t-123"

    def test_load_all_round_task_records(self, project_with_state):
        r1 = TaskRecord(task_id="t-1", model_id="mj-v7")
        r2 = TaskRecord(task_id="t-2", model_id="niji7")
        project_store.save_round_task_record("test-proj", 1, "mj-v7", r1)
        project_store.save_round_task_record("test-proj", 1, "niji7", r2)
        all_records = project_store.load_all_round_task_records("test-proj", 1)
        assert len(all_records) == 2
        assert "mj-v7" in all_records
        assert "niji7" in all_records

    def test_round_trip_evaluation(self, project_with_state):
        ev = RoundEvaluation(
            round=1,
            evaluations=[
                RoundScore(
                    model="mj-v7",
                    scores=DimensionScores(
                        color_palette=8, line_style=7, lighting=8,
                        texture=6, overall_mood=8,
                    ),
                    total=7.4,
                    analysis="Good color match",
                ),
            ],
            recommendation="continue_refine",
            next_direction="improve texture",
        )
        project_store.save_round_evaluation("test-proj", 1, ev)
        loaded = project_store.load_round_evaluation("test-proj", 1)
        assert loaded.recommendation == "continue_refine"
        assert loaded.evaluations[0].scores.color_palette == 8
        assert loaded.evaluations[0].total == 7.4

    def test_empty_round_records(self, project_with_state):
        records = project_store.load_all_round_task_records("test-proj", 99)
        assert records == {}
