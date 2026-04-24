from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from styleclaw.core.models import (
    BatchCase,
    BatchConfig,
    DimensionScores,
    ModelEvaluation,
    ModelScore,
    Phase,
    ProjectConfig,
    ProjectState,
    PromptConfig,
    RoundEvaluation,
    RoundScore,
    StyleAnalysis,
    TaskRecord,
    UploadRecord,
)
from styleclaw.scripts.report import (
    _img_to_data_uri,
    generate_batch_i2i_report,
    generate_batch_t2i_report,
    generate_model_select_report,
    generate_style_refine_report,
)
from styleclaw.storage import project_store


@pytest.fixture(autouse=True)
def use_tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(project_store, "DATA_ROOT", tmp_path / "projects")


@pytest.fixture
def setup_project(tmp_path):
    config = ProjectConfig(name="test-proj", ip_info="anime", ref_images=["refs/ref-001.png"])
    root = project_store.create_project(config)

    ref_img = root / "refs" / "ref-001.png"
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(ref_img)

    state = ProjectState(phase=Phase.MODEL_SELECT, current_round=1, current_batch=1)
    project_store.save_state("test-proj", state)

    return root


class TestImgToDataUri:
    def test_returns_data_uri_for_png(self, tmp_path: Path) -> None:
        p = tmp_path / "test.png"
        Image.new("RGB", (10, 10)).save(p, "PNG")
        uri = _img_to_data_uri(p)
        assert uri.startswith("data:image/png;base64,")

    def test_returns_data_uri_for_jpg(self, tmp_path: Path) -> None:
        p = tmp_path / "test.jpg"
        Image.new("RGB", (10, 10)).save(p, "JPEG")
        uri = _img_to_data_uri(p)
        assert uri.startswith("data:image/jpeg;base64,")

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "missing.png"
        assert _img_to_data_uri(p) == ""


class TestGenerateModelSelectReport:
    def test_generates_html(self, setup_project) -> None:
        analysis = StyleAnalysis(trigger_phrase="bold anime")
        project_store.save_analysis("test-proj", analysis)

        evaluation = ModelEvaluation(
            evaluations=[
                ModelScore(
                    model="mj-v7",
                    scores=DimensionScores(color_palette=8.0, line_style=7.5, lighting=7.0, texture=7.5, overall_mood=8.0),
                    total=8.0,
                    analysis="great",
                    suggestions="none",
                ),
            ],
            recommendation="mj-v7",
        )
        project_store.save_evaluation("test-proj", evaluation)

        results_dir = project_store.model_results_dir("test-proj", "mj-v7")
        Image.new("RGB", (100, 100)).save(results_dir / "output-001.png")

        path = generate_model_select_report("test-proj")
        assert path.exists()
        html = path.read_text()
        assert "bold anime" in html
        assert "mj-v7" in html


class TestGenerateStyleRefineReport:
    def test_generates_html(self, setup_project) -> None:
        scores = DimensionScores(color_palette=8.0, line_style=7.5, lighting=7.0, texture=7.5, overall_mood=8.0)
        evaluation = RoundEvaluation(
            round=1,
            evaluations=[RoundScore(model="mj-v7", scores=scores, total=7.6, analysis="good")],
            recommendation="continue",
        )
        project_store.save_round_evaluation("test-proj", 1, evaluation)

        prompt_config = PromptConfig(round=1, trigger_phrase="refined trigger")
        project_store.save_prompt_config("test-proj", 1, prompt_config)

        results_dir = project_store.round_results_dir("test-proj", 1, "mj-v7")
        Image.new("RGB", (100, 100)).save(results_dir / "output-001.png")

        path = generate_style_refine_report("test-proj", 1)
        assert path.exists()
        html = path.read_text()
        assert "refined trigger" in html


class TestGenerateBatchT2iReport:
    def test_generates_html(self, setup_project) -> None:
        cases = [
            BatchCase(id="am-001", category="adult_male", description="test char", status="SUCCESS"),
        ]
        batch_config = BatchConfig(batch=1, trigger_phrase="bold anime", cases=cases)
        project_store.save_batch_config("test-proj", 1, batch_config)

        record = TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS")
        project_store.save_batch_task_record("test-proj", 1, "am-001", record)

        case_dir = project_store.batch_t2i_case_dir("test-proj", 1, "am-001")
        Image.new("RGB", (100, 100)).save(case_dir / "output-001.png")

        path = generate_batch_t2i_report("test-proj", 1)
        assert path.exists()
        html = path.read_text()
        assert "bold anime" in html
        assert "am-001" in html


class TestGenerateBatchI2iReport:
    def test_generates_html(self, setup_project) -> None:
        uploads = [
            UploadRecord(local_path="source.png", url="https://example.com/s.png", file_name="source.png"),
        ]
        project_store.save_i2i_uploads("test-proj", 1, uploads)

        record = TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS")
        project_store.save_i2i_task_record("test-proj", 1, "i2i-001", record)

        source_dir = project_store.batch_i2i_dir("test-proj", 1) / "source-images"
        source_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100)).save(source_dir / "source.png")

        case_dir = project_store.batch_i2i_case_dir("test-proj", 1, "i2i-001")
        Image.new("RGB", (100, 100)).save(case_dir / "output-001.png")

        path = generate_batch_i2i_report("test-proj", 1)
        assert path.exists()
        html = path.read_text()
        assert "i2i-001" in html
