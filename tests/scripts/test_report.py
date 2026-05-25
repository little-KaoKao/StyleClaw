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
    _relative_img_src,
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


class TestRelativeImgSrc:
    def test_returns_relative_path_with_forward_slashes(self, tmp_path: Path) -> None:
        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        img_dir = tmp_path / "assets"
        img_dir.mkdir()
        img = img_dir / "a.png"
        Image.new("RGB", (4, 4)).save(img)
        src = _relative_img_src(img, report_dir)
        assert src == "../assets/a.png"

    def test_returns_same_dir_path(self, tmp_path: Path) -> None:
        img = tmp_path / "a.png"
        Image.new("RGB", (4, 4)).save(img)
        src = _relative_img_src(img, tmp_path)
        assert src == "a.png"

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "missing.png"
        assert _relative_img_src(p, tmp_path) == ""


class TestGenerateModelSelectReport:
    def test_generates_html(self, setup_project) -> None:
        analysis = StyleAnalysis(trigger_phrase="bold anime")
        project_store.save_analysis("test-proj", analysis)

        evaluation = ModelEvaluation(
            evaluations=[
                ModelScore(
                    model="mj-v7",
                    scores=DimensionScores(visual_style=8.0, color_science=8.0, lighting_quality=7.5, material_texture=7.5, post_processing=7.0, spatial_perspective=7.5, dynamic_state=8.0),
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
        html = path.read_text(encoding="utf-8")
        assert "bold anime" in html
        assert "mj-v7" in html
        # P1: images are relative paths, never data URIs
        assert "data:image" not in html
        assert 'src="results/mj-v7/output-001.png"' in html
        assert 'src="../../refs/ref-001.png"' in html

    def test_sref_image_honors_sref_index(self, tmp_path) -> None:
        """The sref thumbnail in the report must reflect config.sref_index,
        not always default to ref_images[0]."""
        config = ProjectConfig(
            name="sref-proj",
            ip_info="anime",
            ref_images=["refs/ref-001.png", "refs/ref-002.png", "refs/ref-003.png"],
            sref_index=2,
        )
        root = project_store.create_project(config)
        for fname in ("ref-001.png", "ref-002.png", "ref-003.png"):
            Image.new("RGB", (100, 100)).save(root / "refs" / fname)
        project_store.save_state(
            "sref-proj",
            ProjectState(phase=Phase.MODEL_SELECT, current_round=1, current_batch=1),
        )

        project_store.save_analysis("sref-proj", StyleAnalysis(trigger_phrase="x"))
        evaluation = ModelEvaluation(
            evaluations=[
                ModelScore(
                    model="mj-v7",
                    scores=DimensionScores(visual_style=7.0, color_science=7.0, lighting_quality=7.0, material_texture=7.0, post_processing=7.0, spatial_perspective=7.0, dynamic_state=7.0),
                    total=7.0,
                    analysis="ok",
                    suggestions="",
                ),
            ],
            recommendation="mj-v7",
        )
        project_store.save_evaluation("sref-proj", evaluation)

        path = generate_model_select_report("sref-proj")
        html = path.read_text(encoding="utf-8")

        sref_section = html.split('Style Reference')[1].split('</div>')[0]
        assert "ref-003.png" in sref_section
        assert "ref-001.png" not in sref_section
        assert "ref-002.png" not in sref_section


class TestGenerateStyleRefineReport:
    def test_generates_html(self, setup_project) -> None:
        scores = DimensionScores(visual_style=8.0, color_science=8.0, lighting_quality=7.5, material_texture=7.5, post_processing=7.0, spatial_perspective=7.5, dynamic_state=8.0)
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
        html = path.read_text(encoding="utf-8")
        assert "refined trigger" in html
        assert "data:image" not in html
        assert 'src="results/mj-v7/output-001.png"' in html
        assert 'src="../../../refs/ref-001.png"' in html


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
        html = path.read_text(encoding="utf-8")
        assert "bold anime" in html
        assert "am-001" in html
        assert "data:image" not in html
        assert 'src="results/am-001/output-001.png"' in html

    def test_surfaces_failed_downloads(self, setup_project) -> None:
        """When poll.py wrote failed_downloads.json next to a case (because
        some result images couldn't be fetched), the report must surface the
        count so the user notices missing images."""
        import json

        cases = [
            BatchCase(id="am-001", category="adult_male", description="test char", status="SUCCESS"),
        ]
        batch_config = BatchConfig(batch=1, trigger_phrase="t", cases=cases)
        project_store.save_batch_config("test-proj", 1, batch_config)
        project_store.save_batch_task_record(
            "test-proj", 1, "am-001",
            TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS"),
        )

        case_dir = project_store.batch_t2i_case_dir("test-proj", 1, "am-001")
        Image.new("RGB", (100, 100)).save(case_dir / "output-001.png")
        (case_dir / "failed_downloads.json").write_text(
            json.dumps({"failed_urls": ["http://cdn/x.png", "http://cdn/y.png"]}),
            encoding="utf-8",
        )

        path = generate_batch_t2i_report("test-proj", 1)
        html = path.read_text(encoding="utf-8")
        assert "2 image" in html and "download" in html.lower()


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
        html = path.read_text(encoding="utf-8")
        assert "i2i-001" in html
        assert "data:image" not in html
        assert 'src="source-images/source.png"' in html
        assert 'src="results/i2i-001/output-001.png"' in html


class TestMultiImageGrid:
    """Models like MJ return 4 images per task; templates lay multi-image
    cells out as 2-col grids. Narrow cells (batch_t2i / batch_i2i) use a
    .imgs container with .single fallback for 1-image cases. Wide cards
    (model_select / style_refine) constrain the grid to ~320px so each
    thumbnail stays around 150px instead of stretching to card width."""

    def test_batch_t2i_multi_image_uses_grid(self, setup_project) -> None:
        cases = [BatchCase(id="am-001", category="adult_male", description="d", status="SUCCESS")]
        project_store.save_batch_config("test-proj", 1, BatchConfig(batch=1, trigger_phrase="t", cases=cases))
        project_store.save_batch_task_record(
            "test-proj", 1, "am-001",
            TaskRecord(task_id="t1", model_id="mj-v7", status="SUCCESS"),
        )
        case_dir = project_store.batch_t2i_case_dir("test-proj", 1, "am-001")
        for i in range(1, 5):
            Image.new("RGB", (50, 50)).save(case_dir / f"output-{i:03d}.png")

        html = generate_batch_t2i_report("test-proj", 1).read_text(encoding="utf-8")
        import re
        block = re.search(r'<div class="imgs[^"]*">.*?</div>', html, re.DOTALL)
        assert block is not None
        assert "single" not in re.search(r'class="(imgs[^"]*)"', block.group(0)).group(1)
        assert block.group(0).count("<img") == 4

    def test_model_select_splits_images_by_gender(self, tmp_path) -> None:
        """Variant evaluations carry male+female sub-dirs; the report must
        render them as two labeled sub-blocks (Male / Female) so each forms
        a tight 2x2 grid instead of a single wide strip."""
        config = ProjectConfig(name="gp", ip_info="x", ref_images=["refs/ref-001.png"])
        root = project_store.create_project(config)
        Image.new("RGB", (50, 50)).save(root / "refs" / "ref-001.png")
        project_store.save_state(
            "gp", ProjectState(phase=Phase.MODEL_SELECT, current_round=1, current_batch=1),
        )
        project_store.save_analysis("gp", StyleAnalysis(trigger_phrase="t"))
        scores = DimensionScores(visual_style=8, color_science=8, lighting_quality=8, material_texture=8, post_processing=8, spatial_perspective=8, dynamic_state=8)
        project_store.save_evaluation(
            "gp",
            ModelEvaluation(
                evaluations=[
                    ModelScore(
                        model="mj-v7", variant="prompt-sref",
                        scores=scores, total=8, analysis="a", suggestions="",
                    ),
                ],
                recommendation="mj-v7",
            ),
        )
        for gender in ("male", "female"):
            d = project_store.model_results_dir("gp", "mj-v7", variant=f"prompt-sref-{gender}")
            for i in range(1, 5):
                Image.new("RGB", (50, 50)).save(d / f"output-{i:03d}.png")

        html = generate_model_select_report("gp").read_text(encoding="utf-8")

        # Both Male and Female labels appear, each followed by its own .images
        # grid. We don't pin exact counts because list_output_images may pick
        # up sibling files; we only assert the structure that drives layout.
        assert html.count('class="image-block-label"') == 2
        assert ">Male<" in html and ">Female<" in html
        assert html.count('class="images"') == 2
