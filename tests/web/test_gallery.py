from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.storage import project_store
from styleclaw.web.gallery import build_gallery


def _project_with_ref(name: str, phase: Phase) -> None:
    project_store.create_project(
        ProjectConfig(name=name, ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state(name, ProjectState(phase=phase))
    # write a dummy ref file so media URLs resolve
    ref = project_store.project_dir(name) / "refs" / "ref-001.png"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"\x89PNG\r\n\x1a\n")


def test_gallery_init_phase_lists_refs(data_root):
    _project_with_ref("g", Phase.INIT)
    gallery = build_gallery("g")
    assert gallery["phase"] == "INIT"
    assert gallery["ref_images"] == ["/media/g/refs/ref-001.png"]
    assert gallery["groups"] == []


def test_media_url_is_relative_to_project_dir(data_root):
    _project_with_ref("g", Phase.INIT)
    gallery = build_gallery("g")
    assert gallery["ref_images"][0].startswith("/media/g/")


def test_gallery_endpoint(client, data_root):
    _project_with_ref("g", Phase.INIT)
    resp = client.get("/api/projects/g/gallery")
    assert resp.status_code == 200
    assert resp.json()["phase"] == "INIT"


def test_media_endpoint_serves_file(client, data_root):
    _project_with_ref("g", Phase.INIT)
    resp = client.get("/media/g/refs/ref-001.png")
    assert resp.status_code == 200
    assert resp.content.startswith(b"\x89PNG")


def test_safe_media_path_rejects_traversal(data_root):
    # Test the guard directly: an httpx client normalizes `../` out of the URL
    # before it reaches the route, so only a direct call exercises this branch.
    from styleclaw.web.routes_projects import safe_media_path

    _project_with_ref("g", Phase.INIT)
    assert safe_media_path("g", "refs/ref-001.png") is not None
    assert safe_media_path("g", "../../../etc/passwd") is None


def test_gallery_model_select_attaches_scores(data_root):
    from styleclaw.core.models import (
        DimensionScores,
        ModelEvaluation,
        ModelScore,
        TaskRecord,
        TaskStatus,
    )

    project_store.create_project(
        ProjectConfig(name="m", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("m", ProjectState(phase=Phase.MODEL_SELECT, current_model_select_pass=1))

    rec = TaskRecord(task_id="t1", model_id="mj-v7", status=TaskStatus.SUCCESS)
    project_store.save_task_record("m", "mj-v7", rec, variant="prompt-sref-male", pass_num=1)
    results_dir = project_store.model_results_dir("m", "mj-v7", variant="prompt-sref-male", pass_num=1)
    (results_dir / "output-001.png").write_bytes(b"\x89PNG")

    evaluation = ModelEvaluation(
        evaluations=[
            ModelScore(
                model="mj-v7", variant="prompt-sref", total=8.0,
                scores=DimensionScores(visual_style=8.0),
            )
        ],
        recommendation="mj-v7",
    )
    project_store.save_evaluation("m", evaluation, pass_num=1)

    gallery = build_gallery("m")
    assert gallery["phase"] == "MODEL_SELECT"
    grp = next(g for g in gallery["groups"] if g["label"] == "mj-v7/prompt-sref-male")
    assert len(grp["images"]) == 1
    assert grp["scores"] is not None
    assert grp["scores"]["total"] == 8.0


def test_gallery_batch_t2i_lists_case_images(data_root):
    from styleclaw.core.models import BatchCase, BatchConfig

    project_store.create_project(
        ProjectConfig(name="b", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("b", ProjectState(phase=Phase.BATCH_T2I, current_batch=1))
    project_store.save_batch_config(
        "b", 1,
        BatchConfig(
            batch=1, trigger_phrase="trig",
            cases=[BatchCase(id="case-001", category="adult_male", description="d")],
        ),
    )
    case_dir = project_store.batch_t2i_case_dir("b", 1, "case-001")
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "output-001.png").write_bytes(b"\x89PNG")

    gallery = build_gallery("b")
    assert gallery["phase"] == "BATCH_T2I"
    assert gallery["groups"][0]["images"] == ["/media/b/batch-t2i/batch-001/results/case-001/output-001.png"]
