from __future__ import annotations

from styleclaw.core.models import Phase, ProjectConfig, ProjectState
from styleclaw.core.state_machine import advance
from styleclaw.storage import project_store


def _project_in_batch_t2i(name: str) -> None:
    """Create a project that has passed through INIT -> MODEL_SELECT -> STYLE_REFINE -> BATCH_T2I.

    This means both MODEL_SELECT and STYLE_REFINE are valid rollback targets.
    can_rollback requires target_idx < current_idx, so the project must be
    *past* STYLE_REFINE for a STYLE_REFINE rollback to be allowed.
    """
    project_store.create_project(
        ProjectConfig(name=name, ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    s = ProjectState(phase=Phase.INIT)
    s = advance(s, Phase.MODEL_SELECT)
    s = advance(s, Phase.STYLE_REFINE).with_round(3)
    # Advance one more phase so STYLE_REFINE is a valid (earlier) rollback target.
    s = advance(s, Phase.BATCH_T2I)
    project_store.save_state(name, s)
    # Create round dirs 1, 2, 3 under pass-001 so the round-existence check passes.
    for rnum in (1, 2, 3):
        (
            project_store.project_dir(name)
            / "style-refine"
            / project_store.pass_label(1)
            / project_store.round_label(rnum)
        ).mkdir(parents=True, exist_ok=True)


def test_rollback_to_round(client, data_root):
    _project_in_batch_t2i("r")
    resp = client.post("/api/projects/r/rollback", json={"to": "STYLE_REFINE", "round": 2})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert project_store.load_state("r").current_round == 2


def test_rollback_invalid_phase(client, data_root):
    _project_in_batch_t2i("r2")
    resp = client.post("/api/projects/r2/rollback", json={"to": "NONSENSE"})
    assert resp.status_code == 400


def test_rollback_to_unvisited_phase_fails(client, data_root):
    # A fresh INIT project cannot roll back anywhere earlier.
    project_store.create_project(
        ProjectConfig(name="r3", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    project_store.save_state("r3", ProjectState(phase=Phase.INIT))
    resp = client.post("/api/projects/r3/rollback", json={"to": "MODEL_SELECT"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


def test_rollback_nonexistent_round(client, data_root):
    _project_in_batch_t2i("r4")
    resp = client.post("/api/projects/r4/rollback", json={"to": "STYLE_REFINE", "round": 99})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


def test_rollback_same_phase_round_within_style_refine(client, data_root):
    # Project currently IN STYLE_REFINE at round 3; redo/navigate to round 2.
    project_store.create_project(
        ProjectConfig(name="sr", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    s = ProjectState(phase=Phase.INIT)
    s = advance(s, Phase.MODEL_SELECT)
    s = advance(s, Phase.STYLE_REFINE).with_round(3)
    project_store.save_state("sr", s)
    for rnum in (1, 2, 3):
        (project_store.project_dir("sr") / "style-refine" / project_store.pass_label(1) / project_store.round_label(rnum)).mkdir(parents=True, exist_ok=True)

    resp = client.post("/api/projects/sr/rollback", json={"to": "STYLE_REFINE", "round": 2})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert project_store.load_state("sr").phase == Phase.STYLE_REFINE  # phase unchanged
    assert project_store.load_state("sr").current_round == 2


def test_rollback_same_phase_nonexistent_round(client, data_root):
    project_store.create_project(
        ProjectConfig(name="sr2", ip_info="anime", ref_images=["refs/ref-001.png"])
    )
    s = ProjectState(phase=Phase.INIT)
    s = advance(s, Phase.MODEL_SELECT)
    s = advance(s, Phase.STYLE_REFINE).with_round(3)
    project_store.save_state("sr2", s)
    (project_store.project_dir("sr2") / "style-refine" / project_store.pass_label(1) / project_store.round_label(1)).mkdir(parents=True, exist_ok=True)
    resp = client.post("/api/projects/sr2/rollback", json={"to": "STYLE_REFINE", "round": 99})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False
