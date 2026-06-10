from __future__ import annotations

from styleclaw.storage import project_store


def _nums(parent, prefix: str) -> list[int]:
    """Return sorted integer suffixes of `prefix-NNN` dirs under `parent`."""
    if not parent.is_dir():
        return []
    out = []
    for p in parent.iterdir():
        if p.is_dir() and p.name.startswith(f"{prefix}-"):
            try:
                out.append(int(p.name.split("-")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(out)


def build_history(name: str) -> dict:
    """Enumerate all passes/rounds/batches that exist on disk for a project.

    Raises FileNotFoundError (via load_state) if the project doesn't exist.
    """
    state = project_store.load_state(name)  # raises FileNotFoundError if missing
    root = project_store.project_dir(name)

    ms_passes = _nums(root / "model-select", "pass")

    sr_root = root / "style-refine"
    sr_passes: dict[str, list[int]] = {}
    for pnum in _nums(sr_root, "pass"):
        rounds = _nums(sr_root / project_store.pass_label(pnum), "round")
        sr_passes[str(pnum)] = rounds

    t2i = _nums(root / "batch-t2i", "batch")
    i2i = _nums(root / "batch-i2i", "batch")

    return {
        "model_select": {"passes": ms_passes},
        "style_refine": {"passes": sr_passes},
        "batch_t2i": {"batches": t2i},
        "batch_i2i": {"batches": i2i},
        "current": {
            "phase": state.phase.value,
            "pass": state.current_model_select_pass,
            "round": state.current_round,
            "batch": state.current_batch,
        },
    }
