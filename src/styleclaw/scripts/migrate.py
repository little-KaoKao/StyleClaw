"""One-shot migration from pre-pass storage layout to pass-scoped layout.

Layouts migrated:
  1. model-select/{initial-analysis.json, evaluation.json, results/...}
     → model-select/pass-001/...
  2. style-refine/round-NNN/...
     → style-refine/pass-001/round-NNN/...

The migration is idempotent: if pass-001 targets already exist, the legacy
sources are left in place and the command reports "nothing to do". Moves are
performed with `shutil.move`; if a partial move fails, whatever has already
been moved stays moved (the command is safe to re-run).
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from styleclaw.storage import project_store

logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    model_select_migrated: bool = False
    style_refine_rounds_migrated: list[int] = None

    def __post_init__(self) -> None:
        if self.style_refine_rounds_migrated is None:
            self.style_refine_rounds_migrated = []

    @property
    def anything_migrated(self) -> bool:
        return self.model_select_migrated or bool(self.style_refine_rounds_migrated)


def _move_children(src: Path, dst: Path) -> None:
    """Move every entry under `src` into `dst`. `src` is removed when empty."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in list(src.iterdir()):
        target = dst / entry.name
        if target.exists():
            raise FileExistsError(
                f"Cannot migrate {entry} → {target}: destination already exists. "
                "Resolve the conflict manually before retrying."
            )
        shutil.move(str(entry), str(target))
    try:
        src.rmdir()
    except OSError:
        # src is not empty or was already removed; leave it for the operator.
        pass


def _migrate_model_select(root: Path) -> bool:
    model_select = root / "model-select"
    if not model_select.exists():
        return False

    legacy_files_or_dirs = [
        p for p in model_select.iterdir()
        if not p.name.startswith("pass-")
    ]
    if not legacy_files_or_dirs:
        return False

    pass1 = model_select / "pass-001"
    pass1.mkdir(exist_ok=True)

    moved = False
    for entry in legacy_files_or_dirs:
        target = pass1 / entry.name
        if target.exists():
            raise FileExistsError(
                f"Cannot migrate {entry} → {target}: destination already exists."
            )
        shutil.move(str(entry), str(target))
        logger.info("Migrated %s → %s", entry, target)
        moved = True
    return moved


def _migrate_style_refine(root: Path) -> list[int]:
    style_refine = root / "style-refine"
    if not style_refine.exists():
        return []

    legacy_rounds = sorted(
        d for d in style_refine.iterdir()
        if d.is_dir() and d.name.startswith("round-")
    )
    if not legacy_rounds:
        return []

    pass1 = style_refine / "pass-001"
    pass1.mkdir(exist_ok=True)

    migrated: list[int] = []
    for round_dir in legacy_rounds:
        target = pass1 / round_dir.name
        if target.exists():
            raise FileExistsError(
                f"Cannot migrate {round_dir} → {target}: destination already exists."
            )
        shutil.move(str(round_dir), str(target))
        logger.info("Migrated %s → %s", round_dir, target)
        try:
            num = int(round_dir.name.split("-", 1)[1])
            migrated.append(num)
        except (ValueError, IndexError):
            migrated.append(-1)
    return migrated


def migrate_project(name: str) -> MigrationResult:
    root = project_store.project_dir(name)
    if not root.exists():
        raise FileNotFoundError(f"Project '{name}' does not exist at {root}")

    result = MigrationResult()
    result.model_select_migrated = _migrate_model_select(root)
    result.style_refine_rounds_migrated = _migrate_style_refine(root)
    return result
