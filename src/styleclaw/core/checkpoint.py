from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Checkpoint:
    """Tiny key/value store backed by a JSON file under the project directory.

    Used by long-running batch operations (e.g. batch-t2i submit) so that an
    interrupted run can resume by skipping items already recorded.
    """

    def __init__(self, project_dir: Path, phase: str) -> None:
        self._path = project_dir / f".checkpoint_{phase}.json"

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read checkpoint %s: %s", self._path, exc)
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def all(self) -> dict[str, Any]:
        return self._load()

    def save(self, key: str, value: Any) -> None:
        data = self._load()
        data[key] = value
        self._write(data)

    def update(self, items: dict[str, Any]) -> None:
        data = self._load()
        data.update(items)
        self._write(data)

    def clear(self) -> None:
        if self._path.exists():
            try:
                self._path.unlink()
            except OSError as exc:
                logger.warning("Could not delete checkpoint %s: %s", self._path, exc)

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp.replace(self._path)
        except OSError as exc:
            logger.warning("Failed to write checkpoint %s: %s", self._path, exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
