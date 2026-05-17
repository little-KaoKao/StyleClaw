from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class Checkpoint:
    """Tiny key/value store backed by a JSON file under the project directory.

    Used by long-running batch operations (e.g. batch-t2i submit) so that an
    interrupted run can resume by skipping items already recorded.

    Thread-safe: each instance holds an in-memory copy of the file plus a
    lock, so concurrent writes don't lose updates. Each Checkpoint instance
    owns the on-disk file; do not share the file across instances.

    Amortized flush: ``flush_threshold`` controls how many mutations
    accumulate before a disk write. Default is 1 (flush on every call —
    safest, costs one tmp+replace per call). Higher values batch writes;
    callers should pair with an explicit ``flush()`` at a safe point (e.g.
    inside a ``try/finally`` around a ``gather()``) so an interrupt doesn't
    leave the last batch on the floor.
    """

    def __init__(self, project_dir: Path, phase: str, flush_threshold: int = 1) -> None:
        if flush_threshold < 1:
            raise ValueError(f"flush_threshold must be >= 1, got {flush_threshold}")
        self._path = project_dir / f".checkpoint_{phase}.json"
        self._lock = Lock()
        self._data: dict[str, Any] = self._read_disk()
        self._flush_threshold = flush_threshold
        self._pending_since_flush = 0

    @property
    def path(self) -> Path:
        return self._path

    def _read_disk(self) -> dict[str, Any]:
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
        with self._lock:
            value = self._data.get(key, default)
            # Return a copy of mutable collections so callers can't mutate
            # the cached state without going through save/add_to_set.
            if isinstance(value, list):
                return list(value)
            if isinstance(value, dict):
                return dict(value)
            return value

    def all(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def save(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._maybe_flush_locked()

    def update(self, items: dict[str, Any]) -> None:
        with self._lock:
            self._data.update(items)
            self._maybe_flush_locked()

    def add_to_set(self, key: str, item: Any) -> None:
        """Atomically add `item` to the set stored under `key`.

        The on-disk representation is a sorted list (JSON has no set type).
        Concurrent calls with different items never lose entries because the
        read-modify-write happens under the lock.
        """
        with self._lock:
            current = self._data.get(key, [])
            merged = set(current) if isinstance(current, list) else set()
            merged.add(item)
            self._data[key] = sorted(merged)
            self._maybe_flush_locked()

    def flush(self) -> None:
        """Force a flush regardless of threshold. Pair with an amortizing
        ``flush_threshold > 1`` so the tail of a batch always lands on disk.
        Use inside ``try/finally`` to survive Ctrl-C without losing the
        last few entries (they'd otherwise re-submit on resume)."""
        with self._lock:
            self._flush_locked()

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            self._pending_since_flush = 0
            if self._path.exists():
                try:
                    self._path.unlink()
                except OSError as exc:
                    logger.warning("Could not delete checkpoint %s: %s", self._path, exc)

    def _maybe_flush_locked(self) -> None:
        self._pending_since_flush += 1
        if self._pending_since_flush >= self._flush_threshold:
            self._flush_locked()

    def _flush_locked(self) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            tmp.replace(self._path)
            self._pending_since_flush = 0
        except OSError as exc:
            logger.warning("Failed to write checkpoint %s: %s", self._path, exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
