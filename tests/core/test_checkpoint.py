from __future__ import annotations

import asyncio
import json

from styleclaw.core.checkpoint import Checkpoint


class TestCheckpoint:
    def test_save_and_get(self, tmp_path) -> None:
        cp = Checkpoint(tmp_path, "phase")
        cp.save("k", "v")
        assert cp.get("k") == "v"

    def test_get_default(self, tmp_path) -> None:
        cp = Checkpoint(tmp_path, "phase")
        assert cp.get("missing", "fallback") == "fallback"

    def test_clear(self, tmp_path) -> None:
        cp = Checkpoint(tmp_path, "phase")
        cp.save("k", "v")
        cp.clear()
        assert cp.get("k") is None

    def test_atomic_write_under_concurrent_add_to_set(self, tmp_path) -> None:
        """Concurrent additions to the same set-valued key must not lose items.

        Regression test for a lost-update race in batch_submit_t2i: two
        coroutines simultaneously read the same `submitted` list, each appended
        a different case id, and the second write clobbered the first.
        """
        cp = Checkpoint(tmp_path, "phase")
        ids = [f"case-{i:03d}" for i in range(50)]

        async def add_all() -> None:
            await asyncio.gather(*(asyncio.to_thread(cp.add_to_set, "submitted", i) for i in ids))

        asyncio.run(add_all())
        stored = set(cp.get("submitted", []))
        assert stored == set(ids)

    def test_add_to_set_persists_unique_sorted(self, tmp_path) -> None:
        cp = Checkpoint(tmp_path, "phase")
        cp.add_to_set("submitted", "b")
        cp.add_to_set("submitted", "a")
        cp.add_to_set("submitted", "b")
        stored = cp.get("submitted", [])
        assert stored == ["a", "b"]

    def test_add_to_set_survives_reopen(self, tmp_path) -> None:
        cp1 = Checkpoint(tmp_path, "phase")
        cp1.add_to_set("submitted", "x")
        cp2 = Checkpoint(tmp_path, "phase")
        assert cp2.get("submitted", []) == ["x"]

    def test_save_writes_to_disk(self, tmp_path) -> None:
        cp = Checkpoint(tmp_path, "phase")
        cp.save("k", {"n": 1})
        data = json.loads((tmp_path / ".checkpoint_phase.json").read_text())
        assert data == {"k": {"n": 1}}
