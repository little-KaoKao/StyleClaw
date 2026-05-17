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


class TestCheckpointFlushThreshold:
    """``flush_threshold>1`` amortizes disk writes — verify the disk only
    sees a write after every N adds, and that explicit ``flush()`` drains
    the tail of an incomplete batch."""

    def test_disk_skipped_until_threshold_hit(self, tmp_path) -> None:
        cp = Checkpoint(tmp_path, "phase", flush_threshold=5)
        # 4 adds — disk should NOT have the entries yet.
        for i in range(4):
            cp.add_to_set("submitted", f"c-{i:03d}")
        assert not (tmp_path / ".checkpoint_phase.json").exists()
        # 5th add trips the threshold.
        cp.add_to_set("submitted", "c-004")
        data = json.loads((tmp_path / ".checkpoint_phase.json").read_text())
        assert sorted(data["submitted"]) == [f"c-{i:03d}" for i in range(5)]

    def test_explicit_flush_drains_pending(self, tmp_path) -> None:
        cp = Checkpoint(tmp_path, "phase", flush_threshold=100)
        for i in range(3):
            cp.add_to_set("submitted", f"c-{i:03d}")
        # Way below threshold, nothing on disk.
        assert not (tmp_path / ".checkpoint_phase.json").exists()
        cp.flush()
        data = json.loads((tmp_path / ".checkpoint_phase.json").read_text())
        assert sorted(data["submitted"]) == [f"c-{i:03d}" for i in range(3)]

    def test_default_threshold_is_synchronous(self, tmp_path) -> None:
        # Backward compat: no threshold arg means flush-every-call, same as
        # before this change.
        cp = Checkpoint(tmp_path, "phase")
        cp.add_to_set("submitted", "c-001")
        data = json.loads((tmp_path / ".checkpoint_phase.json").read_text())
        assert data["submitted"] == ["c-001"]

    def test_clear_resets_pending_counter(self, tmp_path) -> None:
        # If a partial batch is cleared and then we add new items, the
        # accumulated pending counter should not erroneously trip a flush
        # too early or too late.
        cp = Checkpoint(tmp_path, "phase", flush_threshold=5)
        for i in range(3):
            cp.add_to_set("submitted", f"c-{i:03d}")
        cp.clear()
        # Add 4 more — still below threshold, no disk write.
        for i in range(4):
            cp.add_to_set("submitted", f"d-{i:03d}")
        assert not (tmp_path / ".checkpoint_phase.json").exists()

    def test_invalid_threshold_rejected(self, tmp_path) -> None:
        import pytest
        with pytest.raises(ValueError, match=">= 1"):
            Checkpoint(tmp_path, "phase", flush_threshold=0)
