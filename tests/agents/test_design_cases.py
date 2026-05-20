from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from styleclaw.agents.design_cases import _format_skeleton, design_cases
from styleclaw.core.case_generator import CATEGORIES, generate_case_skeleton


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    cases = [
        {"id": f"am-{i:03d}", "category": "adult_male", "description": f"Male char {i}", "aspect_ratio": "9:16"}
        for i in range(1, 11)
    ]
    llm.invoke.return_value = json.dumps({"cases": cases})
    return llm


class TestDesignCases:
    async def test_returns_batch_config(self, mock_llm) -> None:
        result = await design_cases(mock_llm, "anime", "bold style", batch_num=1)
        assert result.batch == 1
        assert result.trigger_phrase == "bold style"
        assert len(result.cases) == 10

    async def test_ip_info_in_system_prompt(self, mock_llm) -> None:
        await design_cases(mock_llm, "Spider-Verse", "trigger", batch_num=1)
        call_args = mock_llm.invoke.call_args
        assert "Spider-Verse" in call_args.kwargs["system"]

    async def test_no_feedback_section_when_empty(self, mock_llm) -> None:
        await design_cases(mock_llm, "anime", "x", batch_num=1)
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        # Placeholder must not leak through and there must be no feedback heading
        assert "{feedback_section}" not in sys_prompt
        assert "User feedback on previous batch" not in sys_prompt

    async def test_feedback_appended_to_prompt(self, mock_llm) -> None:
        feedback = "上一批室内场景太少，多加一些咖啡馆和书房"
        await design_cases(mock_llm, "anime", "x", batch_num=2, feedback=feedback)
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        assert "User feedback on previous batch" in sys_prompt
        assert feedback in sys_prompt

    async def test_whitespace_only_feedback_treated_as_empty(self, mock_llm) -> None:
        await design_cases(mock_llm, "anime", "x", batch_num=2, feedback="   \n  ")
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        assert "User feedback on previous batch" not in sys_prompt


class TestFormatSkeleton:
    def test_formats_categories(self) -> None:
        skeleton = generate_case_skeleton()
        text = _format_skeleton(skeleton)
        assert "adult_male" in text
        assert "adult_female" in text
        assert "creature" in text
        assert "(fill in description)" in text


def _shard_response(category_ids: list[str], cases_per_category: int = 10) -> str:
    """Build a JSON response covering exactly the given categories."""
    cases = []
    for cat_id in category_ids:
        aspect = next(c["aspect"] for c in CATEGORIES if c["id"] == cat_id)
        for i in range(1, cases_per_category + 1):
            cases.append({
                "id": f"case-{cat_id}-{i:02d}",
                "category": cat_id,
                "description": f"placeholder description for {cat_id} #{i:02d}",
                "aspect_ratio": aspect,
            })
    return json.dumps({"cases": cases})


class TestShardedDesignCases:
    """Default config (SHARDS=5) — verify fan-out, partitioning, merging."""

    @pytest.fixture
    def mock_llm_5shards(self) -> AsyncMock:
        """Returns 5 distinct shard responses, one per call."""
        llm = AsyncMock()
        # Partition CATEGORIES into 5 contiguous slices of 2.
        cat_ids = [c["id"] for c in CATEGORIES]
        partitions = [cat_ids[i:i+2] for i in range(0, 10, 2)]
        llm.invoke.side_effect = [_shard_response(p) for p in partitions]
        return llm

    async def test_default_shards_makes_5_calls(self, mock_llm_5shards) -> None:
        await design_cases(mock_llm_5shards, "anime", "trigger", batch_num=1)
        assert mock_llm_5shards.invoke.call_count == 5

    async def test_combines_to_100_cases(self, mock_llm_5shards) -> None:
        result = await design_cases(mock_llm_5shards, "anime", "trigger", batch_num=1)
        assert len(result.cases) == 100
        # Every category present with exactly 10 cases.
        from collections import Counter
        per_cat = Counter(c.category for c in result.cases)
        assert all(per_cat[c["id"]] == 10 for c in CATEGORIES), per_cat

    async def test_each_shard_prompt_has_only_its_categories(
        self, mock_llm_5shards
    ) -> None:
        await design_cases(mock_llm_5shards, "anime", "trigger", batch_num=1)
        cat_ids = [c["id"] for c in CATEGORIES]
        expected_partitions = [cat_ids[i:i+2] for i in range(0, 10, 2)]
        for call, expected_cats in zip(
            mock_llm_5shards.invoke.call_args_list, expected_partitions
        ):
            system = call.kwargs["system"]
            # The two assigned categories appear in the prompt skeleton.
            for cat in expected_cats:
                assert cat in system, f"expected {cat} in shard system prompt"
            # No other category appears as an "### <name>" header.
            others = [c for c in cat_ids if c not in expected_cats]
            for other in others:
                assert f"### {other} " not in system, (
                    f"unexpected category {other} in shard prompt for {expected_cats}"
                )

    async def test_each_shard_prompt_has_worker_marker(
        self, mock_llm_5shards
    ) -> None:
        await design_cases(mock_llm_5shards, "anime", "trigger", batch_num=1)
        markers = ["worker 1/5", "worker 2/5", "worker 3/5", "worker 4/5", "worker 5/5"]
        seen = {m: False for m in markers}
        for call in mock_llm_5shards.invoke.call_args_list:
            system = call.kwargs["system"]
            for m in markers:
                if m in system:
                    seen[m] = True
        assert all(seen.values()), f"missing worker markers: {seen}"

    async def test_single_shard_failure_propagates(self) -> None:
        llm = AsyncMock()
        cat_ids = [c["id"] for c in CATEGORIES]
        partitions = [cat_ids[i:i+2] for i in range(0, 10, 2)]
        # Third shard raises.
        side = [_shard_response(partitions[0]), _shard_response(partitions[1])]
        side.append(RuntimeError("provider exhausted retries"))
        side.extend([_shard_response(partitions[3]), _shard_response(partitions[4])])
        llm.invoke.side_effect = side
        with pytest.raises(RuntimeError, match="provider exhausted retries"):
            await design_cases(llm, "anime", "trigger", batch_num=1)

    async def test_wrong_category_in_shard_rejected(self) -> None:
        llm = AsyncMock()
        cat_ids = [c["id"] for c in CATEGORIES]
        partitions = [cat_ids[i:i+2] for i in range(0, 10, 2)]
        # First shard returns cases from a category it doesn't own (e.g. group).
        bad_first = _shard_response(["adult_male", "group"])
        side = [bad_first] + [_shard_response(p) for p in partitions[1:]]
        llm.invoke.side_effect = side
        with pytest.raises(ValueError, match="outside its assigned categories"):
            await design_cases(llm, "anime", "trigger", batch_num=1)
