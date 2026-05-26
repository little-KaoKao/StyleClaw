from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from styleclaw.agents.design_cases import _format_skeleton, design_cases
from styleclaw.core.case_generator import CATEGORIES, generate_case_skeleton


@pytest.fixture(autouse=False)
def force_single_shard(monkeypatch):
    monkeypatch.setattr(
        "styleclaw.agents.design_cases.DESIGN_CASES_SHARDS", 1
    )


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Mock that returns a complete 100-case response covering all 10 categories."""
    cases = []
    for cat in CATEGORIES:
        for i in range(1, 11):
            cases.append({
                "id": f"case-{cat['id']}-{i:02d}",
                "category": cat["id"],
                "description": f"placeholder {cat['id']} #{i:02d}",
                "aspect_ratio": cat["aspect"],
            })
    llm = AsyncMock()
    llm.invoke.return_value = json.dumps({"cases": cases})
    return llm


class TestDesignCases:
    @pytest.fixture(autouse=True)
    def _force_single_shard(self, force_single_shard):
        return force_single_shard

    async def test_returns_batch_config(self, mock_llm) -> None:
        result = await design_cases(mock_llm, "anime", "bold style", batch_num=1)
        assert result.batch == 1
        assert result.trigger_phrase == "bold style"
        assert len(result.cases) == 100

    async def test_ip_info_in_system_prompt(self, mock_llm) -> None:
        await design_cases(mock_llm, "Spider-Verse", "trigger", batch_num=1)
        call_args = mock_llm.invoke.call_args
        assert "Spider-Verse" in call_args.kwargs["system"]

    async def test_scene_categories_require_no_people(self, mock_llm) -> None:
        await design_cases(mock_llm, "live action", "trigger", batch_num=1)
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        assert "outdoor_scene" in sys_prompt
        assert "indoor_scene" in sys_prompt
        assert "no people, no human figures, no silhouettes, no portraits" in sys_prompt
        assert "commuters" in sys_prompt
        assert "reporters" in sys_prompt

    async def test_character_categories_avoid_default_melodrama(self, mock_llm) -> None:
        await design_cases(mock_llm, "mobile drama", "trigger", batch_num=1)
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        assert "Do NOT infer melodrama from broad genre words" in sys_prompt
        assert "natural, neutral, relaxed, focused, confident, warm" in sys_prompt
        assert "tears, crying, red eyes, worried" in sys_prompt
        assert "divorce, breakup, custody disputes" in sys_prompt

    async def test_character_categories_include_age_contracts(self, mock_llm) -> None:
        await design_cases(mock_llm, "mobile drama", "trigger", batch_num=1)
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
        assert "`adult_male`, `adult_female`: 20-30 years old only" in sys_prompt
        assert "`little_male_child`, `little_female_child`: 8-14 years old only" in sys_prompt
        assert "`elderly_male`, `elderly_female`: 50+ years old" in sys_prompt
        assert "Do not use 30s/40s/50s" in sys_prompt
        assert "Do not use baby, infant, toddler" in sys_prompt

    async def test_explicit_age_contract_violations_are_rejected(self) -> None:
        cases = []
        for cat in CATEGORIES:
            for i in range(1, 11):
                description = f"placeholder {cat['id']} #{i:02d}"
                if cat["id"] == "adult_female" and i == 1:
                    description = (
                        "A woman in her 40s wearing a tailored blazer, "
                        "standing in a bright office"
                    )
                if cat["id"] == "little_male_child" and i == 1:
                    description = "A boy age 6 in a yellow raincoat jumping over a puddle"
                if cat["id"] == "elderly_male" and i == 1:
                    description = (
                        "A man in his forties wearing a cardigan, reading near "
                        "a sunny window"
                    )
                cases.append({
                    "id": f"case-{cat['id']}-{i:02d}",
                    "category": cat["id"],
                    "description": description,
                    "aspect_ratio": cat["aspect"],
                })
        llm = AsyncMock()
        llm.invoke.return_value = json.dumps({"cases": cases})

        with pytest.raises(ValueError, match="age contract violation") as exc_info:
            await design_cases(llm, "mobile drama", "trigger", batch_num=1)
        message = str(exc_info.value)
        assert "case-adult_female-01" in message
        assert "case-little_male_child-01" in message
        assert "case-elderly_male-01" in message

    async def test_no_feedback_section_when_empty(self, mock_llm) -> None:
        await design_cases(mock_llm, "anime", "x", batch_num=1)
        sys_prompt = mock_llm.invoke.call_args.kwargs["system"]
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

    async def test_single_shard_runtime_error_retried_successfully(self) -> None:
        llm = AsyncMock()
        cat_ids = [c["id"] for c in CATEGORIES]
        partitions = [cat_ids[i:i+2] for i in range(0, 10, 2)]
        # Third shard raises once after provider retries, then succeeds on the
        # targeted shard retry.
        side = [_shard_response(partitions[0]), _shard_response(partitions[1])]
        side.append(RuntimeError("provider exhausted retries"))
        side.extend([_shard_response(partitions[3]), _shard_response(partitions[4])])
        side.append(_shard_response(partitions[2]))
        llm.invoke.side_effect = side
        result = await design_cases(llm, "anime", "trigger", batch_num=1)
        assert len(result.cases) == 100
        assert llm.invoke.call_count == 6

    async def test_single_shard_failure_propagates_after_retry(self) -> None:
        llm = AsyncMock()
        cat_ids = [c["id"] for c in CATEGORIES]
        partitions = [cat_ids[i:i+2] for i in range(0, 10, 2)]
        side = [_shard_response(partitions[0]), _shard_response(partitions[1])]
        side.append(RuntimeError("provider exhausted retries"))
        side.extend([_shard_response(partitions[3]), _shard_response(partitions[4])])
        side.append(RuntimeError("provider exhausted retries again"))
        llm.invoke.side_effect = side
        with pytest.raises(RuntimeError, match="provider exhausted retries"):
            await design_cases(llm, "anime", "trigger", batch_num=1)
        assert llm.invoke.call_count == 6

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

    async def test_short_shard_rejected_with_shard_context(self) -> None:
        """A shard returning fewer cases than its assigned count is rejected
        with a message that identifies which shard and what was missing."""
        import json as _json
        llm = AsyncMock()
        cat_ids = [c["id"] for c in CATEGORIES]
        partitions = [cat_ids[i:i+2] for i in range(0, 10, 2)]
        # First shard returns only 5 cases per category (= 10 total) instead of 20.
        short_cases = []
        for cat_id in partitions[0]:
            aspect = next(c["aspect"] for c in CATEGORIES if c["id"] == cat_id)
            for i in range(1, 6):
                short_cases.append({
                    "id": f"case-{cat_id}-{i:02d}",
                    "category": cat_id,
                    "description": f"truncated #{i}",
                    "aspect_ratio": aspect,
                })
        short_first = _json.dumps({"cases": short_cases})
        side = [short_first] + [_shard_response(p) for p in partitions[1:]]
        llm.invoke.side_effect = side
        with pytest.raises(ValueError, match=r"shard 1 returned 10 cases.*expected 20"):
            await design_cases(llm, "anime", "trigger", batch_num=1)
