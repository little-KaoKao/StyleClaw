from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from styleclaw.core.case_generator import CASES_PER_CATEGORY, CATEGORIES, generate_case_skeleton
from styleclaw.core.config import DESIGN_CASES_SHARDS
from styleclaw.core.models import BatchCase, BatchConfig
from styleclaw.core.text_utils import clean_json, recover_truncated_json, sanitize_braces
from styleclaw.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)
DESIGN_CASES_SHARD_RETRIES = 1

_AGE_CONTRACTS = {
    "adult_male": (20, 30),
    "adult_female": (20, 30),
    "little_male_child": (8, 14),
    "little_female_child": (8, 14),
    "elderly_male": (50, None),
    "elderly_female": (50, None),
}
_SINGLE_PERSON_CATEGORIES = set(_AGE_CONTRACTS)

_FULL_BODY_FRAMING_CUES = {
    1: "Full-body shot",
    2: "Full-length shot",
    3: "Head-to-toe shot",
    4: "Wide full-body shot",
    5: "Full-body portrait",
}
_MEDIUM_OR_CLOSE_FRAMING_CUES = {
    6: "Medium full shot",
    7: "Medium shot",
    8: "Waist-up shot",
    9: "Medium close-up",
    10: "Close-up portrait",
}

_FULL_BODY_FRAMING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bfull[- ]body\b",
        r"\bfull[- ]length\b",
        r"\bhead[- ]to[- ]toe\b",
        r"\bwhole[- ]body\b",
        r"\bentire body\b",
        r"\bwide full[- ]body\b",
        r"\bwide shot\b",
    )
]
_MEDIUM_OR_CLOSE_FRAMING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bmedium long shot\b",
        r"\bmedium full shot\b",
        r"\bmedium shot\b",
        r"\bmedium close[- ]up\b",
        r"\bmid[- ]shot\b",
        r"\bwaist[- ]up\b",
        r"\bknee[- ]up\b",
        r"\bupper[- ]body\b",
        r"\bchest[- ]up\b",
        r"\bshoulders[- ]up\b",
        r"\bbust shot\b",
        r"\bclose[- ]up\b",
        r"\bportrait crop\b",
        r"\bclose[- ]up portrait\b",
    )
]

_EXACT_AGE_PATTERNS = [
    re.compile(r"\bage(?:d)?\s+(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*[- ]year[- ]old\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s+years old\b", re.IGNORECASE),
]

_DECADE_AGE_PATTERNS = [
    re.compile(
        r"\b(?:in\s+)?(?:his|her|their)\s+(?:early\s+|mid\s+|late\s+)?(\d{2})s\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:early|mid|late)\s+(\d{2})s\b", re.IGNORECASE),
    re.compile(r"\b(\d{2})s\b", re.IGNORECASE),
]

_WORD_DECADES = {
    "twenties": 20,
    "thirties": 30,
    "forties": 40,
    "fifties": 50,
    "sixties": 60,
    "seventies": 70,
    "eighties": 80,
    "nineties": 90,
}
_WORD_DECADE_PATTERN = re.compile(
    r"\b(?:in\s+)?(?:(?:his|her|their)\s+)?"
    r"(?:early\s+|mid\s+|late\s+)?"
    r"(twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b",
    re.IGNORECASE,
)

_AGE_TERM_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "adult_male": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bmiddle[- ]aged\b",
            r"\belderly\b",
            r"\bsenior\b",
            r"\bsilver[- ](?:haired|streaked)\b",
            r"\bgr[ae]y[- ]haired\b",
            r"\bsalt[- ]and[- ]pepper\b",
        )
    ],
    "adult_female": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bmiddle[- ]aged\b",
            r"\belderly\b",
            r"\bsenior\b",
            r"\bsilver[- ](?:haired|streaked)\b",
            r"\bgr[ae]y[- ]haired\b",
            r"\bsalt[- ]and[- ]pepper\b",
        )
    ],
    "little_male_child": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bbaby\b",
            r"\binfant\b",
            r"\btoddler\b",
            r"\bpreschool(?:er)?\b",
            r"\bkindergarten[- ]age\b",
            r"\bkindergartener\b",
        )
    ],
    "little_female_child": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bbaby\b",
            r"\binfant\b",
            r"\btoddler\b",
            r"\bpreschool(?:er)?\b",
            r"\bkindergarten[- ]age\b",
            r"\bkindergartener\b",
        )
    ],
}

PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "providers" / "llm" / "prompts" / "design_cases_shard.md"
)


async def design_cases(
    llm: LLMProvider,
    ip_info: str,
    trigger_phrase: str,
    batch_num: int,
    feedback: str = "",
) -> BatchConfig:
    shards = DESIGN_CASES_SHARDS
    cat_ids = [c["id"] for c in CATEGORIES]
    cats_per_shard = len(CATEGORIES) // shards
    partitions: list[list[str]] = [
        cat_ids[i : i + cats_per_shard]
        for i in range(0, len(cat_ids), cats_per_shard)
    ]

    feedback_section = _build_feedback_section(feedback)
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    shard_specs = [
        dict(
            llm=llm,
            template=template,
            ip_info=ip_info,
            trigger_phrase=trigger_phrase,
            feedback_section=feedback_section,
            shard_index=i + 1,
            total_shards=shards,
            shard_categories=partition,
        )
        for i, partition in enumerate(partitions)
    ]
    shard_results = await _design_shards_with_retry(shard_specs)

    merged: list[BatchCase] = []
    for cases in shard_results:
        merged.extend(cases)

    expected_total = len(CATEGORIES) * CASES_PER_CATEGORY
    if len(merged) != expected_total:
        raise ValueError(
            f"design_cases expected {expected_total} total cases across "
            f"{shards} shards, got {len(merged)}"
        )

    logger.info("Designed %d test cases for batch %d (%d shards).",
                len(merged), batch_num, shards)
    return BatchConfig(
        batch=batch_num,
        trigger_phrase=trigger_phrase,
        cases=merged,
    )


async def _design_shards_with_retry(
    shard_specs: list[dict[str, Any]],
) -> list[list[BatchCase]]:
    """Run shards concurrently, then retry transient shard failures in place.

    The provider already retries transport/5xx/429 errors internally. This
    extra layer handles the common batch-design failure mode where one parallel
    shard exhausts provider retries while the other shards succeed. We retry
    only that failed shard once instead of discarding all successful shard work.
    Validation errors are not retried because the prompt/response contract is
    already broken and retrying can hide deterministic bad output.
    """
    pending: list[tuple[int, dict[str, Any]]] = list(enumerate(shard_specs))
    results: dict[int, list[BatchCase]] = {}
    retries_used = 0

    while pending:
        outcomes = await asyncio.gather(
            *[_design_one_shard(**spec) for _, spec in pending],
            return_exceptions=True,
        )
        retry_pending: list[tuple[int, dict[str, Any]]] = []
        for (idx, spec), outcome in zip(pending, outcomes):
            if isinstance(outcome, BaseException):
                if isinstance(outcome, RuntimeError) and retries_used < DESIGN_CASES_SHARD_RETRIES:
                    logger.warning(
                        "design_cases shard %d/%d failed after provider retries; "
                        "retrying failed shard once: %s",
                        spec["shard_index"], spec["total_shards"], outcome,
                    )
                    retry_pending.append((idx, spec))
                    continue
                raise outcome
            results[idx] = outcome
        pending = retry_pending
        if pending:
            retries_used += 1

    return [results[i] for i in range(len(shard_specs))]


async def _design_one_shard(
    *,
    llm: LLMProvider,
    template: str,
    ip_info: str,
    trigger_phrase: str,
    feedback_section: str,
    shard_index: int,
    total_shards: int,
    shard_categories: list[str],
) -> list[BatchCase]:
    shard_cases = len(shard_categories) * CASES_PER_CATEGORY
    skeleton = [
        case
        for case in generate_case_skeleton()
        if case.category in shard_categories
    ]
    skeleton_text = _format_skeleton(skeleton)

    system_prompt = (
        template
        .replace("{ip_info}", sanitize_braces(ip_info))
        .replace("{trigger_phrase}", trigger_phrase)
        .replace("{case_skeleton}", skeleton_text)
        .replace("{feedback_section}", feedback_section)
        .replace("{shard_index}", str(shard_index))
        .replace("{total_shards}", str(total_shards))
        .replace("{shard_category_count}", str(len(shard_categories)))
        .replace("{shard_cases}", str(shard_cases))
    )

    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"Design {shard_cases} diverse test cases for this shard."},
    ]}]

    raw = await llm.invoke(system=system_prompt, messages=messages, max_tokens=4096)

    cleaned = clean_json(raw)
    recovered = recover_truncated_json(cleaned)
    data = json.loads(recovered)
    if "cases" not in data:
        raise ValueError(f"shard {shard_index} LLM response missing 'cases' key")
    cases = [BatchCase.model_validate(c) for c in data["cases"]]
    if not cases:
        raise ValueError(
            f"shard {shard_index} returned zero cases — response may have been truncated."
        )

    allowed = set(shard_categories)
    stray = [c for c in cases if c.category not in allowed]
    if stray:
        raise ValueError(
            f"shard {shard_index} returned cases outside its assigned categories "
            f"{sorted(allowed)}: got {sorted({c.category for c in stray})}"
        )

    if len(cases) != shard_cases:
        raise ValueError(
            f"shard {shard_index} returned {len(cases)} cases for categories "
            f"{sorted(allowed)}, expected {shard_cases}"
        )

    _validate_age_contracts(cases, shard_index)
    cases = _apply_framing_contracts(cases)
    _validate_framing_contracts(cases, shard_index)

    return cases


def _validate_age_contracts(cases: list[BatchCase], shard_index: int) -> None:
    violations: list[str] = []
    for case in cases:
        contract = _AGE_CONTRACTS.get(case.category)
        if contract is None:
            continue
        min_age, max_age = contract
        desc = case.description

        for pattern in _AGE_TERM_PATTERNS.get(case.category, []):
            match = pattern.search(desc)
            if match:
                violations.append(
                    f"{case.id} uses '{match.group(0)}' outside "
                    f"{_format_age_contract(min_age, max_age)}"
                )

        for label, low, high in _iter_age_mentions(desc):
            upper_ok = max_age is None or high <= max_age
            if low < min_age or not upper_ok:
                violations.append(
                    f"{case.id} uses '{label}' outside "
                    f"{_format_age_contract(min_age, max_age)}"
                )

    if violations:
        shown = "; ".join(violations[:5])
        more = f" (+{len(violations) - 5} more)" if len(violations) > 5 else ""
        raise ValueError(f"shard {shard_index} age contract violation: {shown}{more}")


def _iter_age_mentions(text: str) -> list[tuple[str, int, int]]:
    mentions: list[tuple[str, int, int]] = []
    seen: set[tuple[int, int, str]] = set()

    for pattern in _EXACT_AGE_PATTERNS:
        for match in pattern.finditer(text):
            age = int(match.group(1))
            key = (match.start(), match.end(), match.group(0).lower())
            if key not in seen:
                mentions.append((match.group(0), age, age))
                seen.add(key)

    for pattern in _DECADE_AGE_PATTERNS:
        for match in pattern.finditer(text):
            decade = int(match.group(1))
            key = (match.start(), match.end(), match.group(0).lower())
            if key not in seen:
                mentions.append((match.group(0), decade, decade + 9))
                seen.add(key)

    for match in _WORD_DECADE_PATTERN.finditer(text):
        decade = _WORD_DECADES[match.group(1).lower()]
        key = (match.start(), match.end(), match.group(0).lower())
        if key not in seen:
            mentions.append((match.group(0), decade, decade + 9))
            seen.add(key)

    return mentions


def _format_age_contract(min_age: int, max_age: int | None) -> str:
    if max_age is None:
        return f"age {min_age}+"
    return f"age {min_age}-{max_age}"


def _apply_framing_contracts(cases: list[BatchCase]) -> list[BatchCase]:
    framed: list[BatchCase] = []
    for case in cases:
        cue = _expected_framing_cue(case)
        if cue is None or _has_expected_framing(case):
            framed.append(case)
            continue
        framed.append(
            case.model_copy(update={"description": f"{cue}, {case.description}"})
        )
    return framed


def _validate_framing_contracts(cases: list[BatchCase], shard_index: int) -> None:
    violations: list[str] = []
    per_category: dict[str, list[BatchCase]] = {}
    for case in cases:
        if case.category in _SINGLE_PERSON_CATEGORIES:
            per_category.setdefault(case.category, []).append(case)

    for category, category_cases in per_category.items():
        full_count = sum(
            1 for case in category_cases
            if _has_full_body_framing(case.description)
        )
        medium_or_close_count = sum(
            1 for case in category_cases
            if _has_medium_or_close_framing(case.description)
        )
        if full_count < 5:
            violations.append(
                f"{category} has {full_count}/5 required full-body cases"
            )
        if medium_or_close_count < 5:
            violations.append(
                f"{category} has {medium_or_close_count}/5 required medium-or-close cases"
            )

        for case in category_cases:
            if not _has_expected_framing(case):
                cue = _expected_framing_cue(case) or "expected framing"
                violations.append(f"{case.id} missing {cue}")

    if violations:
        shown = "; ".join(violations[:5])
        more = f" (+{len(violations) - 5} more)" if len(violations) > 5 else ""
        raise ValueError(
            f"shard {shard_index} framing contract violation: {shown}{more}"
        )


def _expected_framing_cue(case: BatchCase) -> str | None:
    if case.category not in _SINGLE_PERSON_CATEGORIES:
        return None
    idx = _case_index(case.id)
    if idx is None:
        return None
    if idx in _FULL_BODY_FRAMING_CUES:
        return _FULL_BODY_FRAMING_CUES[idx]
    return _MEDIUM_OR_CLOSE_FRAMING_CUES.get(idx)


def _has_expected_framing(case: BatchCase) -> bool:
    idx = _case_index(case.id)
    if idx is None or case.category not in _SINGLE_PERSON_CATEGORIES:
        return True
    if idx in _FULL_BODY_FRAMING_CUES:
        return _has_full_body_framing(case.description)
    if idx in _MEDIUM_OR_CLOSE_FRAMING_CUES:
        return _has_medium_or_close_framing(case.description)
    return True


def _has_full_body_framing(text: str) -> bool:
    return any(pattern.search(text) for pattern in _FULL_BODY_FRAMING_PATTERNS)


def _has_medium_or_close_framing(text: str) -> bool:
    return any(pattern.search(text) for pattern in _MEDIUM_OR_CLOSE_FRAMING_PATTERNS)


def _case_index(case_id: str) -> int | None:
    match = re.search(r"-(\d{2})$", case_id)
    if not match:
        return None
    return int(match.group(1))


def _build_feedback_section(feedback: str) -> str:
    if not feedback.strip():
        return ""
    return (
        f"\n\n## User feedback on previous batch\n\n{sanitize_braces(feedback)}\n\n"
        "Apply this feedback when designing the new batch — adjust subjects, "
        "scenes, or angles accordingly while keeping the generalization rule."
    )


def _format_skeleton(cases: list[BatchCase]) -> str:
    lines: list[str] = []
    current_cat = ""
    for c in cases:
        if c.category != current_cat:
            current_cat = c.category
            lines.append(f"\n### {current_cat} (aspect: {c.aspect_ratio})")
        lines.append(f"- {c.id}: (fill in description)")
    return "\n".join(lines)
