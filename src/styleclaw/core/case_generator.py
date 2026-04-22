from __future__ import annotations

from styleclaw.core.models import BatchCase

CATEGORIES: list[dict[str, str]] = [
    {"id": "adult_male", "label": "Adult Male", "aspect": "9:16"},
    {"id": "adult_female", "label": "Adult Female", "aspect": "9:16"},
    {"id": "shota", "label": "Shota (Young Boy)", "aspect": "9:16"},
    {"id": "loli", "label": "Loli (Young Girl)", "aspect": "9:16"},
    {"id": "elderly_male", "label": "Elderly Male", "aspect": "9:16"},
    {"id": "elderly_female", "label": "Elderly Female", "aspect": "9:16"},
    {"id": "creature", "label": "Creature / Mascot", "aspect": "9:16"},
    {"id": "outdoor_scene", "label": "Outdoor Scene", "aspect": "16:9"},
    {"id": "indoor_scene", "label": "Indoor Scene", "aspect": "16:9"},
    {"id": "group", "label": "Group / Multi-character", "aspect": "16:9"},
]

CASES_PER_CATEGORY = 10


def generate_case_skeleton() -> list[BatchCase]:
    cases: list[BatchCase] = []
    for cat in CATEGORIES:
        for i in range(1, CASES_PER_CATEGORY + 1):
            cases.append(BatchCase(
                id=f"case-{cat['id']}-{i:02d}",
                category=cat["id"],
                description="",
                aspect_ratio=cat["aspect"],
            ))
    return cases


def category_labels() -> dict[str, str]:
    return {c["id"]: c["label"] for c in CATEGORIES}


def landscape_categories() -> set[str]:
    return {c["id"] for c in CATEGORIES if c["aspect"] == "16:9"}
