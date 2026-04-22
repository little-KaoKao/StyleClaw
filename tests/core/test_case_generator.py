from styleclaw.core.case_generator import (
    CASES_PER_CATEGORY,
    CATEGORIES,
    category_labels,
    generate_case_skeleton,
    landscape_categories,
)


class TestCaseGenerator:
    def test_total_cases(self):
        cases = generate_case_skeleton()
        assert len(cases) == len(CATEGORIES) * CASES_PER_CATEGORY
        assert len(cases) == 100

    def test_ten_categories(self):
        assert len(CATEGORIES) == 10

    def test_each_category_has_ten_cases(self):
        cases = generate_case_skeleton()
        counts: dict[str, int] = {}
        for c in cases:
            counts[c.category] = counts.get(c.category, 0) + 1
        for cat in CATEGORIES:
            assert counts[cat["id"]] == 10

    def test_case_ids_unique(self):
        cases = generate_case_skeleton()
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))

    def test_landscape_categories_use_16_9(self):
        cases = generate_case_skeleton()
        ls = landscape_categories()
        for c in cases:
            if c.category in ls:
                assert c.aspect_ratio == "16:9"
            else:
                assert c.aspect_ratio == "9:16"

    def test_category_labels_returns_all(self):
        labels = category_labels()
        assert len(labels) == 10
        assert "adult_male" in labels
        assert "outdoor_scene" in labels

    def test_landscape_categories_correct(self):
        ls = landscape_categories()
        assert ls == {"outdoor_scene", "indoor_scene", "group"}

    def test_descriptions_empty_in_skeleton(self):
        cases = generate_case_skeleton()
        for c in cases:
            assert c.description == ""

    def test_all_cases_start_pending(self):
        cases = generate_case_skeleton()
        for c in cases:
            assert c.status == "pending"
