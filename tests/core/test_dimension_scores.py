import pytest

from styleclaw.core.models import (
    DimensionScores,
    RoundEvaluation,
    RoundScore,
)


class TestDimensionScores:
    def test_average(self):
        s = DimensionScores(color_palette=8, line_style=7, lighting=9, texture=6, overall_mood=10)
        assert s.average() == 8.0

    def test_min_score(self):
        s = DimensionScores(color_palette=8, line_style=3, lighting=9, texture=6, overall_mood=10)
        assert s.min_score() == 3.0

    def test_all_above_true(self):
        s = DimensionScores(color_palette=8, line_style=7, lighting=9, texture=7, overall_mood=8)
        assert s.all_above(7.0) is True

    def test_all_above_false(self):
        s = DimensionScores(color_palette=8, line_style=6.9, lighting=9, texture=7, overall_mood=8)
        assert s.all_above(7.0) is False

    def test_defaults_are_zero(self):
        s = DimensionScores()
        assert s.average() == 0.0
        assert s.min_score() == 0.0


class TestRoundEvaluation:
    def _make_score(self, cp, ls, lt, tx, om, total):
        return RoundScore(
            model="test",
            scores=DimensionScores(
                color_palette=cp, line_style=ls, lighting=lt,
                texture=tx, overall_mood=om,
            ),
            total=total,
        )

    def test_should_approve_all_high(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 7, 8, 7, 8, 7.6),
        ])
        assert ev.should_approve() is True

    def test_should_approve_fails_low_dimension(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 6, 8, 7, 8, 7.4),
        ])
        assert ev.should_approve() is False

    def test_should_approve_fails_low_total(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(7, 7, 7, 7, 7, 7.0),
        ])
        assert ev.should_approve() is False

    def test_should_approve_empty(self):
        ev = RoundEvaluation(round=1)
        assert ev.should_approve() is False

    def test_needs_human_with_low_score(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 4, 8, 7, 8, 7.0),
        ])
        assert ev.needs_human() is True

    def test_needs_human_false(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 5, 8, 7, 8, 7.2),
        ])
        assert ev.needs_human() is False

    def test_multiple_models_mixed(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 8, 8, 8, 8, 8.0),
            self._make_score(7, 7, 7, 7, 7, 7.0),
        ])
        assert ev.should_approve() is False

    def test_multiple_models_all_pass(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 8, 8, 8, 8, 8.0),
            self._make_score(7.5, 7.5, 7.5, 7.5, 7.5, 7.5),
        ])
        assert ev.should_approve() is True
