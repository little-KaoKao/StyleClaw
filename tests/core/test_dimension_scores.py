import pytest

from styleclaw.core.models import (
    DimensionScores,
    RoundEvaluation,
    RoundScore,
)


def _ds(vs=0.0, cs=0.0, lq=0.0, mt=0.0, pp=0.0, sp=0.0, ds=0.0) -> DimensionScores:
    """Compact builder so tests stay readable when only a few dims matter."""
    return DimensionScores(
        visual_style=vs,
        color_science=cs,
        lighting_quality=lq,
        material_texture=mt,
        post_processing=pp,
        spatial_perspective=sp,
        dynamic_state=ds,
    )


class TestDimensionScores:
    def test_average(self):
        # 7+7+7+7+7+7+7 = 49 / 7 = 7.0
        s = _ds(7, 7, 7, 7, 7, 7, 7)
        assert s.average() == 7.0

    def test_min_score(self):
        s = _ds(8, 3, 9, 6, 10, 7, 7)
        assert s.min_score() == 3.0

    def test_all_above_true(self):
        s = _ds(8, 7, 9, 7, 8, 7, 8)
        assert s.all_above(7.0) is True

    def test_all_above_false(self):
        s = _ds(8, 6.9, 9, 7, 8, 7, 8)
        assert s.all_above(7.0) is False

    def test_defaults_are_zero(self):
        s = DimensionScores()
        assert s.average() == 0.0
        assert s.min_score() == 0.0


class TestRoundEvaluation:
    def _make_score(self, vs, cs, lq, mt, pp, sp, ds, total):
        return RoundScore(
            model="test",
            scores=_ds(vs, cs, lq, mt, pp, sp, ds),
            total=total,
        )

    def test_should_approve_all_high(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 7, 8, 7, 8, 7, 8, 7.6),
        ])
        assert ev.should_approve() is True

    def test_should_approve_fails_low_dimension(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 6, 8, 7, 8, 7, 8, 7.4),
        ])
        assert ev.should_approve() is False

    def test_should_approve_fails_low_total(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(7, 7, 7, 7, 7, 7, 7, 7.0),
        ])
        assert ev.should_approve() is False

    def test_should_approve_empty(self):
        ev = RoundEvaluation(round=1)
        assert ev.should_approve() is False

    def test_needs_human_with_low_score(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 4, 8, 7, 8, 7, 8, 7.0),
        ])
        assert ev.needs_human() is True

    def test_needs_human_false(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 5, 8, 7, 8, 7, 8, 7.2),
        ])
        assert ev.needs_human() is False

    def test_multiple_models_mixed(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 8, 8, 8, 8, 8, 8, 8.0),
            self._make_score(7, 7, 7, 7, 7, 7, 7, 7.0),
        ])
        assert ev.should_approve() is False

    def test_multiple_models_all_pass(self):
        ev = RoundEvaluation(round=1, evaluations=[
            self._make_score(8, 8, 8, 8, 8, 8, 8, 8.0),
            self._make_score(7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5),
        ])
        assert ev.should_approve() is True
