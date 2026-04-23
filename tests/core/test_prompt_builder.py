import pytest

from styleclaw.core.prompt_builder import build_params


class TestBuildParams:
    def test_mj_v7_basic(self):
        params = build_params("mj-v7", "watercolor style", aspect_ratio="16:9")
        assert params["prompt"] == "watercolor style"
        assert params["aspectRatio"] == "16:9"
        assert params["stylize"] == 200

    def test_mj_v7_with_sref(self):
        params = build_params("mj-v7", "test", sref_url="https://example.com/ref.png")
        assert params["sref"] == "https://example.com/ref.png"
        assert params["sw"] == 100

    def test_nb2_has_resolution(self):
        params = build_params("nb2", "test prompt")
        assert params["resolution"] == "2k"
        assert params["aspectRatio"] == "9:16"

    def test_nb2_ignores_sref(self):
        params = build_params("nb2", "test", sref_url="https://example.com/ref.png")
        assert "sref" not in params

    def test_seedream_uses_width_height(self):
        params = build_params("seedream", "test", aspect_ratio="16:9")
        assert params["width"] == 2560
        assert params["height"] == 1440
        assert "aspectRatio" not in params
        assert "resolution" not in params

    def test_seedream_truncates_long_prompt(self):
        long_prompt = "x" * 3000
        params = build_params("seedream", long_prompt)
        assert len(params["prompt"]) == 2000

    def test_character_desc_concatenation(self):
        params = build_params("mj-v7", "watercolor", character_desc="a young girl")
        assert params["prompt"] == "watercolor, a young girl"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_params("nonexistent", "test")

    def test_extra_params_override(self):
        params = build_params("mj-v7", "test", extra_params={"stylize": 500, "chaos": 20})
        assert params["stylize"] == 500
        assert params["chaos"] == 20
