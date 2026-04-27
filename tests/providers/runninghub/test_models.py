import pytest

from styleclaw.providers.runninghub.models import (
    MODEL_REGISTRY,
    get_model,
)


class TestModelRegistry:
    def test_five_models_registered(self):
        assert len(MODEL_REGISTRY) == 5
        assert set(MODEL_REGISTRY.keys()) == {"mj-v7", "niji7", "nb2", "seedream", "gpt-image-2"}

    def test_mj_v7_config(self):
        m = get_model("mj-v7")
        assert m.max_prompt_length == 8192
        assert m.supports_sref is True
        assert "/text-to-image-v7" in m.t2i_endpoint

    def test_niji7_config(self):
        m = get_model("niji7")
        assert m.supports_sref is True
        assert "/text-to-image-niji7" in m.t2i_endpoint

    def test_nb2_config(self):
        m = get_model("nb2")
        assert m.max_prompt_length == 20000
        assert m.supports_sref is False
        assert m.i2i_endpoint != m.t2i_endpoint

    def test_seedream_config(self):
        m = get_model("seedream")
        assert m.max_prompt_length == 2000
        assert m.uses_width_height is True

    def test_gpt_image_2_config(self):
        m = get_model("gpt-image-2")
        assert m.max_prompt_length == 20000
        assert m.supports_sref is False
        assert m.uses_width_height is False
        assert m.i2i_endpoint != m.t2i_endpoint
        assert m.default_params == {"resolution": "2k", "quality": "medium"}
        assert "4:5" in m.aspect_ratio_values
        assert "21:9" in m.aspect_ratio_values

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent")
