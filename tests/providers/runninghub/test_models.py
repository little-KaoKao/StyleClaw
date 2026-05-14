import pytest

from styleclaw.providers.runninghub.models import (
    MODEL_REGISTRY,
    I2IParamStyle,
    SrefMode,
    build_i2i_params,
    get_model,
)


class TestModelRegistry:
    def test_five_models_registered(self):
        assert len(MODEL_REGISTRY) == 5
        assert set(MODEL_REGISTRY.keys()) == {"mj-v7", "niji7", "nb2", "seedream", "gpt-image-2"}

    def test_mj_v7_config(self):
        m = get_model("mj-v7")
        assert m.max_prompt_length == 8192
        assert m.sref_mode == SrefMode.PARAM
        assert "/text-to-image-v7" in m.t2i_endpoint

    def test_niji7_config(self):
        m = get_model("niji7")
        assert m.sref_mode == SrefMode.PARAM
        assert "/text-to-image-niji7" in m.t2i_endpoint

    def test_nb2_config(self):
        m = get_model("nb2")
        assert m.max_prompt_length == 20000
        assert m.sref_mode == SrefMode.PROMPT
        assert m.i2i_endpoint != m.t2i_endpoint

    def test_seedream_config(self):
        m = get_model("seedream")
        assert m.max_prompt_length == 2000
        assert m.uses_width_height is True
        assert m.sref_mode == SrefMode.PROMPT

    def test_gpt_image_2_config(self):
        m = get_model("gpt-image-2")
        assert m.max_prompt_length == 20000
        assert m.sref_mode == SrefMode.PROMPT
        assert m.uses_width_height is False
        assert m.i2i_endpoint != m.t2i_endpoint
        assert m.default_params == {"resolution": "2k", "quality": "medium"}
        assert "4:5" in m.aspect_ratio_values
        assert "21:9" in m.aspect_ratio_values

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent")


class TestI2IParamStyle:
    """Each model declares how its i2i endpoint expects image params, so
    batch_submit_i2i doesn't need to hardcode an if-chain when new models
    are added."""

    def test_mj_uses_single_url_iw(self):
        m = get_model("mj-v7")
        assert m.i2i_param_style == I2IParamStyle.SINGLE_URL_IW

    def test_niji_uses_single_url_iw(self):
        m = get_model("niji7")
        assert m.i2i_param_style == I2IParamStyle.SINGLE_URL_IW

    def test_nb2_uses_multi_urls(self):
        m = get_model("nb2")
        assert m.i2i_param_style == I2IParamStyle.MULTI_URLS

    def test_seedream_uses_multi_urls(self):
        m = get_model("seedream")
        assert m.i2i_param_style == I2IParamStyle.MULTI_URLS

    def test_gpt_image_2_uses_multi_urls(self):
        m = get_model("gpt-image-2")
        assert m.i2i_param_style == I2IParamStyle.MULTI_URLS


class TestBuildI2IParams:
    def test_mj_emits_single_image_url_with_iw(self):
        params = build_i2i_params(get_model("mj-v7"), "bold anime", "https://cdn/1.png")
        assert params["prompt"] == "bold anime"
        assert params["imageUrl"] == "https://cdn/1.png"
        assert params["iw"] == 0.5
        assert "imageUrls" not in params

    def test_niji_emits_single_image_url_with_iw(self):
        params = build_i2i_params(get_model("niji7"), "anime style", "https://cdn/2.png")
        assert params["imageUrl"] == "https://cdn/2.png"
        assert params["iw"] == 0.5

    def test_non_mj_emits_image_urls_list(self):
        params = build_i2i_params(get_model("nb2"), "trigger", "https://cdn/x.png")
        assert params["prompt"] == "trigger"
        assert params["imageUrls"] == ["https://cdn/x.png"]
        assert "imageUrl" not in params
        assert "iw" not in params

    def test_seedream_emits_image_urls_list(self):
        params = build_i2i_params(get_model("seedream"), "t", "https://cdn/a.png")
        assert params["imageUrls"] == ["https://cdn/a.png"]

    def test_gpt_image_2_emits_image_urls_list(self):
        params = build_i2i_params(get_model("gpt-image-2"), "t", "https://cdn/b.png")
        assert params["imageUrls"] == ["https://cdn/b.png"]
