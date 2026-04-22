from styleclaw.scripts.generate import IMAGES_PER_MODEL_REFINE


class TestImagesPerModelRefine:
    def test_default_value_is_three(self):
        assert IMAGES_PER_MODEL_REFINE == 3
