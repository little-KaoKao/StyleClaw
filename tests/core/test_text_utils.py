from styleclaw.core.text_utils import clean_json


class TestCleanJson:
    def test_plain_json(self) -> None:
        assert clean_json('{"key": "val"}') == '{"key": "val"}'

    def test_strips_markdown_fences_with_lang(self) -> None:
        raw = '```json\n{"key": "val"}\n```'
        assert clean_json(raw) == '{"key": "val"}'

    def test_strips_fences_without_language(self) -> None:
        raw = '```\n{"key": "val"}\n```'
        assert clean_json(raw) == '{"key": "val"}'

    def test_strips_surrounding_whitespace(self) -> None:
        assert clean_json('  \n  {"a": 1}  \n  ') == '{"a": 1}'

    def test_multiline_json(self) -> None:
        raw = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
        result = clean_json(raw)
        assert '"a": 1' in result
        assert '"b": 2' in result
