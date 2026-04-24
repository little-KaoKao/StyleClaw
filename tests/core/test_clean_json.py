from styleclaw.core.text_utils import clean_json as _clean_json


class TestCleanJson:
    def test_plain_json(self):
        assert _clean_json('{"key": "val"}') == '{"key": "val"}'

    def test_strips_markdown_fences(self):
        raw = '```json\n{"key": "val"}\n```'
        assert _clean_json(raw) == '{"key": "val"}'

    def test_strips_fences_without_language(self):
        raw = '```\n{"key": "val"}\n```'
        assert _clean_json(raw) == '{"key": "val"}'

    def test_strips_whitespace(self):
        raw = '  \n  {"key": "val"}  \n  '
        assert _clean_json(raw) == '{"key": "val"}'

    def test_nested_backticks_preserved(self):
        raw = '```json\n{"code": "a```b"}\n```'
        assert '"code": "a```b"' in _clean_json(raw)
