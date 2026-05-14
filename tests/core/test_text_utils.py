from styleclaw.core.text_utils import clean_json, recover_truncated_json


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


class TestRecoverTruncatedJson:
    """LLMs sometimes truncate large array-valued JSON mid-element. This
    helper closes the nearest complete object and resyncs the structure."""

    def test_passes_through_valid_json(self) -> None:
        valid = '{"cases": [{"id": "a"}, {"id": "b"}]}'
        assert recover_truncated_json(valid) == valid

    def test_recovers_truncated_array_of_objects(self) -> None:
        truncated = '{"cases": [{"id": "a"}, {"id": "b"}, {"id":'
        recovered = recover_truncated_json(truncated)
        import json
        data = json.loads(recovered)
        assert data["cases"] == [{"id": "a"}, {"id": "b"}]

    def test_returns_input_when_no_recoverable_structure(self) -> None:
        bad = "not json at all and nothing closeable here"
        # Unrecoverable input is returned as-is so the caller's json.loads
        # raises its own clear JSONDecodeError.
        assert recover_truncated_json(bad) == bad
