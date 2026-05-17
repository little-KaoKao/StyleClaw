from __future__ import annotations

import pytest
from pydantic import BaseModel

from styleclaw.core.text_utils import (
    MAX_LLM_RESPONSE_BYTES,
    clean_json,
    parse_llm_response,
)


class TestCleanJsonEdgeCases:
    def test_extracts_json_object_from_surrounding_text(self) -> None:
        raw = 'Here is the result: {"key": "value"} and more text'
        assert clean_json(raw) == '{"key": "value"}'

    def test_extracts_json_array_from_surrounding_text(self) -> None:
        raw = 'Here is the list: [1, 2, 3] done'
        assert clean_json(raw) == "[1, 2, 3]"

    def test_handles_nested_braces(self) -> None:
        raw = 'prefix {"a": {"b": 1}} suffix'
        assert clean_json(raw) == '{"a": {"b": 1}}'

    def test_no_json_found_returns_cleaned(self) -> None:
        raw = "  no json here  "
        assert clean_json(raw) == "no json here"

    def test_json_with_leading_text_no_closing_brace(self) -> None:
        raw = "prefix {incomplete"
        result = clean_json(raw)
        assert isinstance(result, str)

    def test_extracts_array_with_objects(self) -> None:
        raw = 'output: [{"id": 1}, {"id": 2}] end'
        assert clean_json(raw) == '[{"id": 1}, {"id": 2}]'


class _Demo(BaseModel):
    key: str


class TestParseLlmResponseSizeCap:
    def test_normal_size_parses(self) -> None:
        obj = parse_llm_response('{"key": "val"}', _Demo)
        assert obj.key == "val"

    def test_oversized_raw_refused_before_parsing(self) -> None:
        # Padding with whitespace keeps the JSON valid, but the raw byte count
        # blows past the cap — we should refuse before even trying to parse.
        huge = " " * (MAX_LLM_RESPONSE_BYTES + 100) + '{"key": "val"}'
        with pytest.raises(ValueError, match="too large"):
            parse_llm_response(huge, _Demo)
