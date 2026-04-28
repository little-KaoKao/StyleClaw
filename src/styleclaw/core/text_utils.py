from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def clean_json(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl >= 0:
            cleaned = cleaned[first_nl + 1:]
    if cleaned.endswith("```"):
        last_fence = cleaned.rfind("\n```")
        if last_fence >= 0:
            cleaned = cleaned[:last_fence]
        else:
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    if cleaned.startswith(("{", "[")):
        return cleaned
    match = re.search(r"[{\[]", cleaned)
    if match:
        tail = cleaned[match.start():]
        brace = tail[0]
        close = "]" if brace == "[" else "}"
        last = tail.rfind(close)
        if last >= 0:
            return tail[: last + 1]
    return cleaned


def parse_llm_response(raw: str, model_cls: type[T], label: str = "") -> T:
    desc = label or model_cls.__name__
    cleaned = clean_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON for {desc}: {exc}") from exc
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"LLM response failed validation for {desc}: {exc}") from exc


def sanitize_braces(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")
