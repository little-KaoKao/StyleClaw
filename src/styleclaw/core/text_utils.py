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


def recover_truncated_json(cleaned: str) -> str:
    """Try to make a valid JSON string out of an LLM response that was cut
    off mid-output. Strategy: find the last fully-closed object, then close
    the enclosing array and root object.

    Returns the input unchanged when it already parses, or when no recovery
    is possible (so the caller's json.loads surfaces a useful error)."""
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass
    last_brace = cleaned.rfind("}")
    if last_brace < 0:
        return cleaned
    truncated = cleaned[: last_brace + 1]
    bracket = truncated.rfind("]")
    # `bracket` may be inside the last complete object; rsplit-then-close
    # handles the common "{...},{...}, {incomplete" pattern.
    if bracket < 0:
        candidate = truncated + "]}"
    else:
        candidate = truncated[: bracket + 1].rsplit(",", 1)[0] + "]}"
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return cleaned


def parse_llm_response(raw: str, model_cls: type[T], label: str = "") -> T:
    desc = label or model_cls.__name__
    cleaned = clean_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        preview = cleaned[:200].replace("\n", " ")
        hint = (
            "Hint: the LLM may have wrapped JSON in extra prose, "
            "truncated mid-output, or used single quotes. "
            "Re-running often resolves transient issues."
        )
        raise ValueError(
            f"LLM returned invalid JSON for {desc}: {exc}\n"
            f"Cleaned preview ({len(cleaned)} chars): {preview!r}\n"
            f"{hint}"
        ) from exc
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"LLM response failed validation for {desc}: {exc}") from exc


def sanitize_braces(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")

