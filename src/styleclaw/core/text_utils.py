from __future__ import annotations

import re


def clean_json(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
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
