from __future__ import annotations

import os
import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{20,}")
_BEARER_RE = re.compile(
    r"(?i)"
    r"(?P<prefix>"
    r"Bearer|Token|sk[-_]|"
    r"Authorization|"
    r"x[-_]?api[-_]?key|"
    r"api[-_]?key|"
    r"access[-_]?token|"
    r"secret[-_]?key"
    r")"
    r"(?P<sep>\s*[:=]?\s*)"
    r"(?P<value>[A-Za-z0-9_\-\.]+)"
)
_SAFE_PREFIX_RE = re.compile(
    r"^(https?|file|s3|gs|attempt|model|task|case|round|pass|batch|i2i|t2i)$",
    re.IGNORECASE,
)


def _known_secrets() -> Iterable[str]:
    for var in (
        "RUNNINGHUB_API_KEY",
        "OPENAI_COMPAT_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
    ):
        val = os.getenv(var)
        if val and len(val) >= 8:
            yield val


def _bearer_sub(m: re.Match[str]) -> str:
    return f"{m.group('prefix')}{m.group('sep')}***"


def redact(text: str) -> str:
    """Best-effort scrub of secrets and high-entropy tokens from ``text``.

    Three passes, in order:
    1. Exact replacement of any value set via the env vars we know carry
       credentials (``RUNNINGHUB_API_KEY`` / ``OPENAI_COMPAT_API_KEY`` /
       ``AWS_BEARER_TOKEN_BEDROCK``).
    2. Header/keyword scrub: ``Bearer ...`` / ``Token: ...`` / ``sk-...`` /
       ``Authorization: ...`` / ``api_key=...`` / ``x-api-key: ...`` etc.
       The matched prefix + separator are preserved so the redacted form
       still reads naturally; only the value is replaced with ``***``.
    3. Long ``[A-Za-z0-9_-]{20,}`` runs that aren't obviously a URL path
       segment or a familiar identifier prefix (``attempt``, ``task``,
       ``round``, etc.) get partially masked.

    The intent is logging-only — never use this for security boundaries.
    """
    if not text:
        return text
    out = text
    for secret in _known_secrets():
        out = out.replace(secret, "***")
    out = _BEARER_RE.sub(_bearer_sub, out)

    def _maybe_mask(match: re.Match[str]) -> str:
        tok = match.group(0)
        if _SAFE_PREFIX_RE.match(tok):
            return tok
        return tok[:4] + "***" + tok[-2:] if len(tok) > 24 else "***"

    return _TOKEN_RE.sub(_maybe_mask, out)


def redact_exc(exc: BaseException) -> str:
    """Return a redacted string representation of ``exc`` for log output.

    Joins ``type(exc).__name__`` with ``str(exc)`` after running the message
    through :func:`redact`. Walks the ``__cause__`` chain (one level) so we
    keep a hint of the underlying failure without exposing the credentialed
    HTTP request that produced it.
    """
    head = f"{type(exc).__name__}: {redact(str(exc))}"
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        head += f" (cause: {type(cause).__name__}: {redact(str(cause))})"
    return head
