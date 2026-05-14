from __future__ import annotations

import os
import sys


def _int_env(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Invalid value for {name}: '{raw}'. Expected an integer.") from None


def _float_env(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Invalid value for {name}: '{raw}'. Expected a number.") from None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


MAX_AUTO_ROUNDS: int = _int_env("STYLECLAW_MAX_ROUNDS", "5")
CONCURRENCY_LIMIT: int = _int_env("STYLECLAW_CONCURRENCY", "10")
LLM_CONCURRENCY_LIMIT: int = _int_env("STYLECLAW_LLM_CONCURRENCY", "4")
IMAGE_ENCODE_CONCURRENCY: int = _int_env("STYLECLAW_IMAGE_ENCODE_CONCURRENCY", "8")
TASK_TIMEOUT: float = _float_env("STYLECLAW_TASK_TIMEOUT", "300")
POLL_INTERVAL: float = _float_env("STYLECLAW_POLL_INTERVAL", "3")
POLL_MAX_CONSECUTIVE_FAILURES: int = _int_env("STYLECLAW_POLL_MAX_CONSEC_FAIL", "5")
ORCHESTRATOR_POLL_INTERVAL: float = _float_env("STYLECLAW_ORCH_POLL_INTERVAL", "30")
MAX_POLL_CYCLES: int = _int_env("STYLECLAW_MAX_POLL_CYCLES", "60")

# Stream LLM response deltas to stdout. Default is "True iff stdout is a TTY"
# so piped/CI invocations don't blast partial tokens and don't block the event
# loop on synchronous prints during parallel LLM calls.
STREAM_DISPLAY: bool = _bool_env("STYLECLAW_STREAM_DISPLAY", sys.stdout.isatty())

# httpx timeouts for LLM providers (in seconds).
# WriteTimeout is the killer when evaluate POSTs many base64 images at once —
# the default httpx 5s is way too short, and even the historical 60s here
# wasn't enough on slow upload links. Default to 300s (matching read) and
# expose env knobs so users on flaky networks can crank further.
LLM_WRITE_TIMEOUT: float = _float_env("STYLECLAW_LLM_WRITE_TIMEOUT", "300")
LLM_READ_TIMEOUT: float = _float_env("STYLECLAW_LLM_READ_TIMEOUT", "300")
LLM_CONNECT_TIMEOUT: float = _float_env("STYLECLAW_LLM_CONNECT_TIMEOUT", "30")

# httpx timeouts for the RunningHub image-gen client (in seconds).
# submit/query POSTs are tiny JSON bodies; the read budget mainly covers
# RunningHub's own queue-side latency.
RH_CLIENT_TIMEOUT: float = _float_env("STYLECLAW_RH_TIMEOUT", "60")
RH_CLIENT_CONNECT_TIMEOUT: float = _float_env("STYLECLAW_RH_CONNECT_TIMEOUT", "30")


def env_truthy(name: str) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def validate_env() -> list[str]:
    """Check that required environment variables are present.

    Returns a list of human-readable error messages (empty if all checks pass).
    """
    errors: list[str] = []
    if not os.getenv("RUNNINGHUB_API_KEY"):
        errors.append("RUNNINGHUB_API_KEY is not set (required for image generation).")
    has_bedrock = bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))
    has_openai = bool(os.getenv("OPENAI_COMPAT_API_KEY"))
    has_runninghub_llm = env_truthy("RUNNINGHUB_LLM")
    if not (has_bedrock or has_openai or has_runninghub_llm):
        errors.append(
            "No LLM credentials found. Set AWS_BEARER_TOKEN_BEDROCK, "
            "OPENAI_COMPAT_API_KEY, or RUNNINGHUB_LLM=1 (uses RUNNINGHUB_API_KEY)."
        )
    return errors

