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
# Cap on concurrent image downloads when fanning out result URLs in poll.py.
# A 100-task batch with 4 image URLs per task fans out to 400 sockets without
# a bound; 8 keeps the connection pool sane on shared CDN endpoints.
DOWNLOAD_CONCURRENCY: int = _int_env("STYLECLAW_DOWNLOAD_CONCURRENCY", "8")
TASK_TIMEOUT: float = _float_env("STYLECLAW_TASK_TIMEOUT", "300")
POLL_INTERVAL: float = _float_env("STYLECLAW_POLL_INTERVAL", "3")
POLL_MAX_CONSECUTIVE_FAILURES: int = _int_env("STYLECLAW_POLL_MAX_CONSEC_FAIL", "5")
ORCHESTRATOR_POLL_INTERVAL: float = _float_env("STYLECLAW_ORCH_POLL_INTERVAL", "30")
MAX_POLL_CYCLES: int = _int_env("STYLECLAW_MAX_POLL_CYCLES", "60")

# Number of parallel LLM shards used by design_cases. Must divide 10 (the
# fixed category count) evenly — allowed values are 1, 2, 5, 10. Smaller
# shards = smaller per-request token budgets, lower 429/500 risk, but more
# total system-prompt overhead.
DESIGN_CASES_SHARDS: int = _int_env("STYLECLAW_DESIGN_CASES_SHARDS", "5")

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

# Hard cap on a single downloaded image file. Trips early during streaming
# so a runaway URL can't fill the disk. Default is 50MB — typical model
# outputs are well under 5MB; 50MB leaves room for occasional large i2i
# results.
MAX_DOWNLOAD_BYTES_PER_FILE: int = _int_env("STYLECLAW_MAX_DOWNLOAD_BYTES", "52428800")

# httpx timeouts for the RunningHub image-gen client (in seconds).
# submit/query POSTs are tiny JSON bodies; the read budget mainly covers
# RunningHub's own queue-side latency.
RH_CLIENT_TIMEOUT: float = _float_env("STYLECLAW_RH_TIMEOUT", "60")
RH_CLIENT_CONNECT_TIMEOUT: float = _float_env("STYLECLAW_RH_CONNECT_TIMEOUT", "30")


def _list_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


# --- Three-model panel toggles (default OFF). When either is on,
# STYLECLAW_PANEL_MODELS must list exactly 3 model ids; labels (optional)
# must match length. validate_panel_config() reports problems instead of
# raising at import time so unit tests can still load config_mod cleanly.
PANEL_REFINE_ENABLED: bool = _bool_env("STYLECLAW_PANEL_REFINE", False)
PANEL_MODEL_SELECT_ENABLED: bool = _bool_env("STYLECLAW_PANEL_MODEL_SELECT", False)
PANEL_MODELS: list[str] = _list_env("STYLECLAW_PANEL_MODELS")
_PANEL_LABELS_RAW: list[str] = _list_env("STYLECLAW_PANEL_LABELS")
PANEL_LABELS: list[str] = _PANEL_LABELS_RAW or list(PANEL_MODELS)
# When a panel run comes back as degraded (one or more proposer/scorer
# failures, or no winner), default is to refuse persisting the result so a
# bogus winner doesn't silently propagate downstream. Set this env to allow
# saving degraded results anyway (useful for triage / forensics).
ALLOW_DEGRADED_PANEL: bool = _bool_env("STYLECLAW_ALLOW_DEGRADED_PANEL", False)


def validate_panel_config() -> list[str]:
    """Return error strings if panel envs are inconsistent.

    Only checks the global STYLECLAW_PANEL_MODELS when per-role pools are
    NOT set. Per-role pool validation lives in
    llm_routing.validate_routing_env (called separately from validate_env).
    """
    errors: list[str] = []
    if not (PANEL_REFINE_ENABLED or PANEL_MODEL_SELECT_ENABLED):
        return errors
    # If both panel toggles have role-specific pools, skip the global check.
    refine_overridden = bool(os.getenv("STYLECLAW_PANEL_MODELS_VISION_ANALYST"))
    select_overridden = bool(os.getenv("STYLECLAW_PANEL_MODELS_VISION_CRITIC"))
    refine_needs_global = PANEL_REFINE_ENABLED and not refine_overridden
    select_needs_global = PANEL_MODEL_SELECT_ENABLED and not select_overridden
    if refine_needs_global or select_needs_global:
        if len(PANEL_MODELS) != 3:
            errors.append(
                "STYLECLAW_PANEL_MODELS must list exactly 3 comma-separated model "
                f"ids when STYLECLAW_PANEL_REFINE or STYLECLAW_PANEL_MODEL_SELECT "
                f"is set (got {len(PANEL_MODELS)}: {PANEL_MODELS!r})."
            )
    if _PANEL_LABELS_RAW and len(_PANEL_LABELS_RAW) != len(PANEL_MODELS):
        errors.append(
            "STYLECLAW_PANEL_LABELS length must match STYLECLAW_PANEL_MODELS "
            f"(got {len(_PANEL_LABELS_RAW)} labels for {len(PANEL_MODELS)} models)."
        )
    return errors


_ALLOWED_DESIGN_CASES_SHARDS = (1, 2, 5, 10)


def validate_design_cases_config() -> list[str]:
    """Return error strings if DESIGN_CASES_SHARDS is not a value that
    evenly partitions the 10 fixed categories."""
    errors: list[str] = []
    if DESIGN_CASES_SHARDS not in _ALLOWED_DESIGN_CASES_SHARDS:
        errors.append(
            f"STYLECLAW_DESIGN_CASES_SHARDS={DESIGN_CASES_SHARDS} must be one of "
            f"{_ALLOWED_DESIGN_CASES_SHARDS} (each evenly partitions the 10 fixed "
            f"categories)."
        )
    return errors


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
    if not os.getenv("OPENAI_COMPAT_API_KEY"):
        errors.append(
            "OPENAI_COMPAT_API_KEY is not set (required for LLM access)."
        )
    errors.extend(validate_panel_config())
    errors.extend(validate_design_cases_config())

    # Per-role routing checks (vision_critic / vision_analyst / writer / planner).
    # Late import to avoid circular: llm_routing imports from config.
    from styleclaw.core.llm_routing import validate_routing_env
    errors.extend(validate_routing_env())

    return errors

