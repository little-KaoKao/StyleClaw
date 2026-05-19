"""Shared test fixtures.

Currently provides one autouse cleanup fixture that resets the panel-related
attributes on ``styleclaw.core.config`` after every test.

Why this exists: several test classes flip ``STYLECLAW_PANEL_*`` env vars and
call ``importlib.reload(config_mod)`` so the freshly-flipped values take
effect. ``monkeypatch`` reverts the env vars at teardown — but it does NOT
re-reload the module. Without this fixture, ``PANEL_REFINE_ENABLED=True``
leaks across tests, and unrelated tests then enter the panel branch of
``do_refine`` / ``do_evaluate`` and fail when ``build_panel_providers`` is
called without the toggle env vars set.

We reset the panel attributes *directly* on the module instead of
``importlib.reload`` to avoid clobbering other config attributes that the
session loaded from ``.env`` via ``styleclaw.cli.load_dotenv()``.
"""

from __future__ import annotations

import os

import pytest


_PANEL_ENV_KEYS = (
    "STYLECLAW_PANEL_REFINE",
    "STYLECLAW_PANEL_MODEL_SELECT",
    "STYLECLAW_PANEL_MODELS",
    "STYLECLAW_PANEL_LABELS",
    # Role-specific pools — populated from .env by load_dotenv() the moment
    # styleclaw.cli is imported (e.g. via tests/core/test_max_rounds.py). If
    # left in place, they make config.validate_panel_config() skip the global
    # "exactly 3" check, which silently fails unrelated TestPanelConfig tests
    # depending on test ordering.
    "STYLECLAW_PANEL_MODELS_VISION_CRITIC",
    "STYLECLAW_PANEL_MODELS_VISION_ANALYST",
)


@pytest.fixture(autouse=True)
def _reset_panel_config_state():
    """Restore panel attributes on the config module to their unset defaults."""
    yield
    for key in _PANEL_ENV_KEYS:
        os.environ.pop(key, None)
    import styleclaw.core.config as config_mod
    config_mod.PANEL_REFINE_ENABLED = False
    config_mod.PANEL_MODEL_SELECT_ENABLED = False
    config_mod.PANEL_MODELS = []
    config_mod.PANEL_LABELS = []
    config_mod._PANEL_LABELS_RAW = []


@pytest.fixture(autouse=True)
def _reset_encode_semaphore():
    """Drop the per-loop _ENCODE_SEMAPHORE so the next test rebinds on its own loop.

    The semaphore in ``styleclaw.core.image_utils`` is keyed by event loop in a
    WeakKeyDictionary, so once the loop is collected the entry disappears on
    its own. We still clear the map at teardown to make test isolation explicit
    and to avoid relying on GC timing.
    """
    yield
    try:
        import styleclaw.core.image_utils as image_utils
        image_utils._ENCODE_SEMAPHORES.clear()
    except (ImportError, AttributeError):
        pass

