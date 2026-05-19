"""Shared test helpers for routing-aware action tests."""
from __future__ import annotations


class MockRouter:
    """Minimal router stub that returns a fixed provider for every role.

    Real action code only calls .get(role) and .get_panel(role) — never
    inspects which role it received — so a single shared provider is enough
    for action-level tests. Tagging the provider with `_model_id` lets the
    action's `getattr(llm, "_model_id", "")` recording path pick it up.
    """

    def __init__(self, llm, model_id: str = "test-model") -> None:
        self._llm = llm
        self._model_id = model_id
        setattr(llm, "_model_id", model_id)

    def get(self, role):
        return self._llm

    def get_panel(self, role):
        return ([self._llm, self._llm, self._llm], ["m1", "m2", "m3"])

    async def close(self) -> None:
        return None
