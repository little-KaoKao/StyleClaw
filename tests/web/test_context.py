from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from styleclaw.web.context import build_context


@pytest.mark.asyncio
async def test_context_without_client_or_llm(data_root):
    async with build_context("proj", needs_client=False, needs_llm=False) as ctx:
        assert ctx.project == "proj"
        assert ctx.client is None
        assert ctx.llm_router is None


# ---------------------------------------------------------------------------
# Ownership / cleanup contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_caller_supplied_router_not_closed(data_root):
    """A router passed by the caller is NOT closed — caller owns the lifecycle."""
    mock_router = MagicMock()
    mock_router.close = AsyncMock()

    async with build_context("proj", needs_llm=True, router=mock_router) as ctx:
        assert ctx.llm_router is mock_router

    mock_router.close.assert_not_called()


@pytest.mark.asyncio
async def test_created_router_is_closed(data_root, monkeypatch):
    """When build_context creates the router it must close it on exit."""
    mock_router = MagicMock()
    mock_router.close = AsyncMock()

    mock_from_env = MagicMock(return_value=mock_router)
    monkeypatch.setattr(
        "styleclaw.core.llm_routing.RoleRouter.from_env", mock_from_env
    )

    async with build_context("proj", needs_llm=True, router=None) as ctx:
        assert ctx.llm_router is mock_router

    mock_router.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_created_client_is_closed(data_root, monkeypatch):
    """When build_context creates the HTTP client it must close it on exit."""
    mock_client = MagicMock()
    mock_client.close = AsyncMock()

    mock_cls = MagicMock(return_value=mock_client)
    monkeypatch.setattr(
        "styleclaw.providers.runninghub.client.RunningHubClient", mock_cls
    )

    async with build_context("proj", needs_client=True) as ctx:
        assert ctx.client is mock_client

    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_api_key_raises(data_root, monkeypatch):
    """Missing RUNNINGHUB_API_KEY must raise RuntimeError on entry."""
    monkeypatch.delenv("RUNNINGHUB_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="RUNNINGHUB_API_KEY"):
        async with build_context("proj", needs_client=True):
            pass
