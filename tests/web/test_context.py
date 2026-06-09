import pytest

from styleclaw.web.context import build_context


@pytest.mark.asyncio
async def test_context_without_client_or_llm(data_root):
    async with build_context("proj", needs_client=False, needs_llm=False) as ctx:
        assert ctx.project == "proj"
        assert ctx.client is None
        assert ctx.llm_router is None


@pytest.mark.asyncio
async def test_context_reuses_passed_router(data_root):
    sentinel = object()
    async with build_context("proj", needs_llm=True, router=sentinel) as ctx:
        assert ctx.llm_router is sentinel
    # passed-in router is NOT closed by build_context (caller owns it)
