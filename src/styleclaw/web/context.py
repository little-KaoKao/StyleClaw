from __future__ import annotations

import asyncio
import inspect
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from styleclaw.orchestrator.actions import ExecutionContext

logger = logging.getLogger(__name__)


async def _close_resource(resource: Any, label: str) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await asyncio.wait_for(result, timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Timed out closing %s after 5s.", label)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error closing %s: %s", label, exc)


@asynccontextmanager
async def build_context(
    project: str,
    *,
    needs_client: bool = False,
    needs_llm: bool = False,
    router: Any = None,
) -> AsyncIterator[ExecutionContext]:
    """Build an ExecutionContext for a web request/run.

    Mirrors ``cli._build_context`` but standalone (no Typer dependency) to
    avoid a web<->cli import cycle. A ``router`` passed in by the caller is
    reused and NOT closed here (caller owns its lifecycle).
    """
    from styleclaw.core.llm_routing import RoleRouter
    from styleclaw.providers.runninghub.client import RunningHubClient

    client = None
    owns_router = False
    try:
        if needs_client:
            key = os.getenv("RUNNINGHUB_API_KEY")
            if not key:
                raise RuntimeError("RUNNINGHUB_API_KEY not set")
            client = RunningHubClient(api_key=key)
        if needs_llm and router is None:
            router = RoleRouter.from_env()
            owns_router = True
        yield ExecutionContext(project=project, client=client, llm_router=router)
    finally:
        if client is not None:
            await _close_resource(client, "client")
        if router is not None and owns_router:
            await _close_resource(router, "llm_router")
