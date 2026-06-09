from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Build the StyleClaw local web app.

    Single-user, bound to 127.0.0.1 by the launcher. No auth by design.
    """
    app = FastAPI(title="StyleClaw", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
