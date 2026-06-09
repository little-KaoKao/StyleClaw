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

    from styleclaw.web.routes_projects import router as projects_router
    app.include_router(projects_router)

    from styleclaw.web.routes_projects import media_router
    app.include_router(media_router)

    return app
