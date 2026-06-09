from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Module-level constant so tests can monkeypatch it before calling create_app().
_STATIC_DIR: Path = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    """Build the StyleClaw local web app.

    Single-user, bound to 127.0.0.1 by the launcher. No auth by design.
    """
    app = FastAPI(title="StyleClaw", docs_url="/api/docs", openapi_url="/api/openapi.json")

    from styleclaw.web.run_manager import RunManager
    app.state.run_manager = RunManager()

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    from styleclaw.web.routes_confirm import router as confirm_router
    app.include_router(confirm_router)

    from styleclaw.web.routes_projects import router as projects_router
    app.include_router(projects_router)

    from styleclaw.web.routes_projects import media_router
    app.include_router(media_router)

    from styleclaw.web.routes_runs import router as runs_router
    app.include_router(runs_router)

    # SPA serving — only when the frontend has been built.
    # Read the module global at call time so monkeypatch.setattr works in tests.
    static_dir = _STATIC_DIR
    if (static_dir / "index.html").exists():
        app.mount(
            "/assets",
            StaticFiles(directory=static_dir / "assets"),
            name="assets",
        )

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str) -> FileResponse:
            # Let genuine API/media 404s surface as JSON, not HTML.
            if full_path.startswith("api/") or full_path.startswith("media/"):
                raise HTTPException(status_code=404, detail="not found")
            return FileResponse(static_dir / "index.html")

    return app
