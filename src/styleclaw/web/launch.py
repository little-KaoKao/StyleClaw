from __future__ import annotations

import logging
import threading
import webbrowser

logger = logging.getLogger(__name__)

_HOST = "127.0.0.1"


def build_server_config(port: int = 8800, open_browser: bool = True) -> dict:
    """Return the uvicorn config dict. Isolated so tests can assert it without
    actually binding a socket."""
    return {"host": _HOST, "port": port, "open_browser": open_browser}


def serve(port: int = 8800, open_browser: bool = True) -> None:  # pragma: no cover - runs a real server
    import uvicorn

    from styleclaw.web.app import create_app

    cfg = build_server_config(port=port, open_browser=open_browser)
    if cfg["open_browser"]:
        url = f"http://{cfg['host']}:{cfg['port']}"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        logger.info("Opening browser at %s", url)
    uvicorn.run(create_app(), host=cfg["host"], port=cfg["port"], log_level="info")
