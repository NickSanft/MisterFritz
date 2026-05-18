"""Read-only HTML admin panel for Mister Fritz (Phase 9a).

Mounted on its own port (default 8001), bound to 127.0.0.1 only — admins
port-forward over SSH if they want remote access. Shared-password HTTP Basic
auth gates everything; if ADMIN_PANEL_PASSWORD is unset the panel doesn't
start at all.

Built on Starlette (already a transitive dep via the LLM stack) so we don't
need to add FastAPI.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route
from starlette.templating import Jinja2Templates

import privacy
import workspace_store
from fritz_utils import (
    ADMIN_PANEL_PASSWORD,
    ADMIN_PANEL_PORT,
    DOC_FOLDER,
    __version__,
)
from observability import get_health_snapshot

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "admin_templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ── Auth ────────────────────────────────────────────────────────────────────

class _BasicAuthMiddleware(BaseHTTPMiddleware):
    """HTTP Basic auth keyed by ADMIN_PANEL_PASSWORD.

    Username is ignored; only the password matters. Constant-time compare
    so timing attacks can't fingerprint a partial match.
    """

    def __init__(self, app, password: str):
        super().__init__(app)
        self._password = password

    async def dispatch(self, request: Request, call_next):
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("basic "):
            return _unauthorized()
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8", errors="replace")
        except Exception:
            return _unauthorized()
        if ":" not in decoded:
            return _unauthorized()
        _, _, password = decoded.partition(":")
        if not secrets.compare_digest(password, self._password):
            return _unauthorized()
        return await call_next(request)


def _unauthorized() -> Response:
    return Response(
        status_code=401,
        content="Authentication required.",
        headers={"WWW-Authenticate": 'Basic realm="MisterFritz admin"'},
    )


# ── Page handlers ───────────────────────────────────────────────────────────

def _collect_users(schedule_manager) -> list[str]:
    """Union of every user_id we have data on (schedules + workspaces).

    Chroma namespaces aren't easily enumerated; users with memories but no
    schedule or workspace won't appear in the listing. Acceptable for v1.
    """
    seen: set[str] = set()
    if schedule_manager is not None:
        try:
            for s in schedule_manager.list_all_schedules():
                seen.add(s["user_id"])
        except Exception as e:
            logger.debug("collecting users from schedules failed: %s", e)
    try:
        for w in workspace_store.list_all():
            seen.add(w["user_id"])
    except Exception as e:
        logger.debug("collecting users from workspaces failed: %s", e)
    return sorted(seen)


def _schedule_manager_from_request(request: Request):
    """The schedule manager is attached to app.state at startup."""
    return getattr(request.app.state, "schedule_manager", None)


async def overview(request: Request) -> HTMLResponse:
    snap = get_health_snapshot()
    return templates.TemplateResponse(request, "overview.html", {
        "version": __version__,
        "uptime_sec": int(snap["uptime_sec"]),
        "counters": snap["counters"],
        "errors": snap["errors"],
        "latencies": snap["latencies"],
        "last_error": snap["last_error"],
    })


async def users_list(request: Request) -> HTMLResponse:
    schedule_manager = _schedule_manager_from_request(request)
    user_ids = _collect_users(schedule_manager)
    rows = []
    for uid in user_ids:
        memories = privacy.export_memories(uid)
        schedules = privacy.export_schedules(uid, schedule_manager)
        rows.append({
            "user_id": uid,
            "memory_count": len(memories),
            "schedule_count": len(schedules),
            "workspace": privacy.get_workspace_for_export(uid),
        })
    return templates.TemplateResponse(request, "users.html", {"users": rows})


async def user_detail(request: Request) -> HTMLResponse:
    user_id = request.path_params["user_id"]
    schedule_manager = _schedule_manager_from_request(request)
    data = privacy.export_user_data(user_id, schedule_manager)
    return templates.TemplateResponse(request, "user_detail.html", {
        "data": data,
        "user_id": user_id,
    })


async def schedules_list(request: Request) -> HTMLResponse:
    schedule_manager = _schedule_manager_from_request(request)
    schedules = []
    if schedule_manager is not None:
        try:
            schedules = schedule_manager.list_all_schedules()
        except Exception as e:
            logger.warning("list_all_schedules failed: %s", e)
    return templates.TemplateResponse(request, "schedules.html", {"schedules": schedules})


async def documents_list(request: Request) -> HTMLResponse:
    docs = []
    try:
        for root, _, files in os.walk(DOC_FOLDER):
            for f in files:
                if f.startswith("~$"):
                    continue
                path = Path(root) / f
                stat = path.stat()
                docs.append({
                    "name": str(path.relative_to(DOC_FOLDER)),
                    "size_bytes": stat.st_size,
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                })
    except (FileNotFoundError, NotADirectoryError):
        docs = []
    docs.sort(key=lambda d: d["mtime"], reverse=True)
    return templates.TemplateResponse(request, "documents.html", {
        "doc_folder": DOC_FOLDER,
        "docs": docs,
    })


async def health_json(request: Request) -> JSONResponse:
    """JSON snapshot, same shape as :8000/health but exposed here for
    admins who don't want to remember a second port."""
    return JSONResponse(get_health_snapshot())


# ── App factory + server boot ───────────────────────────────────────────────

def create_app(password: str, schedule_manager=None) -> Starlette:
    """Build the Starlette app with the password baked in. Exposed as a
    factory so tests can construct it without spinning up uvicorn."""
    routes = [
        Route("/", overview, name="overview"),
        Route("/users", users_list, name="users"),
        Route("/users/{user_id}", user_detail, name="user_detail"),
        Route("/schedules", schedules_list, name="schedules"),
        Route("/documents", documents_list, name="documents"),
        Route("/health", health_json, name="health"),
    ]
    app = Starlette(
        routes=routes,
        middleware=[Middleware(_BasicAuthMiddleware, password=password)],
    )
    app.state.schedule_manager = schedule_manager
    return app


def start_admin_panel(schedule_manager=None) -> Optional[int]:
    """Spin up the admin panel in a background thread.

    Returns the port if started, or None if disabled (no password set).
    Bound to 127.0.0.1 only — port-forward over SSH for remote access.
    """
    if not ADMIN_PANEL_PASSWORD:
        logger.info("ADMIN_PANEL_PASSWORD not set — admin panel disabled.")
        return None

    app = create_app(ADMIN_PANEL_PASSWORD, schedule_manager=schedule_manager)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=ADMIN_PANEL_PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, name="admin-panel", daemon=True)
    t.start()
    logger.info(
        "Admin panel started at http://127.0.0.1:%d/ — HTTP Basic, any username + ADMIN_PANEL_PASSWORD",
        ADMIN_PANEL_PORT,
    )
    return ADMIN_PANEL_PORT
