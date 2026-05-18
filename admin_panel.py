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
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
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
from observability import audit_log, get_health_snapshot

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
        username, _, password = decoded.partition(":")
        if not secrets.compare_digest(password, self._password):
            return _unauthorized()
        # Stash the supplied username for audit log attribution. Shared password
        # means anyone could enter any name; we still record what they typed so
        # admins can distinguish each other in practice.
        request.state.admin_username = username or "(unset)"
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


# ── Mutating actions (Phase 9b) ──────────────────────────────────────────────
# These are POST-only so a stray GET (link prefetch, browser preview, etc.)
# can't trigger destructive actions. Every action writes an entry to the
# audit log with the admin's Basic-auth username and the target resource.

def _admin(request: Request) -> str:
    return getattr(request.state, "admin_username", "(unknown)")


async def forget_user_action(request: Request) -> Response:
    user_id = request.path_params["user_id"]
    schedule_manager = _schedule_manager_from_request(request)
    result = privacy.forget_all(user_id, schedule_manager)
    audit_log(
        "admin_forget_all", admin=_admin(request),
        target_user=user_id, result=result,
    )
    return RedirectResponse(url="/users", status_code=303)


async def disable_workspace_action(request: Request) -> Response:
    user_id = request.path_params["user_id"]
    removed = privacy.forget_workspace(user_id)
    audit_log(
        "admin_disable_workspace", admin=_admin(request),
        target_user=user_id, removed=removed,
    )
    return RedirectResponse(url=f"/users/{user_id}", status_code=303)


async def cancel_schedule_action(request: Request) -> Response:
    schedule_id = request.path_params["schedule_id"]
    schedule_manager = _schedule_manager_from_request(request)
    removed = False
    target_user = None
    if schedule_manager is not None:
        # Find the owner so we can delete via the normal API (which goes through
        # APScheduler too, not just the DB row).
        for s in schedule_manager.list_all_schedules():
            if s["id"] == schedule_id:
                target_user = s["user_id"]
                break
        if target_user is not None:
            try:
                removed = schedule_manager.remove_schedule(schedule_id, target_user)
            except Exception as e:
                logger.warning("admin cancel_schedule(%s) failed: %s", schedule_id, e)
    audit_log(
        "admin_cancel_schedule", admin=_admin(request),
        schedule_id=schedule_id, target_user=target_user, removed=removed,
    )
    return RedirectResponse(url="/schedules", status_code=303)


async def reindex_document_action(request: Request) -> Response:
    """Re-enqueue a document so the watchdog worker re-ingests it.

    Imports document_engine lazily so admin_panel can be imported in tests
    without dragging in the entire LLM stack.
    """
    form = await request.form()
    doc_name = (form.get("name") or "").strip()
    if not doc_name:
        audit_log("admin_reindex_document", admin=_admin(request), error="missing-name")
        return RedirectResponse(url="/documents", status_code=303)
    full_path = os.path.abspath(os.path.join(DOC_FOLDER, doc_name))
    # Defence-in-depth: the path must stay inside DOC_FOLDER.
    doc_folder_abs = os.path.abspath(DOC_FOLDER)
    if not full_path.startswith(doc_folder_abs + os.sep) and full_path != doc_folder_abs:
        audit_log(
            "admin_reindex_document", admin=_admin(request),
            error="path-escape", attempted=doc_name,
        )
        return RedirectResponse(url="/documents", status_code=303)
    enqueued = False
    if os.path.isfile(full_path):
        try:
            import document_engine
            document_engine.INGESTION_QUEUE.put(("update", full_path))
            enqueued = True
        except Exception as e:
            logger.warning("admin reindex(%s) failed: %s", full_path, e)
    audit_log(
        "admin_reindex_document", admin=_admin(request),
        document=doc_name, enqueued=enqueued,
    )
    return RedirectResponse(url="/documents", status_code=303)


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
        # Mutating actions (POST only).
        Route("/users/{user_id}/forget", forget_user_action,
              methods=["POST"], name="forget_user"),
        Route("/users/{user_id}/workspace/disable", disable_workspace_action,
              methods=["POST"], name="disable_workspace"),
        Route("/schedules/{schedule_id}/cancel", cancel_schedule_action,
              methods=["POST"], name="cancel_schedule"),
        Route("/documents/reindex", reindex_document_action,
              methods=["POST"], name="reindex_document"),
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
