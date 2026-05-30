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
import json
import logging
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

import queue
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.templating import Jinja2Templates

import markdown as md_lib
from starlette.responses import FileResponse

import chat_auth
import fritz_utils
import privacy
import workspace_store
from fritz_utils import (
    ADMIN_PANEL_PASSWORD,
    ADMIN_PANEL_PORT,
    CHAT_ALLOWED_IMAGE_TYPES,
    CHAT_COOKIE_SECRET,
    CHAT_DOC_UPLOAD_MAX_BYTES,
    CHAT_IMAGE_UPLOAD_MAX_BYTES,
    DOC_FOLDER,
    MessageSource,
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
        # The chat surface has its own cookie-based identity model and must
        # NOT require the admin password. Everything else needs Basic auth.
        if request.url.path == "/chat" or request.url.path.startswith("/chat/"):
            return await call_next(request)

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


# ── /chat — local chat UI (Phase web-chat-1) ────────────────────────────────
# A separate identity model from the admin panel: cookie-based, no password.
# Threat model is "me + my friends on a port-forwarded local network." For
# per-user namespacing of memories/schedules/threads, that's enough.

def _chat_user(request: Request) -> str | None:
    """Return the verified chat username from the cookie, or None."""
    token = request.cookies.get(chat_auth.COOKIE_NAME)
    return chat_auth.verify_cookie(token, CHAT_COOKIE_SECRET)


# Markdown extensions enabled for Fritz's replies. fenced_code handles ```py
# blocks; tables / nl2br polish the natural prose he tends to emit.
_MARKDOWN_EXTENSIONS = ["fenced_code", "tables", "nl2br"]


def _render_markdown(text: str) -> str:
    """Render markdown to HTML. Used for Fritz's replies in the chat history
    and for the final state of a streamed message."""
    if not text:
        return ""
    return md_lib.markdown(text, extensions=_MARKDOWN_EXTENSIONS)


def _doc_to_message(checkpoint_msg) -> dict | None:
    """Convert a LangGraph checkpoint message into a chat-view dict.
    Returns None for messages we shouldn't render (system, empty, tool calls)."""
    # langchain messages have .type ("human" | "ai" | "system" | "tool") and .content
    msg_type = getattr(checkpoint_msg, "type", None)
    content = getattr(checkpoint_msg, "content", None)
    if not content or not isinstance(content, str):
        return None
    content = content.strip()
    if not content:
        return None
    if msg_type == "human":
        return {"role": "user", "content": content, "html": None}
    if msg_type == "ai":
        return {"role": "fritz", "content": content, "html": _render_markdown(content)}
    return None


def _load_chat_history(user_id: str, limit: int = 40) -> list[dict]:
    """Load the recent message history from the LangGraph SqliteSaver. Returns
    a list of {role, content, html} dicts ready for the template. Best-effort
    — returns [] on any failure."""
    if not user_id:
        return []
    try:
        # Lazy import — same reasoning as in chat_send.
        from mister_fritz import app as agent_app, get_config_values
        thread_id = re.sub(r"[^a-zA-Z0-9]", "", user_id)
        if not thread_id:
            return []
        config = get_config_values({"metadata": {"user_id": thread_id, "thread_id": thread_id}})
        snapshot = agent_app.get_state(config)
        messages = (snapshot.values or {}).get("messages", []) if snapshot else []
    except Exception as e:
        logger.debug("Could not load chat history for %s: %s", user_id, e)
        return []

    out: list[dict] = []
    for m in messages[-limit:]:
        converted = _doc_to_message(m)
        if converted is not None:
            out.append(converted)
    return out


def _set_chat_cookie(response: Response, username: str) -> None:
    token = chat_auth.make_cookie(username, CHAT_COOKIE_SECRET)
    response.set_cookie(
        chat_auth.COOKIE_NAME, token,
        max_age=chat_auth.COOKIE_MAX_AGE_SECONDS,
        httponly=True, samesite="lax",
    )


async def chat_page(request: Request) -> HTMLResponse:
    user = _chat_user(request)
    if not user:
        return templates.TemplateResponse(request, "chat_login.html", {})
    # Refresh the cookie on every page view so active users don't get logged
    # out on day 31.
    # Phase 3: hydrate the page with the last 40 messages from the LangGraph
    # checkpoint so a refresh doesn't lose context.
    history = await asyncio.get_running_loop().run_in_executor(
        None, _load_chat_history, user, 40,
    )
    response = templates.TemplateResponse(request, "chat.html", {
        "username": user,
        "messages": history,
        "is_admin": fritz_utils.is_admin(user),
    })
    _set_chat_cookie(response, user)
    return response


async def chat_login(request: Request) -> Response:
    form = await request.form()
    username = (form.get("username") or "").strip()
    # Same sanitisation as mister_fritz uses for thread_id.
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", username)[:64]
    if not safe:
        # Re-render with an error.
        return templates.TemplateResponse(request, "chat_login.html", {
            "error": "Pick a username with at least one letter or number.",
        })
    response = RedirectResponse(url="/chat", status_code=303)
    _set_chat_cookie(response, safe)
    audit_log("chat_login", user_id=safe)
    return response


async def chat_logout(request: Request) -> Response:
    user = _chat_user(request)
    response = RedirectResponse(url="/chat", status_code=303)
    response.delete_cookie(chat_auth.COOKIE_NAME)
    if user:
        audit_log("chat_logout", user_id=user)
    return response


async def chat_send(request: Request) -> Response:
    """Synchronous fallback endpoint — submits a message and renders the full
    chat page with the reply. Used by graceful-degradation clients (no JS).
    HTMX clients go to /chat/stream instead for live streaming."""
    user = _chat_user(request)
    if not user:
        return RedirectResponse(url="/chat", status_code=303)

    form = await request.form()
    message = (form.get("message") or "").strip()
    if not message:
        return RedirectResponse(url="/chat", status_code=303)

    # Run the synchronous agent in a worker thread so the event loop is free.
    loop = asyncio.get_running_loop()
    schedule_manager = _schedule_manager_from_request(request)
    pending_images = _drain_pending_images(user)
    source = MessageSource.DISCORD_TEXT_AND_IMAGE if pending_images else MessageSource.LOCAL

    def _invoke_agent() -> dict:
        # Lazy import — keeps admin_panel importable without pulling the LLM
        # stack until the first chat send.
        from mister_fritz import ask_stuff
        return ask_stuff(
            message,
            source,
            user,
            user_image_paths=pending_images or None,
            schedule_manager=schedule_manager,
        )

    try:
        response_data = await loop.run_in_executor(None, _invoke_agent)
    except Exception as e:
        logger.exception("Chat send failed for %s", user)
        audit_log("chat_message", user_id=user, chars=len(message),
                  attached_images=len(pending_images), result="error", error=str(e))
        return templates.TemplateResponse(request, "chat.html", {
            "username": user,
            "messages": [
                {"role": "user", "content": message},
                {"role": "fritz", "content": f"❌ An error occurred: {e}"},
            ],
        })

    reply = response_data.get("text") if isinstance(response_data, dict) else str(response_data)
    audit_log("chat_message", user_id=user, chars=len(message), result="ok", reply_chars=len(reply or ""))

    return templates.TemplateResponse(request, "chat.html", {
        "username": user,
        "messages": [
            {"role": "user", "content": message, "html": None},
            {"role": "fritz", "content": reply or "(no response)",
             "html": _render_markdown(reply or "(no response)")},
        ],
    })


# Sentinel pushed onto the streaming queue when the agent finishes.
_CHAT_STREAM_DONE = object()


async def chat_stream(request: Request) -> Response:
    """Server-Sent Events endpoint. Runs ask_stuff in a worker thread; the
    streaming_callback puts events on a queue.Queue that the SSE generator
    drains. Emits:
      - event=token data=<accumulated text so far>   (zero or more)
      - event=done  data=<final text>                (exactly one)
      - event=error data=<message>                   (instead of done on failure)
    """
    user = _chat_user(request)
    if not user:
        return Response(status_code=401, content="Not signed in.")

    form = await request.form()
    message = (form.get("message") or "").strip()
    if not message:
        return Response(status_code=400, content="Empty message.")

    schedule_manager = _schedule_manager_from_request(request)
    pending_images = _drain_pending_images(user)
    source = MessageSource.DISCORD_TEXT_AND_IMAGE if pending_images else MessageSource.LOCAL
    event_queue: queue.Queue = queue.Queue()

    def _streaming_callback(partial_text: str) -> None:
        # ask_stuff invokes this from a worker thread; queue.Queue is thread-safe.
        event_queue.put(("token", partial_text))

    def _progress_callback(progress_text: str) -> None:
        # Tool progress messages: "Searching the web...", etc. Surfaced to the
        # client as a separate event so the UI can render them as ephemeral
        # italic lines instead of replacing the streaming bubble's content.
        event_queue.put(("progress", progress_text))

    def _invoke_agent() -> None:
        from mister_fritz import ask_stuff
        try:
            response_data = ask_stuff(
                message,
                source,
                user,
                progress_callback=_progress_callback,
                streaming_callback=_streaming_callback,
                user_image_paths=pending_images or None,
                schedule_manager=schedule_manager,
            )
            reply = response_data.get("text") if isinstance(response_data, dict) else str(response_data)
            image_paths = response_data.get("image_paths", []) if isinstance(response_data, dict) else []
            audit_log("chat_message", user_id=user, chars=len(message),
                      attached_images=len(pending_images),
                      result="ok", reply_chars=len(reply or ""), streamed=True)
            # The 'done' frame carries both the raw text and the rendered HTML
            # so the client doesn't have to ship a markdown parser. Image paths
            # are encoded as a JSON-ish pipe-separated list to avoid newline
            # issues in the SSE data field.
            done_payload = {
                "text": reply or "",
                "html": _render_markdown(reply or ""),
                "images": [_chat_asset_url(p) for p in image_paths],
            }
            event_queue.put(("done", json.dumps(done_payload)))
        except Exception as e:
            logger.exception("Chat stream failed for %s", user)
            audit_log("chat_message", user_id=user, chars=len(message),
                      result="error", error=str(e), streamed=True)
            event_queue.put(("error", str(e)))
        finally:
            event_queue.put(_CHAT_STREAM_DONE)

    # Use a plain daemon thread (not run_in_executor) so the worker is
    # independent of whatever event loop is driving the response. This matters
    # under TestClient, which uses per-request loops; an executor task on the
    # request loop can be torn down before the generator runs.
    threading.Thread(target=_invoke_agent, name="chat-stream-worker", daemon=True).start()

    async def _event_generator():
        # Hand-format SSE frames so we avoid sse-starlette's internal
        # asyncio.Event (used for keep-alive), which gets bound to the request
        # loop and explodes under TestClient's per-request loops.
        while True:
            try:
                item = event_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if item is _CHAT_STREAM_DONE:
                return
            event, data = item
            # SSE wire format: an optional `event: <name>` line, one or more
            # `data: <value>` lines (one per logical line in the payload),
            # then a blank line as the frame terminator.
            data_lines = "".join(f"data: {chunk}\n" for chunk in str(data).split("\n"))
            yield f"event: {event}\n{data_lines}\n".encode("utf-8")

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


# Roots that /chat/assets/<path> is allowed to serve from. Anything outside
# these dirs is a path-escape attempt.
def _chat_asset_roots() -> list[str]:
    return [
        os.path.abspath("output"),
        os.path.abspath("temp_images"),
    ]


def _chat_asset_url(path: str) -> str | None:
    """Translate an absolute filesystem path (as returned by Fritz's image
    tools) into a /chat/assets URL the browser can fetch. Returns None for
    paths outside the allowed roots."""
    if not path:
        return None
    abs_path = os.path.abspath(path)
    for root in _chat_asset_roots():
        if abs_path == root or abs_path.startswith(root + os.sep):
            rel = os.path.relpath(abs_path, os.path.dirname(root))
            # Use forward slashes for URL even on Windows.
            return "/chat/assets/" + rel.replace(os.sep, "/")
    return None


async def chat_asset(request: Request) -> Response:
    """Serve a file from one of the chat-asset roots (./output, ./temp_images).
    Requires an authed chat user; rejects paths that escape the roots."""
    user = _chat_user(request)
    if not user:
        return Response(status_code=401, content="Not signed in.")
    rel = request.path_params["path"]
    if not rel:
        return Response(status_code=404)
    for root in _chat_asset_roots():
        root_parent = os.path.dirname(root)
        full = os.path.abspath(os.path.join(root_parent, rel))
        if full == root or full.startswith(root + os.sep):
            if os.path.isfile(full):
                return FileResponse(full)
    return Response(status_code=404)


async def chat_history(request: Request) -> JSONResponse:
    """Return the recent message history for the cookie's user as JSON.
    Used by clients that want to refresh without a full page reload."""
    user = _chat_user(request)
    if not user:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    history = await asyncio.get_running_loop().run_in_executor(
        None, _load_chat_history, user, 40,
    )
    return JSONResponse({"username": user, "messages": history})


async def chat_forget(request: Request) -> Response:
    """Clear the LangGraph checkpoint for the current user — the next message
    starts a fresh conversation. Doesn't touch memories or schedules."""
    user = _chat_user(request)
    if not user:
        return RedirectResponse(url="/chat", status_code=303)
    removed = await asyncio.get_running_loop().run_in_executor(
        None, privacy.forget_conversation, user,
    )
    audit_log("chat_forget_conversation", user_id=user, removed=removed)
    return RedirectResponse(url="/chat", status_code=303)


# ── Phase web-chat-4: file uploads ───────────────────────────────────────────
# Pending-image registry: maps username → list of absolute paths attached to
# the user's next chat message. Single-process, in-memory — fine for the local
# admin/chat panel deployment model.
_pending_images: dict[str, list[str]] = {}
_pending_images_lock = threading.Lock()


def _stash_pending_image(user_id: str, path: str) -> None:
    with _pending_images_lock:
        _pending_images.setdefault(user_id, []).append(path)


def _drain_pending_images(user_id: str) -> list[str]:
    with _pending_images_lock:
        return _pending_images.pop(user_id, [])


def _safe_filename(name: str) -> str:
    """Strip path separators and exotic punctuation from an uploaded filename.
    Keeps the extension; replaces unsafe characters with underscores."""
    base = os.path.basename(name or "upload")
    stem, dot, ext = base.rpartition(".")
    safe_stem = re.sub(r"[^a-zA-Z0-9._-]", "_", stem if stem else base)[:80]
    safe_ext = re.sub(r"[^a-zA-Z0-9]", "", ext)[:8] if dot else ""
    return f"{safe_stem}.{safe_ext}" if safe_ext else safe_stem


async def chat_upload_image(request: Request) -> Response:
    """Accept a multipart-form image, save it to ./temp_images/, and stash
    the path in the per-user pending list. The next /chat/send or
    /chat/stream picks it up and passes as user_image_paths to ask_stuff."""
    user = _chat_user(request)
    if not user:
        return JSONResponse({"error": "not signed in"}, status_code=401)

    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse({"error": "missing file field"}, status_code=400)

    content_type = (getattr(upload, "content_type", "") or "").lower()
    if content_type not in CHAT_ALLOWED_IMAGE_TYPES:
        audit_log("chat_upload_image", user_id=user, result="rejected_type",
                  content_type=content_type)
        return JSONResponse(
            {"error": f"unsupported image type '{content_type}'"},
            status_code=415,
        )

    raw = await upload.read()
    if len(raw) > CHAT_IMAGE_UPLOAD_MAX_BYTES:
        audit_log("chat_upload_image", user_id=user, result="rejected_size",
                  bytes=len(raw), cap=CHAT_IMAGE_UPLOAD_MAX_BYTES)
        return JSONResponse(
            {"error": f"image exceeds {CHAT_IMAGE_UPLOAD_MAX_BYTES} byte cap"},
            status_code=413,
        )

    os.makedirs("temp_images", exist_ok=True)
    ts = int(time.time())
    safe_name = _safe_filename(getattr(upload, "filename", "upload"))
    safe_user = re.sub(r"[^a-zA-Z0-9_-]", "_", user)
    target = os.path.abspath(os.path.join("temp_images", f"{safe_user}_{ts}_{safe_name}"))
    with open(target, "wb") as f:
        f.write(raw)

    _stash_pending_image(user, target)
    audit_log("chat_upload_image", user_id=user, result="ok",
              path=target, bytes=len(raw))
    return JSONResponse({
        "ok": True,
        "url": _chat_asset_url(target),
        "name": safe_name,
    })


async def chat_upload_document(request: Request) -> Response:
    """Accept a multipart-form document and drop it into DOC_FOLDER for the
    watchdog to ingest. Admin-only — DOC_FOLDER is shared knowledge."""
    user = _chat_user(request)
    if not user:
        return JSONResponse({"error": "not signed in"}, status_code=401)
    if not fritz_utils.is_admin(user):
        audit_log("chat_upload_document", user_id=user, result="not_admin")
        return JSONResponse(
            {"error": "document upload is admin-only — DOC_FOLDER is shared knowledge"},
            status_code=403,
        )

    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse({"error": "missing file field"}, status_code=400)

    raw_name = getattr(upload, "filename", "upload")
    safe_name = _safe_filename(raw_name)
    # Defer to document_engine's supported extensions list.
    try:
        from document_engine import SUPPORTED_EXTENSIONS
    except Exception:
        SUPPORTED_EXTENSIONS = (".docx", ".pdf", ".xlsx", ".csv", ".txt", ".md")
    _, dot, ext = safe_name.rpartition(".")
    if not dot or f".{ext.lower()}" not in SUPPORTED_EXTENSIONS:
        audit_log("chat_upload_document", user_id=user, result="bad_ext",
                  filename=safe_name)
        return JSONResponse(
            {"error": f"unsupported extension; allowed: {', '.join(SUPPORTED_EXTENSIONS)}"},
            status_code=415,
        )

    raw = await upload.read()
    if len(raw) > CHAT_DOC_UPLOAD_MAX_BYTES:
        audit_log("chat_upload_document", user_id=user, result="rejected_size",
                  bytes=len(raw), cap=CHAT_DOC_UPLOAD_MAX_BYTES)
        return JSONResponse(
            {"error": f"document exceeds {CHAT_DOC_UPLOAD_MAX_BYTES} byte cap"},
            status_code=413,
        )

    os.makedirs(DOC_FOLDER, exist_ok=True)
    target = os.path.abspath(os.path.join(DOC_FOLDER, safe_name))
    # Guard against path-escape via filename (extra belt to _safe_filename).
    doc_root = os.path.abspath(DOC_FOLDER)
    if not (target == doc_root or target.startswith(doc_root + os.sep)):
        audit_log("chat_upload_document", user_id=user, result="path_escape",
                  filename=safe_name)
        return JSONResponse({"error": "invalid filename"}, status_code=400)

    with open(target, "wb") as f:
        f.write(raw)

    audit_log("chat_upload_document", user_id=user, result="ok",
              path=target, bytes=len(raw))
    return JSONResponse({"ok": True, "name": safe_name})


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
        # Chat surface (cookie auth; no admin password required).
        Route("/chat", chat_page, name="chat"),
        Route("/chat/login", chat_login, methods=["POST"], name="chat_login"),
        Route("/chat/logout", chat_logout, methods=["POST"], name="chat_logout"),
        Route("/chat/send", chat_send, methods=["POST"], name="chat_send"),
        Route("/chat/stream", chat_stream, methods=["POST"], name="chat_stream"),
        Route("/chat/forget", chat_forget, methods=["POST"], name="chat_forget"),
        Route("/chat/history", chat_history, name="chat_history"),
        Route("/chat/assets/{path:path}", chat_asset, name="chat_asset"),
        Route("/chat/upload/image", chat_upload_image,
              methods=["POST"], name="chat_upload_image"),
        Route("/chat/upload/document", chat_upload_document,
              methods=["POST"], name="chat_upload_document"),
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
