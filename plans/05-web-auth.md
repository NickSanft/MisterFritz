# 5. Authenticate and harden the web chat surface

[← back to index](README.md)

**Effort:** L (1-3 days)  
**Depends on:** nothing

## Goal
Today anyone who can reach :8001 can type any username into `/chat/login`, receive a validly-signed identity cookie, and read + write another person's Discord conversation thread; they can also upload an HTML/SVG file that the server stores with its client-chosen extension and serves back same-origin, and every LLM reply is rendered through python-markdown (which passes `<script>` and `onerror=` verbatim) into `| safe` and `innerHTML`. When this lands: obtaining a chat cookie requires a shared secret (`CHAT_PASSWORD`, defaulting to `ADMIN_PANEL_PASSWORD`, fail-closed if neither is set); the web surface has its own LangGraph thread so a web session cannot read Discord history; uploaded images are validated by decoding them with Pillow and stored under a canonical extension derived from the sniffed format, never the client's; `/chat/assets/*` serves only an allowlisted set of image content-types with `nosniff` + a `default-src 'none'; sandbox` CSP and a per-user ownership check on `temp_images`; markdown output is sanitised with nh3 before it reaches the DOM; and no admin-privileged action is reachable from the self-asserted chat identity at all (document upload moves to the Basic-auth'd `/documents` page).

## Definition of done

- [ ] POST /chat/login without a `password` field, or with a wrong one, returns 401 and sets no cookie; the attempt is written to audit.log with the attempted username and client host.
- [ ] With neither CHAT_PASSWORD nor ADMIN_PANEL_PASSWORD set, /chat/login returns 503 and mints no identity (fail closed). start_admin_panel logs a warning naming CHAT_PASSWORD.
- [ ] chat_auth.py's docstring no longer says the cookie is not authentication, and chat_login.html no longer says "No password — anyone reaching this port can pick any name".
- [ ] Signing in as user B and loading /chat shows none of user A's Discord messages; `SELECT DISTINCT thread_id FROM checkpoints` in fritz.db shows a `web-<user>` row distinct from the Discord `<user>` row. Setting CHAT_SHARE_DISCORD_THREAD=true restores the shared thread.
- [ ] POST /chat/upload/image with an HTML or SVG body and a declared Content-Type of image/png returns 415 and writes nothing to ./temp_images.
- [ ] A valid PNG uploaded with filename `evil.html` is stored with a `.png` extension; the stored filename never contains an extension taken from the client.
- [ ] GET /chat/assets/<anything not in {.png,.jpg,.jpeg,.gif,.webp}> returns 404. Successful asset responses carry an explicit allowlisted Content-Type, `X-Content-Type-Options: nosniff`, and `Content-Security-Policy: default-src 'none'; sandbox; frame-ancestors 'none'`.
- [ ] User A cannot fetch a file under ./temp_images uploaded by user B (404 plus a `chat_asset_denied` audit entry).
- [ ] _render_markdown on '<script>alert(1)</script>', '<img src=x onerror=alert(1)>' and '[l](javascript:alert(1))' each produce output containing none of `<script`, `onerror`, or `javascript:` — while `<strong>`, `<pre><code>` and `<table>` still survive.
- [ ] Every response from the panel carries a nonce-based `script-src` CSP plus nosniff / X-Frame-Options / Referrer-Policy; the inline script in chat.html executes under it.
- [ ] `grep -n is_admin admin_panel.py admin_templates/chat.html` returns nothing, and `import fritz_utils` is gone from admin_panel.py. POST /chat/upload/document no longer routes; POST /documents/upload exists and 401s without Basic auth.
- [ ] `ruff check .` is clean and `pytest tests/ --cov=. --cov-fail-under=60` is green (after installing python-multipart, which is missing from the local .venv today).
- [ ] Every new env var appears in .env.example, and the three pre-existing-but-undocumented chat vars (CHAT_COOKIE_SECRET, CHAT_IMAGE_UPLOAD_MAX_BYTES, CHAT_DOC_UPLOAD_MAX_BYTES) are backfilled there.
- [ ] CHANGELOG.md has a `### Security` entry under `## [Unreleased]` in the existing 'Web chat — Phase N' style, flagging both breaking changes (thread split, document-upload route move).

## Current state (verified against the working tree)
All three audit findings confirmed by reading the code this session; line numbers below are verified, with two corrections noted.

AUTH / IDENTITY
- `admin_panel.py:74-78` — `_BasicAuthMiddleware.dispatch` returns `await call_next(request)` unconditionally when `request.url.path == "/chat"` or starts with `"/chat/"`. Everything under `/chat` is therefore password-free.
- `admin_panel.py:400-413` — `chat_login` reads only `form.get("username")`, sanitises with `re.sub(r"[^a-zA-Z0-9_-]", "", username)[:64]`, and mints a cookie. There is no credential of any kind.
- `chat_auth.py:7-10` — docstring states plainly: "This is **not** real authentication — anyone reaching `:8001/chat` can claim any username they like." The HMAC only prevents *tampering* an already-issued cookie (verified: `tests/test_chat_auth.py::TestVerifyCookie::test_tampered_username_rejected` passes).
- `admin_panel.py:353` — `_load_chat_history` does `thread_id = re.sub(r"[^a-zA-Z0-9]", "", user_id)`, byte-identical to `mister_fritz.py:536` (`user_id_clean = _re.sub(r'[^a-zA-Z0-9]', '', user_id)`). CORRECTION/ADDITION to the audit: it is not only a *read* leak. `chat_send` (`admin_panel.py:448-454`) and `chat_stream` (`admin_panel.py:522-530`) both call `ask_stuff(..., user, ...)` with the cookie name as `user_id`, and `mister_fritz.py:548-560` uses `user_id_clean` for BOTH `configurable.thread_id` and `configurable.user_id`. So a claimed username also *writes into* the victim's Discord checkpoint and *writes into their Chroma memory namespace*.
- `admin_panel.py:732` — `chat_upload_document` gates on `fritz_utils.is_admin(user)` where `user` is the self-asserted cookie name. `admin_panel.py:394` passes `is_admin` into the chat template; `admin_templates/chat.html:151-158` renders the upload control from it.

UPLOAD / STORED XSS
- `admin_panel.py:690-697` — `content_type = (getattr(upload, "content_type", "") or "").lower()` checked against `CHAT_ALLOWED_IMAGE_TYPES`. Client-controlled header only; no byte inspection.
- `admin_panel.py:667-674` — `_safe_filename` does `stem, dot, ext = base.rpartition(".")` and keeps `re.sub(r"[^a-zA-Z0-9]", "", ext)[:8]`. "html" and "svg" survive intact.
- `admin_panel.py:712-714` — writes to `temp_images/{safe_user}_{ts}_{safe_name}`.
- `admin_panel.py:606-621` — `chat_asset` returns bare `FileResponse(full)`. Starlette guesses the media type from the extension, so a stored `.html` is served as `text/html` on the same origin as `/chat`. No `X-Content-Type-Options`, no CSP, no `Content-Disposition`, and no check that the requesting user owns the file.
- `fritz_utils.py:167-169` — `CHAT_ALLOWED_IMAGE_TYPES` correctly excludes SVG, but only the header is compared against it.

MARKDOWN XSS — reproduced live this session. Running `markdown.markdown(s, extensions=['fenced_code','tables','nl2br'])` on a string containing `<script>alert(2)</script>`, `<img src=x onerror=alert(1)>` and `[link](javascript:alert(3))` emits all three unchanged.
- `admin_panel.py:318-323` — `_render_markdown` calls `md_lib.markdown(text, extensions=_MARKDOWN_EXTENSIONS)` with no sanitiser. (Minor correction: 315 is the `_MARKDOWN_EXTENSIONS` list; the function body is 318-323.)
- Sinks: `admin_templates/chat.html:122` `{{ m.html | safe }}`, and `admin_templates/chat.html:365` `body.innerHTML = payload.html || ...`. The `html` field is produced at `admin_panel.py:542` (`"html": _render_markdown(reply or "")`) and `admin_panel.py:340` / `:478`.

ENVIRONMENT NOTE (blocks any local verification)
- `python-multipart` is in `requirements.txt:290` but is NOT installed in `.venv`. `pytest tests/test_admin_panel.py tests/test_chat_auth.py` currently reports 30 failed, 48 passed; every failure is `AssertionError: The 'python-multipart' library must be installed to use form parsing` from `starlette/requests.py:262`. This is a local-env problem, not a repo regression — CI installs it. `ruff` is also not installed in `.venv`.
- `nh3` is installable cleanly: `pip install --dry-run nh3` resolves `nh3-0.3.6-cp38-abi3-win_amd64.whl` — abi3, no Rust toolchain. `bleach` (6.4.0) is also available and `html5lib` is already a dep, but nh3 is the maintained choice and the one upstream bleach points at.
- `pillow==12.0.0` is installed (`requirements.txt:171`). Verified this session: `Image.open` raises `UnidentifiedImageError` on `b"<html><script>alert(1)</script></html>"`, on an SVG body, and on the truncated `_TINY_PNG` fixture used by the tests; it returns `format == "PNG"` on a real 1x1 PNG.
- `admin_panel.py:39` `import fritz_utils` is used at exactly two places — lines 394 and 732 — both of which this plan deletes, so the import must go too or ruff F401 fires.
- Generated images are named `output/generated_image-{timestamp}.png` (`image_generator.py:203-211`) — no user identifier, so per-user ownership on `./output` is not possible without a new registry.
- `ask_stuff` callers: `bot_commands.py:395`, `main_discord.py:195`, `main_telegram.py:28` and `:54`, `scheduler.py:111`, plus the two in `admin_panel.py`. All pass `(prompt, source, user_id)` positionally, so a new keyword-only parameter is safe.

## Change sites

### `fritz_utils.py:121-124 (insert after ADMIN_PANEL_PORT) and 155-169 (replace image-type block)`

New chat auth/threading knobs, and an image-format table that maps a sniffed Pillow format name to the canonical extension + MIME we will store and serve. CHAT_ALLOWED_IMAGE_TYPES is kept as a derived name so the existing import in admin_panel.py:45 keeps working.

# after line 124 (ADMIN_PANEL_PORT)

# Shared secret required to obtain a /chat identity cookie. Defaults to
# ADMIN_PANEL_PASSWORD so there is one secret to manage; set it separately to
# hand out chat access without admin-panel access. If BOTH are unset the chat
# surface refuses every login (fail closed) rather than minting free identities.
CHAT_PASSWORD: str | None = os.environ.get("CHAT_PASSWORD") or ADMIN_PANEL_PASSWORD or None

# Optional allowlist of usernames that may be claimed at /chat/login. Empty
# (default) = any sanitised name once the password checks out.
CHAT_ALLOWED_USERS: frozenset[str] = frozenset(
    u.strip() for u in os.environ.get("CHAT_ALLOWED_USERS", "").split(",") if u.strip()
)

# Mark the chat cookie Secure. Off by default: the panel is normally reached
# over plain http through an SSH tunnel, where Secure would break login.
CHAT_COOKIE_SECURE: bool = os.environ.get("CHAT_COOKIE_SECURE", "").lower() in ("1", "true", "yes")

# The web chat gets its own LangGraph thread so a chat identity cannot read or
# write the Discord conversation. Set true to restore the old shared thread.
CHAT_SHARE_DISCORD_THREAD: bool = os.environ.get("CHAT_SHARE_DISCORD_THREAD", "").lower() in ("1", "true", "yes")
CHAT_THREAD_PREFIX: str = os.environ.get("CHAT_THREAD_PREFIX", "web")

# replace lines 165-169
# Image formats accepted on chat upload, keyed by the format name Pillow
# reports after actually decoding the header. The value is the canonical
# (extension, mime) we store under and serve back — the Content-Type the
# *client* declares is advisory only. No SVG: Pillow won't decode it anyway,
# which is the point.
CHAT_ALLOWED_IMAGE_FORMATS: dict[str, tuple[str, str]] = {
    "JPEG": ("jpg", "image/jpeg"),
    "PNG":  ("png", "image/png"),
    "WEBP": ("webp", "image/webp"),
    "GIF":  ("gif", "image/gif"),
}
CHAT_ALLOWED_IMAGE_TYPES: frozenset[str] = frozenset(
    mime for _ext, mime in CHAT_ALLOWED_IMAGE_FORMATS.values()
)

### `mister_fritz.py:523-560`

Add a keyword-only `thread_id` parameter so a caller can select the LangGraph checkpoint independently of the memory namespace. Discord/Telegram/scheduler behaviour is byte-identical (they pass nothing). The sanitiser for an explicit thread_id keeps `_` and `-` so `web-alice` cannot collide with a Discord user literally named `webalice`.

def ask_stuff(
    base_prompt: str,
    source: MessageSource,
    user_id: str,
    progress_callback=None,
    streaming_callback=None,
    user_image_paths: list[str] = None,
    workspace_root: str = None,
    channel_id: int | None = None,
    schedule_manager=None,
    thread_id: str | None = None,
) -> dict:
    """Process user input and return structured output with text and attachments.

    `user_id` selects the memory namespace; `thread_id` selects the LangGraph
    checkpoint. They are the same for Discord/Telegram. The web chat passes its
    own thread so a web session cannot read or overwrite Discord history.
    """
    import re as _re
    user_id_clean = _re.sub(r'[^a-zA-Z0-9]', '', user_id)
    # Explicit thread ids keep _ and - so a namespaced thread ("web-alice")
    # cannot collide with an alnum-only Discord thread.
    thread_id_clean = _re.sub(r'[^a-zA-Z0-9_-]', '', thread_id) if thread_id else user_id_clean
    ...
    config = {
        "configurable": {"user_id": user_id_clean, "thread_id": thread_id_clean},
        "metadata": {
            "user_id": user_id_clean,
            "thread_id": thread_id_clean,
            ...

### `privacy.py:61-71`

`forget_conversation` accepts an optional explicit thread_id so /chat/forget clears the *web* thread, not the Discord one. Existing positional callers are unaffected.

def forget_conversation(user_id: str, thread_id: str | None = None) -> int:
    """Drop the LangGraph SqliteSaver state for a thread.

    Defaults to the thread derived from user_id (the Discord thread). Pass
    thread_id explicitly to target a surface-specific thread, e.g. the web
    chat's "web-<user>".
    """
    if not user_id and not thread_id:
        return 0
    if thread_id:
        thread_id = re.sub(r"[^a-zA-Z0-9_-]", "", thread_id)
    else:
        thread_id = _sanitise_thread_id(user_id)
    if not thread_id:
        return 0
    # ...unchanged DELETE FROM checkpoints / writes body...

### `admin_panel.py:35-58`

New imports (io, html, nh3, PIL.Image) and the CSP-nonce Jinja global. Delete `import fritz_utils` (line 39) — it is only used at 394 and 732, both removed.

import html as html_lib
import io

import markdown as md_lib
from starlette.responses import FileResponse

try:
    import nh3
except ImportError:  # pinned in requirements.txt; fail closed if absent
    nh3 = None

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

# DELETE: import fritz_utils      <- was line 39, now unused (ruff F401)
import chat_auth
import privacy
import workspace_store
from fritz_utils import (
    ADMIN_PANEL_PASSWORD,
    ADMIN_PANEL_PORT,
    CHAT_ALLOWED_IMAGE_FORMATS,
    CHAT_ALLOWED_IMAGE_TYPES,
    CHAT_ALLOWED_USERS,
    CHAT_COOKIE_SECRET,
    CHAT_COOKIE_SECURE,
    CHAT_DOC_UPLOAD_MAX_BYTES,
    CHAT_IMAGE_UPLOAD_MAX_BYTES,
    CHAT_PASSWORD,
    CHAT_SHARE_DISCORD_THREAD,
    CHAT_THREAD_PREFIX,
    DOC_FOLDER,
    MessageSource,
    __version__,
)


def _csp_nonce(request: Request) -> str:
    """Per-response nonce for the inline <script> in chat.html. Generated by
    _SecurityHeadersMiddleware; the lazy fallback keeps templates renderable if
    a handler is exercised without the middleware stack."""
    nonce = getattr(request.state, "csp_nonce", None)
    if nonce is None:
        nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = nonce
    return nonce


templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
templates.env.globals["csp_nonce"] = _csp_nonce

### `admin_panel.py:63-104 (add a sibling class after _unauthorized)`

New `_SecurityHeadersMiddleware` that mints the CSP nonce before the handler runs and applies default hardening headers after. Uses `setdefault` (verified present on Starlette's MutableHeaders) so chat_asset's stricter own CSP wins. style-src deliberately stays 'unsafe-inline': base.html:6, chat.html:5 and chat_login.html's many `style="..."` attributes rely on it, and inline CSS is not the injection vector here — script is.

class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """CSP nonce + baseline hardening headers for every response.

    script-src is nonce-only, which kills both injected <script> tags and
    inline event handlers (onerror=) if anything ever slips past the nh3
    sanitiser in _render_markdown. style-src keeps 'unsafe-inline' because the
    admin templates are full of style="..." attributes and inline <style>.
    Handlers that set their own CSP (chat_asset) are left alone.
    """

    async def dispatch(self, request: Request, call_next):
        nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = nonce
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; "
            f"script-src 'nonce-{nonce}'; "
            "style-src 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; form-action 'self'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        return response

### `admin_panel.py:318-323`

Sanitise the markdown output. This is the primary fix for finding (c); the CSP is defence in depth.

def _render_markdown(text: str) -> str:
    """Render markdown to HTML and sanitise it.

    The input is LLM output — and, via tool results, arbitrary scraped web and
    document text — so it is untrusted. python-markdown passes raw HTML through
    by design: verified that '<script>alert(1)</script>', '<img src=x
    onerror=...>' and '[x](javascript:...)' all survive it unchanged. nh3
    (ammonia) strips script/style/iframe, on* handlers, and non-http(s)/mailto
    URLs before this reaches `{{ m.html | safe }}` (chat.html:122) and
    `body.innerHTML` (chat.html:365).
    """
    if not text:
        return ""
    raw_html = md_lib.markdown(text, extensions=_MARKDOWN_EXTENSIONS)
    if nh3 is None:
        # No sanitiser installed: fail closed, show the markup as text.
        logger.warning("nh3 unavailable — rendering chat reply as escaped text")
        return html_lib.escape(raw_html)
    return nh3.clean(raw_html)

### `admin_panel.py:344-368 (line 353 is the site) plus a new helper above it`

Replace the Discord-identical thread derivation with a web-specific one, and pass it to get_config_values.

def _chat_thread_id(user_id: str) -> str:
    """LangGraph thread for the *web* surface.

    Until this change the web chat reused the Discord thread verbatim (the same
    re.sub as mister_fritz.py:536), so anyone who typed your username into the
    login form read — and appended to — your Discord conversation. Set
    CHAT_SHARE_DISCORD_THREAD=true to restore the old single-thread behaviour.
    """
    clean = re.sub(r"[^a-zA-Z0-9]", "", user_id or "")
    if not clean:
        return ""
    return clean if CHAT_SHARE_DISCORD_THREAD else f"{CHAT_THREAD_PREFIX}-{clean}"


# inside _load_chat_history, replacing line 353:
        thread_id = _chat_thread_id(user_id)
        if not thread_id:
            return []
        config = get_config_values({"metadata": {
            "user_id": re.sub(r"[^a-zA-Z0-9]", "", user_id),
            "thread_id": thread_id,
        }})

### `admin_panel.py:371-377, 380-397, 400-413`

Secure-flag the cookie; drop `is_admin` from the chat page context; gate chat_login on the shared password + optional username allowlist. The empty-username branch keeps its 200 status so only the password field is a test-visible change there.

def _set_chat_cookie(response: Response, username: str) -> None:
    token = chat_auth.make_cookie(username, CHAT_COOKIE_SECRET)
    response.set_cookie(
        chat_auth.COOKIE_NAME, token,
        max_age=chat_auth.COOKIE_MAX_AGE_SECONDS,
        httponly=True, samesite="lax", secure=CHAT_COOKIE_SECURE,
    )


# chat_page: delete the is_admin key entirely (was line 394)
    response = templates.TemplateResponse(request, "chat.html", {
        "username": user,
        "messages": history,
    })


async def chat_login(request: Request) -> Response:
    """Exchange the shared chat password + a username for an identity cookie.

    The password is the perimeter ("may you be here at all"); the username is
    namespacing ("which memories/schedules/thread"). Anyone holding the
    password can still claim any name — set CHAT_ALLOWED_USERS to close that,
    or see the per-user invite-token design in the plan's open questions.
    """
    chat_password = getattr(request.app.state, "chat_password", None)
    form = await request.form()
    username = (form.get("username") or "").strip()
    password = form.get("password") or ""

    if not chat_password:
        audit_log("chat_login", result="disabled")
        return templates.TemplateResponse(request, "chat_login.html", {
            "error": "Chat is disabled: set CHAT_PASSWORD (or ADMIN_PANEL_PASSWORD).",
        }, status_code=503)

    if not secrets.compare_digest(password, chat_password):
        audit_log("chat_login", result="bad_password",
                  attempted_user=re.sub(r"[^a-zA-Z0-9_-]", "", username)[:64],
                  client=request.client.host if request.client else None)
        return templates.TemplateResponse(request, "chat_login.html", {
            "error": "Wrong password.",
        }, status_code=401)

    safe = re.sub(r"[^a-zA-Z0-9_-]", "", username)[:64]
    if not safe:
        return templates.TemplateResponse(request, "chat_login.html", {
            "error": "Pick a username with at least one letter or number.",
        })
    if CHAT_ALLOWED_USERS and safe not in CHAT_ALLOWED_USERS:
        audit_log("chat_login", result="not_allowlisted", user_id=safe)
        return templates.TemplateResponse(request, "chat_login.html", {
            "error": "That username is not on the allowlist for this instance.",
        }, status_code=403)

    response = RedirectResponse(url="/chat", status_code=303)
    _set_chat_cookie(response, safe)
    audit_log("chat_login", user_id=safe, result="ok")
    return response

### `admin_panel.py:448-454, 522-530, 642-644`

Thread the web thread_id through both agent entry points and through /chat/forget. Note chat_forget needs functools.partial (or a lambda) because run_in_executor takes positional args only.

# chat_send._invoke_agent (was 448-454)
        return ask_stuff(
            message,
            source,
            user,
            user_image_paths=pending_images or None,
            schedule_manager=schedule_manager,
            thread_id=_chat_thread_id(user),
        )

# chat_stream._invoke_agent (was 522-530): add the same
            thread_id=_chat_thread_id(user),

# chat_forget (was 642-644)
    removed = await asyncio.get_running_loop().run_in_executor(
        None, functools.partial(privacy.forget_conversation, user,
                                thread_id=_chat_thread_id(user)),
    )

### `admin_panel.py:606-621`

Serve assets with an extension-to-Content-Type allowlist, nosniff, a sandboxing CSP, and an ownership check on temp_images. Deliberately NOT Content-Disposition: attachment — the thumbnails and inline generated images must render; an allowlisted image Content-Type + nosniff + `default-src 'none'; sandbox` achieves the same safety without breaking them.

# module-level, next to _chat_asset_roots
_CHAT_ASSET_CONTENT_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}
_CHAT_ASSET_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; sandbox; frame-ancestors 'none'",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "private, max-age=60",
}


async def chat_asset(request: Request) -> Response:
    """Serve a file from ./output or ./temp_images to a signed-in chat user.

    The Content-Type is derived from an extension allowlist rather than left to
    Starlette's guess, so a file that somehow lands here with an executable
    extension is a 404 instead of a same-origin HTML document.
    """
    user = _chat_user(request)
    if not user:
        return Response(status_code=401, content="Not signed in.")
    rel = request.path_params["path"]
    if not rel:
        return Response(status_code=404)
    media_type = _CHAT_ASSET_CONTENT_TYPES.get(os.path.splitext(rel)[1].lower())
    if media_type is None:
        return Response(status_code=404)
    safe_user = re.sub(r"[^a-zA-Z0-9_-]", "_", user)
    for root in _chat_asset_roots():
        root_parent = os.path.dirname(root)
        full = os.path.abspath(os.path.join(root_parent, rel))
        if full == root or full.startswith(root + os.sep):
            if not os.path.isfile(full):
                continue
            # temp_images holds uploads, named "<safe_user>_<ts>_<name>" by
            # chat_upload_image — only the uploader may fetch their own.
            # ./output holds images Fritz generated; those are named
            # generated_image-<ts>.png with no owner, so they stay readable by
            # any signed-in chat user (see risks).
            if os.path.basename(root) == "temp_images" and \
                    not os.path.basename(full).startswith(f"{safe_user}_"):
                audit_log("chat_asset_denied", user_id=user, path=rel)
                return Response(status_code=404)
            return FileResponse(full, media_type=media_type,
                                headers=dict(_CHAT_ASSET_HEADERS))
    return Response(status_code=404)

### `admin_panel.py:667-674 (add _safe_stem alongside) and 690-714 (rewrite the validation half of chat_upload_image)`

Decode-based sniffing with Pillow, canonical extension derived from the sniffed format. Order is: cheap header check, then size cap, then decode. Keeping the size cap before the decode both preserves the existing 413 test and bounds what Pillow is asked to parse. `_safe_filename` stays for the document path.

def _safe_stem(name: str) -> str:
    """Sanitised basename with any client-supplied extension discarded. The
    caller appends a canonical extension derived from the sniffed content —
    never from the upload's filename or declared Content-Type."""
    base = os.path.basename(name or "upload")
    stem, dot, _ext = base.rpartition(".")
    stem = stem if dot else base
    return re.sub(r"[^a-zA-Z0-9._-]", "_", stem)[:80] or "upload"


def _sniff_image_format(raw: bytes) -> str | None:
    """Return the Pillow format name ("PNG", "JPEG", "GIF", "WEBP") if `raw` is
    genuinely one of CHAT_ALLOWED_IMAGE_FORMATS, else None.

    The declared Content-Type is not evidence: curl (or any script) can send
    `image/png` with an HTML or SVG body. Verified that Image.open raises
    UnidentifiedImageError on '<html><script>...' and on an SVG document.
    verify() parses structure without decoding pixels, so a decompression bomb
    is not materialised here; the size cap upstream is the other half.
    """
    if not raw or _PILImage is None:
        return None
    try:
        with _PILImage.open(io.BytesIO(raw)) as img:
            fmt = img.format
            img.verify()
    except Exception:
        return None
    return fmt if fmt in CHAT_ALLOWED_IMAGE_FORMATS else None


# inside chat_upload_image, replacing 690-714:
    content_type = (getattr(upload, "content_type", "") or "").lower().split(";")[0].strip()
    if content_type not in CHAT_ALLOWED_IMAGE_TYPES:
        audit_log("chat_upload_image", user_id=user, result="rejected_type",
                  content_type=content_type)
        return JSONResponse({"error": f"unsupported image type '{content_type}'"}, status_code=415)

    raw = await upload.read()
    if len(raw) > CHAT_IMAGE_UPLOAD_MAX_BYTES:
        audit_log("chat_upload_image", user_id=user, result="rejected_size",
                  bytes=len(raw), cap=CHAT_IMAGE_UPLOAD_MAX_BYTES)
        return JSONResponse({"error": f"image exceeds {CHAT_IMAGE_UPLOAD_MAX_BYTES} byte cap"}, status_code=413)

    fmt = _sniff_image_format(raw)
    if fmt is None:
        audit_log("chat_upload_image", user_id=user, result="rejected_content",
                  declared_type=content_type, bytes=len(raw))
        return JSONResponse({"error": "file content is not a supported image"}, status_code=415)
    canonical_ext, _canonical_mime = CHAT_ALLOWED_IMAGE_FORMATS[fmt]

    os.makedirs("temp_images", exist_ok=True)
    ts = int(time.time())
    safe_name = f"{_safe_stem(getattr(upload, 'filename', 'upload'))}.{canonical_ext}"
    safe_user = re.sub(r"[^a-zA-Z0-9_-]", "_", user)
    target = os.path.abspath(os.path.join("temp_images", f"{safe_user}_{ts}_{safe_name}"))

### `admin_panel.py:726-783 (delete) and a new handler placed after reindex_document_action (~line 300)`

Delete chat_upload_document entirely and re-home the capability on the Basic-auth'd admin surface as POST /documents/upload. This is what actually removes is_admin from the chat surface: instead of trusting a self-asserted cookie name, the route is protected by the middleware that already exists. Body logic (extension check, size cap, path-escape guard) is lifted verbatim.

async def upload_document_action(request: Request) -> Response:
    """Accept a document and drop it into DOC_FOLDER for the watchdog.

    Lives on the admin surface, not /chat: DOC_FOLDER is shared knowledge and
    the chat cookie carries a self-asserted name, not a credential.
    _BasicAuthMiddleware has already required ADMIN_PANEL_PASSWORD by the time
    we get here.
    """
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return RedirectResponse(url="/documents", status_code=303)

    safe_name = _safe_filename(getattr(upload, "filename", "upload"))
    try:
        from document_engine import SUPPORTED_EXTENSIONS
    except Exception:
        SUPPORTED_EXTENSIONS = (".docx", ".pdf", ".xlsx", ".csv", ".txt", ".md")
    _, dot, ext = safe_name.rpartition(".")
    if not dot or f".{ext.lower()}" not in SUPPORTED_EXTENSIONS:
        audit_log("admin_upload_document", admin=_admin(request),
                  result="bad_ext", filename=safe_name)
        return RedirectResponse(url="/documents", status_code=303)

    raw = await upload.read()
    if len(raw) > CHAT_DOC_UPLOAD_MAX_BYTES:
        audit_log("admin_upload_document", admin=_admin(request),
                  result="rejected_size", bytes=len(raw), cap=CHAT_DOC_UPLOAD_MAX_BYTES)
        return RedirectResponse(url="/documents", status_code=303)

    os.makedirs(DOC_FOLDER, exist_ok=True)
    target = os.path.abspath(os.path.join(DOC_FOLDER, safe_name))
    doc_root = os.path.abspath(DOC_FOLDER)
    if not (target == doc_root or target.startswith(doc_root + os.sep)):
        audit_log("admin_upload_document", admin=_admin(request),
                  result="path_escape", filename=safe_name)
        return RedirectResponse(url="/documents", status_code=303)
    with open(target, "wb") as f:
        f.write(raw)
    audit_log("admin_upload_document", admin=_admin(request),
              result="ok", path=target, bytes=len(raw))
    return RedirectResponse(url="/documents", status_code=303)

### `admin_panel.py:788-826 and 829-858`

create_app grows a chat_password argument and the security middleware; the /chat/upload/document route is removed and /documents/upload added. start_admin_panel passes CHAT_PASSWORD and warns when the chat surface is disabled.

def create_app(password: str, schedule_manager=None,
               chat_password: str | None = None) -> Starlette:
    routes = [
        ...
        Route("/documents/reindex", reindex_document_action,
              methods=["POST"], name="reindex_document"),
        Route("/documents/upload", upload_document_action,
              methods=["POST"], name="upload_document"),
        ...
        # DELETE: Route("/chat/upload/document", chat_upload_document, ...)
    ]
    app = Starlette(
        routes=routes,
        middleware=[
            # Outermost: sets request.state.csp_nonce before auth runs and
            # stamps headers on 401s too.
            Middleware(_SecurityHeadersMiddleware),
            Middleware(_BasicAuthMiddleware, password=password),
        ],
    )
    app.state.schedule_manager = schedule_manager
    app.state.chat_password = chat_password or CHAT_PASSWORD
    return app


# start_admin_panel
    app = create_app(ADMIN_PANEL_PASSWORD, schedule_manager=schedule_manager,
                     chat_password=CHAT_PASSWORD)
    ...
    if not CHAT_PASSWORD:
        logger.warning("CHAT_PASSWORD unset — /chat will refuse every login.")

### `admin_templates/chat.html:151-158, 162, 177-178, 209-224, 232-234`

Remove the admin-only document-upload UI (element, JS uploader, and its listener) now that the route is gone, and nonce the inline script so the new CSP admits it.

<!-- delete the whole {% if is_admin %} ... {% endif %} block (151-158) -->

<script nonce="{{ csp_nonce(request) }}">

// delete: const docInput  = document.getElementById("chat-doc-input");   (177)
// delete: const docStatus = document.getElementById("chat-doc-status");  (178)
// delete: function uploadDocument(file) { ... }                          (209-224)
// delete: if (docInput) docInput.addEventListener("change", ...)         (232-234)

### `admin_templates/chat_login.html:18-33`

Add the password field and replace the "No password — anyone reaching this port can pick any name" disclaimer with an accurate one.

    <form method="post" action="/chat/login">
        <label for="username" ...>Username</label>
        <input type="text" id="username" name="username" autofocus
               placeholder="e.g. nick" required maxlength="64" ...>
        <label for="password" ...>Chat password</label>
        <input type="password" id="password" name="password" required
               autocomplete="current-password" ...>
        <button type="submit">Start chatting</button>
    </form>

    <p class="muted" ...>
        The password is the same one that guards the admin panel unless the
        operator set CHAT_PASSWORD separately. Everyone who holds it can pick
        any username, so the username namespaces your memories and schedules —
        it does not isolate you from other password-holders. Web conversations
        are kept in their own thread, separate from Discord.
    </p>

### `admin_templates/documents.html:5-8 (inside the card, above the table)`

Upload form replacing the chat-side control.

<div class="card">
    <h2>Documents in <code>{{ doc_folder }}</code> <span class="pill">{{ docs | length }}</span></h2>
    <form method="post" action="/documents/upload" enctype="multipart/form-data"
          style="margin-bottom: 1rem;">
        <label class="muted">Add to shared docs:</label>
        <input type="file" name="file" accept=".pdf,.docx,.xlsx,.csv,.txt,.md" required>
        <button type="submit">Upload</button>
    </form>

### `chat_auth.py:1-17`

The module docstring currently asserts the opposite of what will be true. Correct it — it is the first thing a future reader trusts.

"""HMAC-signed cookies for the local chat UI.

The chat surface (`/chat/*`) has its own session mechanism, separate from the
admin panel's HTTP Basic auth. A user authenticates once at `/chat/login` with
the shared chat password (`CHAT_PASSWORD`, defaulting to
`ADMIN_PANEL_PASSWORD`) and picks a username; this module mints the signed
cookie that carries that username for the next 30 days.

The cookie is a *session* token, not a credential: it proves the bearer passed
the password check and cannot forge or edit the name inside. It does not
distinguish two people who both know the password — set `CHAT_ALLOWED_USERS`
to restrict which names may be claimed, or move to per-user invite tokens.
"""

### `requirements.txt:291-292 (append)`

Pin nh3. Verified `pip install --dry-run nh3` resolves nh3-0.3.6-cp38-abi3-win_amd64.whl (abi3, no Rust toolchain); manylinux abi3 wheels cover the ubuntu-latest CI runner.

# Server-side markdown rendering for Fritz's chat replies (Phase web-chat-3).
Markdown>=3.5
# HTML sanitiser applied to that markdown before it reaches the DOM
# (Phase web-chat-5). Rust-backed (ammonia) with cp38-abi3 wheels for
# win_amd64 / manylinux / macos, so CI needs no toolchain.
nh3>=0.3,<0.4

### `.env.example:62-69 (extend the admin-panel block)`

Document every new knob, and backfill the three chat knobs that already exist in code and README but were never added here (CHAT_COOKIE_SECRET, CHAT_IMAGE_UPLOAD_MAX_BYTES, CHAT_DOC_UPLOAD_MAX_BYTES).

# ----- Web chat (:8001/chat) -----
# Shared password required to sign in to the chat UI. Defaults to
# ADMIN_PANEL_PASSWORD. If BOTH are unset, /chat refuses every login.
# CHAT_PASSWORD=
# Comma-separated allowlist of usernames that may be claimed at /chat/login.
# Blank = any name once the password checks out.
# CHAT_ALLOWED_USERS=nick,alice
# Mark the chat cookie Secure. Leave off for plain-http SSH tunnels; turn on
# if you terminate TLS in front of the panel.
# CHAT_COOKIE_SECURE=false
# By default the web chat keeps its own LangGraph thread ("web-<user>") so a
# chat session cannot read or append to your Discord conversation. Set true to
# restore the pre-hardening shared thread.
# CHAT_SHARE_DISCORD_THREAD=false
# CHAT_THREAD_PREFIX=web
# HMAC secret for the identity cookie. Auto-generated into .chat_cookie_secret
# (gitignored) if unset; set explicitly for stable cookies across redeploys.
# CHAT_COOKIE_SECRET=
# CHAT_IMAGE_UPLOAD_MAX_BYTES=10485760
# CHAT_DOC_UPLOAD_MAX_BYTES=10485760

### `README.md:312-352`

The Chat UI section currently documents the exact behaviour being removed. Rewrite: line 318 ("use the same Discord username ... continue the same conversation thread") is now false by default; the line-320 trust-model blockquote must be replaced; the line-335 "Document upload (admin only)" table row points at /documents; the 341-346 config table gains the new knobs.

> **Trust model.** `/chat` now requires a shared password (`CHAT_PASSWORD`,
> defaulting to `ADMIN_PANEL_PASSWORD`). Everyone who holds it can still claim
> any username, so the username namespaces memories and schedules rather than
> isolating people from each other — use `CHAT_ALLOWED_USERS` if you want that
> closed. Keep the port on localhost and tunnel over SSH; there is no TLS here.

... | **Conversation thread** | The web chat keeps its own thread (`web-<user>`),
separate from Discord, so a web session cannot read your Discord history. Set
`CHAT_SHARE_DISCORD_THREAD=true` for the old shared-thread behaviour. |

... | **Document upload** | Moved to the admin panel's **Documents** page,
behind `ADMIN_PANEL_PASSWORD`. |

### `CHANGELOG.md:8 (add a ### Security block under ## [Unreleased])`

Phase-style entry matching the existing 'Web chat — Phase N' convention already used at lines 30-38.

## [Unreleased]

### Security
- **Web chat — Phase 5: authentication and XSS hardening.**
  - `/chat/login` now requires a shared password (`CHAT_PASSWORD`, falling back to `ADMIN_PANEL_PASSWORD`). With neither set the chat surface refuses every login instead of minting free identities. Failed attempts are audited with the attempted username and client address. Optional `CHAT_ALLOWED_USERS` restricts which names may be claimed.
  - **Breaking:** the web chat now uses its own LangGraph thread (`web-<user>`), so it no longer reads or writes the Discord conversation checkpoint. Existing web conversations start fresh; set `CHAT_SHARE_DISCORD_THREAD=true` to restore the old shared thread. `ask_stuff` gained a keyword-only `thread_id`; `privacy.forget_conversation` gained an optional `thread_id`.
  - Image uploads are validated by decoding them with Pillow, and stored under an extension derived from the sniffed format — a client that declares `image/png` while sending HTML is rejected, and a file named `x.html` is stored as `x.png`.
  - `/chat/assets/*` serves only allowlisted image content-types, with `X-Content-Type-Options: nosniff`, `Content-Security-Policy: default-src 'none'; sandbox`, and an uploader-ownership check on `./temp_images`.
  - Fritz's markdown replies are sanitised with `nh3` before reaching `| safe` / `innerHTML`. New nonce-based `script-src` CSP on every panel response as defence in depth.
  - **Breaking:** `POST /chat/upload/document` removed. Document upload moved to `POST /documents/upload` on the Basic-auth'd admin panel; `fritz_utils.is_admin` is no longer consulted anywhere on the chat surface.

## Steps

1. Fix the local env first: `.venv/Scripts/python -m pip install python-multipart nh3 ruff`. Confirm the baseline before touching code — `pytest tests/test_admin_panel.py tests/test_chat_auth.py -q` must go from 30 failed / 48 passed to 0 failed. Any failure that survives this step is pre-existing and unrelated to this work.
2. Verify nh3's default allowlist keeps what the chat needs, before writing code that depends on it: run `nh3.clean(markdown.markdown("**b**\n\n```py\nprint(1)\n```\n\n| a |\n|---|\n| 1 |\n\n<script>x</script><img src=x onerror=y>[l](javascript:1)", extensions=['fenced_code','tables','nl2br']))`. Expect <strong>, <pre><code>, the full table, and NO script/onerror/javascript:. `class="language-py"` being stripped is fine — chat.html:39 styles `pre code` by element, not class. If tables or <pre> do not survive, pass an explicit `tags=` set instead of relying on defaults.
3. Commit 1 — sanitiser. Add `nh3>=0.3,<0.4` to requirements.txt; add the nh3 + `html as html_lib` imports to admin_panel.py; rewrite `_render_markdown` (admin_panel.py:318-323). Add tests `test_script_tag_is_stripped`, `test_onerror_attribute_is_stripped`, `test_javascript_url_is_stripped` to `tests/test_admin_panel.py::TestRenderMarkdown`. The existing `test_bold_renders_to_strong` and `test_code_fence_renders_pre_code` are the regression guard that the sanitiser is not too aggressive.
4. Commit 2 — CSP + security headers. Add `_csp_nonce` and `templates.env.globals["csp_nonce"]` near admin_panel.py:57-58; add `_SecurityHeadersMiddleware`; register it as the FIRST entry in create_app's middleware list so it wraps `_BasicAuthMiddleware`; nonce the `<script>` at chat.html:162. Add `TestSecurityHeaders` asserting the CSP/nosniff headers appear on `GET /` (401) and `GET /chat`, and that the rendered chat page's script tag carries a nonce matching the header.
5. Commit 3 — image sniffing. Add `_safe_stem` and `_sniff_image_format`; add `CHAT_ALLOWED_IMAGE_FORMATS` to fritz_utils.py; rewrite the validation half of `chat_upload_image` (admin_panel.py:690-714). Replace the `_TINY_PNG` fixture at tests/test_admin_panel.py:738 with a real 1x1 PNG and add `_TINY_GIF`. Add `test_rejects_html_body_declared_as_png` and `test_extension_is_derived_from_content_not_filename`.
6. Commit 4 — asset serving. Add `_CHAT_ASSET_CONTENT_TYPES` / `_CHAT_ASSET_HEADERS`; rewrite `chat_asset` (admin_panel.py:606-621). Update `TestChatAsset::test_serves_existing_file_under_output` (tests/test_admin_panel.py:703-717) to write a real PNG named `alice_0_marker.png` under temp_images rather than a `.txt` under output. Add `test_non_image_extension_returns_404`, `test_other_users_upload_returns_404`, `test_asset_response_carries_nosniff_and_sandbox_csp`.
7. Commit 5 — auth gate. Add `CHAT_PASSWORD` / `CHAT_ALLOWED_USERS` / `CHAT_COOKIE_SECURE` to fritz_utils.py; add the `chat_password` argument to `create_app` and `app.state.chat_password`; rewrite `chat_login` (admin_panel.py:400-413); add `secure=CHAT_COOKIE_SECURE` to `_set_chat_cookie`; wire `start_admin_panel`. In tests/test_admin_panel.py add `def _login(client, username="alice", password=PASSWORD)` and update `_build_client` (lines 65-67) to pass `chat_password=PASSWORD`; mechanically replace all 27 `client.post("/chat/login", data={"username": ...})` call sites with `_login(...)`. Add `TestChatLoginPassword` with `test_missing_password_returns_401`, `test_wrong_password_returns_401_and_sets_no_cookie`, `test_no_chat_password_configured_returns_503`, `test_username_not_in_allowlist_returns_403`.
8. Commit 6 — thread separation. Add the keyword-only `thread_id` to `ask_stuff` (mister_fritz.py:523-560); add the optional `thread_id` to `privacy.forget_conversation` (privacy.py:61-71); add `CHAT_SHARE_DISCORD_THREAD` / `CHAT_THREAD_PREFIX` to fritz_utils.py; add `_chat_thread_id` to admin_panel.py and use it at :353, :448-454, :522-530, :642-644. Update `TestChatForget::test_authed_post_calls_forget_conversation_and_redirects` (tests/test_admin_panel.py:675) from `fc.assert_called_once_with("alice")` to `fc.assert_called_once_with("alice", thread_id="web-alice")`. Add `test_chat_send_passes_web_thread_id` and `test_share_discord_thread_flag_restores_discord_thread`; add a `forget_conversation(user, thread_id=...)` case to tests/test_privacy.py.
9. Commit 7 — remove is_admin from the chat surface. Delete `chat_upload_document` (admin_panel.py:726-783) and its route; delete `"is_admin": fritz_utils.is_admin(user)` (admin_panel.py:394); delete `import fritz_utils` (admin_panel.py:39) or ruff F401 fails CI; delete the `{% if is_admin %}` block and the doc-upload JS from chat.html; add `upload_document_action` + the `/documents/upload` route + the form in documents.html. Rewrite `tests/test_admin_panel.py::TestChatUploadDocument` (lines 813-866) as `TestDocumentUploadAction` against `/documents/upload` with `_auth_header()`, deleting the `patch.object(admin_panel.fritz_utils, "is_admin", ...)` at lines 816-818 (that attribute will no longer exist). Add `test_unauthed_upload_returns_401`.
10. Commit 8 — docs. Update chat_auth.py's docstring (lines 1-17), chat_login.html (18-33), .env.example (extend the admin-panel block, and backfill CHAT_COOKIE_SECRET / CHAT_IMAGE_UPLOAD_MAX_BYTES / CHAT_DOC_UPLOAD_MAX_BYTES, which are documented in README but missing from .env.example today), README.md Chat UI section (312-352), and CHANGELOG.md `## [Unreleased]` with a `### Security` block.
11. Final gate: `ruff check .` clean and `pytest tests/ --cov=. --cov-fail-under=60` green. Then manually drive the browser flow (see manualVerification) — the SSE stream under two stacked BaseHTTPMiddleware instances and the nonce'd inline script are the two things TestClient exercises differently from a real browser.

## Config and env changes

- CHAT_PASSWORD — new. Shared secret required at /chat/login. Defaults to ADMIN_PANEL_PASSWORD; if both are unset the chat surface refuses every login (fail closed). Document in .env.example and README.
- CHAT_ALLOWED_USERS — new. Comma-separated allowlist of usernames claimable at /chat/login. Blank (default) = any sanitised name. Mirrors the existing ADMIN_USERS parsing style at fritz_utils.py:214-218.
- CHAT_COOKIE_SECURE — new, default false. Sets the Secure flag on the chat cookie. Must stay false for plain-http SSH-tunnel access, which is the documented deployment.
- CHAT_SHARE_DISCORD_THREAD — new, default false. True restores the pre-hardening behaviour where the web chat and Discord share one LangGraph thread. This is the rollback switch for the thread split.
- CHAT_THREAD_PREFIX — new, default "web". Prefix for the web thread id, joined with a dash ("web-alice"). The dash is why the explicit-thread_id sanitiser in ask_stuff keeps [_-]: it makes collision with an alnum-only Discord thread impossible.
- CHAT_COOKIE_SECRET — existing (fritz_utils.py:130-153), documented in README but NOT in .env.example. Backfill.
- CHAT_IMAGE_UPLOAD_MAX_BYTES / CHAT_DOC_UPLOAD_MAX_BYTES — existing (fritz_utils.py:159-164), documented in README but NOT in .env.example. Backfill.
- requirements.txt — add nh3>=0.3,<0.4. No other new dependency: Pillow (12.0.0) is already pinned at line 171 via the SDXL stack, and python-multipart at line 290.

## Tests
### New

- tests/test_admin_panel.py::TestRenderMarkdown::test_script_tag_is_stripped — `_render_markdown("<script>alert(1)</script>")` contains no `<script`.
- tests/test_admin_panel.py::TestRenderMarkdown::test_onerror_attribute_is_stripped — `_render_markdown('<img src=x onerror=alert(1)>')` contains no `onerror`.
- tests/test_admin_panel.py::TestRenderMarkdown::test_javascript_url_is_stripped — `_render_markdown('[l](javascript:alert(1))')` contains no `javascript:`.
- tests/test_admin_panel.py::TestChatLoginPassword::{test_missing_password_returns_401, test_wrong_password_returns_401_and_sets_no_cookie, test_no_chat_password_configured_returns_503, test_username_not_in_allowlist_returns_403}. The 503 case builds the app via `admin_panel.create_app(PASSWORD, chat_password="")` with `admin_panel.CHAT_PASSWORD` patched to None; the allowlist case patches `admin_panel.CHAT_ALLOWED_USERS` to `frozenset({"nick"})`.
- tests/test_admin_panel.py::TestChatUploadImage::test_rejects_html_body_declared_as_png — POST `files={"file": ("x.png", b"<html><script>alert(1)</script></html>", "image/png")}` expects 415 and an audit `result="rejected_content"`. This is the direct regression test for finding (b).
- tests/test_admin_panel.py::TestChatUploadImage::test_extension_is_derived_from_content_not_filename — upload a real PNG named `evil.html`; assert the stashed pending path ends with `.png` and contains no `html`.
- tests/test_admin_panel.py::TestChatAsset::test_non_image_extension_returns_404 — write `output/x.html`, GET `/chat/assets/output/x.html`, expect 404.
- tests/test_admin_panel.py::TestChatAsset::test_other_users_upload_returns_404 — write `temp_images/bob_1_a.png`, sign in as alice, expect 404 and a `chat_asset_denied` audit entry.
- tests/test_admin_panel.py::TestChatAsset::test_asset_response_carries_nosniff_and_sandbox_csp — assert `x-content-type-options == "nosniff"`, `"sandbox" in content-security-policy`, and `content-type == "image/png"`.
- tests/test_admin_panel.py::TestSecurityHeaders::test_csp_and_nosniff_on_every_response (covering the 401 from `GET /` and the 200 from `GET /chat`) and ::test_chat_page_script_nonce_matches_header.
- tests/test_admin_panel.py::TestChatThreadIsolation::test_chat_send_passes_web_thread_id — capture the `thread_id` kwarg from a fake `ask_stuff`, assert `"web-alice"`; ::test_share_discord_thread_flag_restores_discord_thread with `patch.object(admin_panel, "CHAT_SHARE_DISCORD_THREAD", True)` asserting `"alice"`.
- tests/test_admin_panel.py::TestDocumentUploadAction::{test_unauthed_upload_returns_401, test_admin_upload_writes_to_doc_folder, test_bad_extension_is_rejected_and_audited} — replaces TestChatUploadDocument.
- tests/test_admin_panel.py::TestChatUploadDocumentRouteGone::test_chat_upload_document_no_longer_routes (404 or 405).
- tests/test_privacy.py — add `test_forget_conversation_honours_explicit_thread_id` to the existing conversation-checkpoint class (around line 103): insert rows for thread_id `web-alice` and `alice`, call `forget_conversation("alice", thread_id="web-alice")`, assert only the web rows went.

### Existing tests affected

- tests/test_admin_panel.py `_build_client` (lines 65-67) — must pass `chat_password=PASSWORD` to `create_app`.
- tests/test_admin_panel.py — all 27 `client.post("/chat/login", data={"username": ...})` call sites (lines 344, 352, 361, 375, 388, 411, 433, 451, 495, 525, 549, 573, 599, 624, 653, 669, 698, 705, 769, 787, 797, 808, 836, 846, 860, 878, 907) now need a `password` field. Introduce `_login(client, username="alice")` and replace them all.
- tests/test_admin_panel.py::TestChatLogin::test_post_login_sets_cookie_and_redirects (342), ::test_empty_username_renders_error (350), ::test_username_is_sanitised (358) — all three post to /chat/login directly and must supply the password. The empty-username case still expects 200 (that branch deliberately keeps its status).
- tests/test_admin_panel.py::TestChatBypassesAdminAuth::test_chat_landing_does_not_require_basic_auth (328) — still passes (GET /chat renders the login form without Basic auth), but its class name now means "the chat surface has its own password", not "the chat surface has no password". Rename to TestChatUsesItsOwnAuth.
- tests/test_admin_panel.py::TestChatForget::test_authed_post_calls_forget_conversation_and_redirects (667) — `fc.assert_called_once_with("alice")` at line 675 breaks; becomes `fc.assert_called_once_with("alice", thread_id="web-alice")`.
- tests/test_admin_panel.py::TestChatAsset::test_serves_existing_file_under_output (703-717) — writes `output/_admin_panel_test_marker.txt` and asserts `r.text == "hello world"`. Breaks on the new extension allowlist. Rewrite to a real PNG at `temp_images/alice_0_marker.png` and assert bytes + headers.
- tests/test_admin_panel.py `_TINY_PNG` fixture (line 738) — `b"\x89PNG\r\n\x1a\n" + b"\x00"*32` is NOT a decodable PNG; verified this session that `Image.open` raises UnidentifiedImageError on it. Replace with base64 `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC` (69 bytes, a real 1x1 PNG). Optional second-format fixture `_TINY_GIF` = `R0lGODdhAQABAIAAAAAAAAAAACwAAAAAAQABAAAIBAABBAQAOw==`.
- tests/test_admin_panel.py::TestChatUploadImage::test_happy_path_saves_file_and_stashes_pending (767) — breaks on the current fixture; fixed by the fixture swap above.
- tests/test_admin_panel.py::TestChatUploadImage::test_rejects_oversized_image (795) — patches the cap to 32 bytes and sends the fixture; the real PNG is 69 bytes so it still 413s, PROVIDED the size check stays ahead of the sniff. Keep that ordering.
- tests/test_admin_panel.py::TestChatUploadImage::test_rejects_unsupported_content_type (785) — unchanged behaviour (415), now rejected by the header check before the sniff even runs.
- tests/test_admin_panel.py::TestChatUploadDocument (813-866, 4 tests) — deleted wholesale. Its setUp at 816-818 does `patch.object(admin_panel.fritz_utils, "is_admin", ...)`, which raises AttributeError once `import fritz_utils` is removed from admin_panel.py:39.
- tests/test_admin_panel.py::TestPendingImagePlumbing (869-924) — the fake `ask_stuff` signatures end in `**_`, so the new `thread_id` kwarg is absorbed; only the `_login` helper swap is needed.
- tests/test_admin_panel.py::TestChatStreamSuccess / TestChatStreamError / TestChatStreamEmptyMessage / TestChatStreamDonePayload / TestChatStreamProgressEvents (492-642) — login helper only. They are also the canary for the second BaseHTTPMiddleware not breaking the hand-rolled SSE generator; watch them closely after commit 2.
- tests/test_chat_auth.py — no API change to chat_auth; all 13 tests stay green. Only the module docstring moves.
- tests/test_mister_fritz.py — verified it exercises `planner()` only and never calls `ask_stuff`; unaffected by the new parameter.

### Manual verification

- Start with both CHAT_PASSWORD and ADMIN_PANEL_PASSWORD unset: the panel does not start at all (existing gate at admin_panel.py:835). Set only ADMIN_PANEL_PASSWORD: /chat/login accepts it. Set CHAT_PASSWORD to something different: only the new value works.
- Log in as `alice` in the browser, send a message, refresh — history hydrates. Then `sqlite3 fritz.db "SELECT DISTINCT thread_id FROM checkpoints"` and confirm a `web-alice` row exists and the Discord `alice` row is untouched.
- With devtools open, send a message and confirm the SSE token/progress/done frames still arrive and the final bubble renders formatted HTML — this is what two stacked BaseHTTPMiddleware instances could break in a way TestClient will not show.
- Confirm no CSP violations in the console on /chat, /, /users, /documents. If the inline <style> blocks are reported, the style-src decision needs revisiting (it should not be — 'unsafe-inline' there is intentional).
- curl the sniffing bypass directly: `curl -b fritz_chat_id=<token> -F 'file=@evil.html;type=image/png' http://127.0.0.1:8001/chat/upload/image` must return 415, and nothing may appear in ./temp_images.
- Upload a real photo, confirm the thumbnail renders in the user bubble, then `curl -I` the asset URL and check for `content-type: image/png`, `x-content-type-options: nosniff`, and the sandbox CSP.
- Ask Fritz to reply with a literal `<script>alert(1)</script>` (e.g. "repeat this exactly, once in a code block and once outside one"). Neither rendering may execute; the outside-code-block one should simply vanish.
- Log in as `bob` in a private window and try to GET one of alice's `/chat/assets/temp_images/alice_*` URLs — expect 404 plus a `chat_asset_denied` line in audit.log.
- Upload a document from the admin panel's Documents page and confirm the watchdog ingests it (the old chat-side control is gone).

## Risks

- nh3 wheel availability in CI. Verified locally that `nh3-0.3.6-cp38-abi3-win_amd64.whl` resolves with no toolchain. CI is ubuntu-latest + Python 3.12; nh3 publishes abi3 manylinux wheels, but confirm before merging with `pip download nh3 --no-deps --only-binary :all: --python-version 3.12 --platform manylinux_2_17_x86_64 -d /tmp/x`. If a wheel is ever missing, `bleach` is the drop-in fallback and `html5lib` (requirements.txt:86) is already present.
- nh3's default allowlist could be stricter than expected and eat table markup or <pre>. Detected immediately by the existing `TestRenderMarkdown::test_code_fence_renders_pre_code`. Known and accepted: `class="language-py"` on fenced code is stripped, which is cosmetically invisible because chat.html:39 styles `pre code` by element.
- Existing web conversations reset. Anyone who has been chatting through /chat has their context in the Discord thread; after the split, `web-<user>` is empty and Fritz appears to have forgotten. Detected by the user saying so. Mitigations in order of preference: (a) ship it and say so in the CHANGELOG — the shared thread was the vulnerability; (b) set CHAT_SHARE_DISCORD_THREAD=true for a single-user deployment; (c) a one-time `INSERT INTO checkpoints SELECT ... WHERE thread_id='alice'` with thread_id rewritten to 'web-alice', which duplicates rather than moves and is only safe while the bot is stopped.
- Memories are still shared between Discord and web, by design — only the thread splits. Someone who guesses the chat password and claims your username still writes into your Chroma namespace via ask_stuff's `user_id`. This is the residual hole that CHAT_ALLOWED_USERS narrows and only per-user credentials (the identity-threads item) closes. Say so plainly in the README rather than implying the surface is now safe for untrusted users.
- ./output assets remain readable by any signed-in chat user. image_generator.py:208 names files `generated_image-{timestamp}.png` with no owner, so ownership cannot be enforced without a new registry. Accepted and noted in the README. If it matters, the fix is to record generated paths in the same per-user structure as `_pending_images`.
- Adding a second BaseHTTPMiddleware around the hand-rolled SSE generator. The Phase-2 comment at admin_panel.py:554-557 records that this code path is sensitive to event-loop ownership. The five TestChatStream* classes are the detector; run them right after commit 2, before building anything on top.
- Pillow decompression bomb via a crafted 10 MB PNG. Bounded two ways: the size cap runs before the decode, and verify() parses structure without materialising pixels. Pillow's own MAX_IMAGE_PIXELS guard raises, which _sniff_image_format's bare `except Exception` turns into a clean 415.
- Password brute force against /chat/login. The route is unthrottled and, unlike the admin panel, is a form rather than Basic auth. `secrets.compare_digest` prevents timing leaks and every failure is audited, but nothing rate-limits. Acceptable while bound to 127.0.0.1; if the port is ever exposed, add a per-IP failure counter. Detect by grepping audit.log for `result="bad_password"`.
- CSRF: the design leans entirely on `samesite="lax"` (already set at admin_panel.py:376) to keep cross-site POSTs to /chat/send, /chat/forget and /chat/upload/* from carrying the cookie. Correct for modern browsers, but it is not a token. Leave a comment in the code so a future change to `samesite="none"` is recognised as a security regression.

## Rollback
Each of the eight commits is independently revertable, and they are ordered so reverting a later one never breaks an earlier one. Two runtime escape hatches exist without a deploy: CHAT_SHARE_DISCORD_THREAD=true restores the pre-change shared LangGraph thread (the only change with data consequences), and CHAT_THREAD_PREFIX can be repointed at an existing prefix. There is deliberately NO flag to disable the password gate — that is the point of the item; an operator who needs the old passwordless behaviour must revert commit 5. The nh3 sanitiser degrades safely rather than failing: if the package is missing, _render_markdown escapes instead of rendering, so a bad install shows ugly output rather than executing script. The one-way door is the document-upload route move (commit 7): reverting it restores an is_admin check on a self-asserted name, so revert it only together with commit 5. No schema migration is involved — the thread split writes new thread_id values into the existing checkpoints/writes tables and touches no existing rows.

## Open questions for you to decide

- Ship the shared password now, or go straight to per-user invite tokens? Costing the token design: a `chat_users(username TEXT PRIMARY KEY, token_hash TEXT, created REAL, revoked INTEGER)` table (reuse sqlite_store.py or a small module beside workspace_store.py), an admin-panel page + POST route to mint and revoke, a `/chat/login?invite=<token>` exchange that binds the cookie to the named user, and a revocation check inside `_chat_user`. That is roughly another L on top of this item and would make CHAT_ALLOWED_USERS redundant. It is the only design that actually stops one friend from reading another's memories. Recommendation: ship the password now — it closes the anonymous-internet hole, which is the urgent part — and file the token work under identity-threads.
- Should ./output assets be per-user, given generated images carry no owner in the filename? Making them so means recording every generated path against the requesting user; the cheapest version extends the existing `_pending_images` pattern into a `_recent_assets: dict[str, set[str]]`, which is process-local and lost on restart. Worth it, or is 'any signed-in user can see any generated image' acceptable at this trust level?
- Keep the no-JS synchronous fallback at POST /chat/send? It doubles the number of places `ask_stuff` and `_render_markdown` are wired, and nothing in the test suite proves a real no-JS browser works. If it goes, chat_send (admin_panel.py:425-480) and its four tests disappear and the attack surface shrinks.
- Is CHAT_PASSWORD defaulting to ADMIN_PANEL_PASSWORD the right ergonomic? It means one secret to manage, but it also means everyone with chat access holds the admin-panel password — they just have not been told the panel exists. If that is unacceptable, drop the fallback and require CHAT_PASSWORD explicitly; chat is then off by default after upgrade, which is a louder but safer failure.
- If the panel is ever put behind TLS or a reverse proxy rather than an SSH tunnel, CHAT_COOKIE_SECURE must flip to true and _BasicAuthMiddleware becomes the weakest link (shared password over the network). Worth deciding now whether 'never expose this without a tunnel' is a documented constraint or something to enforce — e.g. refuse to start if the bind host is not 127.0.0.1 while CHAT_COOKIE_SECURE is off.
- Not verifiable statically: whether LangGraph's SqliteSaver treats thread_id as fully opaque. Nothing in the code suggests it parses the string, and `web-alice` is well within what SQLite stores, but the settling experiment is one manual run — send a web message, then `sqlite3 fritz.db "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id"` and confirm the row appears and reloads on refresh.
