"""
Tests for the read-only admin panel (Phase 9a).

We use Starlette's TestClient against the app built by create_app() so we
exercise routing, templating, and auth without spinning up uvicorn.
"""
import base64
import importlib
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ddgs (required transitively by privacy → workspace_store) is stubbed in
# tests/conftest.py before any test module is collected.

from starlette.testclient import TestClient  # noqa: E402

# Stand up a temp DOC_FOLDER before importing admin_panel so its module-level
# DOC_FOLDER constant points somewhere we control.
_TMP = Path(tempfile.mkdtemp())
_DOC_FOLDER = _TMP / "docs"
_DOC_FOLDER.mkdir()
(_DOC_FOLDER / "alpha.txt").write_text("hello", encoding="utf-8")
(_DOC_FOLDER / "beta.pdf").write_bytes(b"pdfdata")

_env = patch.dict(os.environ, {
    "DOC_FOLDER": str(_DOC_FOLDER),
    "SCHEDULE_DB": str(_TMP / "test_fritz.db"),
    "DB_NAME": str(_TMP / "test_fritz.db"),
    "WORKSPACES_ROOT": str(_TMP / "workspaces"),
    "ADMIN_PANEL_PASSWORD": "secret",
    # Deliberately different from ADMIN_PANEL_PASSWORD: there is no fallback
    # between them, and a test that shared one value would not notice if one
    # were reintroduced.
    "CHAT_PASSWORD": "chatsecret",
})
_env.start()

import fritz_utils  # noqa: E402
importlib.reload(fritz_utils)
import workspace_store  # noqa: E402
importlib.reload(workspace_store)
import privacy  # noqa: E402
importlib.reload(privacy)
import admin_panel  # noqa: E402
importlib.reload(admin_panel)


PASSWORD = "secret"
CHAT_PASSWORD = "chatsecret"


def _auth_header(password: str = PASSWORD) -> dict:
    encoded = base64.b64encode(f"admin:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _build_client(schedule_manager=None, chat_password="__default__") -> TestClient:
    kwargs = {} if chat_password == "__default__" else {"chat_password": chat_password}
    app = admin_panel.create_app(PASSWORD, schedule_manager=schedule_manager, **kwargs)
    return TestClient(app)


def _login(client: TestClient, username: str = "alice",
           password: str = CHAT_PASSWORD, **kwargs):
    """Obtain a chat identity cookie. /chat/login now needs a password, so
    every test that wants an authed chat session goes through here rather than
    repeating the form fields."""
    return client.post(
        "/chat/login", data={"username": username, "password": password}, **kwargs,
    )


class TestAuth(unittest.TestCase):
    def test_missing_auth_returns_401(self):
        client = _build_client()
        r = client.get("/")
        self.assertEqual(r.status_code, 401)
        self.assertIn("Basic", r.headers.get("www-authenticate", ""))

    def test_bad_password_returns_401(self):
        client = _build_client()
        r = client.get("/", headers=_auth_header("wrong"))
        self.assertEqual(r.status_code, 401)

    def test_correct_password_returns_200(self):
        client = _build_client()
        r = client.get("/", headers=_auth_header())
        self.assertEqual(r.status_code, 200)

    def test_garbage_auth_header_returns_401(self):
        client = _build_client()
        r = client.get("/", headers={"Authorization": "Basic !!!notbase64!!!"})
        self.assertEqual(r.status_code, 401)


_TEMPLATE_DIR = Path(admin_panel.__file__).parent / "admin_templates"


def _template(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _relative_luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    channels = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(fg: str, bg: str) -> float:
    a, b = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


class TestThemeSplit(unittest.TestCase):
    """The theme splits three ways (DECISIONS.md #12): a shared base, the admin
    palette, and the chat surface's dark academia. These guard the seams — a
    dropped include breaks every page at once, and the contrast fix is easy to
    undo by anyone 'restoring' the design's declared hexes."""

    def test_theme_includes_exist(self):
        for name in ("_theme_base.html", "_theme_admin.html", "_theme_chat.html"):
            self.assertTrue((_TEMPLATE_DIR / name).is_file(), name)

    def test_base_consumes_base_and_admin_themes(self):
        src = _template("base.html")
        self.assertIn('include "_theme_base.html"', src)
        self.assertIn('include "_theme_admin.html"', src)

    def test_admin_panel_does_not_get_the_chat_palette(self):
        # Decision 12: candle-lit purple buys nothing on a data grid, and the
        # faceted clip-paths fight the mobile `table { display: block }` rule.
        self.assertNotIn('include "_theme_chat.html"', _template("base.html"))

    def test_admin_muted_passes_aa_on_every_ground(self):
        # WAS #738291: 3.67:1 on --bg, 3.94:1 on --card, 3.30:1 on the pill
        # ground. `h2 { color: var(--muted) }` means this governed every section
        # heading on every admin page, so it was never merely decorative.
        src = _template("_theme_admin.html")
        match = re.search(r"--muted:\s*(#[0-9a-fA-F]{6})", src)
        self.assertIsNotNone(match, "--muted not found in the admin theme")
        muted = match.group(1)
        for ground, label in (("#f7f7f5", "--bg"), ("#ffffff", "--card"),
                              ("#ecebe5", "pill/code")):
            with self.subTest(ground=label):
                self.assertGreaterEqual(
                    _contrast(muted, ground), 4.5,
                    f"--muted {muted} fails AA against {label} ({ground})",
                )

    def test_chat_theme_uses_layered_tokens_not_literal_rgba(self):
        # Layer 1 channel triplets exist so an accent can be retuned in one
        # place instead of across ~41 hand-typed rgba() literals.
        src = _template("_theme_chat.html")
        self.assertIn("--amethyst-rgb:", src)
        self.assertIn("rgba(var(--amethyst-rgb)", src)

    def test_chat_theme_has_no_border_radius(self):
        # "No rounded corners anywhere; facets via clip-path" is the design
        # rule. Strip comments first — the prose explaining the rule naturally
        # names the property, and matching that would be a false positive.
        src = _template("_theme_chat.html")
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)      # CSS comments
        src = re.sub(r"\{#.*?#\}", "", src, flags=re.S)      # Jinja comments
        self.assertNotRegex(src, r"border-radius\s*:")

    def test_chat_theme_font_stacks_have_real_fallbacks(self):
        # The woff2 files are not in the repo; a bare "'EB Garamond', serif"
        # would drop straight to Times.
        src = _template("_theme_chat.html")
        self.assertIn("--font-body:", src)
        self.assertIn("Georgia", src)

    def test_reduced_motion_block_lives_in_the_shared_base(self):
        self.assertIn("prefers-reduced-motion", _template("_theme_base.html"))

    def test_focus_ring_survives_clip_path(self):
        # clip-path clips outline and outer box-shadow, so faceted controls
        # carry their ring on an unclipped .facet wrapper via :focus-within.
        self.assertIn(".facet:focus-within", _template("_theme_base.html"))


class TestStaticMount(unittest.TestCase):
    """Fonts/CSS at /static are public: gating them would render /chat/login
    unstyled behind the admin password prompt."""

    def test_static_is_exempt_from_basic_auth(self):
        # README.md exists in admin_static/; served without any auth header.
        client = _build_client()
        r = client.get("/static/README.md")
        self.assertNotEqual(r.status_code, 401)
        self.assertEqual(r.status_code, 200)

    def test_missing_static_file_is_404_not_401(self):
        # A miss must not fall through to the auth gate.
        client = _build_client()
        r = client.get("/static/does-not-exist.woff2")
        self.assertEqual(r.status_code, 404)


class TestOverviewPage(unittest.TestCase):
    def test_renders_version_and_uptime(self):
        client = _build_client()
        r = client.get("/", headers=_auth_header())
        self.assertEqual(r.status_code, 200)
        self.assertIn("Overview", r.text)
        self.assertIn(fritz_utils.__version__, r.text)


class TestUsersListPage(unittest.TestCase):
    def test_lists_users_from_workspaces_and_schedules(self):
        manager = MagicMock()
        manager.list_all_schedules.return_value = [
            {"user_id": "alice", "id": "s1", "prompt": "p", "schedule": "1h",
             "description": "", "created": "now"},
        ]
        # Give bob a workspace so he shows up too.
        workspace_store.enable_sandboxed("bob")

        client = _build_client(schedule_manager=manager)
        with patch.object(privacy, "export_memories", return_value=[]), \
             patch.object(privacy, "export_schedules", return_value=[]):
            r = client.get("/users", headers=_auth_header())
        self.assertEqual(r.status_code, 200)
        self.assertIn("alice", r.text)
        self.assertIn("bob", r.text)


class TestUserDetailPage(unittest.TestCase):
    def test_renders_user_data(self):
        manager = MagicMock()
        client = _build_client(schedule_manager=manager)
        fake_data = {
            "user_id": "alice",
            "memories": [{"id": "m1", "content": "hello world", "metadata": {}}],
            "schedules": [{"id": "s1", "prompt": "p", "schedule": "1h",
                          "description": "morning", "created": "now"}],
            "conversation_checkpoint_count": 7,
            "workspace_path": "/tmp/workspaces/alice",
        }
        with patch.object(privacy, "export_user_data", return_value=fake_data):
            r = client.get("/users/alice", headers=_auth_header())
        self.assertEqual(r.status_code, 200)
        self.assertIn("alice", r.text)
        self.assertIn("morning", r.text)  # schedule description rendered
        self.assertIn("7", r.text)         # checkpoint count rendered


class TestSchedulesPage(unittest.TestCase):
    def test_lists_all_schedules_with_user_id(self):
        manager = MagicMock()
        manager.list_all_schedules.return_value = [
            {"id": "s1", "user_id": "alice", "prompt": "p1",
             "schedule": "1h", "description": "", "created": "now"},
            {"id": "s2", "user_id": "bob", "prompt": "weather",
             "schedule": "0 9 * * *", "description": "morning", "created": "now"},
        ]
        client = _build_client(schedule_manager=manager)
        r = client.get("/schedules", headers=_auth_header())
        self.assertEqual(r.status_code, 200)
        self.assertIn("alice", r.text)
        self.assertIn("bob", r.text)
        self.assertIn("morning", r.text)

    def test_empty_state_renders(self):
        manager = MagicMock()
        manager.list_all_schedules.return_value = []
        client = _build_client(schedule_manager=manager)
        r = client.get("/schedules", headers=_auth_header())
        self.assertEqual(r.status_code, 200)
        self.assertIn("No schedules", r.text)


class TestDocumentsPage(unittest.TestCase):
    def test_lists_files_in_doc_folder(self):
        client = _build_client()
        r = client.get("/documents", headers=_auth_header())
        self.assertEqual(r.status_code, 200)
        self.assertIn("alpha.txt", r.text)
        self.assertIn("beta.pdf", r.text)


class TestHealthJsonRoute(unittest.TestCase):
    def test_returns_health_snapshot(self):
        client = _build_client()
        r = client.get("/health", headers=_auth_header())
        self.assertEqual(r.status_code, 200)
        body = r.json()
        # health_snapshot is a dict with at least "uptime_sec"
        self.assertIn("uptime_sec", body)


class TestStartAdminPanelGate(unittest.TestCase):
    def test_returns_none_when_password_unset(self):
        with patch.object(admin_panel, "ADMIN_PANEL_PASSWORD", None):
            result = admin_panel.start_admin_panel()
        self.assertIsNone(result)


class TestForgetUserMutation(unittest.TestCase):
    def test_post_calls_forget_all_and_redirects(self):
        manager = MagicMock()
        client = _build_client(schedule_manager=manager)
        fake_result = {"memories": 3, "conversation_rows": 1, "schedules": 2, "workspace_dropped": True}
        with patch.object(privacy, "forget_all", return_value=fake_result) as forget, \
             patch.object(admin_panel, "audit_log") as audit:
            r = client.post("/users/alice/forget", headers=_auth_header(),
                            follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/users")
        forget.assert_called_once_with("alice", manager)
        audit.assert_called_once()
        kwargs = audit.call_args.kwargs
        self.assertEqual(kwargs.get("target_user"), "alice")
        self.assertEqual(kwargs.get("result"), fake_result)

    def test_get_on_mutation_route_returns_405(self):
        client = _build_client()
        r = client.get("/users/alice/forget", headers=_auth_header())
        self.assertEqual(r.status_code, 405)

    def test_unauthed_post_still_returns_401(self):
        client = _build_client()
        r = client.post("/users/alice/forget", follow_redirects=False)
        self.assertEqual(r.status_code, 401)


class TestDisableWorkspaceMutation(unittest.TestCase):
    def test_post_drops_workspace_and_redirects_to_user(self):
        client = _build_client()
        with patch.object(privacy, "forget_workspace", return_value=True) as fw, \
             patch.object(admin_panel, "audit_log"):
            r = client.post("/users/alice/workspace/disable", headers=_auth_header(),
                            follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/users/alice")
        fw.assert_called_once_with("alice")


class TestCancelScheduleMutation(unittest.TestCase):
    def test_post_cancels_schedule_via_manager(self):
        manager = MagicMock()
        manager.list_all_schedules.return_value = [
            {"id": "sid1", "user_id": "alice", "prompt": "p", "schedule": "1h",
             "description": "", "created": "now"},
        ]
        manager.remove_schedule.return_value = True
        client = _build_client(schedule_manager=manager)
        with patch.object(admin_panel, "audit_log") as audit:
            r = client.post("/schedules/sid1/cancel", headers=_auth_header(),
                            follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/schedules")
        manager.remove_schedule.assert_called_once_with("sid1", "alice")
        self.assertEqual(audit.call_args.kwargs.get("target_user"), "alice")

    def test_unknown_schedule_id_does_not_call_remove(self):
        manager = MagicMock()
        manager.list_all_schedules.return_value = []
        client = _build_client(schedule_manager=manager)
        with patch.object(admin_panel, "audit_log"):
            r = client.post("/schedules/sid_missing/cancel", headers=_auth_header(),
                            follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        manager.remove_schedule.assert_not_called()


class TestReindexDocumentMutation(unittest.TestCase):
    def test_post_enqueues_existing_document(self):
        client = _build_client()
        fake_queue = MagicMock()
        fake_document_engine = MagicMock()
        fake_document_engine.INGESTION_QUEUE = fake_queue
        with patch.dict(sys.modules, {"document_engine": fake_document_engine}), \
             patch.object(admin_panel, "audit_log"):
            r = client.post("/documents/reindex",
                            data={"name": "alpha.txt"},
                            headers=_auth_header(), follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/documents")
        fake_queue.put.assert_called_once()
        action, path = fake_queue.put.call_args.args[0]
        self.assertEqual(action, "update")
        self.assertTrue(path.endswith("alpha.txt"))

    def test_rejects_path_outside_doc_folder(self):
        client = _build_client()
        fake_queue = MagicMock()
        fake_document_engine = MagicMock()
        fake_document_engine.INGESTION_QUEUE = fake_queue
        with patch.dict(sys.modules, {"document_engine": fake_document_engine}), \
             patch.object(admin_panel, "audit_log") as audit:
            r = client.post("/documents/reindex",
                            data={"name": "../../../etc/passwd"},
                            headers=_auth_header(), follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        fake_queue.put.assert_not_called()
        self.assertEqual(audit.call_args.kwargs.get("error"), "path-escape")

    def test_missing_name_field_audits_error_and_does_not_enqueue(self):
        client = _build_client()
        fake_queue = MagicMock()
        fake_document_engine = MagicMock()
        fake_document_engine.INGESTION_QUEUE = fake_queue
        with patch.dict(sys.modules, {"document_engine": fake_document_engine}), \
             patch.object(admin_panel, "audit_log") as audit:
            r = client.post("/documents/reindex",
                            data={"name": ""},
                            headers=_auth_header(), follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        fake_queue.put.assert_not_called()
        self.assertEqual(audit.call_args.kwargs.get("error"), "missing-name")


class TestAdminUsernameInAudit(unittest.TestCase):
    def test_basic_auth_username_appears_in_audit_log(self):
        manager = MagicMock()
        client = _build_client(schedule_manager=manager)
        with patch.object(privacy, "forget_all", return_value={}), \
             patch.object(admin_panel, "audit_log") as audit:
            # Build a custom header with a specific username.
            encoded = base64.b64encode(f"nick:{PASSWORD}".encode()).decode()
            r = client.post("/users/alice/forget",
                            headers={"Authorization": f"Basic {encoded}"},
                            follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(audit.call_args.kwargs.get("admin"), "nick")


# ── /chat (Phase web-chat-1) ────────────────────────────────────────────────

class TestChatBypassesAdminAuth(unittest.TestCase):
    """The chat surface has its own cookie-based identity — Basic auth
    should NOT be required to reach /chat or its sub-routes."""

    def test_chat_landing_does_not_require_basic_auth(self):
        client = _build_client()
        r = client.get("/chat")
        # Login form rendered (200), not 401.
        self.assertEqual(r.status_code, 200)
        self.assertIn("Sign in to chat", r.text)

    def test_admin_pages_still_require_basic_auth(self):
        client = _build_client()
        r = client.get("/")
        self.assertEqual(r.status_code, 401)


class TestChatLogin(unittest.TestCase):
    def test_post_login_sets_cookie_and_redirects(self):
        client = _build_client()
        r = _login(client, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/chat")
        self.assertIn("fritz_chat_id", r.cookies)

    def test_empty_username_renders_error(self):
        client = _build_client()
        r = _login(client, username="  ", follow_redirects=False)
        # Re-renders login form with error message (200, not redirect).
        self.assertEqual(r.status_code, 200)
        self.assertIn("at least one letter", r.text)

    def test_username_is_sanitised(self):
        # Path-like or punctuation-heavy usernames get stripped to safe chars.
        client = _build_client()
        r = _login(client, username="../bad/name", follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        # The set cookie should contain "badname" (slashes + dots stripped).
        import chat_auth
        from fritz_utils import CHAT_COOKIE_SECRET
        token = r.cookies.get("fritz_chat_id")
        self.assertEqual(chat_auth.verify_cookie(token, CHAT_COOKIE_SECRET), "badname")


class TestChatPasswordGate(unittest.TestCase):
    """The password is the perimeter: before this, anyone who could reach the
    port could type any username and read that person's conversation."""

    def test_missing_password_is_rejected(self):
        client = _build_client()
        with patch.object(admin_panel, "audit_log"):
            r = client.post("/chat/login", data={"username": "alice"},
                            follow_redirects=False)
        self.assertEqual(r.status_code, 401)
        self.assertNotIn("fritz_chat_id", r.cookies)

    def test_wrong_password_is_rejected(self):
        client = _build_client()
        with patch.object(admin_panel, "audit_log"):
            r = _login(client, password="wrong", follow_redirects=False)
        self.assertEqual(r.status_code, 401)
        self.assertNotIn("fritz_chat_id", r.cookies)

    def test_bad_password_attempt_is_audited(self):
        client = _build_client()
        with patch.object(admin_panel, "audit_log") as audit:
            _login(client, password="wrong", follow_redirects=False)
        kwargs = audit.call_args.kwargs
        self.assertEqual(kwargs["result"], "bad_password")
        self.assertEqual(kwargs["attempted_user"], "alice")

    def test_correct_password_still_works(self):
        client = _build_client()
        r = _login(client, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("fritz_chat_id", r.cookies)

    def test_chat_is_disabled_when_no_password_configured(self):
        # Fail closed rather than minting free identities.
        client = _build_client(chat_password=None)
        with patch.object(admin_panel, "audit_log"):
            r = _login(client, follow_redirects=False)
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("fritz_chat_id", r.cookies)
        self.assertIn("CHAT_PASSWORD", r.text)

    def test_admin_password_is_not_accepted_for_chat(self):
        # DECISIONS #2b: no fallback between the two secrets. Giving someone
        # chat access must never hand them the admin panel's password.
        client = _build_client()
        with patch.object(admin_panel, "audit_log"):
            r = _login(client, password=PASSWORD, follow_redirects=False)
        self.assertEqual(r.status_code, 401)

    def test_allowlist_blocks_unlisted_username(self):
        client = _build_client()
        with patch.object(admin_panel, "CHAT_ALLOWED_USERS", frozenset({"alice"})), \
             patch.object(admin_panel, "audit_log"):
            r = _login(client, username="mallory", follow_redirects=False)
        self.assertEqual(r.status_code, 403)
        self.assertNotIn("fritz_chat_id", r.cookies)

    def test_allowlist_admits_listed_username(self):
        client = _build_client()
        with patch.object(admin_panel, "CHAT_ALLOWED_USERS", frozenset({"alice"})):
            r = _login(client, username="alice", follow_redirects=False)
        self.assertEqual(r.status_code, 303)

    def test_login_page_no_longer_claims_there_is_no_password(self):
        client = _build_client()
        self.assertNotIn("No password", client.get("/chat").text)


class TestChatLogout(unittest.TestCase):
    def test_logout_clears_cookie(self):
        client = _build_client()
        # First log in.
        _login(client)
        # Then log out.
        r = client.post("/chat/logout", follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        # Logout sets a max-age=0 / expired cookie; httpx may strip it from
        # the jar. Either way, the next /chat should land on the login form.
        r2 = client.get("/chat")
        self.assertIn("Sign in to chat", r2.text)


class TestChatPageWithCookie(unittest.TestCase):
    def test_authed_user_sees_chat_ui(self):
        client = _build_client()
        _login(client)
        r = client.get("/chat")
        self.assertEqual(r.status_code, 200)
        # Identity moved into the presence row but is still on the page.
        self.assertIn("alice", r.text)
        # The placeholder changed with the restyle: "Type a message..." became
        # "Address the butler…". This assertion is the reason the copy change
        # is not silent.
        self.assertIn("Address the butler", r.text)

    def test_tampered_cookie_renders_login(self):
        client = _build_client()
        client.cookies.set("fritz_chat_id", "alice:9999999999:deadbeef")
        r = client.get("/chat")
        self.assertIn("Sign in to chat", r.text)


class TestChatChrome(unittest.TestCase):
    """The chat surface has its own chrome. Nothing else in the suite reads
    chat.html's markup, so without these a completely broken page would keep
    every other chat test green."""

    def _page(self) -> str:
        client = _build_client()
        _login(client)
        return client.get("/chat").text

    def test_chat_extends_chat_base_not_admin_base(self):
        self.assertIn('extends "chat_base.html"', _template("chat.html"))

    def test_chat_base_has_no_admin_nav(self):
        # Chat users would otherwise see six admin links, five of which lead
        # straight to a Basic-auth prompt they do not have the password for.
        page = self._page()
        for href in ('href="/users"', 'href="/schedules"', 'href="/documents"',
                     'href="/health"'):
            with self.subTest(href=href):
                self.assertNotIn(href, page)

    def test_chat_uses_the_chat_palette_not_the_admin_one(self):
        page = self._page()
        self.assertIn("--amethyst", page)
        self.assertNotIn("--muted: #5b6a78", page)

    def test_identity_and_signout_are_present(self):
        page = self._page()
        self.assertIn("presence-user", page)
        self.assertIn("/chat/logout", page)

    def test_faceted_controls_carry_a_focus_wrapper(self):
        # clip-path clips outline AND outer box-shadow, so a faceted control
        # without the unclipped .facet wrapper has NO visible focus indicator.
        # Guarding this because any new faceted control repeats the bug.
        src = _template("chat.html")
        for control in ("btn-ghost", "attach-btn", "send-btn"):
            with self.subTest(control=control):
                idx = src.index(f'class="{control}"')
                # The wrapper opens within a few hundred chars before it.
                self.assertIn('<span class="facet">', src[max(0, idx - 400):idx])

    def test_transcript_scrolls_its_own_container(self):
        # The shell is fixed/overflow-hidden, so the window-level scroll APIs
        # are wrong here; scroll helpers must drive scrollTop on the transcript.
        # Strip comments first — the prose explaining the rule names the very
        # APIs being banned, and matching that would be a false positive.
        src = _template("chat.html")
        self.assertIn("transcript.scrollTop", src)
        code = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
        code = re.sub(r"\{#.*?#\}", "", code, flags=re.S)
        self.assertNotIn("scrollIntoView", code)
        self.assertNotIn("window.scrollTo", code)

    def test_composer_font_size_avoids_ios_zoom(self):
        self.assertRegex(_template("chat.html"), r"font-size:\s*16\.5px")

    def test_enter_to_send_guards_ime_and_touch(self):
        # A bare Enter handler swallows the character an IME is composing, and
        # soft keyboards have no reliable Shift for Shift+Enter.
        src = _template("chat.html")
        self.assertIn("isComposing", src)
        self.assertIn("keyCode === 229", src)
        self.assertIn("pointer: coarse", src)

    def test_empty_state_offers_suggestions(self):
        src = _template("chat.html")
        self.assertIn("The study is lit", src)
        self.assertIn("suggestion", src)


class TestChatStreamingUx(unittest.TestCase):
    """Status chip, Stop, notices and the confirm dialog. These are template
    guards — the behaviour itself was exercised in a browser."""

    def _code(self) -> str:
        """chat.html with comments stripped, so prose describing a banned API
        cannot masquerade as a use of it."""
        src = _template("chat.html")
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        src = re.sub(r"^\s*//.*$", "", src, flags=re.M)
        return re.sub(r"\{#.*?#\}", "", src, flags=re.S)

    def test_no_alert_calls_remain(self):
        # alert() steals focus, cannot be styled, and blocks the event loop the
        # stream is running on. Replaced by the notice bar.
        self.assertNotRegex(self._code(), r"\balert\s*\(")

    def test_window_confirm_replaced_by_the_dialog(self):
        code = self._code()
        self.assertNotRegex(code, r"window\.confirm\s*\(")
        self.assertIn("confirm-veil", code)
        self.assertIn("Burn the correspondence?", code)

    def test_dialog_is_a_labelled_modal(self):
        src = _template("chat.html")
        self.assertIn('role="dialog"', src)
        self.assertIn('aria-modal="true"', src)
        self.assertIn('aria-labelledby="confirm-title"', src)

    def test_dialog_can_be_dismissed_three_ways(self):
        code = self._code()
        self.assertIn("confirm-cancel", code)          # Spare it
        self.assertIn('veil.addEventListener("click"', code)
        self.assertIn('e.key === "Escape"', code)
        # A click inside the card must not bubble out to the cancelling veil.
        self.assertIn("stopPropagation", code)

    def test_apply_token_seam_exists(self):
        # The wire format is cumulative today. This function is the only place
        # that knows, so flipping to deltas is a two-line change.
        code = self._code()
        self.assertIn("function applyToken(delta, cumulative, restart)", code)

    def test_stop_is_labelled_honestly(self):
        # Aborting the client fetch cannot stop the server: chat_stream runs
        # ask_stuff on a daemon thread and never checks is_disconnected().
        # Claiming otherwise in the UI would be a lie.
        src = _template("chat.html")
        self.assertIn("Enough", src)
        self.assertIn("AbortController", src)
        self.assertIn("mutters on in the servants", src)

    def test_stream_teardown_clears_every_timer(self):
        # A missed timer leaves a zombie status chip that never goes away.
        code = self._code()
        self.assertIn("function clearStreamState()", code)
        for cleared in ("statusSwapTimer", "activeCaret", "abortController"):
            with self.subTest(cleared=cleared):
                self.assertIn(cleared, code)

    def test_sending_while_streaming_is_blocked(self):
        self.assertIn("Fritz is mid-sentence", _template("chat.html"))

    def test_forget_uses_fetch_not_a_reload(self):
        code = self._code()
        self.assertIn('fetch("/chat/forget"', code)
        self.assertIn("renderEmptyState", code)


class TestChatCorrectnessAndA11y(unittest.TestCase):
    """Guards for the mobile / confirm-dialog / screen-reader fixes."""

    def _chat_page(self) -> str:
        client = _build_client()
        _login(client)
        return client.get("/chat").text

    def test_viewport_meta_present(self):
        # Lives in base.html, so this covers all eight templates at once.
        self.assertIn('name="viewport"', self._chat_page())

    def test_login_page_also_has_viewport_meta(self):
        client = _build_client()
        self.assertIn('name="viewport"', client.get("/chat").text)

    def test_forget_form_has_no_inline_onsubmit(self):
        # The attribute it replaced contained a literal \\' sequence. HTML does
        # not unescape backslashes, so the handler was a SyntaxError, compiled
        # to null, and POST /chat/forget fired with no confirmation at all —
        # destroying the thread's checkpoint unrecoverably.
        page = self._chat_page()
        self.assertNotIn("onsubmit", page)
        self.assertIn('id="chat-forget-form"', page)

    def test_confirm_is_registered_above_the_early_return_guard(self):
        # Placement is the whole fix: below `if (!form || !list) return;` a
        # missing composer would silently disarm the confirmation again.
        page = self._chat_page()
        listener = page.index("chat-forget-form")
        guard = page.index("if (!form || !list) return;")
        self.assertLess(listener, guard)

    def test_transcript_is_a_log_region(self):
        page = self._chat_page()
        self.assertIn('role="log"', page)
        self.assertIn('aria-live="polite"', page)

    def test_status_region_exists(self):
        self.assertIn('id="chat-status"', self._chat_page())

    def test_composer_font_size_avoids_ios_zoom(self):
        # iOS Safari force-zooms a focused input whose text is under 16px and
        # never zooms back out.
        self.assertIn("font-size: 16px", self._chat_page())

    def test_reduced_motion_block_present(self):
        self.assertIn("prefers-reduced-motion", self._chat_page())


class TestChatSend(unittest.TestCase):
    def test_unauthed_send_redirects_to_chat(self):
        client = _build_client()
        r = client.post("/chat/send", data={"message": "hi"},
                        follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/chat")

    def test_authed_send_invokes_ask_stuff_with_username(self):
        client = _build_client()
        _login(client)

        fake_module = MagicMock()
        fake_module.ask_stuff.return_value = {
            "text": "Very well.",
            "image_paths": [],
            "timestamp": "now",
        }
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log"):
            r = client.post("/chat/send", data={"message": "hello fritz"})

        self.assertEqual(r.status_code, 200)
        self.assertIn("Very well.", r.text)
        # ask_stuff received the cookie's username as user_id.
        args, kwargs = fake_module.ask_stuff.call_args
        # ask_stuff(message, source, user_id, ...)
        self.assertEqual(args[0], "hello fritz")
        self.assertEqual(args[2], "alice")

    def test_send_audit_log_records_message_chars(self):
        client = _build_client()
        _login(client)

        fake_module = MagicMock()
        fake_module.ask_stuff.return_value = {"text": "okay", "image_paths": [], "timestamp": "now"}
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log") as audit:
            client.post("/chat/send", data={"message": "this is a test message"})

        # First call should be chat_message with ok result.
        calls = [c for c in audit.call_args_list if c.args and c.args[0] == "chat_message"]
        self.assertEqual(len(calls), 1)
        kwargs = calls[0].kwargs
        self.assertEqual(kwargs["user_id"], "alice")
        self.assertEqual(kwargs["chars"], len("this is a test message"))
        self.assertEqual(kwargs["result"], "ok")

    def test_empty_message_redirects_without_invoking_agent(self):
        client = _build_client()
        _login(client)

        fake_module = MagicMock()
        with patch.dict(sys.modules, {"mister_fritz": fake_module}):
            r = client.post("/chat/send", data={"message": "   "},
                            follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        fake_module.ask_stuff.assert_not_called()


# ── /chat/stream (Phase web-chat-2: SSE streaming) ──────────────────────────

def _parse_sse(text: str):
    """Parse a raw SSE response body into a list of (event, data) tuples."""
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = None
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                # SSE protocol: strip exactly one leading space if present.
                d = line[len("data:"):]
                if d.startswith(" "):
                    d = d[1:]
                data_lines.append(d)
        if event_name or data_lines:
            events.append((event_name, "\n".join(data_lines)))
    return events


class TestChatStreamUnauthed(unittest.TestCase):
    def test_no_cookie_returns_401(self):
        client = _build_client()
        r = client.post("/chat/stream", data={"message": "hi"})
        self.assertEqual(r.status_code, 401)


class TestChatStreamSuccess(unittest.TestCase):
    def test_streams_token_events_then_done(self):
        client = _build_client()
        _login(client)

        # Fake ask_stuff calls its streaming_callback a few times, then returns.
        def fake_ask_stuff(message, source, user, *,
                          streaming_callback=None, schedule_manager=None, **_):
            assert streaming_callback is not None
            streaming_callback("Very")
            streaming_callback("Very well")
            streaming_callback("Very well, sir.")
            return {"text": "Very well, sir.", "image_paths": [], "timestamp": "now"}

        fake_module = MagicMock()
        fake_module.ask_stuff = fake_ask_stuff
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log"):
            r = client.post("/chat/stream", data={"message": "hello"})

        self.assertEqual(r.status_code, 200)
        events = _parse_sse(r.text)
        tokens = [d for ev, d in events if ev == "token"]
        dones = [d for ev, d in events if ev == "done"]
        self.assertEqual(tokens, ["Very", "Very well", "Very well, sir."])
        # The 'done' frame is now a JSON payload (Phase web-chat-3); the
        # detailed shape is asserted in TestChatStreamDonePayload.
        self.assertEqual(len(dones), 1)
        import json as _json
        self.assertEqual(_json.loads(dones[0])["text"], "Very well, sir.")

    def test_audit_log_records_streamed_message(self):
        client = _build_client()
        _login(client)

        def fake_ask_stuff(message, source, user, *,
                          streaming_callback=None, **_):
            streaming_callback("ok")
            return {"text": "ok", "image_paths": [], "timestamp": "now"}

        fake_module = MagicMock()
        fake_module.ask_stuff = fake_ask_stuff
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log") as audit:
            client.post("/chat/stream", data={"message": "ping"})

        calls = [c for c in audit.call_args_list if c.args and c.args[0] == "chat_message"]
        self.assertEqual(len(calls), 1)
        kwargs = calls[0].kwargs
        self.assertEqual(kwargs["user_id"], "alice")
        self.assertEqual(kwargs["result"], "ok")
        self.assertTrue(kwargs.get("streamed"))


class TestChatStreamError(unittest.TestCase):
    def test_agent_exception_yields_error_event(self):
        client = _build_client()
        _login(client)

        def fake_ask_stuff(message, source, user, **_):
            raise RuntimeError("ollama down")

        fake_module = MagicMock()
        fake_module.ask_stuff = fake_ask_stuff
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log"):
            r = client.post("/chat/stream", data={"message": "hello"})

        self.assertEqual(r.status_code, 200)
        events = _parse_sse(r.text)
        errors = [d for ev, d in events if ev == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("ollama down", errors[0])
        # No 'done' event when the agent failed.
        dones = [d for ev, d in events if ev == "done"]
        self.assertEqual(dones, [])


class TestChatStreamEmptyMessage(unittest.TestCase):
    def test_empty_message_returns_400(self):
        client = _build_client()
        _login(client)
        r = client.post("/chat/stream", data={"message": "   "})
        self.assertEqual(r.status_code, 400)


# ── Phase web-chat-3 polish: markdown, progress, forget, history, assets ───

class TestRenderMarkdown(unittest.TestCase):
    def test_bold_renders_to_strong(self):
        html = admin_panel._render_markdown("This is **bold** text.")
        self.assertIn("<strong>bold</strong>", html)

    def test_code_fence_renders_pre_code(self):
        html = admin_panel._render_markdown("```py\nprint('hi')\n```")
        self.assertIn("<pre>", html)
        self.assertIn("print", html)

    def test_empty_string_renders_empty(self):
        self.assertEqual(admin_panel._render_markdown(""), "")
        self.assertEqual(admin_panel._render_markdown(None), "")


class TestSanitiseHtml(unittest.TestCase):
    """python-markdown emits raw HTML verbatim and chat.html renders replies
    with `| safe`, so anything Fritz can be talked into saying reaches the DOM.
    """

    def test_script_tag_is_stripped(self):
        html = admin_panel._render_markdown("hello <script>alert(1)</script> there")
        self.assertNotIn("<script", html)
        self.assertNotIn("alert(1)", html)

    def test_event_handler_attribute_is_stripped(self):
        html = admin_panel._render_markdown('<img src="x" onerror="steal()">')
        self.assertNotIn("onerror", html)
        self.assertNotIn("steal()", html)

    def test_javascript_url_is_neutralised(self):
        html = admin_panel._render_markdown('<a href="javascript:alert(1)">click</a>')
        self.assertNotIn("javascript:", html)

    def test_iframe_is_stripped(self):
        html = admin_panel._render_markdown('<iframe src="https://evil.test"></iframe>')
        self.assertNotIn("<iframe", html)

    def test_benign_formatting_survives(self):
        html = admin_panel._render_markdown(
            "**bold** and [a link](https://example.test) and ![pic](/img.png)"
        )
        self.assertIn("<strong>bold</strong>", html)
        self.assertIn("https://example.test", html)   # href kept
        self.assertIn("/img.png", html)               # img src kept

    def test_codehilite_classes_survive_sanitiser(self):
        # The guard for the trap in DECISIONS.md #20: nh3's default attribute
        # map has no entry for div/pre/code/span, so a plain nh3.clean() strips
        # every Pygments class and highlighting dies silently — no error, and a
        # test that only checks for "<pre>" and "print" still passes.
        highlighted = (
            '<div class="codehilite"><pre><code>'
            '<span class="nb">print</span><span class="p">(</span>'
            "</code></pre></div>"
        )
        cleaned = admin_panel._sanitise_html(highlighted)
        self.assertIn('class="codehilite"', cleaned)
        self.assertIn('class="nb"', cleaned)

    def test_class_allowlist_does_not_disarm_other_attributes(self):
        # Building the map from anything other than dict(nh3.ALLOWED_ATTRIBUTES)
        # silently drops href/src, because `attributes=` replaces the defaults
        # rather than extending them.
        cleaned = admin_panel._sanitise_html(
            '<span class="ok" onclick="evil()">x</span>'
        )
        self.assertIn('class="ok"', cleaned)
        self.assertNotIn("onclick", cleaned)

    def test_empty_input_is_empty(self):
        self.assertEqual(admin_panel._sanitise_html(""), "")
        self.assertEqual(admin_panel._sanitise_html(None), "")


class TestChatStreamDonePayload(unittest.TestCase):
    def test_done_event_carries_html_and_text(self):
        import json as _json
        client = _build_client()
        _login(client)

        def fake_ask_stuff(message, source, user, *,
                          streaming_callback=None, progress_callback=None, **_):
            streaming_callback("Very well")
            return {"text": "Very well **sir**.", "image_paths": [], "timestamp": "now"}

        fake_module = MagicMock()
        fake_module.ask_stuff = fake_ask_stuff
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log"):
            r = client.post("/chat/stream", data={"message": "hi"})

        events = _parse_sse(r.text)
        dones = [d for ev, d in events if ev == "done"]
        self.assertEqual(len(dones), 1)
        payload = _json.loads(dones[0])
        self.assertEqual(payload["text"], "Very well **sir**.")
        self.assertIn("<strong>sir</strong>", payload["html"])
        self.assertEqual(payload["images"], [])


class TestChatStreamProgressEvents(unittest.TestCase):
    def test_progress_callback_yields_progress_events(self):
        client = _build_client()
        _login(client)

        def fake_ask_stuff(message, source, user, *,
                          streaming_callback=None, progress_callback=None, **_):
            assert progress_callback is not None
            progress_callback("Searching the web...")
            progress_callback("Reading results...")
            streaming_callback("Here is what I found.")
            return {"text": "Here is what I found.", "image_paths": [], "timestamp": "now"}

        fake_module = MagicMock()
        fake_module.ask_stuff = fake_ask_stuff
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log"):
            r = client.post("/chat/stream", data={"message": "look it up"})

        events = _parse_sse(r.text)
        progresses = [d for ev, d in events if ev == "progress"]
        self.assertEqual(progresses, ["Searching the web...", "Reading results..."])


class TestChatHistory(unittest.TestCase):
    def test_unauthed_returns_401(self):
        client = _build_client()
        r = client.get("/chat/history")
        self.assertEqual(r.status_code, 401)

    def test_authed_returns_messages_from_loader(self):
        client = _build_client()
        _login(client)
        fake_history = [
            {"role": "user", "content": "hi", "html": None},
            {"role": "fritz", "content": "Greetings.", "html": "<p>Greetings.</p>"},
        ]
        with patch.object(admin_panel, "_load_chat_history", return_value=fake_history):
            r = client.get("/chat/history")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["username"], "alice")
        self.assertEqual(body["messages"], fake_history)


class TestChatForget(unittest.TestCase):
    def test_authed_post_calls_forget_conversation_and_redirects(self):
        client = _build_client()
        _login(client)
        with patch.object(privacy, "forget_conversation", return_value=3) as fc, \
             patch.object(admin_panel, "audit_log") as audit:
            r = client.post("/chat/forget", follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/chat")
        # Must target the WEB thread — clearing "alice" here would wipe the
        # Discord conversation instead of the one the user is looking at.
        fc.assert_called_once_with("alice", thread_id="web-alice")
        # Audit entry recorded.
        calls = [c for c in audit.call_args_list if c.args and c.args[0] == "chat_forget_conversation"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["user_id"], "alice")
        self.assertEqual(calls[0].kwargs["removed"], 3)

    def test_unauthed_redirects_without_touching_privacy(self):
        client = _build_client()
        with patch.object(privacy, "forget_conversation") as fc:
            r = client.post("/chat/forget", follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        fc.assert_not_called()


class TestChatThreadId(unittest.TestCase):
    """The chat cookie's username is self-asserted, so the web chat must not
    share a LangGraph checkpoint with the Discord user of the same name."""

    def test_web_thread_is_namespaced(self):
        self.assertEqual(admin_panel._chat_thread_id("alice"), "web-alice")

    def test_web_thread_cannot_collide_with_a_discord_user(self):
        # A Discord user literally named "webalice" gets thread "webalice"
        # (alphanumerics only). Keeping the dash is what keeps these apart.
        self.assertNotEqual(
            admin_panel._chat_thread_id("alice"),
            re.sub(r"[^a-zA-Z0-9]", "", "webalice"),
        )

    def test_punctuation_is_stripped_but_dash_and_underscore_survive(self):
        self.assertEqual(admin_panel._chat_thread_id("al.i/ce"), "web-alice")
        self.assertEqual(admin_panel._chat_thread_id("a_b-c"), "web-a_b-c")

    def test_empty_user_gives_empty_thread(self):
        self.assertEqual(admin_panel._chat_thread_id(""), "")

    def test_share_flag_restores_the_discord_thread(self):
        # The rollback lever, and the single-user case where continuing one
        # conversation across Discord and the browser is the point.
        with patch.object(admin_panel, "CHAT_SHARE_DISCORD_THREAD", True):
            self.assertEqual(admin_panel._chat_thread_id("alice"), "alice")

    def test_history_load_uses_the_same_thread_as_the_write_path(self):
        # The read path used to re-derive the thread id with its own inline
        # copy of the regex; that drift is the /forget bug for punctuated names.
        captured = {}

        def _fake_get_state(config):
            captured["thread_id"] = config["configurable"]["thread_id"]
            raise RuntimeError("stop here — we only want the config")

        fake_mf = MagicMock()
        fake_mf.app.get_state.side_effect = _fake_get_state
        fake_mf.get_config_values.side_effect = lambda c: {
            "configurable": {"thread_id": c["metadata"]["thread_id"]},
        }
        with patch.dict(sys.modules, {"mister_fritz": fake_mf}):
            admin_panel._load_chat_history("alice")
        self.assertEqual(captured["thread_id"], admin_panel._chat_thread_id("alice"))


class TestChatAsset(unittest.TestCase):
    def test_unauthed_returns_401(self):
        client = _build_client()
        r = client.get("/chat/assets/output/whatever.png")
        self.assertEqual(r.status_code, 401)

    def test_path_escape_returns_404(self):
        client = _build_client()
        _login(client)
        # Try to escape the asset roots.
        r = client.get("/chat/assets/../etc/passwd")
        self.assertEqual(r.status_code, 404)

    def test_serves_existing_image_under_output(self):
        client = _build_client()
        _login(client)

        os.makedirs("output", exist_ok=True)
        marker = os.path.join("output", "_admin_panel_test_marker.png")
        with open(marker, "wb") as f:
            f.write(_TINY_PNG)
        try:
            r = client.get("/chat/assets/output/_admin_panel_test_marker.png")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.content, _TINY_PNG)
        finally:
            os.unlink(marker)

    def test_response_carries_hardening_headers(self):
        client = _build_client()
        _login(client)
        os.makedirs("output", exist_ok=True)
        marker = os.path.join("output", "_admin_panel_hdr_marker.png")
        with open(marker, "wb") as f:
            f.write(_TINY_PNG)
        try:
            r = client.get("/chat/assets/output/_admin_panel_hdr_marker.png")
            self.assertEqual(r.headers["content-type"], "image/png")
            self.assertEqual(r.headers["x-content-type-options"], "nosniff")
            csp = r.headers["content-security-policy"]
            self.assertIn("default-src 'none'", csp)
            self.assertIn("sandbox", csp)
        finally:
            os.unlink(marker)

    def test_non_image_extension_is_404(self):
        # The whole point: a stored .html must never be served back as a
        # same-origin document next to the chat session's cookie.
        client = _build_client()
        _login(client)
        os.makedirs("output", exist_ok=True)
        marker = os.path.join("output", "_admin_panel_test_marker.html")
        with open(marker, "w") as f:
            f.write("<script>alert(1)</script>")
        try:
            r = client.get("/chat/assets/output/_admin_panel_test_marker.html")
            self.assertEqual(r.status_code, 404)
        finally:
            os.unlink(marker)

    def test_user_cannot_fetch_another_users_upload(self):
        client = _build_client()
        _login(client, "bob")
        os.makedirs("temp_images", exist_ok=True)
        # Named as chat_upload_image would name one of alice's uploads.
        victim = os.path.join("temp_images", "alice_123_secret.png")
        with open(victim, "wb") as f:
            f.write(_TINY_PNG)
        try:
            with patch.object(admin_panel, "audit_log") as audit:
                r = client.get("/chat/assets/temp_images/alice_123_secret.png")
            self.assertEqual(r.status_code, 404)
            self.assertEqual(audit.call_args.args[0], "chat_asset_denied")
        finally:
            os.unlink(victim)

    def test_user_can_fetch_their_own_upload(self):
        client = _build_client()
        _login(client, "alice")
        os.makedirs("temp_images", exist_ok=True)
        mine = os.path.join("temp_images", "alice_123_mine.png")
        with open(mine, "wb") as f:
            f.write(_TINY_PNG)
        try:
            r = client.get("/chat/assets/temp_images/alice_123_mine.png")
            self.assertEqual(r.status_code, 200)
        finally:
            os.unlink(mine)


class TestChatAssetUrlHelper(unittest.TestCase):
    def test_path_in_output_root_returns_chat_assets_url(self):
        abs_path = os.path.abspath(os.path.join("output", "abc.png"))
        url = admin_panel._chat_asset_url(abs_path)
        self.assertEqual(url, "/chat/assets/output/abc.png")

    def test_path_outside_roots_returns_none(self):
        self.assertIsNone(admin_panel._chat_asset_url("/etc/passwd"))

    def test_empty_returns_none(self):
        self.assertIsNone(admin_panel._chat_asset_url(""))
        self.assertIsNone(admin_panel._chat_asset_url(None))


# ── Phase web-chat-4: file uploads ──────────────────────────────────────────

# A real, decodable 1x1 PNG. The previous fixture was PNG magic bytes followed
# by nulls, which Pillow rejects with UnidentifiedImageError — it only ever
# passed because the upload route trusted the declared Content-Type and never
# looked at the body.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Payloads that declare an image type but are not one.
_HTML_PAYLOAD = b"<html><script>alert(1)</script></html>"
_SVG_PAYLOAD = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'


def _drain_for_user(user: str):
    """Clear any leftover pending image so tests can run independently."""
    admin_panel._drain_pending_images(user)


class TestChatUploadImage(unittest.TestCase):
    def setUp(self):
        _drain_for_user("alice")

    def tearDown(self):
        _drain_for_user("alice")
        # Clean up the temp_images dir of any test artefacts.
        if os.path.isdir("temp_images"):
            for f in os.listdir("temp_images"):
                if f.startswith("alice_") or f.startswith("_admin_panel_test_"):
                    try:
                        os.unlink(os.path.join("temp_images", f))
                    except OSError:
                        pass

    def test_unauthed_returns_401(self):
        client = _build_client()
        r = client.post("/chat/upload/image",
                        files={"file": ("a.png", _TINY_PNG, "image/png")})
        self.assertEqual(r.status_code, 401)

    def test_happy_path_saves_file_and_stashes_pending(self):
        client = _build_client()
        _login(client)
        with patch.object(admin_panel, "audit_log"):
            r = client.post(
                "/chat/upload/image",
                files={"file": ("photo.png", _TINY_PNG, "image/png")},
            )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertIsNotNone(body["url"])
        # The pending dict has an entry for alice now.
        with admin_panel._pending_images_lock:
            pending = list(admin_panel._pending_images.get("alice", []))
        self.assertEqual(len(pending), 1)
        self.assertTrue(os.path.isfile(pending[0]))

    def test_rejects_unsupported_content_type(self):
        client = _build_client()
        _login(client)
        with patch.object(admin_panel, "audit_log"):
            r = client.post(
                "/chat/upload/image",
                files={"file": ("a.svg", b"<svg/>", "image/svg+xml")},
            )
        self.assertEqual(r.status_code, 415)

    def test_rejects_oversized_image(self):
        client = _build_client()
        _login(client)
        with patch.object(admin_panel, "CHAT_IMAGE_UPLOAD_MAX_BYTES", 32), \
             patch.object(admin_panel, "audit_log"):
            r = client.post(
                "/chat/upload/image",
                files={"file": ("a.png", _TINY_PNG, "image/png")},
            )
        self.assertEqual(r.status_code, 413)

    def test_missing_file_field_returns_400(self):
        client = _build_client()
        _login(client)
        r = client.post("/chat/upload/image", data={"not_file": "x"})
        self.assertEqual(r.status_code, 400)


class TestChatUploadImageSniffing(unittest.TestCase):
    """The declared Content-Type is not evidence — the bytes are."""

    def setUp(self):
        _drain_for_user("alice")

    def tearDown(self):
        _drain_for_user("alice")
        if os.path.isdir("temp_images"):
            for f in os.listdir("temp_images"):
                if f.startswith("alice_"):
                    try:
                        os.unlink(os.path.join("temp_images", f))
                    except OSError:
                        pass

    def _upload(self, filename, body, declared="image/png"):
        client = _build_client()
        _login(client)
        with patch.object(admin_panel, "audit_log"):
            return client.post(
                "/chat/upload/image",
                files={"file": (filename, body, declared)},
            )

    def test_html_declared_as_png_is_rejected(self):
        r = self._upload("evil.png", _HTML_PAYLOAD)
        self.assertEqual(r.status_code, 415)

    def test_svg_declared_as_png_is_rejected(self):
        r = self._upload("evil.png", _SVG_PAYLOAD)
        self.assertEqual(r.status_code, 415)

    def test_png_magic_followed_by_html_is_rejected(self):
        # Magic-byte prefix matching alone would let this through.
        r = self._upload("evil.png", b"\x89PNG\r\n\x1a\n<html><script>x</script>")
        self.assertEqual(r.status_code, 415)

    def test_rejected_content_writes_nothing_to_temp_images(self):
        before = set(os.listdir("temp_images")) if os.path.isdir("temp_images") else set()
        self._upload("evil.png", _HTML_PAYLOAD)
        after = set(os.listdir("temp_images")) if os.path.isdir("temp_images") else set()
        self.assertEqual(before, after)

    def test_extension_comes_from_content_not_filename(self):
        # A genuine PNG uploaded as "evil.html" must be stored as .png.
        r = self._upload("evil.html", _TINY_PNG)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["name"].endswith(".png"))
        with admin_panel._pending_images_lock:
            pending = list(admin_panel._pending_images.get("alice", []))
        self.assertEqual(len(pending), 1)
        self.assertTrue(pending[0].endswith(".png"))
        self.assertNotIn(".html", os.path.basename(pending[0]))

    def test_gif_body_declared_png_is_stored_as_gif(self):
        # The canonical extension follows the sniffed format, not the claim.
        import io as _io

        from PIL import Image as _Image
        buf = _io.BytesIO()
        _Image.new("RGB", (1, 1), (0, 255, 0)).save(buf, format="GIF")
        r = self._upload("thing.png", buf.getvalue())
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["name"].endswith(".gif"))

    def test_sniff_helper_rejects_non_images(self):
        self.assertIsNone(admin_panel._sniff_image_format(_HTML_PAYLOAD))
        self.assertIsNone(admin_panel._sniff_image_format(_SVG_PAYLOAD))
        self.assertIsNone(admin_panel._sniff_image_format(b""))
        self.assertEqual(admin_panel._sniff_image_format(_TINY_PNG), "PNG")

    def test_safe_stem_discards_client_extension(self):
        self.assertEqual(admin_panel._safe_stem("evil.html"), "evil")
        self.assertEqual(admin_panel._safe_stem("../../etc/passwd"), "passwd")
        self.assertEqual(admin_panel._safe_stem(""), "upload")


class TestAdminUploadDocument(unittest.TestCase):
    """Document upload moved off /chat. DOC_FOLDER is shared knowledge every
    user's RAG queries read from, and the chat cookie is a self-asserted name —
    gating on is_admin(cookie_name) meant anyone holding the chat password
    could type the owner's username and inject into the shared corpus.
    """

    def setUp(self):
        self.tmp_doc_dir = tempfile.mkdtemp()
        self._doc_patch = patch.object(admin_panel, "DOC_FOLDER", self.tmp_doc_dir)
        self._doc_patch.start()

    def tearDown(self):
        self._doc_patch.stop()

    def test_chat_route_is_gone(self):
        client = _build_client()
        _login(client)
        r = client.post("/chat/upload/document",
                        files={"file": ("notes.md", b"# hi", "text/markdown")})
        self.assertEqual(r.status_code, 404)

    def test_requires_basic_auth(self):
        client = _build_client()
        r = client.post("/documents/upload",
                        files={"file": ("notes.md", b"# hi", "text/markdown")})
        self.assertEqual(r.status_code, 401)

    def test_a_chat_cookie_does_not_open_it(self):
        # The regression this move exists to prevent.
        client = _build_client()
        _login(client, "alice")
        r = client.post("/documents/upload",
                        files={"file": ("notes.md", b"# hi", "text/markdown")})
        self.assertEqual(r.status_code, 401)

    def test_admin_happy_path_writes_to_doc_folder(self):
        client = _build_client()
        with patch.object(admin_panel, "audit_log"):
            r = client.post(
                "/documents/upload",
                files={"file": ("notes.md", b"# hi from the admin", "text/markdown")},
                headers=_auth_header(),
                follow_redirects=False,
            )
        self.assertEqual(r.status_code, 303)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp_doc_dir, "notes.md")))

    def test_bad_extension_is_rejected_and_audited(self):
        client = _build_client()
        with patch.object(admin_panel, "audit_log") as audit:
            r = client.post(
                "/documents/upload",
                files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
                headers=_auth_header(),
                follow_redirects=False,
            )
        self.assertEqual(r.status_code, 303)
        self.assertFalse(os.path.isfile(os.path.join(self.tmp_doc_dir, "malware.exe")))
        self.assertEqual(audit.call_args.kwargs["result"], "bad_ext")

    def test_oversized_document_is_rejected(self):
        client = _build_client()
        with patch.object(admin_panel, "CHAT_DOC_UPLOAD_MAX_BYTES", 4), \
             patch.object(admin_panel, "audit_log") as audit:
            client.post(
                "/documents/upload",
                files={"file": ("notes.md", b"far too long", "text/markdown")},
                headers=_auth_header(),
                follow_redirects=False,
            )
        self.assertFalse(os.path.isfile(os.path.join(self.tmp_doc_dir, "notes.md")))
        self.assertEqual(audit.call_args.kwargs["result"], "rejected_size")

    def test_chat_page_no_longer_offers_document_upload(self):
        client = _build_client()
        _login(client)
        page = client.get("/chat").text
        self.assertNotIn("chat-doc-input", page)
        self.assertNotIn("/chat/upload/document", page)


class TestPendingImagePlumbing(unittest.TestCase):
    def setUp(self):
        _drain_for_user("alice")

    def tearDown(self):
        _drain_for_user("alice")

    def test_send_picks_up_pending_image_and_clears(self):
        client = _build_client()
        _login(client)
        admin_panel._stash_pending_image("alice", "/tmp/fake-img.png")

        captured = {}

        def fake_ask_stuff(message, source, user, *,
                          user_image_paths=None, **_):
            captured["source"] = source
            captured["images"] = user_image_paths
            return {"text": "ok", "image_paths": [], "timestamp": "now"}

        fake_module = MagicMock()
        fake_module.ask_stuff = fake_ask_stuff
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log"):
            client.post("/chat/send", data={"message": "look at this"})

        # ask_stuff received the stashed image path.
        self.assertEqual(captured["images"], ["/tmp/fake-img.png"])
        # Compare by enum name, not identity — other tests in the suite
        # (test_workspace_store) reload fritz_utils, which creates a new
        # MessageSource class that won't `==` the one admin_panel captured.
        self.assertEqual(captured["source"].name, "DISCORD_TEXT_AND_IMAGE")
        # The pending registry is empty after consumption.
        with admin_panel._pending_images_lock:
            self.assertNotIn("alice", admin_panel._pending_images)

    def test_send_without_pending_uses_local_source(self):
        client = _build_client()
        _login(client)

        captured = {}

        def fake_ask_stuff(message, source, user, *,
                          user_image_paths=None, **_):
            captured["source"] = source
            captured["images"] = user_image_paths
            return {"text": "ok", "image_paths": [], "timestamp": "now"}

        fake_module = MagicMock()
        fake_module.ask_stuff = fake_ask_stuff
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log"):
            client.post("/chat/send", data={"message": "just text"})

        self.assertIsNone(captured["images"])
        self.assertEqual(captured["source"].name, "LOCAL")


if __name__ == "__main__":
    unittest.main()
