"""
Tests for the read-only admin panel (Phase 9a).

We use Starlette's TestClient against the app built by create_app() so we
exercise routing, templating, and auth without spinning up uvicorn.
"""
import base64
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _ensure_mock(name: str):
    if name not in sys.modules:
        sys.modules[name] = MagicMock()
    return sys.modules[name]


# ddgs is required transitively by privacy → workspace_store → fritz_utils
# is fine, but agent_tools (which privacy doesn't actually need) is imported
# nowhere; nothing else to stub.
_ensure_mock("ddgs")

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


def _auth_header(password: str = PASSWORD) -> dict:
    encoded = base64.b64encode(f"admin:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _build_client(schedule_manager=None) -> TestClient:
    app = admin_panel.create_app(PASSWORD, schedule_manager=schedule_manager)
    return TestClient(app)


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
        r = client.post("/chat/login", data={"username": "alice"},
                        follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/chat")
        self.assertIn("fritz_chat_id", r.cookies)

    def test_empty_username_renders_error(self):
        client = _build_client()
        r = client.post("/chat/login", data={"username": "  "},
                        follow_redirects=False)
        # Re-renders login form with error message (200, not redirect).
        self.assertEqual(r.status_code, 200)
        self.assertIn("at least one letter", r.text)

    def test_username_is_sanitised(self):
        # Path-like or punctuation-heavy usernames get stripped to safe chars.
        client = _build_client()
        r = client.post("/chat/login", data={"username": "../bad/name"},
                        follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        # The set cookie should contain "badname" (slashes + dots stripped).
        import chat_auth
        from fritz_utils import CHAT_COOKIE_SECRET
        token = r.cookies.get("fritz_chat_id")
        self.assertEqual(chat_auth.verify_cookie(token, CHAT_COOKIE_SECRET), "badname")


class TestChatLogout(unittest.TestCase):
    def test_logout_clears_cookie(self):
        client = _build_client()
        # First log in.
        client.post("/chat/login", data={"username": "alice"})
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
        client.post("/chat/login", data={"username": "alice"})
        r = client.get("/chat")
        self.assertEqual(r.status_code, 200)
        self.assertIn("alice", r.text)
        self.assertIn("Type a message", r.text)

    def test_tampered_cookie_renders_login(self):
        client = _build_client()
        client.cookies.set("fritz_chat_id", "alice:9999999999:deadbeef")
        r = client.get("/chat")
        self.assertIn("Sign in to chat", r.text)


class TestChatSend(unittest.TestCase):
    def test_unauthed_send_redirects_to_chat(self):
        client = _build_client()
        r = client.post("/chat/send", data={"message": "hi"},
                        follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/chat")

    def test_authed_send_invokes_ask_stuff_with_username(self):
        client = _build_client()
        client.post("/chat/login", data={"username": "alice"})

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
        client.post("/chat/login", data={"username": "alice"})

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
        client.post("/chat/login", data={"username": "alice"})

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
        client.post("/chat/login", data={"username": "alice"})

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
        self.assertEqual(dones, ["Very well, sir."])

    def test_audit_log_records_streamed_message(self):
        client = _build_client()
        client.post("/chat/login", data={"username": "alice"})

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
        client.post("/chat/login", data={"username": "alice"})

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
        client.post("/chat/login", data={"username": "alice"})
        r = client.post("/chat/stream", data={"message": "   "})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
