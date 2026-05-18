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


if __name__ == "__main__":
    unittest.main()
