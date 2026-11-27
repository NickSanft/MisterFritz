"""
Tests for the read-only admin panel (Phase 9a).

We use Starlette's TestClient against the app built by create_app() so we
exercise routing, templating, and auth without spinning up uvicorn.
"""
import base64
import contextlib
import datetime
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


# The cookie carries "alice"; every store keys off the namespaced identity
# derived from it, and uploads are named from a filesystem-safe rendering of
# that identity. Three different strings for the same person, deliberately.
ALICE_ID = "web-alice"
ALICE_FILE_PREFIX = "web-alice_"
# Uploads live in a per-user DIRECTORY under temp_images/, not behind a
# filename prefix — '_' is a legal identity character, so a prefix test also
# matched web-alice_2's files. See TestChatAssetOwnership.
ALICE_UPLOAD_DIR = "web-alice"


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


class TestDestructiveConfirmsSurviveTheCSP(unittest.TestCase):
    """Regression: the global CSP silently disabled every admin confirmation.

    _SecurityHeadersMiddleware applies script-src 'nonce-…' with no
    'unsafe-inline' to EVERY response, not just /chat. A nonce cannot be
    attached to an inline event-handler attribute, so the three
    onsubmit="return confirm(...)" handlers on the admin pages were blocked
    and their destructive POSTs — cancel schedule, disable workspace, and
    "Permanently delete ALL data for <user>" — fired with no prompt at all.
    This is the same defect class chat.html was fixed for.
    """

    _DESTRUCTIVE = ("schedules.html", "user_detail.html")

    def test_no_admin_template_uses_an_inline_event_handler(self):
        for name in self._DESTRUCTIVE:
            with self.subTest(template=name):
                src = _template(name)
                for attr in ("onsubmit=", "onclick=", "onchange="):
                    self.assertNotIn(attr, src)

    def test_every_destructive_form_declares_data_confirm(self):
        for name in self._DESTRUCTIVE:
            with self.subTest(template=name):
                src = _template(name)
                posts = re.findall(r"<form[^>]*method=\"post\"[^>]*>", src)
                self.assertTrue(posts, f"{name} has no POST forms")
                for form in posts:
                    self.assertIn("data-confirm=", form)

    def test_base_registers_the_confirm_handler_under_a_nonce(self):
        src = _template("base.html")
        self.assertIn('<script nonce="{{ csp_nonce(request) }}">', src)
        self.assertIn("data-confirm", src)
        self.assertIn("preventDefault", src)


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

    def test_a_crons_zone_on_the_users_page(self):
        """Plan 14 PR 14: as /schedule list says it."""
        client = _build_client(schedule_manager=MagicMock())
        fake_data = {
            "user_id": "alice", "memories": [], "conversation_checkpoint_count": 0, "workspace_path": None,
            "schedules": [
                {"id": "s1", "prompt": "p", "schedule": "0 9 * * *", "description": "", "created": "now",
                 "timezone": "Asia/Tokyo"},
                {"id": "s2", "prompt": "p", "schedule": "0 7 * * *", "description": "", "created": "now",
                 "timezone": None},
                {"id": "s3", "prompt": "p", "schedule": "1h", "description": "", "created": "now",
                 "timezone": "Europe/Paris"},
            ],
        }
        with patch.object(privacy, "export_user_data", return_value=fake_data):
            r = client.get("/users/alice", headers=_auth_header())
        self.assertIn('<code>0 9 * * *</code> <span class="muted">Asia/Tokyo</span>', r.text)
        self.assertIn('<code>0 7 * * *</code> <span class="muted">the host&#39;s zone</span>', r.text)
        self.assertNotIn("Europe/Paris", r.text)


class TestALinkedIdentityIsNotItsOwnPerson(unittest.TestCase):
    """IDENTITY_LINKS treats the desk as its owner, and every privacy function
    follows the link before it reads or deletes. The desk still has a row of
    its own (its handshake records the PC's name), and that row used to be the
    owner's data under the PC's name, with a Forget button that forgot the
    owner."""

    DESK, OWNER = "desktop-nick-pc", "discord-1"

    def setUp(self):
        import identity_store
        self.identity_store = identity_store
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        names = stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        for name, value in (("SCHEDULE_DB", os.path.join(names, "names.db")),
                            ("_INITIALISED", False), ("_NAME_CACHE", {})):
            stack.enter_context(patch.object(identity_store, name, value))
        # Nobody else's rows: other tests leave workspaces in the shared store.
        stack.enter_context(patch.object(workspace_store, "list_all", return_value=[]))
        self.workspaces = {}
        stack.enter_context(patch.object(workspace_store, "get", side_effect=self.workspaces.get))
        # The desk's handshake wrote this; nothing ever wrote the owner's.
        identity_store.record(self.DESK, "Nick's PC", "desktop")
        self.stack = stack
        self.manager = MagicMock()
        self.manager.list_all_schedules.return_value = []
        self.schedules = {}
        self.manager.list_schedules.side_effect = lambda uid: self.schedules.get(uid, [])
        self.client = _build_client(schedule_manager=self.manager)

    def linked(self, links=None):
        return patch.dict(fritz_utils.IDENTITY_LINKS, links or {self.DESK: self.OWNER})

    def get(self, path):
        r = self.client.get(path, headers=_auth_header())
        self.assertEqual(r.status_code, 200, r.text[:300])
        return r.text

    def post(self, path):
        return self.client.post(path, headers=_auth_header(), follow_redirects=False)

    @staticmethod
    def row_of(page: str, user_id: str) -> str:
        """The <tr> of /users whose first link is to `user_id`, whitespace squashed."""
        for row in page.split("<tr>")[1:]:
            if f'<a href="/users/{user_id}">' in row.split("</td>")[0]:
                return " ".join(row.split("</tr>")[0].split())
        raise AssertionError(f"no row for {user_id}")

    @staticmethod
    def squashed(page: str) -> str:
        return " ".join(page.split())

    def counted(self, memories=3):
        """privacy's readers, mocked: each identity has `memories` memories."""
        stack = contextlib.ExitStack()
        reads = [stack.enter_context(patch.object(privacy, name, return_value=value))
                 for name, value in (("export_memories", [1] * memories), ("export_schedules", []),
                                     ("get_workspace_for_export", None))]
        return stack, reads

    def test_who_it_is_linked_to(self):
        with self.linked({"web-b": self.OWNER, "web-a": "web-a", self.DESK: self.OWNER, "telegram-7": "web-c"}):
            self.assertEqual(admin_panel._linked_to(self.DESK), self.OWNER)
            self.assertEqual(admin_panel._linked_to("web-b"), self.OWNER)          # not only a desk
            self.assertEqual(admin_panel._linked_to("telegram-7"), "web-c")        # not only to Discord
            self.assertIsNone(admin_panel._linked_to(self.OWNER))
            self.assertIsNone(admin_panel._linked_to("web-a"))        # to itself: not linked
            self.assertIsNone(admin_panel._linked_to("web-nobody"))
            self.assertEqual(admin_panel._linked_from(self.OWNER), [self.DESK, "web-b"])
            self.assertEqual(admin_panel._linked_from("web-a"), [])
        self.assertIsNone(admin_panel._linked_to(self.DESK))

    def test_the_list_counts_its_memories_with_who_it_is_linked_to(self):
        stack, reads = self.counted()
        with self.linked(), stack:
            page = self.get("/users")
        desk = self.row_of(page, self.DESK)
        self.assertIn("Nick&#39;s PC", desk)
        self.assertIn(f'<td> <code>{self.DESK}</code> <span class="muted">treated as</span> '
                      f'<a href="/users/{self.OWNER}"><code>{self.OWNER}</code></a> </td>', desk)
        self.assertIn(f'<td class="muted">counted with <code>{self.OWNER}</code> (IDENTITY_LINKS)</td> <td>0</td>',
                      desk)
        self.assertNotIn("<td>3</td>", desk)
        # Nothing was read for the desk through the link: each read would have been the owner's.
        for read in reads:
            asked = [call.args[0] for call in read.call_args_list]
            self.assertNotIn(self.DESK, asked)
            self.assertIn(self.OWNER, asked)
        # The owner is listed, though no store has a row for them, and counted once.
        owner = self.row_of(page, self.OWNER)
        self.assertIn("<td>3</td>", owner)
        self.assertNotIn("treated as", owner)
        self.assertEqual(page.count("<td>3</td>"), 1)

    def test_whoever_it_is_linked_to_is_listed_whichever_store_listed_it(self):
        """Not only a desk, and not only from the alias table: /chat/login
        records web-alice unresolved, and /schedule add and /workspace enable
        store under the id they are given."""
        self.identity_store.record("web-alice", "alice", "web")
        self.manager.list_all_schedules.return_value = [{"user_id": "telegram-99"}]
        links = {self.DESK: self.OWNER, "web-alice": "discord-2", "telegram-99": "discord-3", "discord-40": "discord-4"}
        stack, _reads = self.counted()
        with self.linked(links), stack, \
             patch.object(workspace_store, "list_all", return_value=[{"user_id": "discord-40"}]):
            self.assertEqual(admin_panel._collect_users(self.manager), sorted([*links, *links.values()]))
            page = self.get("/users")
        for alias, primary in links.items():
            self.assertIn(f"counted with <code>{primary}</code>", self.row_of(page, alias))
            self.assertIn("<td>3</td>", self.row_of(page, primary))
        self.assertEqual(page.count("<td>3</td>"), len(links))

    def test_unlinked_it_is_counted_as_anyone_is(self):
        stack, reads = self.counted()
        with stack:
            page = self.get("/users")
        desk = self.row_of(page, self.DESK)
        self.assertIn("<td>3</td>", desk)
        self.assertNotIn("treated as", desk)
        self.assertIn(self.DESK, [call.args[0] for call in reads[0].call_args_list])
        self.assertNotIn(f'<a href="/users/{self.OWNER}">', page)

    def test_its_page_shows_nothing_of_the_owners_and_offers_no_action(self):
        with self.linked(), patch.object(privacy, "export_user_data") as export:
            page = self.get(f"/users/{self.DESK}")
        export.assert_not_called()
        card = self.squashed(page)
        self.assertIn("<h2>Nick&#39;s PC</h2>", card)
        self.assertIn(f"<th>Identity</th><td><code>{self.DESK}</code></td>", card)
        self.assertIn(f'<th>Treated as</th><td> <a href="/users/{self.OWNER}"><code>{self.OWNER}</code></a>', card)
        self.assertIn(f"What <code>{self.DESK}</code> says and remembers is <code>{self.OWNER}</code>'s: its "
                      f'memories and conversations are on <a href="/users/{self.OWNER}">that page</a>, and so is '
                      "Forget. Nothing can be forgotten from here.", card)
        self.assertIn(f"Whatever <code>{self.DESK}</code> said before it was linked is kept, unread, for as long "
                      "as the link stays.", card)
        for gone in ("<form", 'data-confirm="', "/forget", "workspace/disable", "<h2>Memories",
                     "<h2>Schedules", "Conversation checkpoints", "Kept under", "<th>Also</th>"):
            self.assertNotIn(gone, page)

    def test_what_a_discord_alias_keeps_under_its_own_id_is_shown_as_its_own(self):
        """/schedule add and /workspace enable store under the id that ran them,
        which the link does not move: not the owner's, and not forgotten with them."""
        alt = "discord-77"
        self.identity_store.record(alt, "nick_alt", "discord")
        self.schedules[alt] = [{"id": "s77", "schedule": "0 9 * * *", "prompt": "the weather"}]
        self.workspaces[alt] = "/w/alt"
        stack, reads = self.counted()
        with self.linked({alt: self.OWNER}), stack, patch.object(privacy, "export_user_data") as export:
            row = self.row_of(self.get("/users"), alt)
            page = self.get(f"/users/{alt}")
        self.assertIn(f"counted with <code>{self.OWNER}</code> (IDENTITY_LINKS)</td> <td>1</td> <td> <code>/w/alt</code>",
                      row)
        card = self.squashed(page)
        self.assertIn(f"<h2>Kept under <code>{alt}</code> itself</h2>", card)
        self.assertIn(f"so these are not on <code>{self.OWNER}</code>'s page and its Forget does not remove them. "
                      f"A schedule here still runs, as <code>{self.OWNER}</code>", card)
        self.assertIn('<th>Workspace</th><td><code>/w/alt</code> <span class="muted">(this page cannot disable '
                      'it while the link stays: take the link out, then disable it here)</span>', card)
        self.assertIn("<td><code>s77</code></td> <td><code>0 9 * * *</code> "
                      "<span class=\"muted\">the host&#39;s zone</span></td> <td>the weather</td>", card)
        self.assertIn('<form class="inline" method="post" action="/schedules/s77/cancel" '
                      'data-confirm="Cancel schedule s77 for discord-77?">', card)
        self.assertEqual(page.count("<form"), 1)
        self.assertNotIn("could not be read", page)
        self.assertNotIn(f"/users/{alt}/", page)
        export.assert_not_called()
        for read in reads:
            self.assertNotIn(alt, [call.args[0] for call in read.call_args_list])
        self.assertIn(alt, [call.args[0] for call in self.manager.list_schedules.call_args_list])
        # A workspace alone is shown as well, with nothing to cancel.
        self.workspaces["discord-78"] = "/w/alt2"
        with self.linked({"discord-78": self.OWNER}):
            page = self.squashed(self.get("/users/discord-78"))
        self.assertIn("<h2>Kept under <code>discord-78</code> itself</h2>", page)
        self.assertIn("<th>Workspace</th><td><code>/w/alt2</code>", page)
        self.assertNotIn("<form", page)
        self.assertNotIn("<th>ID</th>", page)
        self.assertNotIn("could not be read", page)
        # And schedules alone, with no workspace to speak of.
        self.schedules["discord-79"] = [{"id": "s79", "schedule": "0 8 * * *", "prompt": "the news"}]
        with self.linked({"discord-79": self.OWNER}):
            page = self.squashed(self.get("/users/discord-79"))
        self.assertIn("<h2>Kept under <code>discord-79</code> itself</h2>", page)
        self.assertIn("<td><code>s79</code></td>", page)
        self.assertNotIn("<th>Workspace</th>", page)

    def test_schedules_that_cannot_be_read_are_not_none(self):
        """A cell that cannot be counted says so: "?" is not 0."""
        alt = "discord-77"
        self.identity_store.record(alt, "nick_alt", "discord")
        self.workspaces[alt] = "/w/alt"
        self.manager.list_schedules.side_effect = RuntimeError("database is locked")
        stack, _reads = self.counted()
        with self.linked({alt: self.OWNER}), stack, self.assertLogs("admin_panel", level="WARNING") as logs:
            row = self.row_of(self.get("/users"), alt)
            page = self.squashed(self.get(f"/users/{alt}"))
        self.assertIn("(IDENTITY_LINKS)</td> <td>?</td> <td> <code>/w/alt</code>", row)
        self.assertIn(f"The schedules kept under <code>{alt}</code> could not be read (the log says why).", page)
        self.assertIn("database is locked", "; ".join(logs.output))
        self.schedules.clear()
        with self.linked({"discord-78": self.OWNER}), self.assertLogs("admin_panel", level="WARNING"):
            page = self.squashed(self.get("/users/discord-78"))
        self.assertIn("<h2>Kept under <code>discord-78</code> itself</h2>", page)
        # A workspace that cannot be read takes nothing down either.
        self.manager.list_schedules.side_effect = lambda uid: self.schedules.get(uid, [])
        stack, _reads = self.counted()
        with self.linked({alt: self.OWNER}), stack, \
             patch.object(workspace_store, "get", side_effect=RuntimeError("workspace db locked")), \
             self.assertLogs("admin_panel", level="WARNING") as logs:
            row = self.row_of(self.get("/users"), alt)
            self.get(f"/users/{alt}")
        self.assertIn('(IDENTITY_LINKS)</td> <td>0</td> <td> <span class="muted">—</span>', row)
        self.assertIn("workspace db locked", "; ".join(logs.output))

    def test_a_chain_says_no_page_holds_it_and_names_its_end(self):
        """Links are not chained: what desktop-x says is kept under web-b, whose
        own page reads one hop further."""
        self.identity_store.record("desktop-x", "X PC", "desktop")
        stack, _reads = self.counted()
        with self.linked({"desktop-x": "web-b", "web-b": "web-c", "jean luc?": "web-b"}), stack, \
             patch.object(privacy, "forget_all") as forget, patch.object(admin_panel, "audit_log"), \
             self.assertLogs("admin_panel", level="WARNING"):
            first = self.squashed(self.get("/users/desktop-x"))
            middle = self.squashed(self.get("/users/web-b"))
            legacy = self.squashed(self.get("/users/jean%20luc%3F"))
            self.assertEqual(self.post("/users/desktop-x/forget").status_code, 303)
            self.assertEqual(self.post("/users/web-b/forget").status_code, 303)
        forget.assert_not_called()
        self.assertIn("<code>web-b</code> is itself linked on to <code>web-c</code>, and links are not chained: "
                      "what <code>desktop-x</code> says and remembers is kept under <code>web-b</code>, where no page "
                      "here shows or forgets it while that link stays. Link <code>desktop-x</code> to "
                      "<code>web-c</code> itself in IDENTITY_LINKS", first)
        self.assertNotIn("that page", first)
        self.assertIn('<th>Treated as</th><td> <a href="/users/web-b"><code>web-b</code></a>', first)
        self.assertIn('that page</a>', middle)
        self.assertIn('<th>Also</th><td> <a href="/users/desktop-x"><code>desktop-x</code></a> (X PC), '
                      '<a href="/users/jean%20luc%3F"><code>jean luc?</code></a> <span class="muted">treated as this '
                      "identity in turn: links are not chained, so what is said and remembered there is kept under "
                      "<code>web-b</code>, where no page here shows it, except that a web name's memories, and a "
                      "reminder booked in a Discord conversation when it fires, go one link further, to "
                      "<code>web-c</code>.</span>", middle)
        self.assertNotIn("<form", first + middle)
        # Only a web name's memories take the second hop: not a legacy name's.
        self.assertIn("what <code>jean luc?</code> says and remembers is kept under <code>web-b</code>", legacy)
        self.assertNotIn("one link further", legacy)
        # A chain of three is mended in one edit: the remedy names its end, as the startup log does.
        links = {"desktop-x": "web-b", "web-b": "web-c", "web-c": "discord-9"}
        with self.linked(links):
            self.assertIn("Link <code>desktop-x</code> to <code>discord-9</code> itself in IDENTITY_LINKS (the brain's "
                          "startup log says the same)", self.squashed(self.get("/users/desktop-x")))
            self.assertTrue(any(said.endswith("Link desktop-x to discord-9 itself")
                                for said in fritz_utils.identity_link_problems()))
            # The middle one's own next link, not the chain's end.
            self.assertIn("go one link further, to <code>web-c</code>.</span>", self.squashed(self.get("/users/web-b")))
        # A chain that ends at an id with "/" in it has no end to link to, however long.
        for links in ({"desktop-x": "web-b", "web-b": "discord-1/2"},
                      {"desktop-x": "web-b", "web-b": "web-c", "web-c": "discord-1/2"}):
            with self.subTest(links=links), self.linked(links):
                page = self.squashed(self.get("/users/desktop-x"))
            self.assertIn("The links end at <code>discord-1/2</code>, which no identity is (no identity has a <code>/</code> "
                          "in it): correct that link in IDENTITY_LINKS, then link <code>desktop-x</code> straight to what "
                          "it points to.", page)
            self.assertNotIn("to <code>discord-1/2</code> itself", page)

    def test_a_chained_web_names_memories_go_one_link_further(self):
        """The web chat resolves its name before ask_stuff resolves it again."""
        with self.linked({"web-a": "web-b", "web-b": "web-c"}):
            page = self.squashed(self.get("/users/web-a"))
        self.assertIn("its conversation is kept under <code>web-b</code>, where no page here shows or forgets it "
                      "while that link stays, and as a web name its memories go one link further, to "
                      '<code>web-c</code>: they are on <a href="/users/web-c">that page</a>, and its Forget removes '
                      "them.", page)
        self.assertNotIn("says and remembers is kept under", page)
        with self.linked({"web-a": "web-b", "web-b": "web-c", "web-c": "discord-9"}):
            page = self.squashed(self.get("/users/web-a"))
        self.assertIn("its memories go one link further, to <code>web-c</code>, which no page here shows.", page)
        with self.linked({"web-a": "web-b", "web-b": "discord-1/2"}):
            page = self.squashed(self.get("/users/web-a"))
        self.assertIn("its memories go one link further, to <code>discord-1/2</code>, which no page here shows.", page)

    def test_links_that_go_round_in_a_loop_have_no_end_to_link_to(self):
        for links, loop, kept in (
                ({"web-a": "web-b", "web-b": "web-a"}, "<code>web-a</code> to <code>web-b</code> to <code>web-a</code>",
                 "what <code>web-a</code> says is kept under <code>web-b</code>, and as a web name its memories "
                 "under <code>web-a</code>, and no page here shows or forgets it."),
                ({"web-a": "desktop-y", "desktop-y": "web-c", "web-c": "web-a"},
                 "<code>web-a</code> to <code>desktop-y</code> to <code>web-c</code> to <code>web-a</code>",
                 "what <code>web-a</code> says is kept under <code>desktop-y</code>, and as a web name its "
                 "memories under <code>web-c</code>, and no page here shows or forgets it."),
                ({"desktop-x": "web-b", "web-b": "desktop-x"},
                 "<code>desktop-x</code> to <code>web-b</code> to <code>desktop-x</code>",
                 "what <code>desktop-x</code> says is kept under <code>web-b</code>, and no page here shows or forgets it."),
                # Into a loop from outside it: the loop named is the one it runs into.
                ({"desktop-t": "web-a", "web-a": "web-b", "web-b": "web-a"},
                 "<code>web-a</code> to <code>web-b</code> to <code>web-a</code>",
                 "what <code>desktop-t</code> says is kept under <code>web-a</code>, and no page here shows or forgets it.")):
            alias = next(iter(links))
            with self.subTest(links=links), self.linked(links):
                page = self.squashed(self.get(f"/users/{alias}"))
            self.assertIn(f"These links go round in a loop ({loop}), and links are not chained: {kept} Choose the one "
                          "identity the others are, link each of the others straight to it, and take that one's own "
                          "link out of IDENTITY_LINKS.", page)
            self.assertNotIn(" itself in IDENTITY_LINKS", page)
            self.assertNotIn("that page", page)

    def test_the_list_says_a_chained_one_is_counted_nowhere(self):
        for name in ("web-a", "web-b", "desktop-x", "discord-77"):
            self.identity_store.record(name, name, "web")
        stack, _reads = self.counted()
        with self.linked({"web-a": "web-b", "web-b": "web-c", "desktop-x": "web-b", "discord-77": "web-b"}), stack:
            page = self.get("/users")
        self.assertIn("conversation kept under <code>web-b</code>, counted nowhere; memories counted with "
                      "<code>web-c</code>", self.row_of(page, "web-a"))
        self.assertIn("kept under <code>web-b</code>, counted nowhere: links are not chained", self.row_of(page, "desktop-x"))
        self.assertIn('treated as</span> <a href="/users/web-b"><code>web-b</code></a>', self.row_of(page, "desktop-x"))
        self.assertNotIn("memories counted with", self.row_of(page, "discord-77"))
        self.assertIn("counted with <code>web-c</code> (IDENTITY_LINKS)", self.row_of(page, "web-b"))
        # A web name's memories with no page to be counted on.
        for links, home in (({"web-a": "web-b", "web-b": "web-c", "web-c": "discord-9"}, "web-c"),
                            ({"web-a": "web-b", "web-b": "discord-1/2"}, "discord-1/2")):
            stack, _reads = self.counted()
            with self.subTest(links=links), self.linked(links), stack:
                row = self.row_of(self.get("/users"), "web-a")
            self.assertIn(f"conversation kept under <code>web-b</code>, counted nowhere; memories kept under "
                          f"<code>{home}</code>, which no page here shows", row)
            self.assertNotIn("memories counted with", row)
        stack, _reads = self.counted()
        with self.linked({"web-a": "web-b", "web-b": "web-a"}), stack:
            row = self.row_of(self.get("/users"), "web-a")
        self.assertIn("kept under <code>web-b</code> (memories under <code>web-a</code>), counted nowhere: these links go "
                      "round in a loop", row)

    def test_one_linked_to_an_id_with_a_slash_says_it_has_no_page(self):
        """The detail route takes one path segment, so such a link would 404."""
        stack, _reads = self.counted()
        with self.linked({self.DESK: "discord-1/2"}), stack:
            page = self.get(f"/users/{self.DESK}")
            listing = self.get("/users")
        row = self.row_of(listing, self.DESK)
        for html in (page, listing):
            self.assertNotIn('href="/users/discord-1/2"', html)
        # Its own row is still listed, with its count: only the link is left out.
        self.assertIn("<td><code>discord-1/2</code></td> <td>3</td>", self.squashed(listing))
        data = {"user_id": self.OWNER, "memories": [], "schedules": [],
                "conversation_checkpoint_count": 0, "workspace_path": None}
        with self.linked({"discord-1/2": self.OWNER}), patch.object(privacy, "export_user_data", return_value=data):
            owner = self.get(f"/users/{self.OWNER}")
        self.assertNotIn('href="/users/discord-1/2"', owner)
        self.assertIn("<th>Also</th><td> <code>discord-1/2</code>", self.squashed(owner))
        with self.linked({"discord-1/2": "web-b", "web-b": "web-c"}):
            middle = self.squashed(self.get("/users/web-b"))
        self.assertNotIn('href="/users/discord-1/2"', middle)
        self.assertIn('<th>Also</th><td> <code>discord-1/2</code> <span class="muted">treated as this identity in turn',
                      middle)
        card = self.squashed(page)
        self.assertIn(f'<td> <code>{self.DESK}</code> <span class="muted">treated as</span> <code>discord-1/2</code> </td>',
                      row)
        self.assertIn('<th>Treated as</th><td> <code>discord-1/2</code> <span class="muted">(IDENTITY_LINKS)</span>', card)
        self.assertIn(f"What <code>{self.DESK}</code> says and remembers is kept under <code>discord-1/2</code>, which "
                      "has no page here: no identity has a <code>/</code> in it. Correct the link in IDENTITY_LINKS",
                      card)
        self.assertNotIn("that page", page)

    def test_ids_that_need_quoting_are_quoted_in_every_new_link(self):
        """A legacy id may carry anything; ? # and % would misroute unquoted."""
        stack, _reads = self.counted()
        with self.linked({self.DESK: "jean luc?"}), stack:
            row = self.row_of(self.get("/users"), self.DESK)
            page = self.get(f"/users/{self.DESK}")
        self.assertIn('<a href="/users/jean%20luc%3F"><code>jean luc?</code></a>', row)
        self.assertEqual(page.count('<a href="/users/jean%20luc%3F">'), 2)
        data = {"user_id": self.OWNER, "memories": [], "schedules": [],
                "conversation_checkpoint_count": 0, "workspace_path": None}
        with self.linked({"jean luc?": self.OWNER, "a&b <c>": self.OWNER}), \
             patch.object(privacy, "export_user_data", return_value=data):
            owner = self.get(f"/users/{self.OWNER}")
            alias = self.get("/users/jean%20luc%3F")
        self.assertIn('<a href="/users/jean%20luc%3F"><code>jean luc?</code></a>', owner)
        self.assertIn("<code>a&amp;b &lt;c&gt;</code>", owner)
        self.assertIn("<th>Treated as</th>", alias)
        self.assertNotIn("<form", alias)
        with self.linked({"web-a": "web-b", "web-b": "jean luc?"}):
            self.assertIn('<a href="/users/jean%20luc%3F">that page</a>', self.get("/users/web-a"))

    def test_the_owners_page_names_them_and_keeps_its_actions(self):
        data = {"user_id": self.OWNER, "memories": [], "schedules": [],
                "conversation_checkpoint_count": 4, "workspace_path": "/w/owner"}
        # A web name is its login, the id's own tail: no name worth repeating.
        self.identity_store.record("web-zed", "zed", "web")
        with self.linked({"web-b": self.OWNER, self.DESK: self.OWNER, "web-zed": self.OWNER}), \
             patch.object(privacy, "export_user_data", return_value=data) as export:
            page = self.get(f"/users/{self.OWNER}")
        export.assert_called_once_with(self.OWNER, self.manager)
        self.assertIn(f'<th>Also</th><td> <a href="/users/{self.DESK}"><code>{self.DESK}</code></a> (Nick&#39;s PC), '
                      '<a href="/users/web-b"><code>web-b</code></a>, <a href="/users/web-zed"><code>web-zed</code></a> '
                      '<span class="muted">treated as this identity by IDENTITY_LINKS: their memories and '
                      "conversations are here.</span>", self.squashed(page))
        self.assertIn(f'action="/users/{self.OWNER}/forget"', page)
        self.assertIn(f'action="/users/{self.OWNER}/workspace/disable"', page)
        self.assertNotIn("<th>Treated as</th>", page)

    def test_a_page_nobody_is_linked_to_says_nothing_of_links(self):
        data = {"user_id": self.DESK, "memories": [], "schedules": [],
                "conversation_checkpoint_count": 0, "workspace_path": None}
        with patch.object(privacy, "export_user_data", return_value=data) as export:
            page = self.get(f"/users/{self.DESK}")
        export.assert_called_once_with(self.DESK, self.manager)
        self.assertNotIn("<th>Also</th>", page)
        self.assertNotIn("<th>Treated as</th>", page)
        self.assertIn(f'action="/users/{self.DESK}/forget"', page)

    def test_forgetting_it_forgets_nobody(self):
        """A page loaded before the link was made, or a request made by hand."""
        for alias in (self.DESK, "web-alice"):
            with self.subTest(alias=alias), self.linked({alias: self.OWNER}), \
                 patch.object(privacy, "forget_all") as forget, \
                 patch.object(admin_panel, "audit_log") as audit, \
                 self.assertLogs("admin_panel", level="WARNING") as logs:
                r = self.post(f"/users/{alias}/forget")
            forget.assert_not_called()
            self.assertEqual((r.status_code, r.headers["location"]), (303, f"/users/{alias}"))
            audit.assert_called_once_with("admin_forget_all", admin="admin", target_user=alias,
                                          refused="linked", linked_to=self.OWNER)
            self.assertIn(f"admin_forget_all refused for '{alias}': it is linked to '{self.OWNER}', whose data "
                          "it would have reached", logs.output[0])

    def test_disabling_its_workspace_disables_nobodys(self):
        with self.linked(), patch.object(privacy, "forget_workspace") as forget, \
             patch.object(admin_panel, "audit_log") as audit, \
             self.assertLogs("admin_panel", level="WARNING"):
            r = self.post(f"/users/{self.DESK}/workspace/disable")
        forget.assert_not_called()
        self.assertEqual((r.status_code, r.headers["location"]), (303, f"/users/{self.DESK}"))
        audit.assert_called_once_with("admin_disable_workspace", admin="admin", target_user=self.DESK,
                                      refused="linked", linked_to=self.OWNER)

    def test_the_owner_is_still_forgotten_from_their_own_page(self):
        with self.linked(), patch.object(privacy, "forget_all", return_value={}) as forget, \
             patch.object(privacy, "forget_workspace", return_value=True) as workspace, \
             patch.object(admin_panel, "audit_log"):
            self.assertEqual(self.post(f"/users/{self.OWNER}/forget").status_code, 303)
            self.assertEqual(self.post(f"/users/{self.OWNER}/workspace/disable").status_code, 303)
        forget.assert_called_once_with(self.OWNER, self.manager)
        workspace.assert_called_once_with(self.OWNER)

    def test_unlinked_it_is_forgotten_as_anyone_is(self):
        with patch.object(privacy, "forget_all", return_value={}) as forget, \
             patch.object(admin_panel, "audit_log"):
            r = self.post(f"/users/{self.DESK}/forget")
        forget.assert_called_once_with(self.DESK, self.manager)
        self.assertEqual(r.headers["location"], "/users")

    def test_a_refusal_goes_back_to_a_page_that_exists(self):
        """A legacy id may carry anything; the address it is sent back to is quoted."""
        with self.linked({"jean luc?": self.OWNER}), patch.object(privacy, "forget_all") as forget, \
             patch.object(admin_panel, "audit_log"), self.assertLogs("admin_panel", level="WARNING"):
            r = self.post("/users/jean%20luc%3F/forget")
        forget.assert_not_called()
        self.assertEqual(r.headers["location"], "/users/jean%20luc%3F")


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

    def test_a_crons_zone_is_shown_and_an_intervals_isnt(self):
        """Plan 14 PR 14: each schedule keeps the zone it was made in."""
        manager = MagicMock()
        manager.list_all_schedules.return_value = [
            {"id": "s1", "user_id": "alice", "prompt": "p1", "schedule": "1h", "description": "",
             "created": "now", "timezone": "Europe/Paris"},
            {"id": "s2", "user_id": "bob", "prompt": "weather", "schedule": "0 9 * * *", "description": "",
             "created": "now", "timezone": "Asia/Tokyo"},
            {"id": "s3", "user_id": "carol", "prompt": "old", "schedule": "0 7 * * *", "description": "",
             "created": "now", "timezone": None},
        ]
        client = _build_client(schedule_manager=manager)
        r = client.get("/schedules", headers=_auth_header())
        self.assertIn("Asia/Tokyo", r.text)
        self.assertNotIn("Europe/Paris", r.text)
        self.assertIn("the host&#39;s zone", r.text, "an older cron says whose zone")

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
        # The seam that isolated the cumulative→delta wire-format change. Only
        # this function knows the format, so the migration was a two-line edit.
        code = self._code()
        self.assertIn("function applyToken(delta, cumulative, restart)", code)

    def test_client_appends_deltas_and_clears_on_reset(self):
        # Frames now carry only the NEW text. Passing `data` as the accumulated
        # value again would silently restore the O(n^2) behaviour AND render
        # every reply as just its final token.
        code = self._code()
        self.assertIn("applyToken(data, null, false)", code)
        self.assertIn('eventName === "reset"', code)
        self.assertIn('applyToken("", "", true)', code)

    def test_stop_is_labelled_honestly(self):
        # This asserted the opposite until plan 14 P3: aborting the fetch did
        # not stop the server, and the notice said so. It does now (see
        # tests/test_cancel.py), so the notice must stop apologising for it.
        src = _template("chat.html")
        self.assertIn("Enough", src)
        self.assertIn("AbortController", src)
        self.assertNotIn("mutters on in the servants", src)
        # Nor overclaim the other way: a tool already running is not stopped.
        self.assertIn("already in hand will be finished, and nothing further begun", src)
        self.assertNotIn("fallen silent", src)

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


class TestChatStreamDeltaWireFormat(unittest.TestCase):
    """`token` frames carry deltas and `reset` marks a new answer segment.

    The failure this guards is subtle: if the server reverted to sending
    accumulated text, the client (which appends) would render every reply as a
    growing pile of duplicated prefixes, and the wire cost would go quadratic.
    """

    def _stream(self, fake_ask_stuff):
        client = _build_client()
        _login(client)
        fake_module = MagicMock()
        fake_module.ask_stuff = fake_ask_stuff
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log"):
            r = client.post("/chat/stream", data={"message": "hi"})
        return _parse_sse(r.text)

    def test_reset_precedes_each_segment_in_order(self):
        # Preamble, tool call, fresh answer — verified to be what a real model
        # produces, since the two halves arrive under different chunk ids.
        def fake(message, source, user, *, streaming_callback=None, **_):
            streaming_callback("Let me look.", "Let me look.", True)
            streaming_callback("I found", "I found", True)
            streaming_callback(" it.", "I found it.", False)
            return {"text": "I found it.", "image_paths": [], "timestamp": "now"}

        events = [(ev, d) for ev, d in self._stream(fake) if ev in ("reset", "token")]
        self.assertEqual(events, [
            ("reset", ""), ("token", "Let me look."),
            ("reset", ""), ("token", "I found"), ("token", " it."),
        ])

    def test_deltas_since_the_last_reset_reassemble_the_reply(self):
        def fake(message, source, user, *, streaming_callback=None, **_):
            streaming_callback("scratch", "scratch", True)
            streaming_callback("Very", "Very", True)
            streaming_callback(" well", "Very well", False)
            return {"text": "Very well", "image_paths": [], "timestamp": "now"}

        acc = ""
        for ev, data in self._stream(fake):
            if ev == "reset":
                acc = ""
            elif ev == "token":
                acc += data
        self.assertEqual(acc, "Very well")

    def test_token_delta_preserves_a_leading_space(self):
        # Guards the `data: ` prefix + single-leading-space-strip round trip.
        # Losing this silently runsallthewordstogether.
        def fake(message, source, user, *, streaming_callback=None, **_):
            streaming_callback(" well", "Very well", False)
            return {"text": "Very well", "image_paths": [], "timestamp": "now"}

        tokens = [d for ev, d in self._stream(fake) if ev == "token"]
        self.assertEqual(tokens, [" well"])

    def test_token_delta_with_embedded_newlines_round_trips(self):
        # Each logical line gets its own `data:` line; the blank-line frame
        # terminator must not be produced by the payload itself.
        def fake(message, source, user, *, streaming_callback=None, **_):
            streaming_callback("a\n\nb", "a\n\nb", True)
            return {"text": "a\n\nb", "image_paths": [], "timestamp": "now"}

        tokens = [d for ev, d in self._stream(fake) if ev == "token"]
        self.assertEqual(tokens, ["a\n\nb"])

    def test_wire_cost_is_linear_not_quadratic(self):
        # 40 words sent cumulatively would cost ~O(n^2) bytes.
        words = [f"w{i} " for i in range(40)]

        def fake(message, source, user, *, streaming_callback=None, **_):
            acc = ""
            for i, w in enumerate(words):
                acc += w
                streaming_callback(w, acc, i == 0)
            return {"text": acc, "image_paths": [], "timestamp": "now"}

        tokens = [d for ev, d in self._stream(fake) if ev == "token"]
        reply_len = len("".join(words))
        self.assertEqual(sum(len(t) for t in tokens), reply_len)


class TestCodeHighlighting(unittest.TestCase):
    """codehilite is inert unless nh3 allows `class` on div/pre/code/span. That
    failure is silent — Pygments runs, spans are emitted, styling hooks are
    stripped, and both '<pre>' and the code text still appear."""

    _FENCE = "Here:\n\n```python\n# note\ndef f(x):\n    return x + 1\n```\n"

    def test_codehilite_classes_survive_the_sanitiser(self):
        html = admin_panel._render_markdown(self._FENCE)
        self.assertIn('class="codehilite"', html)
        self.assertIn('class="k"', html)      # keyword
        self.assertIn('class="c1"', html)     # comment

    def test_language_is_reattached_as_data_lang(self):
        # codehilite strips the class="language-python" fenced_code emits, so
        # the language is read back off the markdown source.
        html = admin_panel._render_markdown(self._FENCE)
        self.assertIn('data-lang="python"', html)

    def test_unlabelled_fence_gets_no_language(self):
        # Degrades to the client's plain "code" chip rather than guessing.
        html = admin_panel._render_markdown("```\nplain\n```\n")
        self.assertNotIn("data-lang", html)

    def test_data_lang_survives_the_sanitiser(self):
        self.assertIn("data-lang", admin_panel._sanitise_html(
            '<div class="codehilite" data-lang="python"><pre>x</pre></div>'))

    def test_highlighting_is_gated_by_the_knob(self):
        self.assertIn("codehilite", admin_panel._MARKDOWN_EXTENSIONS)
        self.assertFalse(
            admin_panel._MARKDOWN_CONFIGS["codehilite"]["guess_lang"],
            "guess_lang must stay off, or Pygments colours prose and logs "
            "as though they were code",
        )

    def test_theme_targets_real_pygments_classes(self):
        # The design handoff's colour table used prototype-only names that
        # never appear in production HTML — styling those would look like a
        # working theme sitting over completely uncoloured code. Comments are
        # stripped first because the note explaining that names them.
        src = _template("chat.html")
        rules = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        self.assertNotIn(".hl-kw", rules)
        for cls in (".k,", ".c1,", ".nf,", ".mi,", ".s2,"):
            with self.subTest(cls=cls):
                self.assertIn(cls, rules)

    def test_copy_button_has_a_non_secure_context_fallback(self):
        # navigator.clipboard is undefined over plain http — which is how this
        # panel is normally reached — and rejects transiently when the document
        # is not focused, so the legacy path must be chained, not an else-branch.
        src = _template("chat.html")
        self.assertIn("execCommand", src)
        self.assertIn("isSecureContext", src)
        self.assertIn(".catch(legacyCopy)", src)


class TestSecurityHeaders(unittest.TestCase):
    """CSP is applied last, to finished markup. It is also the single easiest
    way to silently kill the chat client, so the nonce is asserted."""

    def test_csp_present_on_chat(self):
        client = _build_client()
        _login(client)
        csp = client.get("/chat").headers["content-security-policy"]
        self.assertIn("default-src 'none'", csp)
        self.assertIn("script-src 'nonce-", csp)
        self.assertIn("object-src 'none'", csp)
        self.assertIn("frame-ancestors 'none'", csp)

    def test_font_src_is_allowed(self):
        # Without font-src every @font-face is blocked, self-hosted or not.
        client = _build_client()
        csp = client.get("/chat").headers["content-security-policy"]
        self.assertIn("font-src 'self'", csp)

    def test_style_src_keeps_unsafe_inline_and_no_nonce(self):
        # A nonce on style-src makes 'unsafe-inline' be IGNORED, killing every
        # style="" attribute and <style> block on the page.
        client = _build_client()
        csp = client.get("/chat").headers["content-security-policy"]
        style = [d for d in csp.split(";") if d.strip().startswith("style-src")][0]
        self.assertIn("'unsafe-inline'", style)
        self.assertNotIn("nonce-", style)

    def test_inline_chat_script_carries_the_nonce(self):
        # chat.html is one large inline script; without a matching nonce the
        # whole client is dead and the page merely looks fine.
        client = _build_client()
        _login(client)
        # ONE request: the nonce is per-response, so comparing a body from one
        # request against a header from another would always fail.
        r = client.get("/chat")
        self.assertIn("<script nonce=", r.text)
        # The nonce in the body must be the one the header authorises.
        nonce = re.search(r'<script nonce="([^"]+)"', r.text).group(1)
        self.assertIn(f"'nonce-{nonce}'", r.headers["content-security-policy"])

    def test_nonce_differs_per_response(self):
        client = _build_client()
        _login(client)
        a = re.search(r'<script nonce="([^"]+)"', client.get("/chat").text).group(1)
        b = re.search(r'<script nonce="([^"]+)"', client.get("/chat").text).group(1)
        self.assertNotEqual(a, b)

    def test_baseline_headers_present(self):
        client = _build_client()
        r = client.get("/chat")
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")
        self.assertEqual(r.headers["x-frame-options"], "DENY")
        self.assertEqual(r.headers["referrer-policy"], "no-referrer")

    def test_asset_route_keeps_its_stricter_own_csp(self):
        # chat_asset sets a sandboxing CSP; the middleware uses setdefault so
        # it must not be overwritten by the looser page policy.
        client = _build_client()
        _login(client)
        os.makedirs("output", exist_ok=True)
        marker = os.path.join("output", "_csp_marker.png")
        with open(marker, "wb") as f:
            f.write(_TINY_PNG)
        try:
            csp = client.get("/chat/assets/output/_csp_marker.png").headers[
                "content-security-policy"]
            self.assertIn("sandbox", csp)
            self.assertNotIn("nonce-", csp)
        finally:
            os.unlink(marker)


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


# The four TestChatSend cases were removed with the POST /chat/send handler
# they covered (DECISIONS.md #3). /chat/stream is the only send path, and
# TestChatStream* below covers it — including the identity-namespacing
# assertion these tests used to carry.


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
            # (delta, accumulated, restart) — one segment, so only the first
            # emission restarts.
            streaming_callback("Very", "Very", True)
            streaming_callback(" well", "Very well", False)
            streaming_callback(", sir.", "Very well, sir.", False)
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
        # Deltas, not growing prefixes. THIS assertion is the wire contract:
        # the old value was ["Very", "Very well", "Very well, sir."], which is
        # the O(n^2) behaviour this change removes.
        self.assertEqual(tokens, ["Very", " well", ", sir."])
        self.assertEqual("".join(tokens), "Very well, sir.")
        # Exactly one reset, opening the single segment.
        self.assertEqual(len([1 for ev, _ in events if ev == "reset"]), 1)
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
            streaming_callback("ok", "ok", True)
            return {"text": "ok", "image_paths": [], "timestamp": "now"}

        fake_module = MagicMock()
        fake_module.ask_stuff = fake_ask_stuff
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log") as audit:
            client.post("/chat/stream", data={"message": "ping"})

        calls = [c for c in audit.call_args_list if c.args and c.args[0] == "chat_message"]
        self.assertEqual(len(calls), 1)
        kwargs = calls[0].kwargs
        self.assertEqual(kwargs["user_id"], "web-alice")
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
        # The raw exception must NOT reach the browser — this assertion used to
        # require the opposite, which is what the leak looked like in tests.
        # The real message stays in the log and the audit entry.
        self.assertNotIn("ollama down", errors[0])
        self.assertIn("did not go to plan", errors[0])
        self.assertRegex(errors[0], r"ref `[0-9a-f]{8}`")
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
            streaming_callback("Very well", "Very well", True)
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
            streaming_callback("Here is what I found.", "Here is what I found.", True)
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
        # That the delete waits for a turn in flight is forget_conversation's
        # own job, tested in tests/test_forget_race.py through this endpoint.
        client = _build_client()
        _login(client)
        with patch.object(privacy, "forget_conversation", return_value=3) as fc, \
             patch.object(admin_panel, "audit_log") as audit:
            r = client.post("/chat/forget", follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/chat")
        # Must target the WEB thread — clearing "alice" here would wipe the
        # Discord conversation instead of the one the user is looking at.
        fc.assert_called_once_with("web-alice", thread_id="web-alice")
        # Audit entry recorded.
        calls = [c for c in audit.call_args_list if c.args and c.args[0] == "chat_forget_conversation"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["user_id"], "web-alice")
        self.assertEqual(calls[0].kwargs["removed"], 3)

    def test_unauthed_redirects_without_touching_privacy(self):
        client = _build_client()
        with patch.object(privacy, "forget_conversation") as fc:
            r = client.post("/chat/forget", follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        fc.assert_not_called()


class TestChatThreadId(unittest.TestCase):
    """The chat cookie's username is self-asserted, so the web chat must not
    share a LangGraph checkpoint with the Discord user of the same name.

    The namespace now lives in the identity itself, so _chat_thread_id is a
    pass-through and these tests operate on canonical ids.
    """

    def test_web_thread_is_the_identity(self):
        self.assertEqual(admin_panel._chat_thread_id("web-alice"), "web-alice")

    def test_web_thread_cannot_collide_with_a_discord_user(self):
        # The separator makes `web-alice` unreachable from any Discord id,
        # which are all `discord-<snowflake>`.
        self.assertNotEqual(
            admin_panel._chat_thread_id("web-alice"),
            admin_panel._chat_thread_id("discord-alice"),
        )

    def test_empty_user_gives_empty_thread(self):
        self.assertEqual(admin_panel._chat_thread_id(""), "")

    def test_login_mints_the_namespaced_identity(self):
        # The transformation moved from _chat_thread_id to _chat_identity;
        # this is where the `web-` prefix is applied now.
        self.assertEqual(fritz_utils.canonical_user_id("web", "al.i/ce"), "web-alice")
        self.assertEqual(fritz_utils.canonical_user_id("web", "a_b-c"), "web-a_b-c")

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
            admin_panel._load_chat_history("web-alice")
        self.assertEqual(captured["thread_id"], admin_panel._chat_thread_id("web-alice"))


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
        victim_dir = os.path.join("temp_images", ALICE_UPLOAD_DIR)
        os.makedirs(victim_dir, exist_ok=True)
        # Placed as chat_upload_image would place one of alice's uploads.
        victim = os.path.join(victim_dir, "123_secret.png")
        with open(victim, "wb") as f:
            f.write(_TINY_PNG)
        try:
            with patch.object(admin_panel, "audit_log") as audit:
                r = client.get(
                    f"/chat/assets/temp_images/{ALICE_UPLOAD_DIR}/123_secret.png")
            self.assertEqual(r.status_code, 404)
            self.assertEqual(audit.call_args.args[0], "chat_asset_denied")
        finally:
            os.unlink(victim)

    def test_user_can_fetch_their_own_upload(self):
        client = _build_client()
        _login(client, "alice")
        mine_dir = os.path.join("temp_images", ALICE_UPLOAD_DIR)
        os.makedirs(mine_dir, exist_ok=True)
        mine = os.path.join(mine_dir, "123_mine.png")
        with open(mine, "wb") as f:
            f.write(_TINY_PNG)
        try:
            r = client.get(f"/chat/assets/temp_images/{ALICE_UPLOAD_DIR}/123_mine.png")
            self.assertEqual(r.status_code, 200)
        finally:
            os.unlink(mine)


class TestChatAssetOwnership(unittest.TestCase):
    """Regression: ownership is a directory match, not a filename prefix.

    The prefix form was exploitable because '_' is a legal identity character
    (fritz_utils.safe_user_token keeps it), so
    "web-alice_2_<ts>_secret.png".startswith("web-alice_") was True and
    web-alice could read web-alice_2's uploads. Verified live before the fix:
    HTTP 200, image/png, and no chat_asset_denied audit entry.
    """

    def test_identity_that_is_a_prefix_of_another_cannot_read_their_uploads(self):
        client = _build_client()
        _login(client, "alice")          # identity web-alice
        # web-alice_2 is a DIFFERENT person whose identity happens to start
        # with alice's, exactly the collision the old check missed.
        victim_dir = os.path.join("temp_images", "web-alice_2")
        os.makedirs(victim_dir, exist_ok=True)
        victim = os.path.join(victim_dir, "123_secret.png")
        with open(victim, "wb") as f:
            f.write(_TINY_PNG)
        try:
            with patch.object(admin_panel, "audit_log") as audit:
                r = client.get("/chat/assets/temp_images/web-alice_2/123_secret.png")
            self.assertEqual(r.status_code, 404)
            self.assertEqual(audit.call_args.args[0], "chat_asset_denied")
        finally:
            os.unlink(victim)
            os.rmdir(victim_dir)

    def test_upload_lands_in_a_per_user_directory(self):
        """The write path and the ownership check have to agree, or every
        upload 404s for its own uploader."""
        client = _build_client()
        _login(client, "alice")
        r = client.post(
            "/chat/upload/image",
            files={"file": ("shot.png", _TINY_PNG, "image/png")},
        )
        self.assertEqual(r.status_code, 200, r.text)
        url = r.json()["url"]
        self.assertIn(f"/temp_images/{ALICE_UPLOAD_DIR}/", url)
        # And the uploader can actually fetch it back.
        self.assertEqual(client.get(url).status_code, 200)


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
        _drain_for_user(ALICE_ID)

    def tearDown(self):
        _drain_for_user(ALICE_ID)
        # Clean up the temp_images dir of any test artefacts.
        if os.path.isdir("temp_images"):
            for f in os.listdir("temp_images"):
                if f.startswith(ALICE_FILE_PREFIX) or f.startswith("_admin_panel_test_"):
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
            pending = list(admin_panel._pending_images.get(ALICE_ID, []))
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


class TestFacetedControlsKeepAFocusRing(unittest.TestCase):
    """clip-path clips outline, so a faceted control needs an unclipped .facet
    wrapper or it has no visible focus indicator at all (WCAG 2.4.7).

    Source-level because the failure is invisible without a browser: the
    control still works, still focuses, and still activates — the ring simply
    is not painted. Nothing else in the suite would notice.
    """

    def test_suggestion_chips_are_wrapped(self):
        src = _template("chat.html")
        # Server-rendered block.
        self.assertNotIn('<button type="button" class="suggestion">', src.replace(
            '<span class="facet"><button type="button" class="suggestion">', ""))
        # JS-rebuilt block (renderEmptyState) goes through the helper.
        self.assertIn("chips.appendChild(faceted(b))", src)

    def test_copy_buttons_are_wrapped(self):
        src = _template("chat.html")
        self.assertIn("faceted(copy)", src)

    def test_faceted_helper_exists(self):
        src = _template("chat.html")
        self.assertIn("function faceted(", src)
        self.assertIn('span.className = "facet"', src)

    def test_control_inside_a_clipped_parent_uses_a_filled_indicator(self):
        """The attachment × sits inside .attach-chip, which is itself clipped —
        a wrapper there would be clipped by the ancestor, so it needs a focus
        style that paints inside the element's own box."""
        src = _template("chat.html")
        self.assertIn(".attach-chip button:focus-visible", src)



    # ── The generic guard ────────────────────────────────────────────────
    # The assertions above name specific controls, so they can only catch a
    # regression in one someone already thought of — they did NOT catch the
    # suggestion chips or the copy buttons, which is what this class was
    # written for. This derives the set from the stylesheet and demands that
    # every clip-path'd selector be CLASSIFIED, so a new faceted control fails
    # here the day it is added rather than shipping with no focus ring.

    # Focusable, and wrapped in an unclipped .facet that carries the ring.
    WRAPPED = {
        ".attach-btn", ".btn-danger", ".copy-btn", ".send-btn",
        ".stop-btn", ".suggestion",
    }
    # Not focusable — decoration, bubbles and containers.
    DECORATIVE = {
        ".attach-chip", ".attach-chip .gem", ".avatar", ".code-bar .diamond",
        ".confirm-badge", ".confirm-card", ".empty-seal", ".status-chip",
        ".msg-row.fritz .bubble", ".msg-row.user .bubble",
    }

    def _clipped_selectors(self, src):
        """Every selector in the <style> block that sets a clip-path."""
        found = set()
        for block in re.finditer(r"([^{}]+)\{([^{}]*)\}", src):
            selector, body = block.group(1), block.group(2)
            if "clip-path" not in body:
                continue
            for part in selector.split(","):
                part = part.strip()
                if part.startswith("."):
                    found.add(part.split(":")[0].strip())
        return found

    def test_every_clipped_selector_is_classified(self):
        """Fails on a NEW faceted control, which the named assertions cannot.

        clip-path clips outline, so a focusable control that is neither wrapped
        nor deliberately exempt ships with no visible focus ring and nothing
        else in the suite notices.
        """
        clipped = self._clipped_selectors(_template("chat.html"))
        unclassified = sorted(clipped - self.WRAPPED - self.DECORATIVE)
        self.assertEqual(
            unclassified, [],
            "these selectors set clip-path but are not classified: "
            f"{unclassified}. clip-path clips outline, so if any is focusable "
            "it has NO focus ring. Wrap it in .facet and add it to WRAPPED, or "
            "add it to DECORATIVE if it can never take focus.",
        )

    def test_the_wrapped_set_still_matches_the_stylesheet(self):
        """Catches the opposite drift: a control that lost its clip-path, or
        was renamed, leaving a stale entry that guards nothing."""
        clipped = self._clipped_selectors(_template("chat.html"))
        stale = sorted((self.WRAPPED | self.DECORATIVE) - clipped)
        self.assertEqual(stale, [],
                         f"these no longer set clip-path: {stale}")


class TestConfirmDialogTrapsFocus(unittest.TestCase):
    """aria-modal tells assistive tech the rest of the page is inert; it does
    not change the browser's tab order. Without an explicit trap a keyboard
    user tabs out of a modal asking whether to destroy their conversation."""

    def test_tab_is_trapped_inside_the_dialog(self):
        src = _template("chat.html")
        self.assertIn('e.key !== "Tab"', src)
        self.assertIn("veil.contains(active)", src)

    def test_dialog_carries_modal_semantics(self):
        src = _template("chat.html")
        self.assertIn('role="dialog"', src)
        self.assertIn('aria-modal="true"', src)
        self.assertIn('aria-labelledby="confirm-title"', src)

    def test_focus_moves_in_and_is_restored(self):
        src = _template("chat.html")
        self.assertIn("lastFocused = document.activeElement", src)
        self.assertIn("confirmCancel.focus()", src)


class TestMessageTimestamps(unittest.TestCase):
    """DECISIONS.md #16: messages carry a creation timestamp.

    LangGraph checkpoints have no time of their own, so mister_fritz stamps
    additional_kwargs["ts"] when the message is built. Without it the chat
    could only show times for messages it watched arrive live, and every page
    reload would look like the timestamps had been lost.
    """

    def test_ts_is_surfaced_from_additional_kwargs(self):
        from langchain_core.messages import AIMessage, HumanMessage
        human = HumanMessage(content="hello", additional_kwargs={"ts": "2026-08-03T12:00:00+00:00"})
        ai = AIMessage(content="Good day.", additional_kwargs={"ts": "2026-08-03T12:00:05+00:00"})
        self.assertEqual(admin_panel._doc_to_message(human)["ts"], "2026-08-03T12:00:00+00:00")
        self.assertEqual(admin_panel._doc_to_message(ai)["ts"], "2026-08-03T12:00:05+00:00")

    def test_message_without_ts_yields_none_not_a_guess(self):
        """History written before stamping landed must render without a time
        rather than with an invented one."""
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="from before the change")
        converted = admin_panel._doc_to_message(msg)
        self.assertIn("ts", converted)
        self.assertIsNone(converted["ts"])

    def test_mister_fritz_stamps_an_iso_timestamp(self):
        import mister_fritz
        ts = mister_fritz._now_iso()
        # Parses as an aware ISO-8601 instant.
        parsed = datetime.datetime.fromisoformat(ts)
        self.assertIsNotNone(parsed.tzinfo)


class TestUploadImageMetadata(unittest.TestCase):
    """DECISIONS.md #16: the upload response carries the dimensions and format
    the chat's image card captions with. They come from the sniffed bytes, so
    they cannot be spoofed by the filename or the declared Content-Type."""

    def setUp(self):
        _drain_for_user(ALICE_ID)

    def tearDown(self):
        _drain_for_user(ALICE_ID)

    def test_response_carries_sniffed_dimensions_and_format(self):
        client = _build_client()
        _login(client)
        r = client.post("/chat/upload/image",
                        files={"file": ("shot.png", _TINY_PNG, "image/png")})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        # _TINY_PNG is a 1x1 PNG; whatever it is, the server must report the
        # real numbers rather than omitting them.
        self.assertEqual(body["format"], "PNG")
        self.assertIsInstance(body["width"], int)
        self.assertIsInstance(body["height"], int)
        self.assertGreater(body["width"], 0)
        self.assertGreater(body["height"], 0)

    def test_sniff_image_returns_format_and_size(self):
        self.assertIsNone(admin_panel._sniff_image(_HTML_PAYLOAD))
        self.assertIsNone(admin_panel._sniff_image(b""))
        fmt, w, h = admin_panel._sniff_image(_TINY_PNG)
        self.assertEqual(fmt, "PNG")
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)


class TestChatUploadImageSniffing(unittest.TestCase):
    """The declared Content-Type is not evidence — the bytes are."""

    def setUp(self):
        _drain_for_user(ALICE_ID)

    def tearDown(self):
        _drain_for_user(ALICE_ID)
        if os.path.isdir("temp_images"):
            for f in os.listdir("temp_images"):
                if f.startswith(ALICE_FILE_PREFIX):
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
            pending = list(admin_panel._pending_images.get(ALICE_ID, []))
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
        _drain_for_user(ALICE_ID)

    def tearDown(self):
        _drain_for_user(ALICE_ID)

    def test_send_picks_up_pending_image_and_clears(self):
        client = _build_client()
        _login(client)
        admin_panel._stash_pending_image(ALICE_ID, "/tmp/fake-img.png")

        captured = {}

        def fake_ask_stuff(message, source, user, *,
                          user_image_paths=None, streaming_callback=None, **_):
            captured["source"] = source
            captured["images"] = user_image_paths
            return {"text": "ok", "image_paths": [], "timestamp": "now"}

        fake_module = MagicMock()
        fake_module.ask_stuff = fake_ask_stuff
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log"):
            client.post("/chat/stream", data={"message": "look at this"})

        # ask_stuff received the stashed image path.
        self.assertEqual(captured["images"], ["/tmp/fake-img.png"])
        # Compare by enum name, not identity — other tests in the suite
        # (test_workspace_store) reload fritz_utils, which creates a new
        # MessageSource class that won't `==` the one admin_panel captured.
        self.assertEqual(captured["source"].name, "DISCORD_TEXT_AND_IMAGE")
        # The pending registry is empty after consumption.
        with admin_panel._pending_images_lock:
            self.assertNotIn(ALICE_ID, admin_panel._pending_images)

    def test_send_without_pending_uses_local_source(self):
        client = _build_client()
        _login(client)

        captured = {}

        def fake_ask_stuff(message, source, user, *,
                          user_image_paths=None, streaming_callback=None, **_):
            captured["source"] = source
            captured["images"] = user_image_paths
            return {"text": "ok", "image_paths": [], "timestamp": "now"}

        fake_module = MagicMock()
        fake_module.ask_stuff = fake_ask_stuff
        with patch.dict(sys.modules, {"mister_fritz": fake_module}), \
             patch.object(admin_panel, "audit_log"):
            client.post("/chat/stream", data={"message": "just text"})

        self.assertIsNone(captured["images"])
        self.assertEqual(captured["source"].name, "LOCAL")


class TestWebThreadBranchesLikeAnyChannel(unittest.TestCase):
    """The web surface passed no channel key. Identical while
    THREADS_PER_CHANNEL was off — and silently the ONE surface that kept a
    single global thread the moment it was turned on."""

    def test_web_thread_is_per_channel_when_the_flag_is_on(self):
        with patch.object(fritz_utils, "THREADS_PER_CHANNEL", True):
            self.assertEqual(admin_panel._chat_thread_id("web-alice"),
                             "web-alice#web")

    def test_web_thread_is_the_identity_when_the_flag_is_off(self):
        with patch.object(fritz_utils, "THREADS_PER_CHANNEL", False):
            self.assertEqual(admin_panel._chat_thread_id("web-alice"), "web-alice")

    def test_empty_identity_yields_empty(self):
        self.assertEqual(admin_panel._chat_thread_id(""), "")




class TestCodeBlockLabelsStayAligned(unittest.TestCase):
    """Languages were zipped onto rendered blocks by position, but only
    LABELLED fences were collected — so an unlabelled fence (or an indented
    code block) shifted every label one block along and the wrong language
    was announced."""

    # Built with join() rather than escapes: a literal newline in a fixture
    # is easier to read and impossible to mangle.
    UNLABELLED_THEN_PYTHON = chr(10).join(
        ['```', 'SELECT 1;', '```', '', '```python', 'x = 1', '```', ''])
    INDENTED_THEN_PYTHON = chr(10).join(
        ['    indented = 1', '', '```python', 'x = 1', '```', ''])
    TWO_LABELLED = chr(10).join(
        ['```python', 'x = 1', '```', '', '```js', 'let y = 2', '```', ''])

    def _tags(self, src):
        html = admin_panel._render_markdown(src)
        return re.findall(r'<div class="codehilite"[^>]*>', html)

    def test_unlabelled_fence_does_not_steal_the_next_label(self):
        tags = self._tags(self.UNLABELLED_THEN_PYTHON)
        self.assertEqual(len(tags), 2)
        self.assertNotIn('data-lang', tags[0])
        self.assertIn('data-lang="python"', tags[1])

    def test_indented_block_makes_it_label_nothing_rather_than_guess(self):
        """An indented block renders with no fence at all, so the lists
        cannot be aligned. A wrong language is worse than none."""
        tags = self._tags(self.INDENTED_THEN_PYTHON)
        self.assertEqual(len(tags), 2)
        for tag in tags:
            self.assertNotIn('data-lang', tag)

    def test_consecutive_labelled_fences_keep_their_own_languages(self):
        tags = self._tags(self.TWO_LABELLED)
        self.assertIn('data-lang="python"', tags[0])
        self.assertIn('data-lang="js"', tags[1])

    def test_fence_languages_records_openers_only(self):
        langs = admin_panel._fence_languages(self.UNLABELLED_THEN_PYTHON)
        self.assertEqual(langs, [None, 'python'])

if __name__ == "__main__":
    unittest.main()


class TestAnotherSiteCannotDriveThePanel(unittest.TestCase):
    """Nothing stopped it. The destructive routes are plain POSTs with no
    token, and the panel authenticates with HTTP Basic, which has no SameSite
    attribute - once a browser has cached the credentials for
    http://127.0.0.1:8001 it replays them on any request to that origin,
    including one caused by a page on another port or another site.

    Sec-Fetch-Site is what the guard reads, because a page cannot forge it:
    browsers set it themselves and script cannot change it.
    """

    DESTRUCTIVE = (
        ("/users/alice/forget", {}),
        ("/users/alice/workspace/disable", {}),
        ("/schedules/abc123/cancel", {}),
        ("/documents/reindex", {"name": "notes.pdf"}),
    )

    def setUp(self):
        self.client = _build_client(schedule_manager=MagicMock())

    def _post(self, path, data=None, site=None, origin=None, host=None):
        headers = dict(_auth_header())
        if site is not None:
            headers["Sec-Fetch-Site"] = site
        if origin is not None:
            headers["Origin"] = origin
        if host is not None:
            headers["Host"] = host
        return self.client.post(path, data=data or {}, headers=headers,
                                follow_redirects=False)

    def test_a_cross_site_post_is_refused_on_every_destructive_route(self):
        with patch.object(privacy, "forget_all", return_value={}) as forget, \
                patch.object(privacy, "forget_workspace", return_value=True) as workspace:
            for path, data in self.DESTRUCTIVE:
                with self.subTest(path=path):
                    response = self._post(path, data, site="cross-site")
                    self.assertEqual(response.status_code, 403)
            forget.assert_not_called()
            workspace.assert_not_called()

    def test_a_same_site_post_is_refused_too(self):
        """The local case this is really for: on localhost a page served from
        another port is same-SITE but a different origin."""
        with patch.object(privacy, "forget_all", return_value={}) as forget:
            response = self._post("/users/alice/forget", site="same-site")
        self.assertEqual(response.status_code, 403)
        forget.assert_not_called()

    def test_the_panels_own_form_still_works(self):
        with patch.object(privacy, "forget_all", return_value={}) as forget, \
                patch.object(admin_panel, "audit_log"):
            response = self._post("/users/alice/forget", site="same-origin")
        self.assertEqual(response.status_code, 303)
        forget.assert_called_once()

    def test_a_typed_url_or_bookmark_still_works(self):
        """Sec-Fetch-Site: none is a user-initiated request with no initiator
        site, so there is no other site to be acting on behalf of."""
        with patch.object(privacy, "forget_all", return_value={}) as forget, \
                patch.object(admin_panel, "audit_log"):
            response = self._post("/users/alice/forget", site="none")
        self.assertEqual(response.status_code, 303)
        forget.assert_called_once()

    def test_reading_the_panel_from_anywhere_is_still_allowed(self):
        """Only state-changing methods are gated; a cross-site GET cannot
        change anything, and refusing it would break nothing useful."""
        response = self.client.get("/users", headers={
            **_auth_header(), "Sec-Fetch-Site": "cross-site"})
        self.assertEqual(response.status_code, 200)

    def test_a_scripted_client_with_the_password_is_still_allowed(self):
        """Deliberate, not a gap: no browser omits Sec-Fetch-Site on a POST,
        so a header-less request is curl or a script — the operator, who
        already has the password."""
        with patch.object(privacy, "forget_all", return_value={}) as forget, \
                patch.object(admin_panel, "audit_log"):
            response = self._post("/users/alice/forget")
        self.assertEqual(response.status_code, 303)
        forget.assert_called_once()

    def test_an_older_browser_is_covered_by_the_origin_compare(self):
        with patch.object(privacy, "forget_all", return_value={}) as forget:
            response = self._post("/users/alice/forget",
                                  origin="http://evil.example")
        self.assertEqual(response.status_code, 403)
        forget.assert_not_called()

    def test_an_origin_matching_the_host_is_allowed(self):
        with patch.object(privacy, "forget_all", return_value={}) as forget, \
                patch.object(admin_panel, "audit_log"):
            response = self._post("/users/alice/forget",
                                  origin="http://testserver", host="testserver")
        self.assertEqual(response.status_code, 303)
        forget.assert_called_once()

    def test_the_refusal_is_audited(self):
        """An operator going back through the log should see that something
        tried, and from where."""
        with patch.object(admin_panel, "audit_log") as audit:
            self._post("/users/alice/forget", site="cross-site")
        audit.assert_called_once()
        self.assertEqual(audit.call_args.args[0], "admin_cross_origin_refused")
        self.assertIn("cross-site", audit.call_args.kwargs["detail"])

    def test_authentication_is_still_checked_first(self):
        """The guard sits INSIDE the auth middleware: a cross-origin POST with
        no credentials must be told it is unauthenticated, not handed a verdict
        about its origin."""
        response = self.client.post("/users/alice/forget",
                                    headers={"Sec-Fetch-Site": "cross-site"},
                                    follow_redirects=False)
        self.assertEqual(response.status_code, 401)

    def test_the_chat_surface_is_gated_the_same_way(self):
        """It has its own cookie identity and is exempt from the admin
        password, which would make it the obvious way round this."""
        response = self.client.post("/chat/forget",
                                    headers={"Sec-Fetch-Site": "cross-site"},
                                    follow_redirects=False)
        self.assertEqual(response.status_code, 403)
