"""
Regression tests for the admin gating on schedule commands.

Originally added for the Phase 2 ROOT_USER gate; updated in Phase 7a to
exercise the new is_admin() abstraction (ROOT_USER + ADMIN_USERS).

Slash commands in discord.py are wrapped in an app_commands.Command descriptor,
so we exercise the underlying callback (the .callback attribute) directly with
a fake Interaction object. That sidesteps the need for a live Discord client.
"""
import asyncio
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

# bot_commands pulls in mister_fritz → agent_tools → ddgs, plus document_engine
# / image_generator / tts. All are stubbed in tests/conftest.py.

import fritz_utils  # noqa: E402
import bot_commands  # noqa: E402
from bot_commands import FritzCommands, _require_admin  # noqa: E402


ROOT_NAME = "root_test_user"


def _fake_interaction(username: str) -> MagicMock:
    """Build an Interaction-like object that records the response message."""
    interaction = MagicMock()
    interaction.user = MagicMock()
    interaction.user.name = username
    interaction.channel_id = 12345
    interaction.guild_id = 67890
    interaction.response = MagicMock()
    interaction.response.send_message = AsyncMock()
    # /voice, /gen and /lore defer first and answer via followup. Without these
    # AsyncMocks `await interaction.response.defer(...)` raises TypeError against
    # a bare MagicMock — which is precisely why those three commands had never
    # been covered by a test.
    interaction.response.defer = AsyncMock()
    interaction.followup = MagicMock()
    interaction.followup.send = AsyncMock()
    interaction.channel = MagicMock()
    interaction.channel.send = AsyncMock()
    # Force the "not connected to a voice channel" branch of voice_slash.
    interaction.guild = None
    return interaction


def _make_cog(schedule_manager=None) -> FritzCommands:
    return FritzCommands(
        bot=MagicMock(),
        sayer=MagicMock(),
        schedule_manager=schedule_manager,
    )


def _patch_admins(root: str | None = ROOT_NAME, extras: tuple[str, ...] = ()):
    """Patch ROOT_USER and ADMIN_USERS in fritz_utils for the test scope."""
    return unittest.mock.patch.multiple(
        fritz_utils,
        ROOT_USER=root,
        ADMIN_USERS=frozenset(extras),
    )


class TestRequireAdmin(unittest.IsolatedAsyncioTestCase):
    async def test_root_user_allowed(self):
        with _patch_admins():
            interaction = _fake_interaction(ROOT_NAME)
            allowed = await _require_admin(interaction)
        self.assertTrue(allowed)
        interaction.response.send_message.assert_not_called()

    async def test_admin_user_from_list_allowed(self):
        with _patch_admins(extras=("alice",)):
            interaction = _fake_interaction("alice")
            allowed = await _require_admin(interaction)
        self.assertTrue(allowed)
        interaction.response.send_message.assert_not_called()

    async def test_non_admin_user_rejected_with_ephemeral(self):
        with _patch_admins():
            interaction = _fake_interaction("evil_user")
            allowed = await _require_admin(interaction)
        self.assertFalse(allowed)
        interaction.response.send_message.assert_called_once()
        _, kwargs = interaction.response.send_message.call_args
        self.assertTrue(kwargs.get("ephemeral"))


class TestScheduleAddOpenToAll(unittest.IsolatedAsyncioTestCase):
    """Phase 7c: any user can add their own schedules."""

    async def test_non_admin_can_schedule_add(self):
        manager = MagicMock()
        manager.add_schedule.return_value = "abc12345"
        cog = _make_cog(schedule_manager=manager)
        with _patch_admins():
            interaction = _fake_interaction("regular_user")
            await cog.schedule_add.callback(
                cog, interaction, every="1h", prompt="hello", description=None
            )
        manager.add_schedule.assert_called_once()
        # Ownership is keyed by the caller's name, not by ROOT.
        _, kwargs = manager.add_schedule.call_args
        self.assertEqual(kwargs.get("user_id"), "regular_user")

    async def test_value_error_surfaces_as_user_message(self):
        # Schedule cap exceeded, bad cron, etc. all raise ValueError.
        manager = MagicMock()
        manager.add_schedule.side_effect = ValueError("max 10 schedules")
        cog = _make_cog(schedule_manager=manager)
        with _patch_admins():
            interaction = _fake_interaction("regular_user")
            await cog.schedule_add.callback(
                cog, interaction, every="1h", prompt="hello", description=None
            )
        interaction.response.send_message.assert_called_once()
        args, _ = interaction.response.send_message.call_args
        self.assertIn("max 10", args[0])


class TestScheduleRemoveOpenToAll(unittest.IsolatedAsyncioTestCase):
    """Phase 7c: any user can remove their own schedules. The scheduler
    enforces per-user ownership via remove_schedule's PermissionError."""

    async def test_non_admin_can_remove_own_schedule(self):
        manager = MagicMock()
        manager.remove_schedule.return_value = True
        cog = _make_cog(schedule_manager=manager)
        with _patch_admins():
            interaction = _fake_interaction("regular_user")
            await cog.schedule_remove.callback(cog, interaction, schedule_id="abc12345")
        manager.remove_schedule.assert_called_once_with("abc12345", "regular_user")

    async def test_permission_error_surfaces_to_caller(self):
        # Attempting to remove another user's schedule.
        manager = MagicMock()
        manager.remove_schedule.side_effect = PermissionError("not your schedule")
        cog = _make_cog(schedule_manager=manager)
        with _patch_admins():
            interaction = _fake_interaction("regular_user")
            await cog.schedule_remove.callback(cog, interaction, schedule_id="abc12345")
        interaction.response.send_message.assert_called_once()


class TestScheduleListOpenToAll(unittest.IsolatedAsyncioTestCase):
    async def test_non_admin_can_list_their_own_schedules(self):
        manager = MagicMock()
        manager.list_schedules.return_value = []
        cog = _make_cog(schedule_manager=manager)
        with _patch_admins():
            interaction = _fake_interaction("regular_user")
            await cog.schedule_list.callback(cog, interaction)
        # list_schedules is read-only and per-user, so it's open by design.
        manager.list_schedules.assert_called_once_with("regular_user")


class TestScheduleListAllAdminOnly(unittest.IsolatedAsyncioTestCase):
    """Phase 7c: /schedule list_all shows every user's schedules — admin only."""

    async def test_non_admin_blocked_from_list_all(self):
        manager = MagicMock()
        cog = _make_cog(schedule_manager=manager)
        with _patch_admins():
            interaction = _fake_interaction("regular_user")
            await cog.schedule_list_all.callback(cog, interaction)
        interaction.response.send_message.assert_called_once()
        manager.list_all_schedules.assert_not_called()

    async def test_admin_can_view_all_schedules(self):
        manager = MagicMock()
        manager.list_all_schedules.return_value = [
            {"id": "id1", "user_id": "alice", "prompt": "p1", "schedule": "1h",
             "description": "", "created": "now"},
            {"id": "id2", "user_id": "bob", "prompt": "p2", "schedule": "2h",
             "description": "weather", "created": "now"},
        ]
        cog = _make_cog(schedule_manager=manager)
        with _patch_admins():
            interaction = _fake_interaction(ROOT_NAME)
            await cog.schedule_list_all.callback(cog, interaction)
        manager.list_all_schedules.assert_called_once()
        interaction.response.send_message.assert_called_once()


class TestWorkspaceEnableOpenToAll(unittest.IsolatedAsyncioTestCase):
    """Phase 7b: any user can enable their own sandboxed workspace."""

    async def test_non_admin_can_enable_workspace(self):
        cog = _make_cog()
        with _patch_admins(), \
             unittest.mock.patch("bot_commands.workspace_store.enable_sandboxed",
                                 return_value="/tmp/workspaces/regular_user") as enable_mock:
            interaction = _fake_interaction("regular_user")
            await cog.workspace_enable.callback(cog, interaction)
        enable_mock.assert_called_once_with("regular_user")
        interaction.response.send_message.assert_called_once()
        # Confirm the response is ephemeral (workspace path is sensitive).
        _, kwargs = interaction.response.send_message.call_args
        self.assertTrue(kwargs.get("ephemeral"))

    async def test_anyone_can_disable_their_workspace(self):
        cog = _make_cog()
        with _patch_admins(), \
             unittest.mock.patch("bot_commands.workspace_store.remove", return_value=True):
            interaction = _fake_interaction("regular_user")
            await cog.workspace_disable.callback(cog, interaction)
        interaction.response.send_message.assert_called_once()

    async def test_status_shows_current_workspace(self):
        cog = _make_cog()
        with _patch_admins(), \
             unittest.mock.patch("bot_commands.workspace_store.get",
                                 return_value="/tmp/workspaces/alice"):
            interaction = _fake_interaction("alice")
            await cog.workspace_status.callback(cog, interaction)
        interaction.response.send_message.assert_called_once()


class TestWorkspaceSetAdminOnly(unittest.IsolatedAsyncioTestCase):
    """/workspace set <path> is admin-only — it registers an arbitrary host path."""

    async def test_non_admin_blocked_from_workspace_set(self):
        cog = _make_cog()
        with _patch_admins(), \
             unittest.mock.patch("bot_commands.workspace_store.set_path") as set_mock:
            interaction = _fake_interaction("not_root")
            await cog.workspace_set.callback(cog, interaction, path="/tmp/whatever")
        interaction.response.send_message.assert_called_once()
        set_mock.assert_not_called()


class TestBlockingWorkIsOffloaded(unittest.IsolatedAsyncioTestCase):
    """The three commands that used to run multi-second work directly on the
    Discord event loop, freezing heartbeats and every other interaction."""

    async def test_gen_runs_image_generation_off_the_loop(self):
        loop_thread = threading.get_ident()
        seen = {}

        def fake_generate(prompt):
            seen["thread"] = threading.get_ident()
            return "out.png"

        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch("bot_commands.generate_image", fake_generate), \
             unittest.mock.patch("bot_commands.discord.File", MagicMock()):
            await cog.gen_slash.callback(cog, interaction, prompt="a cat")
        self.assertNotEqual(seen["thread"], loop_thread)

    async def test_lore_runs_query_documents_off_the_loop(self):
        loop_thread = threading.get_ident()
        seen = {}

        def fake_query(q):
            seen["thread"] = threading.get_ident()
            return "the answer"

        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch("bot_commands.query_documents", fake_query):
            await cog.lore_slash.callback(cog, interaction, query="what?")
        self.assertNotEqual(seen["thread"], loop_thread)
        interaction.followup.send.assert_awaited_once_with("the answer")

    async def test_voice_runs_agent_and_tts_off_the_loop(self):
        loop_thread = threading.get_ident()
        seen = {}

        def fake_ask(*a, **k):
            seen["ask"] = threading.get_ident()
            return {"text": "spoken words"}

        def fake_speech(text):
            seen["tts"] = threading.get_ident()
            return "out.wav"

        cog = _make_cog()
        cog.sayer.generate_speech = fake_speech
        interaction = _fake_interaction("someone")
        with unittest.mock.patch("bot_commands.ask_stuff", fake_ask), \
             unittest.mock.patch("bot_commands.discord.File", MagicMock()):
            await cog.voice_slash.callback(cog, interaction, message="say this")
        self.assertNotEqual(seen["ask"], loop_thread)
        self.assertNotEqual(seen["tts"], loop_thread)

    async def test_voice_answers_the_deferred_interaction_on_failure(self):
        # Without the try/except a raised exception left the interaction
        # spinning until Discord timed it out.
        cog = _make_cog()
        interaction = _fake_interaction("someone")

        def boom(*a, **k):
            raise RuntimeError("ollama down")

        with unittest.mock.patch("bot_commands.ask_stuff", boom):
            await cog.voice_slash.callback(cog, interaction, message="say this")
        interaction.followup.send.assert_awaited_once()
        self.assertIn("Failed to generate speech",
                      interaction.followup.send.await_args.args[0])


class TestGpuSerialisation(unittest.IsolatedAsyncioTestCase):
    """SDXL and XTTS are serialised so newly-possible concurrency cannot
    thrash or OOM the GPU."""

    async def test_two_concurrent_gens_never_overlap(self):
        peak = {"now": 0, "max": 0}
        lock = threading.Lock()

        def fake_generate(prompt):
            with lock:
                peak["now"] += 1
                peak["max"] = max(peak["max"], peak["now"])
            time.sleep(0.05)
            with lock:
                peak["now"] -= 1
            return "out.png"

        cog = _make_cog()
        with unittest.mock.patch("bot_commands.generate_image", fake_generate), \
             unittest.mock.patch("bot_commands.discord.File", MagicMock()):
            await asyncio.gather(
                cog.gen_slash.callback(cog, _fake_interaction("a"), prompt="one"),
                cog.gen_slash.callback(cog, _fake_interaction("b"), prompt="two"),
            )
        self.assertEqual(peak["max"], 1)

    async def test_the_queued_notice_is_sent_when_busy(self):
        started = asyncio.Event()

        def slow_generate(prompt):
            time.sleep(0.1)
            return "out.png"

        cog = _make_cog()
        first = _fake_interaction("a")
        second = _fake_interaction("b")
        with unittest.mock.patch("bot_commands.generate_image", slow_generate), \
             unittest.mock.patch("bot_commands.discord.File", MagicMock()):
            task = asyncio.create_task(
                cog.gen_slash.callback(cog, first, prompt="one"))
            # Let the first take the semaphore before the second checks it.
            await asyncio.sleep(0.02)
            started.set()
            await cog.gen_slash.callback(cog, second, prompt="two")
            await task
        notices = [c.args[0] for c in second.followup.send.await_args_list if c.args]
        self.assertTrue(any("Queued" in n for n in notices), notices)

    async def test_semaphores_live_on_the_cog_instance(self):
        # Module-level asyncio primitives bind to whichever loop first awaited
        # them — which would wedge /gen after a gateway reconnect built a new
        # loop, and make these tests order-dependent.
        a, b = _make_cog(), _make_cog()
        self.assertIsNot(a._image_semaphore, b._image_semaphore)
        self.assertIsNot(a._tts_semaphore, b._tts_semaphore)

    async def test_semaphores_are_sized_from_config(self):
        # bot_commands imports the values by name at import time, so assert
        # against the module's own binding rather than patching fritz_utils.
        cog = _make_cog()
        self.assertEqual(cog._image_semaphore._value,
                         bot_commands.IMAGE_GEN_MAX_CONCURRENCY)
        self.assertEqual(cog._tts_semaphore._value,
                         bot_commands.TTS_MAX_CONCURRENCY)


if __name__ == "__main__":
    unittest.main()
