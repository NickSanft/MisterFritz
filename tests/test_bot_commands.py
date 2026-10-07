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
from unittest.mock import AsyncMock, MagicMock, patch

# bot_commands pulls in mister_fritz → agent_tools → ddgs, plus document_engine
# / image_generator / tts. All are stubbed in tests/conftest.py.

import discord  # noqa: E402
import fritz_utils  # noqa: E402
import privacy  # noqa: E402  - for PrivacyOperationFailed
import bot_commands  # noqa: E402
from bot_commands import FritzCommands, _require_admin  # noqa: E402


ROOT_NAME = "root_test_user"
# Every fake interaction shares one snowflake unless a test says otherwise, so
# the canonical id a command stores is predictable.
FAKE_USER_ID = 424242
FAKE_IDENTITY = "discord-424242"


def _fake_interaction(username: str, user_id: int = FAKE_USER_ID) -> MagicMock:
    """Build an Interaction-like object that records the response message."""
    interaction = MagicMock()
    interaction.user = MagicMock()
    interaction.user.name = username
    # Without an explicit int, `interaction.user.id` is an auto-created
    # MagicMock and canonical_user_id would stringify it into garbage — the
    # single most likely way for these tests to pass while asserting nonsense.
    interaction.user.id = user_id
    interaction.user.display_name = username
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


def _patch_admins(root: str | None = ROOT_NAME, extras: tuple[str, ...] = (),
                  legacy_names: bool = False):
    """Patch ROOT_USER, ADMIN_USERS and the legacy-name shim in fritz_utils.

    `legacy_names` has to be explicit: these tests are written against
    display names, so without pinning the flag they would pass through the
    compatibility path and silently stop testing the real gate.
    """
    return unittest.mock.patch.multiple(
        fritz_utils,
        ROOT_USER=root,
        ADMIN_USERS=frozenset(extras),
        ADMIN_LEGACY_NAME_MATCH=legacy_names,
    )


class TestRequireAdmin(unittest.IsolatedAsyncioTestCase):
    """The gate keys on the canonical id. The display-name path is reachable
    only via ADMIN_LEGACY_NAME_MATCH, which ships false (DECISIONS #6)."""

    async def test_root_user_by_canonical_id_allowed(self):
        with _patch_admins(root=FAKE_IDENTITY):
            interaction = _fake_interaction("anything-at-all")
            allowed = await _require_admin(interaction)
        self.assertTrue(allowed)
        interaction.response.send_message.assert_not_called()

    async def test_admin_user_from_list_by_canonical_id_allowed(self):
        with _patch_admins(root=None, extras=(FAKE_IDENTITY,)):
            interaction = _fake_interaction("alice")
            allowed = await _require_admin(interaction)
        self.assertTrue(allowed)
        interaction.response.send_message.assert_not_called()

    async def test_rename_does_not_revoke_admin(self):
        """The whole point: admin follows the snowflake, not the name."""
        with _patch_admins(root=FAKE_IDENTITY):
            renamed = _fake_interaction("brand_new_name")
            self.assertTrue(await _require_admin(renamed))

    async def test_legacy_name_rejected_when_flag_is_false(self):
        with _patch_admins(root=ROOT_NAME, legacy_names=False):
            interaction = _fake_interaction(ROOT_NAME)
            allowed = await _require_admin(interaction)
        self.assertFalse(allowed)

    async def test_legacy_name_allowed_when_flag_is_true(self):
        with _patch_admins(root=ROOT_NAME, legacy_names=True):
            interaction = _fake_interaction(ROOT_NAME)
            allowed = await _require_admin(interaction)
        self.assertTrue(allowed)

    async def test_canonical_root_is_not_matchable_by_name(self):
        """Once ROOT_USER is canonical, the legacy shim grants nothing — there
        is no display name that equals `discord-424242`. This is what makes
        migrating the config the thing that actually closes the rename hole,
        rather than flipping the flag."""
        with _patch_admins(root=FAKE_IDENTITY, legacy_names=True):
            impostor = _fake_interaction(ROOT_NAME, user_id=999)
            self.assertFalse(await _require_admin(impostor))

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
        # Ownership is keyed by the caller's canonical identity, not by
        # their display name and not by ROOT.
        _, kwargs = manager.add_schedule.call_args
        self.assertEqual(kwargs.get("user_id"), FAKE_IDENTITY)

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
        manager.remove_schedule.assert_called_once_with("abc12345", FAKE_IDENTITY)

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
        manager.list_schedules.assert_called_once_with(FAKE_IDENTITY)


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
        with _patch_admins(root=FAKE_IDENTITY):
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
        enable_mock.assert_called_once_with(FAKE_IDENTITY)
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
        with unittest.mock.patch("image_generator.generate_image", fake_generate), \
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

        interaction.response.is_done = MagicMock(return_value=True)
        with unittest.mock.patch("bot_commands.ask_stuff", boom):
            await cog.voice_slash.callback(cog, interaction, message="say this")
        interaction.followup.send.assert_awaited_once()
        sent = interaction.followup.send.await_args.args[0]
        # Butler copy plus a log ref — the exception itself stays in the log.
        self.assertNotIn("ollama down", sent)
        self.assertIn("did not go to plan", sent)


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
        with unittest.mock.patch("image_generator.generate_image", fake_generate), \
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
        with unittest.mock.patch("image_generator.generate_image", slow_generate), \
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


class TestAppCommandErrorHandler(unittest.IsolatedAsyncioTestCase):
    """Before this existed, an out-of-range argument or any uncaught exception
    left the user staring at a spinner until Discord timed the interaction out."""

    def _interaction(self, *, done=False):
        interaction = _fake_interaction("someone")
        interaction.response.is_done = MagicMock(return_value=done)
        interaction.command = MagicMock()
        interaction.command.qualified_name = "gen"
        return interaction

    async def test_invoke_error_replies_without_leaking_the_exception(self):
        interaction = self._interaction()
        err = discord.app_commands.CommandInvokeError(
            interaction.command, RuntimeError("secret internal detail"))
        await bot_commands.handle_app_command_error(interaction, err)
        interaction.response.send_message.assert_awaited_once()
        sent = interaction.response.send_message.await_args.args[0]
        self.assertNotIn("secret internal detail", sent)
        self.assertIn("did not go to plan", sent)

    async def test_reply_is_ephemeral(self):
        interaction = self._interaction()
        err = discord.app_commands.CommandInvokeError(
            interaction.command, RuntimeError("x"))
        await bot_commands.handle_app_command_error(interaction, err)
        self.assertTrue(
            interaction.response.send_message.await_args.kwargs.get("ephemeral"))

    async def test_uses_followup_when_already_deferred(self):
        # A command that has deferred must answer via followup; using response
        # would raise and compound the original failure.
        interaction = self._interaction(done=True)
        err = discord.app_commands.CommandInvokeError(
            interaction.command, RuntimeError("x"))
        await bot_commands.handle_app_command_error(interaction, err)
        interaction.followup.send.assert_awaited_once()
        interaction.response.send_message.assert_not_awaited()

    async def test_check_failure_gets_its_own_copy(self):
        interaction = self._interaction()
        await bot_commands.handle_app_command_error(
            interaction, discord.app_commands.CheckFailure("nope"))
        sent = interaction.response.send_message.await_args.args[0]
        self.assertIn("not available to you", sent)

    async def test_cog_hook_delegates_to_the_shared_handler(self):
        cog = _make_cog()
        interaction = self._interaction()
        err = discord.app_commands.CommandInvokeError(
            interaction.command, RuntimeError("x"))
        await cog.cog_app_command_error(interaction, err)
        interaction.response.send_message.assert_awaited_once()


class TestErrorsAreAnsweredOnce(unittest.IsolatedAsyncioTestCase):
    """discord.py runs a command's own handlers AND THEN tree.on_error.

    main_discord installed handle_app_command_error straight onto the tree, so
    every failure in the cog was handled twice: two error replies, two refs,
    two METRICS errors. These drive the real cog through the library's own
    dispatch sequence rather than calling one handler in isolation, because
    the bug only exists in the composition.
    """

    async def _bot_with_real_cog(self):
        from discord.ext import commands
        bot = commands.Bot(command_prefix="$", intents=discord.Intents.default())
        await bot.add_cog(_make_cog())
        bot.tree.on_error = bot_commands.handle_tree_error   # as main_discord does
        return bot

    def _interaction(self, command):
        interaction = _fake_interaction("someone")
        interaction.command = command
        answered = {"done": False}

        async def _answer(*a, **kw):
            answered["done"] = True
        interaction.response.is_done = MagicMock(side_effect=lambda: answered["done"])
        interaction.response.send_message = AsyncMock(side_effect=_answer)
        interaction.followup.send = AsyncMock()
        return interaction

    async def _dispatch_like_the_tree(self, bot, interaction, error):
        # app_commands/tree.py `_call`, verbatim in effect. Pinned against the
        # installed library by test_library_still_calls_both_in_this_order.
        command = interaction.command
        if command is not None:
            await command._invoke_error_handlers(interaction, error)
        await bot.tree.on_error(interaction, error)

    def replies(self, interaction):
        return (interaction.response.send_message.await_count
                + interaction.followup.send.await_count)

    async def test_a_cog_command_failure_is_answered_once(self):
        bot = await self._bot_with_real_cog()
        for path in (("health",), ("forget", "memories")):
            with self.subTest(command=" ".join(path)):
                command = bot.tree.get_command(path[0])
                for part in path[1:]:
                    command = command.get_command(part)
                interaction = self._interaction(command)
                err = discord.app_commands.CommandInvokeError(command, RuntimeError("x"))
                with patch.object(bot_commands, "fritz_error", wraps=bot_commands.fritz_error) as fe:
                    await self._dispatch_like_the_tree(bot, interaction, err)
                self.assertEqual(self.replies(interaction), 1)
                self.assertEqual(fe.call_count, 1, "the error was recorded twice")

    async def test_a_failure_outside_any_command_still_gets_the_backstop(self):
        """The case the tree handler exists for: a stale sync, no command."""
        bot = await self._bot_with_real_cog()
        interaction = self._interaction(None)
        await self._dispatch_like_the_tree(
            bot, interaction, discord.app_commands.CommandNotFound("gone", []))
        self.assertEqual(self.replies(interaction), 1)

    async def test_a_command_with_no_handler_of_its_own_gets_the_backstop(self):
        bot = await self._bot_with_real_cog()

        @discord.app_commands.command(name="bare", description="d")
        async def bare(interaction: discord.Interaction):
            pass
        bot.tree.add_command(bare)
        interaction = self._interaction(bare)
        await self._dispatch_like_the_tree(
            bot, interaction, discord.app_commands.CommandInvokeError(bare, RuntimeError("x")))
        self.assertEqual(self.replies(interaction), 1)

    def test_library_still_calls_both_in_this_order(self):
        """If discord.py ever stops double-dispatching, the replay above is
        stale and this is the test that should say so."""
        import inspect
        from discord.app_commands import tree
        src = inspect.getsource(tree.CommandTree._call)
        first = src.index("await command._invoke_error_handlers(interaction, e)")
        second = src.index("await self.on_error(interaction, e)")
        self.assertLess(first, second)

    def test_main_discord_installs_the_guarded_backstop(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1] / "main_discord.py").read_text(encoding="utf-8")
        self.assertTrue("client.tree.on_error = handle_tree_error" in src,
                        "main_discord no longer installs the guarded backstop")
        self.assertFalse("client.tree.on_error = handle_app_command_error" in src,
                         "the unguarded handler is back on the tree")


class TestEmbedsAndBounds(unittest.IsolatedAsyncioTestCase):
    async def test_health_is_ephemeral_and_an_embed(self):
        # Was public: metrics went into channel history for everyone to read.
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        await cog.health_slash.callback(cog, interaction)
        kwargs = interaction.response.send_message.await_args.kwargs
        self.assertTrue(kwargs.get("ephemeral"))
        self.assertIsInstance(kwargs.get("embed"), discord.Embed)

    async def test_about_reports_a_real_uptime(self):
        # The key is "uptime_sec", not "uptime_seconds" — the .get default
        # swallowed the miss and /about always said "0s".
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch.object(
                bot_commands, "get_health_snapshot",
                return_value={"uptime_sec": 3725, "counters": {}, "errors": {},
                              "latencies": {}, "last_error": None}):
            await cog.about_slash.callback(cog, interaction)
        embed = interaction.response.send_message.await_args.kwargs["embed"]
        uptime = [f.value for f in embed.fields if f.name == "Uptime"][0]
        self.assertNotEqual(uptime, "0s")
        self.assertIn("1h", uptime)

    async def test_help_is_an_embed(self):
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        await cog.help_slash.callback(cog, interaction)
        self.assertIsInstance(
            interaction.response.send_message.await_args.kwargs.get("embed"),
            discord.Embed)

    async def test_schedule_list_is_an_embed(self):
        manager = MagicMock()
        manager.list_schedules.return_value = [
            {"id": "abc", "schedule": "1h", "description": "d", "prompt": "p"},
        ]
        cog = _make_cog(schedule_manager=manager)
        interaction = _fake_interaction("someone")
        await cog.schedule_list.callback(cog, interaction)
        embed = interaction.response.send_message.await_args.kwargs["embed"]
        self.assertIsInstance(embed, discord.Embed)
        self.assertEqual(len(embed.fields), 1)

    async def test_schedule_list_caps_at_25_fields(self):
        # Discord rejects an embed with more than 25 fields outright.
        manager = MagicMock()
        manager.list_schedules.return_value = [
            {"id": str(i), "schedule": "1h", "description": "", "prompt": "p"}
            for i in range(40)
        ]
        cog = _make_cog(schedule_manager=manager)
        interaction = _fake_interaction("someone")
        await cog.schedule_list.callback(cog, interaction)
        embed = interaction.response.send_message.await_args.kwargs["embed"]
        self.assertEqual(len(embed.fields), 25)
        self.assertIn("Showing 25 of 40", embed.footer.text)

    async def test_draw_is_range_bounded(self):
        # /draw 500 previously raised an uncaught HTTPException and the user
        # saw nothing at all. Discord now rejects it before the command runs.
        import typing
        hints = typing.get_type_hints(
            bot_commands.FritzCommands.draw_slash.callback, include_extras=True)
        transformer = hints["num_cards"]
        self.assertEqual(type(transformer).__name__, "RangeTransformer")
        self.assertEqual(transformer.min_value, 1)
        self.assertEqual(transformer.max_value, 40)

    async def test_draw_chunks_its_output(self):
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        long_output = "card line here\n" * 400          # ~6000 chars
        with unittest.mock.patch.object(
                bot_commands, "draw_cards", return_value=long_output):
            await cog.draw_slash.callback(cog, interaction, num_cards=40)
        self.assertGreater(interaction.followup.send.await_count, 1)
        for call in interaction.followup.send.await_args_list:
            with self.subTest():
                self.assertLessEqual(len(call.kwargs["content"]), 2000)

    async def test_lore_continuations_use_followup_not_channel_send(self):
        # channel.send posts detached messages that interleave with other
        # traffic; followup keeps them attached to the interaction.
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch(
                "bot_commands.query_documents", return_value="word " * 900):
            await cog.lore_slash.callback(cog, interaction, query="q")
        self.assertGreater(interaction.followup.send.await_count, 1)
        interaction.channel.send.assert_not_awaited()

    async def test_lore_drops_the_over_2000_header(self):
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch(
                "bot_commands.query_documents", return_value="word " * 900):
            await cog.lore_slash.callback(cog, interaction, query="q")
        first = interaction.followup.send.await_args_list[0].args[0]
        self.assertNotIn("The answer was over", first)


class TestIdentityFromSnowflake(unittest.IsolatedAsyncioTestCase):
    """The cog's single conversion point. Every slash command used to key off
    `interaction.user.name`, so a Discord rename orphaned that person's
    memories, schedules and workspace — and a Telegram user with the same
    handle shared them."""

    def test_identity_is_derived_from_the_snowflake(self):
        interaction = _fake_interaction("Nick")
        self.assertEqual(bot_commands._identity(interaction), FAKE_IDENTITY)

    def test_renaming_does_not_change_the_identity(self):
        before = bot_commands._identity(_fake_interaction("Nick"))
        after = bot_commands._identity(_fake_interaction("Someone Else Entirely"))
        self.assertEqual(before, after)

    def test_two_users_get_two_identities(self):
        a = bot_commands._identity(_fake_interaction("same_name", user_id=1))
        b = bot_commands._identity(_fake_interaction("same_name", user_id=2))
        self.assertNotEqual(a, b)

    def test_display_name_is_kept_separately(self):
        interaction = _fake_interaction("Nick")
        self.assertEqual(bot_commands._display(interaction), "Nick")

    def test_identity_records_the_alias(self):
        # Nothing else knows the human's name once the id is a snowflake.
        with unittest.mock.patch.object(bot_commands.identity_store, "record") as rec:
            bot_commands._identity(_fake_interaction("Nick"))
        rec.assert_called_once_with(FAKE_IDENTITY, "Nick", "discord")

    async def test_forget_confirm_view_is_keyed_on_the_id(self):
        # A rename between opening the dialog and pressing Confirm used to
        # lock the requester out of their own confirmation.
        view = bot_commands._ForgetConfirmView(FAKE_IDENTITY, MagicMock())
        renamed = _fake_interaction("a_completely_new_name")
        self.assertTrue(await view.interaction_check(renamed))

    async def test_forget_confirm_view_rejects_a_different_user(self):
        view = bot_commands._ForgetConfirmView(FAKE_IDENTITY, MagicMock())
        other = _fake_interaction("someone", user_id=999)
        self.assertFalse(await view.interaction_check(other))

    def test_no_colon_reaches_a_filename(self):
        # DECISIONS #5: identities land in Windows paths in several places.
        self.assertNotIn(":", bot_commands._identity(_fake_interaction("Nick")))


class TestDeferredCommandsAlwaysAnswer(unittest.IsolatedAsyncioTestCase):
    """A deferred interaction that never receives a followup leaves the user
    staring at "thinking…" until Discord expires it.

    split_into_chunks returns [] for empty input, so iterating it was a silent
    no-op on an empty result — and empty is a REAL outcome for /lore, whose
    corpus may simply hold nothing on the subject. Note the obvious fix of
    `... or [text]` does not work either: `followup.send(content="")` raises.
    """

    async def test_lore_with_an_empty_result_still_sends_something(self):
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch("bot_commands.query_documents", lambda q: ""):
            await cog.lore_slash.callback(cog, interaction, query="anything")
        interaction.followup.send.assert_awaited_once()
        sent = interaction.followup.send.await_args.args[0]
        self.assertTrue(sent.strip(), "a deferred command sent empty content")

    async def test_draw_with_an_empty_result_still_sends_something(self):
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch("bot_commands.draw_cards", lambda *a, **k: ""):
            await cog.draw_slash.callback(cog, interaction, num_cards=1)
        interaction.followup.send.assert_awaited_once()
        sent = interaction.followup.send.await_args.kwargs.get("content")
        self.assertTrue((sent or "").strip(), "a deferred command sent empty content")

    async def test_draw_records_a_metric_like_every_other_command(self):
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch("bot_commands.draw_cards", lambda *a, **k: "AS"), \
             unittest.mock.patch.object(bot_commands.METRICS, "increment") as inc:
            await cog.draw_slash.callback(cog, interaction, num_cards=1)
        self.assertIn("discord_commands.draw",
                      [c.args[0] for c in inc.call_args_list])


class TestPrivacyWorkRunsOffTheLoop(unittest.IsolatedAsyncioTestCase):
    """privacy.* touches Chroma and SQLite. Left on the event loop it freezes
    every other user's streaming for the duration of a /forget."""

    async def test_forget_memories_is_offloaded(self):
        loop_thread = threading.get_ident()
        seen = {}

        def fake_forget(user_id):
            seen["thread"] = threading.get_ident()
            return 3

        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch.object(bot_commands.privacy, "forget_memories",
                                        fake_forget):
            await cog.forget_memories_slash.callback(cog, interaction)
        self.assertNotEqual(seen["thread"], loop_thread)


if __name__ == "__main__":
    unittest.main()


class TestMetricsMeasureTheRightThing(unittest.IsolatedAsyncioTestCase):
    """These bugs produce plausible-but-wrong numbers, which is worse than no
    numbers: nothing fails, the dashboard just quietly lies."""

    async def test_gen_failure_is_recorded_once_not_twice(self):
        """METRICS.time_block records the exception and re-raises; the except
        block then routed it through fritz_error, which recorded it again."""
        cog = _make_cog()
        interaction = _fake_interaction("someone")

        def boom(prompt):
            raise RuntimeError("gpu on fire")

        with unittest.mock.patch("image_generator.generate_image", boom), \
             unittest.mock.patch.object(bot_commands.METRICS, "record_error") as rec:
            await cog.gen_slash.callback(cog, interaction, prompt="a cat")
        gen_errors = [c for c in rec.call_args_list
                      if c.args and c.args[0] == "discord_commands.gen"]
        self.assertEqual(len(gen_errors), 1,
                         f"recorded {len(gen_errors)} times, expected 1")

    async def test_gen_latency_excludes_time_spent_queuing(self):
        """With IMAGE_GEN_MAX_CONCURRENCY=1 a queued /gen used to report the
        other render's duration as its own."""
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        recorded = {}

        def fake_latency(name, seconds):
            recorded.setdefault(name, []).append(seconds)

        # Hold the semaphore so the command has to wait for it.
        await cog._image_semaphore.acquire()

        async def release_soon():
            await asyncio.sleep(0.15)
            cog._image_semaphore.release()

        with unittest.mock.patch("image_generator.generate_image", lambda p: "out.png"), \
             unittest.mock.patch("bot_commands.discord.File", MagicMock()), \
             unittest.mock.patch.object(bot_commands.METRICS, "record_latency",
                                        fake_latency):
            await asyncio.gather(
                cog.gen_slash.callback(cog, interaction, prompt="a cat"),
                release_soon(),
            )

        work = recorded.get("discord_commands.gen", [None])[0]
        queue = recorded.get("discord_commands.gen.queue", [None])[0]
        self.assertIsNotNone(work, "work latency was not recorded")
        self.assertIsNotNone(queue, "queue latency was not recorded")
        # The wait was ~0.15s; the render is a no-op lambda.
        self.assertGreater(queue, 0.05, "queue time was not measured")
        self.assertLess(work, 0.05, "work latency still includes the queue wait")


class TestTheDestructiveCommandsAreAcknowledgedFirst(unittest.IsolatedAsyncioTestCase):
    """Each of these did its store work BEFORE making any response, so it could
    pass Discord's three-second deadline with the data already gone - and an
    expired token makes the reply fail, telling someone their deletion did not
    happen when it had. /voice, /gen and /lore already deferred.
    """

    SLASH = (
        ("forget_memories_slash", "forget_memories", ("user_id",)),
        ("forget_conversation_slash", "forget_conversation", ("user_id",)),
        ("forget_schedules_slash", "forget_schedules", ("user_id", "manager")),
    )

    async def _run(self, command: str, target: str, returns=3):
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch.object(bot_commands.privacy, target,
                                        MagicMock(return_value=returns)) as store:
            with unittest.mock.patch.object(bot_commands, "audit_log"):
                await getattr(cog, command).callback(cog, interaction)
        return interaction, store

    async def test_the_interaction_is_deferred_before_the_store_is_touched(self):
        for command, target, _args in self.SLASH:
            with self.subTest(command=command):
                interaction, store = await self._run(command, target)
                interaction.response.defer.assert_awaited_once()
                self.assertTrue(store.called, "the store was never called")

    async def test_the_outcome_is_reported_through_followup(self):
        """A deferred interaction can only be answered with a followup;
        response.send_message raises InteractionResponded against it."""
        for command, target, _args in self.SLASH:
            with self.subTest(command=command):
                interaction, _store = await self._run(command, target)
                interaction.followup.send.assert_awaited_once()
                interaction.response.send_message.assert_not_awaited()
                said = interaction.followup.send.await_args.args[0]
                self.assertIn("3", said)

    async def test_the_reply_stays_ephemeral(self):
        """It names what Fritz has stored about the person. The old call said
        so explicitly and a followup does not default to it."""
        for command, target, _args in self.SLASH:
            with self.subTest(command=command):
                interaction, _store = await self._run(command, target)
                self.assertIs(interaction.response.defer.await_args.kwargs["ephemeral"], True)
                self.assertIs(interaction.followup.send.await_args.kwargs["ephemeral"], True)


class TestForgetAllAnswersThePressItWasGiven(unittest.IsolatedAsyncioTestCase):
    """The longest of them: Chroma deletes, checkpoint rows, the workspace
    drop, the alias drop and the relay purge."""

    RESULT = {"memories": 3, "conversation_rows": 1, "schedules": 2,
              "workspace_dropped": True, "alias_dropped": True, "relays": 0,
              "relay_bodies_scrubbed": 0}

    def _view(self):
        view = bot_commands._ForgetConfirmView(FAKE_IDENTITY, MagicMock())
        interaction = _fake_interaction("someone")
        interaction.edit_original_response = AsyncMock()
        return view, interaction

    async def _press(self, result=None, raises=None):
        view, interaction = self._view()
        store = MagicMock(side_effect=raises) if raises else MagicMock(
            return_value=result or self.RESULT)
        with unittest.mock.patch.object(bot_commands.privacy, "forget_all", store):
            with unittest.mock.patch.object(bot_commands, "audit_log") as audit:
                await view.confirm.callback(interaction)
        return view, interaction, audit

    async def test_the_press_is_acknowledged_before_the_deletion(self):
        _view, interaction, _audit = await self._press()
        interaction.response.defer.assert_awaited_once()

    async def test_the_report_edits_the_original_message(self):
        _view, interaction, _audit = await self._press()
        interaction.edit_original_response.assert_awaited_once()
        kwargs = interaction.edit_original_response.await_args.kwargs
        self.assertIsNone(kwargs["view"], "the buttons were left on screen")
        self.assertIn("3", kwargs["content"])

    async def test_the_timeout_cannot_fire_mid_deletion(self):
        """on_timeout says "Nothing was deleted", which would be a lie while
        forget_all is running. stop() therefore happens before the work, not
        after - the opposite of the relay's unblock-all view, whose timeout
        copy stays true."""
        view, interaction = self._view()
        stopped_before = {}

        def slow_forget(*_args):
            stopped_before["value"] = view.is_finished()
            return self.RESULT

        with unittest.mock.patch.object(bot_commands.privacy, "forget_all", slow_forget):
            with unittest.mock.patch.object(bot_commands, "audit_log"):
                await view.confirm.callback(interaction)
        self.assertTrue(stopped_before.get("value"),
                        "the 30-second timeout was still live during the deletion")

    async def test_a_failure_says_so_and_withdraws_the_buttons(self):
        _view, interaction, audit = await self._press(raises=RuntimeError("chroma is down"))
        audit.assert_not_called()
        kwargs = interaction.edit_original_response.await_args.kwargs
        self.assertIsNone(kwargs["view"])
        self.assertIn("could not complete", kwargs["content"])

    async def test_a_failure_does_not_claim_what_was_removed(self):
        """Reporting partial counts after a failure would be a guess. The copy
        says it will not guess, and this is what holds it to that."""
        _view, interaction, _audit = await self._press(raises=RuntimeError("boom"))
        said = interaction.edit_original_response.await_args.kwargs["content"]
        for misleading in ("Removed", "Cleared", "Cancelled"):
            self.assertNotIn(misleading, said)

    async def test_an_unreportable_outcome_is_logged_not_raised(self):
        """The press is already answered by the defer; a failed edit must not
        turn into an unhandled exception in a button callback."""
        view, interaction = self._view()
        interaction.edit_original_response = AsyncMock(
            side_effect=RuntimeError("token expired"))
        with unittest.mock.patch.object(bot_commands.privacy, "forget_all",
                                        MagicMock(return_value=self.RESULT)):
            with unittest.mock.patch.object(bot_commands, "audit_log"):
                await view.confirm.callback(interaction)   # must not raise


class TestARefusedForgetIsNotRecordedAsADeletion(unittest.IsolatedAsyncioTestCase):
    """privacy.py raises now instead of returning 0, so these three have
    something to report. Two things must not happen: the success copy, and an
    audit line - the audit log is the record of what was deleted, and an entry
    for a deletion that never happened is worse than no entry."""

    SLASH = (
        ("forget_memories_slash", "forget_memories", "memories"),
        ("forget_conversation_slash", "forget_conversation", "conversation"),
        ("forget_schedules_slash", "forget_schedules", "schedules"),
    )

    async def _refuse(self, command: str, target: str):
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        broken = MagicMock(side_effect=privacy.PrivacyOperationFailed(
            target, RuntimeError("store is down")))
        with unittest.mock.patch.object(bot_commands.privacy, target, broken):
            with unittest.mock.patch.object(bot_commands, "audit_log") as audit:
                await getattr(cog, command).callback(cog, interaction)
        return interaction, audit

    async def test_the_person_is_told_it_failed(self):
        for command, target, _scope in self.SLASH:
            with self.subTest(command=command):
                interaction, _audit = await self._refuse(command, target)
                said = interaction.followup.send.await_args.args[0]
                self.assertIn("could not complete", said)
                for success in ("Removed", "Cleared", "Cancelled"):
                    self.assertNotIn(success, said)

    async def test_no_audit_line_claims_a_deletion(self):
        for command, target, _scope in self.SLASH:
            with self.subTest(command=command):
                _interaction, audit = await self._refuse(command, target)
                audit.assert_not_called()

    async def test_the_reply_is_still_ephemeral(self):
        for command, target, _scope in self.SLASH:
            with self.subTest(command=command):
                interaction, _audit = await self._refuse(command, target)
                self.assertIs(
                    interaction.followup.send.await_args.kwargs["ephemeral"], True)

    async def test_a_successful_run_still_records_one(self):
        """The other side of it: the audit line must not have been lost."""
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch.object(bot_commands.privacy, "forget_memories",
                                        MagicMock(return_value=4)):
            with unittest.mock.patch.object(bot_commands, "audit_log") as audit:
                await cog.forget_memories_slash.callback(cog, interaction)
        audit.assert_called_once()
        self.assertEqual(audit.call_args.kwargs["removed"], 4)
        self.assertEqual(audit.call_args.kwargs["scope"], "memories")


class TestAnIncompleteExportSaysSo(unittest.IsolatedAsyncioTestCase):
    """An export that silently omits a section is worse than a partial one:
    somebody checking what is held about them reads the absence as the answer.
    """

    COMPLETE = {"user_id": "discord-1", "display_name": "Nick", "memories": [],
                "schedules": [], "conversation_checkpoint_count": 0,
                "workspace_path": None, "relays": {}, "failed": []}

    async def _export(self, data):
        cog = _make_cog()
        interaction = _fake_interaction("someone")
        with unittest.mock.patch.object(bot_commands.privacy, "export_user_data",
                                        MagicMock(return_value=data)):
            with unittest.mock.patch.object(bot_commands, "audit_log"):
                await cog.export_slash.callback(cog, interaction)
        return interaction.followup.send.await_args

    async def test_a_section_that_could_not_be_read_is_named(self):
        sent = await self._export(dict(self.COMPLETE, failed=["export_memories"]))
        note = sent.args[0]
        self.assertIn("Incomplete", note)
        self.assertIn("export_memories", note)
        self.assertIn("missing, not empty", note)

    async def test_every_failed_section_is_named(self):
        sent = await self._export(dict(self.COMPLETE,
                                       failed=["export_memories", "export_relay"]))
        for section in ("export_memories", "export_relay"):
            self.assertIn(section, sent.args[0])

    async def test_a_complete_export_says_nothing_about_failures(self):
        sent = await self._export(self.COMPLETE)
        self.assertNotIn("Incomplete", sent.args[0])
        self.assertIn("nothing was sent to a third party", sent.args[0])

    async def test_the_file_is_still_attached_either_way(self):
        """What was read is still worth having."""
        for failed in ([], ["export_memories"]):
            with self.subTest(failed=failed):
                sent = await self._export(dict(self.COMPLETE, failed=failed))
                self.assertIsNotNone(sent.kwargs.get("file"))
                self.assertIs(sent.kwargs["ephemeral"], True)
