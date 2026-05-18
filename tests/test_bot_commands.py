"""
Regression tests for the admin gating on schedule commands.

Originally added for the Phase 2 ROOT_USER gate; updated in Phase 7a to
exercise the new is_admin() abstraction (ROOT_USER + ADMIN_USERS).

Slash commands in discord.py are wrapped in an app_commands.Command descriptor,
so we exercise the underlying callback (the .callback attribute) directly with
a fake Interaction object. That sidesteps the need for a live Discord client.
"""
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock


def _ensure_mock(name: str) -> MagicMock:
    if name not in sys.modules:
        sys.modules[name] = MagicMock()
    return sys.modules[name]


# bot_commands pulls in mister_fritz → agent_tools → ddgs, plus document_engine
# / image_generator / tts. Stub everything heavy.
_ensure_mock("ddgs")
_ensure_mock("image_generator")
_ensure_mock("document_engine")
_ensure_mock("tts")

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


if __name__ == "__main__":
    unittest.main()
