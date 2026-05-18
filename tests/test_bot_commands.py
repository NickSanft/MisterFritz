"""
Regression tests for the ROOT_USER gating added to schedule commands in Phase 2.

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

import bot_commands  # noqa: E402
from bot_commands import FritzCommands, _require_root  # noqa: E402


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
        user_workspaces={},
        schedule_manager=schedule_manager,
    )


class TestRequireRoot(unittest.IsolatedAsyncioTestCase):
    async def test_root_user_allowed(self):
        with unittest.mock.patch.object(bot_commands, "ROOT_USER", ROOT_NAME):
            interaction = _fake_interaction(ROOT_NAME)
            allowed = await _require_root(interaction)
        self.assertTrue(allowed)
        interaction.response.send_message.assert_not_called()

    async def test_non_root_user_rejected_with_ephemeral(self):
        with unittest.mock.patch.object(bot_commands, "ROOT_USER", ROOT_NAME):
            interaction = _fake_interaction("evil_user")
            allowed = await _require_root(interaction)
        self.assertFalse(allowed)
        interaction.response.send_message.assert_called_once()
        # Confirm the rejection uses ephemeral=True so it doesn't leak in-channel.
        _, kwargs = interaction.response.send_message.call_args
        self.assertTrue(kwargs.get("ephemeral"))


class TestScheduleAddGating(unittest.IsolatedAsyncioTestCase):
    async def test_non_root_user_blocked_from_schedule_add(self):
        manager = MagicMock()
        cog = _make_cog(schedule_manager=manager)
        with unittest.mock.patch.object(bot_commands, "ROOT_USER", ROOT_NAME):
            interaction = _fake_interaction("not_root")
            # .callback is the unwrapped coroutine behind the app_commands.command decorator
            await cog.schedule_add.callback(
                cog, interaction, every="1h", prompt="hello", description=None
            )
        # Should send the ephemeral rejection and never touch the manager.
        interaction.response.send_message.assert_called_once()
        manager.add_schedule.assert_not_called()

    async def test_root_user_can_schedule_add(self):
        manager = MagicMock()
        manager.add_schedule.return_value = "abc12345"
        cog = _make_cog(schedule_manager=manager)
        with unittest.mock.patch.object(bot_commands, "ROOT_USER", ROOT_NAME):
            interaction = _fake_interaction(ROOT_NAME)
            await cog.schedule_add.callback(
                cog, interaction, every="1h", prompt="hello", description="my reminder"
            )
        manager.add_schedule.assert_called_once()
        interaction.response.send_message.assert_called_once()


class TestScheduleRemoveGating(unittest.IsolatedAsyncioTestCase):
    async def test_non_root_user_blocked_from_schedule_remove(self):
        manager = MagicMock()
        cog = _make_cog(schedule_manager=manager)
        with unittest.mock.patch.object(bot_commands, "ROOT_USER", ROOT_NAME):
            interaction = _fake_interaction("not_root")
            await cog.schedule_remove.callback(cog, interaction, schedule_id="abc12345")
        interaction.response.send_message.assert_called_once()
        manager.remove_schedule.assert_not_called()

    async def test_root_user_can_remove_schedule(self):
        manager = MagicMock()
        manager.remove_schedule.return_value = True
        cog = _make_cog(schedule_manager=manager)
        with unittest.mock.patch.object(bot_commands, "ROOT_USER", ROOT_NAME):
            interaction = _fake_interaction(ROOT_NAME)
            await cog.schedule_remove.callback(cog, interaction, schedule_id="abc12345")
        manager.remove_schedule.assert_called_once_with("abc12345", ROOT_NAME)


class TestScheduleListOpenToAll(unittest.IsolatedAsyncioTestCase):
    async def test_non_root_user_can_list_their_own_schedules(self):
        manager = MagicMock()
        manager.list_schedules.return_value = []
        cog = _make_cog(schedule_manager=manager)
        with unittest.mock.patch.object(bot_commands, "ROOT_USER", ROOT_NAME):
            interaction = _fake_interaction("regular_user")
            await cog.schedule_list.callback(cog, interaction)
        # list_schedules is read-only and per-user, so it's open by design.
        manager.list_schedules.assert_called_once_with("regular_user")


if __name__ == "__main__":
    unittest.main()
