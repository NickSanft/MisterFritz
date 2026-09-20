"""
Tests for relay_router and the DM hook it sits on (P0 and PR 4 of
plans/12-direct-message-relay.md).

The hook lives directly above the bot's most load-bearing line: every DM that
reaches the agent passes through it. A router that ate every DM would still
"work" from the code's point of view, so the test that a plain DM still
reaches ask_stuff is a release blocker, not a nicety.
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import discord

import main_discord
import relay_router

REPO = pathlib.Path(__file__).resolve().parent.parent


def _dm(content="hello", author_id=4242, *, reference=None,
        message_type=discord.MessageType.default):
    """A DM to Fritz, shaped like the Message on_message actually receives."""
    ctx = MagicMock()
    ctx.author = MagicMock()
    ctx.author.id = author_id
    ctx.author.name = "bob"
    ctx.author.display_name = "Bob"
    ctx.author.bot = False
    ctx.content = content
    ctx.clean_content = content
    ctx.attachments = []
    ctx.type = message_type
    ctx.reference = reference
    ctx.channel = MagicMock(spec=discord.DMChannel)
    ctx.channel.id = 777000111
    ctx.channel.send = AsyncMock(return_value=MagicMock(edit=AsyncMock()))
    return ctx


class OnMessageTestCase(unittest.IsolatedAsyncioTestCase):
    """Drives the real on_message with the collaborators it reaches for."""

    def setUp(self):
        self.reply = {"text": "a reply from the agent", "image_paths": []}
        self.run_blocking = AsyncMock(side_effect=self._run_blocking)
        self.seen = []
        self._patches = [
            patch.object(main_discord, "run_blocking", self.run_blocking),
            patch.object(main_discord.METRICS, "increment", MagicMock()),
            patch.object(main_discord.identity_store, "record", MagicMock()),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()

    async def _run_blocking(self, fn, *args, **kwargs):
        self.seen.append(getattr(fn, "__name__", repr(fn)))
        return self.reply

    def reached_the_agent(self) -> bool:
        return "ask_stuff" in self.seen


class TestAPlainDmStillReachesTheAgent(OnMessageTestCase):
    """RELEASE BLOCKER. A router that swallowed every DM would look fine from
    the code's side: on_message would simply return, every time."""

    async def test_a_plain_dm_reaches_ask_stuff(self):
        ctx = _dm("what is the time?")
        await main_discord.on_message(ctx)
        self.assertTrue(self.reached_the_agent(), "a plain DM no longer reaches the agent")
        ctx.channel.send.assert_awaited()               # the thinking placeholder
        main_discord.identity_store.record.assert_called_once()

    async def test_a_dm_that_is_not_a_relay_reply_reaches_ask_stuff(self):
        """The router says False for anything it does not recognise."""
        reference = MagicMock()
        reference.message_id = 123456789
        ctx = _dm("sure, 8 works", reference=reference,
                  message_type=discord.MessageType.reply)
        await main_discord.on_message(ctx)
        self.assertTrue(self.reached_the_agent())


class TestARoutedReplyStopsDead(OnMessageTestCase):
    """When the router claims a message, nothing downstream may run: not the
    agent, not the identity roster, not the counter, not the placeholder."""

    async def test_nothing_downstream_runs(self):
        with patch.object(main_discord.relay_router, "try_route_reply",
                          AsyncMock(return_value=True)):
            ctx = _dm("sure, 8 works")
            await main_discord.on_message(ctx)
        self.assertFalse(self.reached_the_agent(), "a routed reply reached the agent")
        ctx.channel.send.assert_not_awaited()
        main_discord.identity_store.record.assert_not_called()
        names = [c.args[0] for c in main_discord.METRICS.increment.call_args_list]
        self.assertNotIn("discord_messages", names)
        self.assertIn("relay.replies_routed", names)

    async def test_a_guild_message_never_reaches_the_router(self):
        """Relayed messages only ever arrive in DMs."""
        router = AsyncMock(return_value=True)
        # A stand-in client: the real one has user=None without a gateway, and
        # the guild path needs client.user.mentioned_in.
        fake_client = MagicMock()
        fake_client.user.mentioned_in = MagicMock(return_value=True)
        with patch.object(main_discord.relay_router, "try_route_reply", router), \
             patch.object(main_discord, "client", fake_client):
            ctx = _dm("hello")
            ctx.channel = MagicMock(spec=discord.TextChannel)
            ctx.channel.id = 5
            ctx.channel.send = AsyncMock(return_value=MagicMock(edit=AsyncMock()))
            await main_discord.on_message(ctx)
        router.assert_not_awaited()
        self.assertTrue(self.reached_the_agent())


class TestTheHookSitsWhereItMust(unittest.TestCase):
    """Source-level, following TestIdentityRecordedOnlyForRealTurns: the bug
    class here is purely statement order, and at runtime everything still
    "works" when the order is wrong."""

    def body(self) -> str:
        src = (REPO / "main_discord.py").read_text(encoding="utf-8")
        return src.split("async def on_message(", 1)[1]

    def test_the_router_runs_after_the_early_returns_and_before_everything_else(self):
        body = self.body()
        self_guard = body.index("ctx.author == client.user")
        prefix_guard = body.index("ctx.content.startswith(command_prefix)")
        hook = body.index("relay_router.try_route_reply")
        mention_guard = body.index("client.user.mentioned_in(ctx)")
        record = body.index("identity_store.record(")
        counter = body.index('METRICS.increment("discord_messages")')
        placeholder = body.index("thinking...")
        agent = body.index("ask_stuff")

        # After: Fritz's own relay DM must not route itself, and a $command in
        # a DM stays a command.
        self.assertGreater(hook, self_guard, "the router runs before the self-check")
        self.assertGreater(hook, prefix_guard, "the router runs before the $ prefix check")
        # Before: that guard is the collision this hook exists to resolve, and
        # everything below it belongs to a conversation turn that is not one.
        for later, what in ((mention_guard, "the mention guard"),
                            (record, "identity_store.record"),
                            (counter, "the discord_messages counter"),
                            (placeholder, "the thinking placeholder"),
                            (agent, "ask_stuff")):
            self.assertGreater(later, hook, f"{what} runs before the relay router")


class TestTheRouterCannotReachTheAgent(unittest.TestCase):
    """Enforced by the import graph, not by a guard.

    A relayed message is attacker-controlled text, and the agent binds file
    tools with read/write/exec whenever that user has a workspace. Code that
    cannot reach ask_stuff cannot contaminate the LangGraph checkpoint or the
    Chroma memories injected into that person's next prompt. Patching one and
    not the other proves nothing.
    """

    def test_importing_relay_router_does_not_import_mister_fritz(self):
        # A subprocess, because by now this process has imported half the repo:
        # asserting on sys.modules in here would prove nothing at all.
        sandbox = tempfile.mkdtemp(prefix="relay-import-")
        env = dict(os.environ,
                   SCHEDULE_DB=os.path.join(sandbox, "fritz.db"),
                   DB_NAME=os.path.join(sandbox, "fritz.db"),
                   CHAT_DB_NAME=os.path.join(sandbox, "chat.db"),
                   CHROMA_DB_PATH=os.path.join(sandbox, "chroma"),
                   WORKSPACES_ROOT=os.path.join(sandbox, "workspaces"),
                   AUDIT_LOG_PATH=os.path.join(sandbox, "audit.log"),
                   FRITZ_WRITE_DIAGRAMS="0",
                   PYTHONPATH=str(REPO))
        probe = ("import sys, relay_router; "
                 "print(sorted(m for m in ('mister_fritz', 'agent_tools', 'bot_commands', "
                 "'langchain', 'langgraph') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO), env=env,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip(), "[]", "relay_router pulled in the agent stack")


class TestTheStubIsInert(unittest.IsolatedAsyncioTestCase):
    async def test_it_routes_nothing_yet(self):
        self.assertIs(await relay_router.try_route_reply(MagicMock(), _dm()), False)


if __name__ == "__main__":
    unittest.main()
