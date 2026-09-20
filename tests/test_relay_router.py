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
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import discord

import fritz_utils
import main_discord
import relay_format
import relay_router
import relay_store
from test_tell_command import RECIPIENT, RECIPIENT_SNOWFLAKE, SENDER, SENDER_SNOWFLAKE
from test_tell_command import TellTestCase, _http_error, _member

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
            # ON, or the router returns False at its first line and this class
            # — the one labelled RELEASE BLOCKER — would pass against a router
            # that swallowed every DM.
            patch.object(fritz_utils, "RELAY_ENABLED", True),
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

    async def test_a_dm_that_merely_starts_with_the_prefix_reaches_ask_stuff(self):
        """process_commands swallowed anything starting with "$", and this bot
        has no prefix commands at all, so "$20 plus tip" went nowhere."""
        ctx = _dm("$20 plus tip")
        await main_discord.on_message(ctx)
        self.assertTrue(self.reached_the_agent(), "a $-leading DM was swallowed")

    async def test_a_real_prefix_command_still_runs(self):
        """A command in a DM stays a command — the reason the hook sits below
        this check at all."""
        @main_discord.client.command(name="ping")
        async def _ping(context):       # pragma: no cover - dispatched by discord.py
            pass
        dispatch = AsyncMock()
        try:
            with patch.object(main_discord.client, "process_commands", dispatch):
                await main_discord.on_message(_dm("$ping and more"))
        finally:
            main_discord.client.remove_command("ping")
        dispatch.assert_awaited_once()
        self.assertFalse(self.reached_the_agent(), "a command reached the agent too")

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


class ReplyTestCase(TellTestCase):
    """A delivered relay, and the recipient standing in the DM it arrived in."""

    def setUp(self):
        super().setUp()
        self.client = MagicMock()
        self.dm_channel = MagicMock()
        self.sent_back = MagicMock()
        self.sent_back.id = 424242424242424242
        self.sent_back.channel.id = 313131313131313131
        self.dm_channel.send = AsyncMock(return_value=self.sent_back)
        self.client.create_dm = AsyncMock(return_value=self.dm_channel)

    async def relayed(self, sender_interaction=None, recipient=None):
        """Send one /tell and return the DM Discord 'delivered'."""
        recipient = recipient or _member()
        await self.tell(sender_interaction, recipient=recipient)
        return recipient.send.return_value

    def reply_to(self, dm, text="sure, 8 works", author_id=RECIPIENT_SNOWFLAKE,
                 message_type=discord.MessageType.reply, attachments=()):
        reference = MagicMock()
        reference.message_id = dm.id
        reference.channel_id = 777000111
        reference.guild_id = None
        ctx = _dm(text, author_id, reference=reference, message_type=message_type)
        ctx.author.name = "bob"
        ctx.author.display_name = "Bob"
        ctx.author.display_avatar = MagicMock()
        ctx.author.display_avatar.url = "https://cdn.example/bob.png"
        ctx.attachments = list(attachments)
        return ctx

    async def route(self, ctx):
        return await relay_router.try_route_reply(self.client, ctx)

    def said(self, ctx) -> str:
        """What the router told the person who replied. Exactly one thing."""
        ctx.channel.send.assert_awaited_once()
        call = ctx.channel.send.await_args
        text = call.args[0] if call.args else call.kwargs.get("content")
        self.assert_no_mentions(call.kwargs["allowed_mentions"])
        return text

    def carried(self):
        """The embed the original sender received."""
        self.dm_channel.send.assert_awaited_once()
        return self.dm_channel.send.await_args.kwargs["embed"]


class TestAReplyGoesBack(ReplyTestCase):
    async def test_the_senders_own_words_reach_them_verbatim(self):
        dm = await self.relayed()
        ctx = self.reply_to(dm, "**yes** — see you at 8 @everyone")
        self.assertIs(await self.route(ctx), True)
        embed = self.carried()
        self.assertEqual(embed.description, "**yes** — see you at 8 @everyone")
        self.assertEqual(embed.author.name, "@bob · Bob")     # the replier, as they are
        self.assert_no_mentions(self.dm_channel.send.await_args.kwargs["allowed_mentions"])
        self.assertIn("Carried back", self.said(ctx))

    async def test_it_goes_to_the_account_that_sent_the_relay(self):
        dm = await self.relayed()
        await self.route(self.reply_to(dm))
        self.client.create_dm.assert_awaited_once()
        self.assertEqual(self.client.create_dm.await_args.args[0].id, SENDER_SNOWFLAKE)

    async def test_it_goes_to_the_account_that_wrote_not_its_linked_main(self):
        """The recipient answers whoever wrote to them. sender_id is that
        identity after IDENTITY_LINKS; sender_account is the account."""
        alt = SENDER_SNOWFLAKE + 900
        with patch.object(fritz_utils, "IDENTITY_LINKS",
                          {f"discord-{alt}": f"discord-{SENDER_SNOWFLAKE + 901}"}):
            recipient = _member()
            await self.tell(self.interaction(user_id=alt, name="alty"), recipient=recipient)
            await self.route(self.reply_to(recipient.send.return_value))
        self.assertEqual(self.client.create_dm.await_args.args[0].id, alt)

    async def test_the_reply_is_a_relay_of_its_own_charged_to_the_replier(self):
        dm = await self.relayed()
        await self.route(self.reply_to(dm))
        [original, reply] = self.rows()
        self.assertEqual(reply["sender_id"], RECIPIENT)       # the replier pays
        self.assertEqual(reply["recipient_id"], SENDER)
        self.assertEqual(reply["origin_id"], original["id"])  # the chain
        self.assertEqual(reply["guild_id"], original["guild_id"])
        self.assertEqual(reply["status"], "delivered")
        self.assertEqual(reply["dm_message_id"], self.sent_back.id)
        self.assertEqual(reply["shown_as"], "@bob · Bob")

    async def test_the_sender_can_reply_to_the_reply(self):
        """Direction-agnostic: the second hop is the first with the ends
        swapped, and it must route as readily."""
        dm = await self.relayed()
        await self.route(self.reply_to(dm))
        back = MagicMock()
        back.id = self.sent_back.id
        second = self.reply_to(back, "perfect", author_id=SENDER_SNOWFLAKE)
        second.author.name = "alice"
        second.author.display_name = "alice"
        self.dm_channel.send.reset_mock()
        self.sent_back.id = 515151515151515151
        self.assertIs(await self.route(second), True)
        self.assertEqual(self.carried().description, "perfect")
        self.assertEqual(self.client.create_dm.await_args.args[0].id, RECIPIENT_SNOWFLAKE)

    async def test_two_open_relays_answer_only_their_own_sender(self):
        """Any last-relay-wins or time-window implementation passes the
        single-relay test and fails this one."""
        first = await self.relayed()
        await self.relayed(self.interaction(user_id=SENDER_SNOWFLAKE + 7, name="dave"))
        await self.route(self.reply_to(first))
        self.assertEqual(self.client.create_dm.await_args.args[0].id, SENDER_SNOWFLAKE)

    async def test_an_attachment_stays_behind_and_both_parties_are_told(self):
        dm = await self.relayed()
        ctx = self.reply_to(dm, "here you go", attachments=[MagicMock()])
        await self.route(ctx)
        self.assertIn("do not carry", self.carried().footer.text)
        self.assertIn("words, not parcels", self.said(ctx))

    async def test_a_delivery_that_cannot_be_recorded_is_still_admitted(self):
        dm = await self.relayed()
        ctx = self.reply_to(dm)
        with patch.object(relay_store, "mark_sent",
                          side_effect=relay_store.RelayStoreError("locked")):
            await self.route(ctx)
        text = self.said(ctx)
        self.assertIn("Carried back", text)
        self.assertIn("failed to note it down", text)

    async def test_metrics_and_audit(self):
        dm = await self.relayed()
        with patch.object(relay_router.METRICS, "increment") as inc:
            await self.route(self.reply_to(dm, "zebra-quartz"))
        self.assertIn("relay.delivered", [c.args[0] for c in inc.call_args_list])
        sent = [e for e in self.audit_events() if e["event"] == "relay_sent"]
        self.assertEqual(len(sent), 2)                        # the /tell and the reply
        self.assertNotIn("zebra", self.audit_path.read_text(encoding="utf-8"))
        self.assertEqual(sent[-1]["sender"], RECIPIENT)
        self.assertEqual(sent[-1]["body_digest"], relay_format.audit_digest("zebra-quartz"))


class TestWhatMustNotRoute(ReplyTestCase):
    async def assert_falls_through(self, ctx):
        """Not ours: the agent gets it, and nobody is told anything."""
        self.assertIs(await self.route(ctx), False)
        ctx.channel.send.assert_not_awaited()
        self.dm_channel.send.assert_not_awaited()

    async def test_bare_text_is_a_conversation_with_fritz(self):
        """A session or a TTL window would capture whatever she typed next."""
        await self.relayed()
        await self.assert_falls_through(
            _dm("remind me to take my meds at 9", RECIPIENT_SNOWFLAKE))

    async def test_a_forward_does_not_route(self):
        """A forward carries reference.message_id for the row we would look up,
        with empty content: without the type gate it would relay nothing."""
        dm = await self.relayed()
        await self.assert_falls_through(
            self.reply_to(dm, "", message_type=discord.MessageType.default))

    async def test_a_reply_to_something_that_is_not_a_relay(self):
        stray = MagicMock()
        stray.id = 999999999999999999
        await self.assert_falls_through(self.reply_to(stray))

    async def test_a_reply_with_no_reference_id_never_reaches_the_store(self):
        """A reference without a message_id is normal for pins, thread
        starters and channel-follow adds. The store would answer None anyway,
        so this asserts the guard by its only observable effect."""
        dm = await self.relayed()
        ctx = self.reply_to(dm)
        ctx.reference.message_id = None
        with patch.object(relay_store, "get_by_dm_message") as lookup:
            await self.assert_falls_through(ctx)
        lookup.assert_not_called()

    async def test_someone_who_is_not_the_recipient(self):
        """The guard against a forged or mis-indexed anchor."""
        dm = await self.relayed()
        await self.assert_falls_through(self.reply_to(dm, author_id=SENDER_SNOWFLAKE + 99))

    async def test_nothing_routes_while_the_relay_is_switched_off(self):
        dm = await self.relayed()
        ctx = self.reply_to(dm)
        with patch.object(fritz_utils, "RELAY_ENABLED", False):
            await self.assert_falls_through(ctx)

    async def test_a_store_outage_leaves_the_message_to_the_agent(self):
        dm = await self.relayed()
        ctx = self.reply_to(dm)
        with patch.object(relay_store, "get_by_dm_message",
                          side_effect=relay_store.RelayStoreError("locked")):
            await self.assert_falls_through(ctx)

    async def test_a_linked_recipient_can_answer_their_own_relay(self):
        """The row holds the recipient RESOLVED; the replier arrives raw."""
        linked = f"discord-{SENDER_SNOWFLAKE + 500}"
        with patch.object(fritz_utils, "IDENTITY_LINKS", {RECIPIENT: linked}):
            recipient = _member()
            await self.tell(recipient=recipient)
            ctx = self.reply_to(recipient.send.return_value)
            self.assertIs(await self.route(ctx), True)
        self.assertIn("Carried back", self.said(ctx))

    async def test_a_link_added_after_delivery_leaves_the_older_relay_alone(self):
        """A consequence worth stating plainly: the row was written under the
        old identity, so a reply to it is a conversation with Fritz, not a
        misrouted answer to someone else."""
        dm = await self.relayed()
        linked = f"discord-{SENDER_SNOWFLAKE + 500}"
        with patch.object(fritz_utils, "IDENTITY_LINKS", {RECIPIENT: linked}):
            await self.assert_falls_through(self.reply_to(dm))


class TestALapsedExchange(ReplyTestCase):
    """A reply to a closed or expired relay must NOT fall through either: it
    would become a Fritz conversation turn containing the recipient's answer
    to someone else."""

    def lapse(self, how):
        with sqlite3.connect(self.db) as conn:
            if how == "expired":
                stale = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
                conn.execute("UPDATE relay_messages SET expires_at = ?", (stale,))
            else:
                conn.execute("UPDATE relay_messages SET closed_at = ?",
                             (datetime.now(timezone.utc).isoformat(),))
            conn.commit()

    async def test_expired_and_closed_are_answered_not_forwarded(self):
        for how in ("expired", "closed"):
            with self.subTest(how):
                dm = await self.relayed()
                self.lapse(how)
                ctx = self.reply_to(dm)
                self.assertIs(await self.route(ctx), True)
                self.assertIn("lapsed", self.said(ctx))
                self.dm_channel.send.assert_not_awaited()

    async def test_a_timestamp_without_a_timezone_does_not_drop_the_dm(self):
        """on_message has no handler around the hook: a TypeError comparing a
        naive timestamp to an aware one would lose the message entirely."""
        dm = await self.relayed()
        with sqlite3.connect(self.db) as conn:
            naive = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(tzinfo=None)
            conn.execute("UPDATE relay_messages SET expires_at = ?", (naive.isoformat(),))
            conn.commit()
        ctx = self.reply_to(dm)
        self.assertIs(await self.route(ctx), True)          # read as UTC, still open
        self.assertIn("Carried back", self.said(ctx))

    async def test_a_reply_with_no_words_carries_nothing(self):
        dm = await self.relayed()
        ctx = self.reply_to(dm, "   ", attachments=[MagicMock()])
        self.assertIs(await self.route(ctx), True)
        self.assertIn("words, not parcels", self.said(ctx))
        self.dm_channel.send.assert_not_awaited()


class TestTheReplierIsToldWhatHappened(ReplyTestCase):
    """The failure copy goes to the person who just replied. They are the only
    one in the room; the original sender cannot see any of it."""

    async def test_a_block_and_closed_dms_read_the_same(self):
        dm = await self.relayed()
        relay_store.block(SENDER, RECIPIENT)                  # the sender blocked them
        blocked_ctx = self.reply_to(dm)
        await self.route(blocked_ctx)
        by_block = self.said(blocked_ctx)
        self.dm_channel.send.assert_not_awaited()
        relay_store.unblock(SENDER, RECIPIENT)

        second = await self.relayed()
        self.dm_channel.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
        closed_ctx = self.reply_to(second)
        await self.route(closed_ctx)
        by_closed = self.said(closed_ctx)

        self.assertEqual(by_block, by_closed)
        self.assertEqual(by_block, relay_store.REFUSED)
        self.assertEqual([r["status"] for r in self.rows() if r["origin_id"]],
                         ["refused", "refused"])

    async def test_both_refusals_keep_the_same_schedule(self):
        """Drawn before anything is decided, waited on, and only then settled.
        Recording just the sleep would miss a deadline drawn after the store
        call, or a reservation released early."""
        order = []
        real_settle = relay_format.settle

        def deadline():
            order.append("draw")
            return 4242.0

        async def sleep_until(value):
            order.append(("sleep", value))

        async def settle(fn, relay_id, *a):
            order.append(("settle", fn.__name__))
            await real_settle(fn, relay_id, *a)

        with patch.object(relay_format, "refusal_deadline", deadline), \
             patch.object(relay_format, "sleep_until", sleep_until), \
             patch.object(relay_format, "settle", settle):
            dm = await self.relayed()
            relay_store.block(SENDER, RECIPIENT)
            await self.route(self.reply_to(dm))
            by_block = list(order)
            order.clear()
            relay_store.unblock(SENDER, RECIPIENT)
            second = await self.relayed()
            self.dm_channel.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
            await self.route(self.reply_to(second))
            by_403 = list(order)

        expected = ["draw", ("sleep", 4242.0), ("settle", "mark_refused")]
        self.assertEqual(by_block, expected)
        self.assertEqual(by_403, expected)

    async def test_both_refusals_write_the_same_audit_line(self):
        """The audit log outlives /relay forget-blocks; a field that differed
        between them would be a durable record of who blocked whom."""
        dm = await self.relayed()
        relay_store.block(SENDER, RECIPIENT)
        await self.route(self.reply_to(dm))
        relay_store.unblock(SENDER, RECIPIENT)
        second = await self.relayed()
        self.dm_channel.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
        await self.route(self.reply_to(second))

        refused = [e for e in self.audit_events() if e["event"] == "relay_refused"]
        self.assertEqual(len(refused), 2)
        self.assertEqual(set(refused[0]), set(refused[1]))
        for event in refused:
            self.assertNotIn("reason", event)
            self.assertNotIn("discord_code", event)

    async def test_every_other_failure_says_nothing_was_sent(self):
        failures = {
            "not found": _http_error(discord.NotFound, 404),
            "rate limited": discord.RateLimited(7.0),
            "server error": _http_error(discord.DiscordServerError, 503),
            "http 429": _http_error(discord.HTTPException, 429),
            "anything else": RuntimeError("secret internal detail"),
        }
        for label, exc in failures.items():
            with self.subTest(label):
                dm = await self.relayed()
                self.dm_channel.send.side_effect = exc
                ctx = self.reply_to(dm)
                self.assertIs(await self.route(ctx), True)
                text = self.said(ctx)
                self.assertIn("Nothing was sent", text)
                self.assertNotIn("secret internal detail", text)
                self.assertEqual(self.rows()[-1]["status"], "failed")

    async def test_content_discord_rejects_is_charged(self):
        dm = await self.relayed()
        self.dm_channel.send.side_effect = _http_error(discord.HTTPException, 400, 240000)
        ctx = self.reply_to(dm)
        await self.route(ctx)
        self.assertIn("declined that message as written", self.said(ctx))
        self.assertEqual(self.rows()[-1]["status"], "rejected")

    async def test_a_rate_limited_replier_is_told_and_nothing_is_sent(self):
        dm = await self.relayed()
        with patch.object(fritz_utils, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            await self.route(self.reply_to(dm))               # spends the cap
            self.dm_channel.send.reset_mock()
            ctx = self.reply_to(dm)
            self.assertIs(await self.route(ctx), True)
        self.assertIn("limit", self.said(ctx))
        self.dm_channel.send.assert_not_awaited()

    async def test_the_failure_is_recorded_through_the_store(self):
        """mark was a default argument, bound at definition time, so patching
        relay_store.mark_failed did nothing and every bookkeeping assertion
        around it passed vacuously."""
        dm = await self.relayed()
        self.dm_channel.send.side_effect = _http_error(discord.NotFound, 404)
        with patch.object(relay_store, "mark_failed") as mark:
            await self.route(self.reply_to(dm))
        mark.assert_called_once()

    async def test_a_reserve_failure_says_nothing_was_sent(self):
        dm = await self.relayed()
        ctx = self.reply_to(dm)
        with patch.object(relay_store, "reserve_send",
                          side_effect=relay_store.RelayStoreError("disk full")):
            await self.route(ctx)
        text = self.said(ctx)
        self.assertIn("Nothing was sent", text)
        self.assertNotIn("disk full", text)

    async def test_a_failure_to_answer_does_not_escape(self):
        """on_message has no error handling around this hook."""
        dm = await self.relayed()
        ctx = self.reply_to(dm)
        ctx.channel.send.side_effect = _http_error(discord.DiscordServerError, 503)
        await self.route(ctx)                                  # must not raise


class TestTheAgentNeverSeesARoutedReply(ReplyTestCase):
    """The contamination the whole module exists to prevent, asserted through
    the real on_message rather than against a mock of it."""

    async def test_a_routed_reply_reaches_nothing_downstream(self):
        dm = await self.relayed()
        ctx = self.reply_to(dm, "meet me at the old mill")
        seen = []

        async def spy(fn, *a, **kw):
            seen.append(getattr(fn, "__name__", repr(fn)))
            return {"text": "x", "image_paths": []}
        # on_message hands the router ITS client, so the stand-in has to be
        # installed there, not just passed to try_route_reply.
        with patch.object(main_discord, "run_blocking", AsyncMock(side_effect=spy)), \
             patch.object(main_discord, "client", self.client), \
             patch.object(main_discord.identity_store, "record") as record:
            await main_discord.on_message(ctx)
        self.assertNotIn("ask_stuff", seen)
        record.assert_not_called()
        self.assertEqual(self.carried().description, "meet me at the old mill")

    async def test_a_reply_that_starts_with_the_prefix_is_still_carried(self):
        """The one path that left someone believing they had answered."""
        dm = await self.relayed()
        ctx = self.reply_to(dm, "$20 plus tip")
        # The real client for the prefix lookup (it registers no prefix
        # commands), the stand-in for create_dm.
        self.client.get_command = main_discord.client.get_command
        with patch.object(main_discord, "run_blocking", AsyncMock()), \
             patch.object(main_discord, "client", self.client):
            await main_discord.on_message(ctx)
        self.assertEqual(self.carried().description, "$20 plus tip")

    async def test_a_plain_dm_still_reaches_the_agent(self):
        await self.relayed()
        seen = []

        async def spy(fn, *a, **kw):
            seen.append(getattr(fn, "__name__", repr(fn)))
            return {"text": "x", "image_paths": []}
        with patch.object(main_discord, "run_blocking", AsyncMock(side_effect=spy)):
            await main_discord.on_message(_dm("what is the time?", RECIPIENT_SNOWFLAKE))
        self.assertIn("ask_stuff", seen)


if __name__ == "__main__":
    unittest.main()
