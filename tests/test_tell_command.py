"""
Tests for /tell message — the relay's delivery path (PR 2 of
plans/12-direct-message-relay.md).

These run the real relay_store against a temp database rather than mocking
it: most of what can go wrong here is the command and the store disagreeing
about what a row, a refusal or a charge means, and a mock would agree with
whatever the command assumed.
"""
import asyncio
import hashlib
import inspect
import itertools
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import discord
from discord import app_commands

import bot_commands
import fritz_utils
import observability
import relay_format
import relay_store
from bot_commands import FritzCommands
from test_bot_commands import _fake_interaction, _make_cog

SENDER_SNOWFLAKE = 111111111111111111
RECIPIENT_SNOWFLAKE = 555555555555555555
SENDER = f"discord-{SENDER_SNOWFLAKE}"
RECIPIENT = f"discord-{RECIPIENT_SNOWFLAKE}"
GUILD_ID = 67890
# Every DM Discord "sends" gets its own id, as real ones do. A shared one made
# the second relay to the same person collide on the routing index.
_SENT_IDS = itertools.count(900_000_000_000_000_001)


def _http_error(cls, status, code=0):
    response = MagicMock()
    response.status = status
    response.reason = "reason"
    return cls(response, {"code": code, "message": "discord says no"})


def _member(uid=RECIPIENT_SNOWFLAKE, name="bob", display="Bob", bot=False):
    member = MagicMock(spec=discord.Member)
    member.id = uid
    member.name = name
    member.display_name = display
    member.bot = bot
    member.mention = f"<@{uid}>"
    member.display_avatar = MagicMock()
    member.display_avatar.url = f"https://cdn.example/{uid}.png"
    sent = MagicMock()
    sent.id = next(_SENT_IDS)
    sent.channel.id = 700_000_000_000_000_001
    member.send = AsyncMock(return_value=sent)
    return member


class TellTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = str(self.tmp / "relay.db")
        self.audit_path = self.tmp / "audit.log"
        self._patches = [
            patch.object(relay_store, "SCHEDULE_DB", self.db),
            patch.object(relay_store, "_INITIALISED", False),
            patch.multiple(fritz_utils, RELAY_ENABLED=True, RELAY_MAX_BODY_CHARS=1000,
                           RELAY_MAX_PER_SENDER_PER_HOUR=10,
                           RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR=10,
                           RELAY_REPLY_WINDOW_MIN=1440),
            patch.object(relay_format, "REFUSAL_WINDOW_SEC", (0, 0)),
            patch.object(observability, "AUDIT_LOG_PATH", str(self.audit_path)),
        ]
        for p in self._patches:
            p.start()
        self.cog = _make_cog()
        self.cog.bot.user.id = 999          # Fritz, unless a test says otherwise

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()

    def interaction(self, *, guild_id=GUILD_ID, name="alice", display="alice",
                    user_id=SENDER_SNOWFLAKE):
        interaction = _fake_interaction(name, user_id=user_id)
        interaction.user.display_name = display
        interaction.user.display_avatar = MagicMock()
        interaction.user.display_avatar.url = "https://cdn.example/sender.png"
        interaction.user.send = AsyncMock()
        interaction.guild_id = guild_id
        interaction.is_guild_integration = MagicMock(return_value=True)
        # Post-defer state, so _reply_error takes the followup path — the
        # same thing the real InteractionResponse reports after defer().
        interaction.response.is_done = MagicMock(return_value=True)
        return interaction

    async def tell(self, interaction=None, recipient=None, message="running late"):
        interaction = interaction or self.interaction()
        recipient = recipient or _member()
        await self.cog.tell_message.callback(self.cog, interaction,
                                             recipient=recipient, message=message)
        return interaction, recipient

    def reply(self, interaction) -> str:
        """The one answer the sender got: deferred ephemerally first, then
        exactly one ephemeral followup, never a second response."""
        interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
        interaction.response.send_message.assert_not_awaited()
        interaction.followup.send.assert_awaited_once()
        call = interaction.followup.send.await_args
        self.assertIs(call.kwargs.get("ephemeral"), True, "a /tell reply was not ephemeral")
        text = call.args[0] if call.args else call.kwargs.get("content")
        self.assertTrue((text or "").strip(), "a deferred /tell answered with nothing")
        return text

    def rows(self):
        try:
            with sqlite3.connect(self.db) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute("SELECT * FROM relay_messages")]
        except sqlite3.OperationalError:
            return []

    def audit_events(self):
        if not self.audit_path.exists():
            return []
        return [json.loads(line) for line in
                self.audit_path.read_text(encoding="utf-8").splitlines()]

    def assert_no_mentions(self, allowed):
        self.assertIsInstance(allowed, discord.AllowedMentions)
        for field in ("everyone", "users", "roles", "replied_user"):
            self.assertFalse(getattr(allowed, field), f"mentions.{field} left on")


class TestDelivery(TellTestCase):
    async def test_the_exact_words_arrive_as_an_embed(self):
        body = "**Late** — <@1> @everyone\n\n— actually from @admin"
        interaction, recipient = await self.tell(message=body)
        recipient.send.assert_awaited_once()
        embed = recipient.send.await_args.kwargs["embed"]
        self.assertEqual(embed.description, body)        # verbatim, unescaped
        self.assertTrue(embed.author.name.startswith("@alice"))
        self.assertIn("Delivered", self.reply(interaction))

    async def test_the_relay_leg_can_ping_nobody(self):
        _, recipient = await self.tell(message="@everyone <@&42> <@1>")
        self.assert_no_mentions(recipient.send.await_args.kwargs["allowed_mentions"])

    async def test_the_row_records_the_routing_key_and_the_server(self):
        _, recipient = await self.tell()
        [row] = self.rows()
        self.assertEqual(row["status"], "delivered")
        self.assertEqual(row["dm_message_id"], recipient.send.return_value.id)
        self.assertEqual(row["guild_id"], GUILD_ID)
        with sqlite3.connect(self.db) as conn:
            kind = conn.execute("SELECT typeof(dm_message_id) FROM relay_messages").fetchone()[0]
        self.assertEqual(kind, "integer")

    async def test_the_sender_gets_a_receipt_naming_the_recipient(self):
        interaction, _ = await self.tell(message="see you at 8 @everyone")
        interaction.user.send.assert_awaited_once()
        kwargs = interaction.user.send.await_args.kwargs
        self.assertEqual(kwargs["embed"].description, "see you at 8 @everyone")
        self.assertIn("@bob", kwargs["embed"].author.name)
        self.assertNotIn("@alice", kwargs["embed"].author.name)
        self.assert_no_mentions(kwargs["allowed_mentions"])

    async def test_a_failed_receipt_does_not_undo_the_relay(self):
        interaction = self.interaction()
        interaction.user.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
        await self.tell(interaction)
        text = self.reply(interaction)
        self.assertIn("Delivered", text)
        self.assertIn("copy", text)
        self.assertEqual(self.rows()[0]["status"], "delivered")

    async def test_the_confirmation_can_ping_nobody(self):
        interaction, _ = await self.tell()
        self.reply(interaction)
        self.assert_no_mentions(interaction.followup.send.await_args.kwargs["allowed_mentions"])

    async def test_a_failure_to_report_a_delivery_never_reaches_the_error_handler(self):
        """Its copy ("Discord declined to carry that message") would tell a
        sender whose message WAS delivered that it was not — and they resend."""
        interaction = self.interaction()
        interaction.followup.send = AsyncMock(side_effect=_http_error(discord.DiscordServerError, 503))
        await self.tell(interaction)                         # must not raise
        self.assertEqual(self.rows()[0]["status"], "delivered")

    async def test_store_calls_run_off_the_event_loop(self):
        """sqlite3 blocks; discord.py coroutines must NOT go through the pool."""
        seen = []
        real = bot_commands.run_blocking

        async def spy(fn, *a, **kw):
            seen.append(getattr(fn, "__name__", repr(fn)))
            return await real(fn, *a, **kw)
        with patch.object(bot_commands, "run_blocking", spy):
            await self.tell()
        self.assertIn("reserve_send", seen)
        self.assertIn("mark_sent", seen)
        self.assertNotIn("send", seen)

    async def test_metrics(self):
        with patch.object(bot_commands.METRICS, "increment") as inc:
            await self.tell()
        names = [c.args[0] for c in inc.call_args_list]
        self.assertIn("discord_commands.tell.message", names)
        self.assertIn("relay.delivered", names)


class TestAuthorLine(TellTestCase):
    async def author(self, name, display):
        _, recipient = await self.tell(self.interaction(name=name, display=display))
        return recipient.send.await_args.kwargs["embed"].author.name

    async def test_the_unborrowable_username_comes_first(self):
        self.assertEqual(await self.author("mallory", "Bob"), "@mallory · Bob")

    async def test_a_nickname_cannot_carry_its_own_handle(self):
        """'Alice (@alice)' used to render as 'Alice (@alice) (@mallory)'."""
        author = await self.author("mallory", "Alice (@alice)")
        self.assertTrue(author.startswith("@mallory"))
        self.assertNotIn("@alice", author)
        self.assertNotIn("(", author)

    async def test_text_that_reorders_or_hides_itself_is_dropped(self):
        author = await self.author("mallory", "‮ecilA​⁦x\n")
        for ch in "‮​⁦\n":
            self.assertNotIn(ch, author)

    async def test_no_repeat_when_the_names_match(self):
        self.assertEqual(await self.author("alice", "alice"), "@alice")

    async def test_an_uncached_server_does_not_leave_a_hole_in_the_footer(self):
        interaction = self.interaction()
        interaction.guild = MagicMock()
        interaction.guild.name = ""
        _, recipient = await self.tell(interaction)
        self.assertIn("a server you share", recipient.send.await_args.kwargs["embed"].footer.text)


class TestRefusals(TellTestCase):
    async def test_closed_dms_read_exactly_like_a_block(self):
        """The property /relay block exists to protect: "has X blocked me"
        must be unanswerable. Equality of the two strings, not two literals."""
        relay_store.block(RECIPIENT, SENDER)
        blocked_i, blocked_r = await self.tell()
        by_block = self.reply(blocked_i)
        blocked_r.send.assert_not_awaited()
        relay_store.unblock(RECIPIENT, SENDER)

        closed_r = _member()
        closed_r.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
        closed_i, _ = await self.tell(recipient=closed_r)
        by_closed_dm = self.reply(closed_i)

        self.assertEqual(by_block, by_closed_dm)
        self.assertEqual(by_block, relay_store.REFUSED)
        self.assertEqual([r["status"] for r in self.rows()], ["refused", "refused"])

    async def test_both_refusals_keep_the_same_schedule(self):
        """Answered the moment each was known, a block (a millisecond) and a
        403 (a Discord round trip) told themselves apart by timing. A pad on
        the block path alone only moved the oracle. Both must wait for the
        SAME deadline, drawn at the start, and settle only after it."""
        order = []

        async def sleep_until(deadline):
            order.append(("sleep", deadline))
        real_settle = bot_commands._settle

        async def settle(fn, relay_id, *a):
            order.append(("settle", fn.__name__))
            await real_settle(fn, relay_id, *a)

        with patch.object(bot_commands, "_refusal_deadline", lambda: 4242.0), \
             patch.object(bot_commands, "_sleep_until", sleep_until), \
             patch.object(bot_commands, "_settle", settle):
            relay_store.block(RECIPIENT, SENDER)
            await self.tell()
            by_block = list(order)
            relay_store.unblock(RECIPIENT, SENDER)
            order.clear()
            closed = _member()
            closed.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
            await self.tell(recipient=closed)
            by_403 = list(order)

        expected = [("sleep", 4242.0), ("settle", "mark_refused")]
        self.assertEqual(by_block, expected)
        self.assertEqual(by_403, expected)

    async def test_deliveries_do_not_wait_for_the_refusal_deadline(self):
        with patch.object(bot_commands, "_sleep_until", AsyncMock()) as sleep_until:
            await self.tell()
        sleep_until.assert_not_awaited()

    async def test_the_deadline_is_real_in_production(self):
        """Every other test zeroes the window. The shipped one must outlast a
        normal Discord round trip — read from source, because setUp patches it."""
        src = Path(relay_format.__file__).read_text(encoding="utf-8")
        self.assertIn("REFUSAL_WINDOW_SEC = (1.0, 2.0)", src)
        with patch.object(relay_format, "REFUSAL_WINDOW_SEC", (1.0, 2.0)):
            loop = asyncio.get_running_loop()
            self.assertGreaterEqual(bot_commands._refusal_deadline() - loop.time(), 0.99)

    async def test_a_second_account_cannot_tell_a_block_from_closed_dms_mid_flight(self):
        """The race review found: a 403-bound send sits `reserved` for its
        Discord round trip, counting toward the recipient's inbox, while a
        block used to be `refused` at once and never counted. Account B,
        sending while A's attempt was still in flight, saw "their inbox is
        full" only when A was NOT blocked."""
        seen = {}
        with patch.object(fritz_utils, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 1):
            for blocked in (False, True):
                gate = asyncio.Event()
                recipient = _member(uid=600 + int(blocked))
                recipient_id = f"discord-{recipient.id}"

                async def slow_403(*a, **kw):
                    await gate.wait()
                    raise _http_error(discord.Forbidden, 403, 50007)
                recipient.send.side_effect = slow_403

                async def hold(deadline):
                    await gate.wait()
                if blocked:
                    relay_store.block(recipient_id, SENDER)
                with patch.object(bot_commands, "_sleep_until", hold):
                    a = asyncio.create_task(self.tell(recipient=recipient))
                    try:
                        for _ in range(200):              # until A holds its reservation
                            if any(r["recipient_id"] == recipient_id for r in self.rows()):
                                break
                            await asyncio.sleep(0.01)
                        b_interaction = self.interaction(user_id=222)
                        # Bounded: if A settled early, B is admitted and its
                        # send waits on the same gate — a deadlock that must
                        # fail this test, not hang the suite.
                        await asyncio.wait_for(self.tell(b_interaction, recipient=recipient), 10)
                        seen[blocked] = self.reply(b_interaction)
                    finally:
                        gate.set()
                        await asyncio.wait_for(a, 10)
        self.assertEqual(seen[False], seen[True])
        self.assertIn("received as many", seen[False])

    async def test_the_sender_cap_refuses_and_sends_nothing(self):
        with patch.object(fritz_utils, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            await self.tell()
            interaction, recipient = await self.tell()
        self.assertIn("limit", self.reply(interaction))
        recipient.send.assert_not_awaited()

    async def test_a_disabled_relay_refuses_and_writes_nothing(self):
        with patch.object(fritz_utils, "RELAY_ENABLED", False):
            interaction, recipient = await self.tell()
        self.assertIn("switched off", self.reply(interaction))
        recipient.send.assert_not_awaited()
        self.assertEqual(self.rows(), [])

    async def test_bots_fritz_and_yourself_are_refused(self):
        # Each case asserts its OWN copy. Fritz is a bot too, so "was it
        # refused" alone cannot tell the Fritz branch from the generic one.
        cases = {
            "bot": (dict(recipient=_member(bot=True)), "Bots don't read"),
            "fritz": (dict(recipient=_member(uid=999, bot=True)), "already here"),
            "self": (dict(recipient=_member(uid=SENDER_SNOWFLAKE)), "yourself"),
        }
        for label, (kw, expected) in cases.items():
            with self.subTest(label):
                interaction, recipient = await self.tell(**kw)
                self.assertIn(expected, self.reply(interaction))
                recipient.send.assert_not_awaited()
        self.assertEqual(self.rows(), [])

    async def test_outside_a_server_nothing_happens(self):
        interaction, recipient = await self.tell(self.interaction(guild_id=None))
        self.assertIn("Nothing was sent", self.reply(interaction))
        recipient.send.assert_not_awaited()
        self.assertEqual(self.rows(), [])

    async def test_a_user_installed_copy_cannot_relay_from_a_server_fritz_is_not_in(self):
        interaction = self.interaction()
        interaction.is_guild_integration = MagicMock(return_value=False)
        _, recipient = await self.tell(interaction)
        self.assertIn("Nothing was sent", self.reply(interaction))
        recipient.send.assert_not_awaited()
        self.assertEqual(self.rows(), [])

    async def test_whitespace_is_not_a_message(self):
        interaction, recipient = await self.tell(message="   \n  ")
        self.assertIn("Nothing was sent", self.reply(interaction))
        recipient.send.assert_not_awaited()

    async def test_no_receipt_unless_something_was_delivered(self):
        """A receipt reads "as delivered". On any other outcome it would lie."""
        relay_store.block(RECIPIENT, SENDER)
        blocked, _ = await self.tell()
        relay_store.unblock(RECIPIENT, SENDER)
        closed = _member()
        closed.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
        refused, _ = await self.tell(recipient=closed)
        broken = _member()
        broken.send.side_effect = RuntimeError("x")
        failed, _ = await self.tell(recipient=broken)
        for interaction in (blocked, refused, failed):
            interaction.user.send.assert_not_awaited()


class TestFailures(TellTestCase):
    """Every way the send can fail on OUR side: one answer, says in words that
    nothing was sent, leaks no exception text, writes 'failed', costs the
    sender nothing."""

    FAILURES = {
        "not found": lambda: _http_error(discord.NotFound, 404, 10013),
        "rate limited": lambda: discord.RateLimited(7.0),
        "server error": lambda: _http_error(discord.DiscordServerError, 503),
        "http 429": lambda: _http_error(discord.HTTPException, 429),
        "http other": lambda: _http_error(discord.HTTPException, 418),
        "anything else": lambda: RuntimeError("secret internal detail"),
    }

    async def test_every_failure_is_answered_once_and_honestly(self):
        for label, make in self.FAILURES.items():
            with self.subTest(label), \
                 patch.object(fritz_utils, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
                recipient = _member()
                recipient.send.side_effect = make()
                interaction, _ = await self.tell(recipient=recipient)
                text = self.reply(interaction)
                self.assertIn("Nothing was sent", text)
                self.assertNotIn("secret internal detail", text)
                self.assertNotIn("Traceback", text)
                self.assertIn("ref", text)                   # greppable in the log
                self.assertEqual(self.rows()[-1]["status"], "failed")
                # Free: with a cap of one, the next attempt still goes through.
                _, ok_r = await self.tell()
                ok_r.send.assert_awaited_once()
                with sqlite3.connect(self.db) as conn:     # reset for the next case
                    conn.execute("DELETE FROM relay_messages")

    async def test_content_discord_rejects_is_charged(self):
        """Discord's harmful-link filter reliably 400s. Free, that was an
        unlimited supply of DM-channel opens on the bot's token."""
        with patch.object(fritz_utils, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            recipient = _member()
            recipient.send.side_effect = _http_error(discord.HTTPException, 400, 240000)
            interaction, _ = await self.tell(recipient=recipient)
            self.assertIn("declined the message as written", self.reply(interaction))
            self.assertEqual(self.rows()[-1]["status"], "rejected")
            again, again_r = await self.tell()
        self.assertIn("limit", self.reply(again))
        again_r.send.assert_not_awaited()

    async def test_a_rate_limit_says_how_long(self):
        recipient = _member()
        recipient.send.side_effect = discord.RateLimited(7.0)
        interaction, _ = await self.tell(recipient=recipient)
        self.assertIn("7 seconds", self.reply(interaction))

    async def test_a_store_failure_before_sending_sends_nothing(self):
        with patch.object(relay_store, "reserve_send",
                          side_effect=relay_store.RelayStoreError("disk full")):
            interaction, recipient = await self.tell()
        text = self.reply(interaction)
        self.assertIn("Nothing was sent", text)
        self.assertNotIn("disk full", text)
        recipient.send.assert_not_awaited()

    async def test_a_bookkeeping_failure_after_delivery_admits_the_delivery(self):
        """Delivered cannot be un-delivered; reporting a failure would be a lie."""
        with patch.object(relay_store, "mark_sent",
                          side_effect=relay_store.RelayStoreError("locked")):
            interaction, recipient = await self.tell()
        text = self.reply(interaction)
        self.assertIn("Delivered", text)
        self.assertIn("note it down", text)

    async def test_a_bookkeeping_failure_on_a_failed_send_still_answers(self):
        recipient = _member()
        recipient.send.side_effect = _http_error(discord.NotFound, 404)
        with patch.object(relay_store, "mark_failed",
                          side_effect=relay_store.RelayStoreError("locked")):
            interaction, _ = await self.tell(recipient=recipient)
        self.assertIn("Nothing was sent", self.reply(interaction))

    async def test_a_bookkeeping_failure_on_a_refusal_still_answers(self):
        recipient = _member()
        recipient.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
        with patch.object(relay_store, "mark_refused",
                          side_effect=relay_store.RelayStoreError("locked")):
            interaction, _ = await self.tell(recipient=recipient)
        self.assertEqual(self.reply(interaction), relay_store.REFUSED)

    async def test_a_transport_error_on_the_failure_reply_does_not_escape(self):
        """_reply_error caught only HTTPException, so a timeout here reached the
        cog error handler and the sender was answered twice."""
        recipient = _member()
        recipient.send.side_effect = RuntimeError("x")
        interaction = self.interaction()
        interaction.followup.send = AsyncMock(side_effect=TimeoutError())
        await self.tell(interaction, recipient=recipient)    # must not raise

    async def test_cancellation_is_not_swallowed(self):
        recipient = _member()
        recipient.send.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.tell(recipient=recipient)


class TestAuditLog(TellTestCase):
    BODY = "zebra-quartz-7731 meet me by the old mill"

    async def test_no_body_reaches_the_audit_log_on_any_path(self):
        await self.tell(message=self.BODY)                          # delivered
        relay_store.block(RECIPIENT)
        await self.tell(message=self.BODY)                          # refused by a block
        relay_store.unblock(RECIPIENT)
        with patch.object(fritz_utils, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            await self.tell(message=self.BODY)                      # denied by a cap
        refused = _member(uid=556)
        refused.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
        await self.tell(recipient=refused, message=self.BODY)       # refused
        failed = _member(uid=557)
        failed.send.side_effect = RuntimeError("x")
        await self.tell(recipient=failed, message=self.BODY)        # failed

        events = self.audit_events()
        self.assertEqual(sorted({e["event"] for e in events}),
                         ["relay_denied", "relay_failed", "relay_refused", "relay_sent"])
        text = self.audit_path.read_text(encoding="utf-8")
        self.assertNotIn("zebra", text)
        self.assertNotIn("old mill", text)
        for event in events:
            self.assertEqual(event["chars"], len(self.BODY), event["event"])
            self.assertEqual(event["body_digest"], bot_commands._audit_digest(self.BODY))

    async def test_the_digest_is_keyed_not_a_bare_hash(self):
        """sha256("ok")[:16] is the message, to anyone with a dictionary."""
        await self.tell(message="ok")
        [sent] = [e for e in self.audit_events() if e["event"] == "relay_sent"]
        self.assertEqual(sent["body_digest"], bot_commands._audit_digest("ok"))
        self.assertNotEqual(sent["body_digest"], hashlib.sha256(b"ok").hexdigest()[:16])
        with patch.object(fritz_utils, "CHAT_COOKIE_SECRET", "another-host"):
            self.assertNotEqual(bot_commands._audit_digest("ok"), sent["body_digest"])


class TestCommandShape(unittest.IsolatedAsyncioTestCase):
    async def test_tell_is_guild_only_and_guild_installed_on_the_wire(self):
        """guild_only alone sends only the deprecated dm_permission field, and
        says nothing about a user-installed copy of Fritz."""
        from discord.ext import commands
        bot = commands.Bot(command_prefix="$", intents=discord.Intents.default())
        await bot.add_cog(_make_cog())
        payload = bot.tree.get_command("tell").to_dict(bot.tree)
        self.assertIs(payload["dm_permission"], False)
        self.assertEqual(payload["contexts"], [0])
        self.assertEqual(payload["integration_types"], [0])

    def test_the_recipient_must_be_a_member(self):
        """Member, not User and not a union: that is what rejects non-members."""
        sig = inspect.signature(FritzCommands.tell_message.callback)
        self.assertIs(sig.parameters["recipient"].annotation, discord.Member)

    def test_the_message_cannot_outgrow_an_embed(self):
        params = {p.name: p for p in FritzCommands.tell_message.parameters}
        self.assertEqual(params["message"].max_value, bot_commands.TELL_MAX_CHARS)
        self.assertEqual(bot_commands._tell_max_chars(99_999), 4096)
        self.assertEqual(bot_commands._tell_max_chars(500), 500)
        self.assertEqual(bot_commands._tell_max_chars(0), 1)


class TestNonMemberRecipient(unittest.IsolatedAsyncioTestCase):
    def _interaction(self, qualified_name):
        interaction = _fake_interaction("someone")
        interaction.response.is_done = MagicMock(return_value=False)
        interaction.command = MagicMock()
        interaction.command.qualified_name = qualified_name
        return interaction

    async def test_a_non_member_gets_its_own_copy(self):
        interaction = self._interaction("tell message")
        err = app_commands.TransformerError("x", discord.AppCommandOptionType.user, MagicMock())
        await bot_commands.handle_app_command_error(interaction, err)
        text = interaction.response.send_message.await_args.args[0]
        self.assertIn("members of this server", text)
        self.assertIn("Nothing was sent", text)

    async def test_other_commands_keep_the_generic_copy(self):
        interaction = self._interaction("draw")
        err = app_commands.TransformerError("x", discord.AppCommandOptionType.integer, MagicMock())
        await bot_commands.handle_app_command_error(interaction, err)
        self.assertIn("permitted range", interaction.response.send_message.await_args.args[0])


class TestShipsDark(unittest.TestCase):
    def test_the_relay_is_off_unless_an_operator_turns_it_on(self):
        """Every commit lands on master. Until /relay block exists, a live /tell
        lets anyone DM anyone with no way to refuse. Read from source so a
        developer's own .env cannot make this pass or fail."""
        src = Path(fritz_utils.__file__).read_text(encoding="utf-8")
        self.assertIn('os.environ.get("RELAY_ENABLED", "false")', src)


if __name__ == "__main__":
    unittest.main()
