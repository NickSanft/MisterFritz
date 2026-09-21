"""
Tests for PR 6 of plans/12-direct-message-relay.md: what /forget and /export
do with the relay, reconciling reservations a crash left open, retention, and
what a reply to a relay whose row has gone is told.

The rule that shapes all of it is DECISIONS #22: /forget all removes every
message relayed to or from you, but keeps blocks — the ones placed AGAINST you
because they are other people's, and the ones YOU placed because a privacy
command must not quietly re-arm a harasser.

The other rules: forgetting must not reset a rate limit, so a row still
inside the rate window is scrubbed to what the caps need rather than deleted;
and a SENDER forgetting must not disarm the recipient's block, so a message
they delivered keeps, until retention, what "Block this sender" needs.
"""
import asyncio
import logging
import os
import pathlib
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import discord
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

import bot_commands
import fritz_utils
import main_discord
import migrate_identity
import privacy
import relay_format
import relay_router
import relay_store
from test_relay_router import ReplyTestCase
from test_tell_command import (RECIPIENT, RECIPIENT_SNOWFLAKE, SENDER, SENDER_SNOWFLAKE,
                               TellTestCase, _member)

REPO = pathlib.Path(__file__).resolve().parent.parent
THIRD = "discord-333333333333333333"
BOT_ID = 999                                # what TellTestCase makes Fritz

# Three accounts under CHAINED links: A is an alt of B, and B of C.
# resolve_identity is one hop, so A's messages are written as B's and B's as
# C's. Resolving twice acts on the wrong person.
ACCOUNT_A = "discord-121212121212121212"
ACCOUNT_B = "discord-232323232323232323"
ACCOUNT_C = "discord-343434343434343434"
CHAINED = {ACCOUNT_A: ACCOUNT_B, ACCOUNT_B: ACCOUNT_C}


def _ago(**delta) -> str:
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


class RelayPrivacyTestCase(TellTestCase):
    """A few relays in both directions, delivered through the real /tell."""

    async def relay(self, sender_interaction=None, recipient=None, message="hi"):
        recipient = recipient or _member()
        await self.tell(sender_interaction, recipient=recipient, message=message)
        return recipient

    def all_rows(self):
        with sqlite3.connect(self.db) as conn:
            return conn.execute("SELECT sender_id, recipient_id FROM relay_messages").fetchall()

    def rows(self):
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM relay_messages ORDER BY created_at")]

    def menu(self):
        """The recipient, invoking Apps -> Block this sender."""
        interaction = self.interaction(user_id=RECIPIENT_SNOWFLAKE, name="bob")
        interaction.response.defer = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=True)
        return interaction

    def execute(self, sql, *params):
        with sqlite3.connect(self.db) as conn:
            conn.execute(sql, params)
            conn.commit()

    def backdate(self, **delta):
        """Move every row back in time, its reply window with it."""
        with sqlite3.connect(self.db) as conn:
            for rid, created, expires in conn.execute(
                    "SELECT id, created_at, expires_at FROM relay_messages").fetchall():
                shift = timedelta(**delta)
                conn.execute(
                    "UPDATE relay_messages SET created_at = ?, expires_at = ? WHERE id = ?",
                    ((datetime.fromisoformat(created) - shift).isoformat(),
                     (datetime.fromisoformat(expires) - shift).isoformat(), rid))
            conn.commit()


# ─── /forget ─────────────────────────────────────────────────────────────────

class TestForgetRelay(RelayPrivacyTestCase):
    async def test_both_directions_go(self):
        """A shared exchange cannot be half-deleted: keeping one half so the
        other party can still export it would mean "forget me" did not."""
        await self.relay(message="sent by the sender")
        await self.relay(self.interaction(user_id=RECIPIENT_SNOWFLAKE, name="bob"),
                         _member(uid=SENDER_SNOWFLAKE, name="alice"),
                         message="sent by the recipient")
        self.assertEqual(privacy.forget_relay(SENDER), 2)
        for who in (SENDER, RECIPIENT):
            exported = privacy.export_relay(who)
            self.assertEqual((exported["sent"], exported["received"]), ([], []), who)
        self.assertEqual({r["body"] for r in self.rows()}, {""})

    async def test_a_recent_message_is_scrubbed_to_what_the_caps_need(self):
        """Deleting it handed back the hour's quota. What stays is the two ids,
        the status and the time — nothing anyone typed or was shown."""
        await self.relay(message="meet me at the old mill")
        [before] = self.rows()
        privacy.forget_relay(RECIPIENT)
        [after] = self.rows()
        for kept in ("id", "sender_id", "recipient_id", "status", "created_at", "expires_at"):
            self.assertEqual(after[kept], before[kept], kept)
        self.assertEqual(after["body"], "")
        for gone in ("shown_as", "sender_account", "recipient_account", "origin_id",
                     "guild_id", "dm_channel_id", "dm_message_id", "error"):
            self.assertIsNone(after[gone], gone)
        self.assertTrue(after["closed_at"], "a scrubbed relay must not stay answerable")
        self.assertEqual(after["forgotten"], 1)
        self.assertIsNone(relay_store.get_by_dm_message(before["dm_message_id"]))

    async def test_an_older_message_is_deleted_outright(self):
        await self.relay()
        self.backdate(minutes=61)
        self.assertEqual(privacy.forget_relay(RECIPIENT), 1)
        self.assertEqual(self.rows(), [])

    async def test_a_senders_undelivered_attempts_go_like_anything_else(self):
        """Nothing reached anyone, so there is nothing to block from."""
        relay_store.block(RECIPIENT, SENDER)
        await self.relay(message="let me in")
        privacy.forget_relay(SENDER)
        [row] = self.rows()
        self.assertEqual((row["forgotten"], row["sender_account"]),
                         (relay_store.FORGOTTEN_SCRUBBED, None))
        self.backdate(minutes=61)
        privacy.forget_relay(SENDER)
        self.assertEqual(self.rows(), [])

    async def test_forgetting_does_not_hand_back_the_senders_quota(self):
        """Send until refused, /forget all, send again — without this, an
        unbounded loop against the one control that stops a DM-spam run."""
        with patch.object(fritz_utils, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            await self.relay()
            privacy.forget_relay(SENDER)
            again = relay_store.reserve_send(SENDER, THIRD, "again")
        self.assertFalse(again.ok)
        self.assertEqual(again.reason, "sender_rate_limited")

    async def test_nor_the_recipients_inbound_cap(self):
        """The only cap that stops a brigade. A recipient who forgets does not
        open their inbox to another hour's worth."""
        with patch.object(fritz_utils, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 1):
            await self.relay()
            privacy.forget_relay(RECIPIENT)
            again = relay_store.reserve_send(THIRD, RECIPIENT, "me too")
        self.assertFalse(again.ok)
        self.assertEqual(again.reason, "recipient_rate_limited")

    async def test_the_count_leaves_out_what_was_refused_on_the_way_in(self):
        """The count is what they could have seen. Refused attempts at them
        would say, in a number, that someone they blocked keeps trying."""
        relay_store.block(RECIPIENT, SENDER)
        await self.relay(message="let me in")
        await self.relay(self.interaction(user_id=333333333333333333, name="carol"),
                         message="hello from carol")
        self.assertEqual(privacy.forget_relay(RECIPIENT), 1)

    async def test_a_sender_counts_their_own_refused_attempts(self):
        relay_store.block(RECIPIENT, SENDER)
        await self.relay(message="let me in")
        self.assertEqual(privacy.forget_relay(SENDER), 1)

    async def test_forgetting_twice_removes_nothing_the_second_time(self):
        await self.relay()
        privacy.forget_relay(SENDER)
        self.assertEqual(privacy.forget_relay(SENDER), 0)

    async def test_other_peoples_relays_stay(self):
        await self.relay()
        await self.relay(self.interaction(user_id=333333333333333333, name="carol"),
                         _member(uid=444444444444444444, name="dave"), message="theirs")
        privacy.forget_relay(SENDER)
        [theirs] = [r for r in self.rows() if r["sender_id"] == THIRD]
        self.assertEqual((theirs["body"], theirs["forgotten"]), ("theirs", 0))
        self.assertEqual(len(privacy.export_relay(THIRD)["sent"]), 1)

    async def test_no_block_is_touched_in_either_direction(self):
        """DECISIONS #22. Blocks against you are other people's data — deleting
        them would launder away every block on you. Blocks you placed stay so
        /forget all never re-arms a harasser; /relay forget-blocks drops them."""
        relay_store.block(SENDER, THIRD)            # placed by the forgetter
        relay_store.block(RECIPIENT, SENDER)        # placed against the forgetter
        privacy.forget_relay(SENDER)
        self.assertEqual(relay_store.list_blocks(SENDER), [THIRD])
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [SENDER])

    async def test_a_forgotten_sender_is_not_offered_as_someone_to_block(self):
        await self.relay()
        self.assertEqual(len(relay_store.recent_senders(RECIPIENT)), 1)
        privacy.forget_relay(RECIPIENT)
        self.assertEqual(relay_store.recent_senders(RECIPIENT), [])

    async def test_forgetting_through_an_alias_forgets_the_person(self):
        await self.relay()
        with patch.object(fritz_utils, "IDENTITY_LINKS", {"web-alice": SENDER}):
            self.assertEqual(privacy.forget_relay("web-alice"), 1)
        self.assertEqual(privacy.export_relay(SENDER)["sent"], [])

    async def test_forget_all_reports_the_relays_it_removed(self):
        await self.relay()
        result = privacy.forget_all(SENDER, None)
        self.assertEqual(result["relays"], 1)
        self.assertEqual(privacy.export_relay(SENDER)["sent"], [])

    async def test_a_store_failure_is_reported_as_nothing_removed(self):
        with patch.object(relay_store, "forget_relay",
                          side_effect=relay_store.RelayStoreError("locked")):
            self.assertEqual(privacy.forget_relay(SENDER), 0)


class TestResolvedExactlyOnce(RelayPrivacyTestCase):
    """The store resolves an identity once, as it did when the rows were
    written. Resolving in privacy as well followed a second hop under chained
    links: A's own messages untouched and C's gone."""

    def write_chained(self):
        with patch.object(fritz_utils, "IDENTITY_LINKS", CHAINED):
            self.assertTrue(relay_store.reserve_send(ACCOUNT_A, RECIPIENT, "from A").ok)
            self.assertTrue(relay_store.reserve_send(ACCOUNT_B, THIRD, "from B").ok)
        self.assertEqual({r["body"]: r["sender_id"] for r in self.rows()},
                         {"from A": ACCOUNT_B, "from B": ACCOUNT_C})

    def test_forget_acts_on_the_accounts_own_messages(self):
        self.write_chained()
        with patch.object(fritz_utils, "IDENTITY_LINKS", CHAINED):
            self.assertEqual(privacy.forget_relay(ACCOUNT_A), 1)
        self.assertEqual({r["body"] for r in self.rows()}, {"", "from B"})

    def test_export_shows_the_accounts_own_messages(self):
        self.write_chained()
        with patch.object(fritz_utils, "IDENTITY_LINKS", CHAINED):
            sent = privacy.export_relay(ACCOUNT_A)["sent"]
        self.assertEqual([s["body"] for s in sent], ["from A"])

    def test_privacy_hands_the_store_the_account_as_typed(self):
        with patch.object(fritz_utils, "IDENTITY_LINKS", {"web-alice": SENDER}), \
             patch.object(relay_store, "forget_relay", return_value=0) as forget, \
             patch.object(relay_store, "export_relay", return_value={}) as export:
            privacy.forget_relay("web-alice")
            privacy.export_relay("web-alice")
        forget.assert_called_once_with("web-alice")
        export.assert_called_once_with("web-alice")

    def test_forget_all_hands_every_store_the_id_as_given(self):
        """Each operation resolves at its own entry point (pinned in
        test_privacy.TestAliasResolution), so forget_all must not."""
        stores = dict(forget_memories=0, forget_conversation=0, forget_schedules=0,
                      forget_workspace=False, forget_alias=False, forget_relay=0)
        with patch.object(fritz_utils, "IDENTITY_LINKS", {"web-alice": SENDER}), \
             patch.multiple(privacy, **{k: MagicMock(return_value=v) for k, v in stores.items()}):
            privacy.forget_all("web-alice", None)
            for name in stores:
                self.assertEqual(getattr(privacy, name).call_args.args[0], "web-alice", name)

    def test_under_chained_links_forget_all_acts_on_the_next_hop_only(self):
        """Resolving in forget_all AND in each operation was two hops: /forget
        all from A deleted C's workspace, among everything else."""
        import workspace_store
        with patch.object(fritz_utils, "IDENTITY_LINKS", CHAINED), \
             patch.object(workspace_store, "remove", return_value=True) as remove, \
             patch.multiple(privacy, forget_memories=MagicMock(return_value=0),
                            forget_conversation=MagicMock(return_value=0),
                            forget_alias=MagicMock(return_value=False)):
            manager = MagicMock()
            manager.remove_all_for_user.return_value = 0
            privacy.forget_all(ACCOUNT_A, manager)
        remove.assert_called_once_with(ACCOUNT_B)
        manager.remove_all_for_user.assert_called_once_with(ACCOUNT_B)

    def test_export_user_data_hands_every_store_the_id_as_given(self):
        stores = dict(export_memories=[], export_schedules=[],
                      count_conversation_checkpoints=0, get_workspace_for_export=None,
                      export_relay={})
        with patch.object(fritz_utils, "IDENTITY_LINKS", CHAINED), \
             patch.multiple(privacy, **{k: MagicMock(return_value=v) for k, v in stores.items()}):
            data = privacy.export_user_data(ACCOUNT_A, None)
            for name in stores:
                self.assertEqual(getattr(privacy, name).call_args.args[0], ACCOUNT_A, name)
        self.assertEqual(data["user_id"], ACCOUNT_B)            # one hop, not two


class TestTheRowsFollowTheAccountThatWroteThem(RelayPrivacyTestCase):
    """A link added or removed after the fact must not strand a row: matching
    only the resolved column left it neither forgotten nor exported."""

    ALT, MAIN = "discord-777777777777777777", "discord-888888888888888888"

    def test_a_link_added_later(self):
        self.assertTrue(relay_store.reserve_send(self.ALT, RECIPIENT, "before the link").ok)
        with patch.object(fritz_utils, "IDENTITY_LINKS", {self.ALT: self.MAIN}):
            self.assertEqual([s["body"] for s in privacy.export_relay(self.ALT)["sent"]],
                             ["before the link"])
            self.assertEqual(privacy.forget_relay(self.ALT), 1)
        self.assertEqual({r["body"] for r in self.rows()}, {""})

    def test_a_link_removed_later(self):
        with patch.object(fritz_utils, "IDENTITY_LINKS", {self.ALT: self.MAIN}):
            self.assertTrue(relay_store.reserve_send(self.ALT, RECIPIENT, "under the link").ok)
        self.assertEqual([s["body"] for s in privacy.export_relay(self.ALT)["sent"]],
                         ["under the link"])
        self.assertEqual(privacy.forget_relay(self.ALT), 1)
        self.assertEqual({r["body"] for r in self.rows()}, {""})

    def test_a_link_added_later_on_the_receiving_side(self):
        reservation = relay_store.reserve_send(SENDER, self.ALT, "to the alt")
        relay_store.mark_sent(reservation.id, 3131, 3232, shown_as="@alice")
        with patch.object(fritz_utils, "IDENTITY_LINKS", {self.ALT: self.MAIN}):
            self.assertEqual([r["body"] for r in privacy.export_relay(self.ALT)["received"]],
                             ["to the alt"])
            self.assertEqual(privacy.forget_relay(self.ALT), 1)
        self.assertEqual(self.rows()[0]["forgotten"], relay_store.FORGOTTEN_SCRUBBED)

    def test_a_row_from_before_the_account_columns_is_still_found(self):
        """NULL account columns: `=` there made NOT (...) NULL, not true."""
        reservation = relay_store.reserve_send(SENDER, RECIPIENT, "legacy")
        relay_store.mark_sent(reservation.id, 5151, 5252, shown_as="@alice")
        self.execute("UPDATE relay_messages SET sender_account = NULL, recipient_account = NULL")
        self.assertEqual(privacy.forget_relay(SENDER), 1)
        [row] = self.rows()
        self.assertEqual((row["body"], row["forgotten"]), ("", relay_store.FORGOTTEN_ANCHOR))


class TestASenderCannotDisarmTheBlock(RelayPrivacyTestCase):
    """Relay, /forget all, relay again: scrubbing everything the sender sent
    left the recipient no way to block them from the message in their inbox."""

    async def a_message_then_the_sender_forgets(self, sender_interaction=None):
        recipient = _member()
        await self.relay(sender_interaction, recipient=recipient, message="you again")
        dm = recipient.send.return_value
        privacy.forget_relay(SENDER)
        return dm

    async def test_block_this_sender_still_works(self):
        dm = await self.a_message_then_the_sender_forgets(self.interaction(display="Mal"))
        menu = self.menu()
        await self.cog.block_sender_from_message(menu, dm)
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [SENDER])
        self.assertIn("@alice", menu.followup.send.await_args.args[0])
        again = relay_store.reserve_send(SENDER, RECIPIENT, "and again")
        self.assertEqual(again.reason, "blocked_sender")

    async def test_and_so_does_the_block_picker(self):
        await self.a_message_then_the_sender_forgets(self.interaction(display="Mal"))
        [offered] = relay_store.recent_senders(RECIPIENT)
        self.assertEqual((offered["account"], offered["shown_as"]), (SENDER, "@alice · Mal"))
        self.assertEqual(bot_commands._block_target(offered["relay_id"], RECIPIENT),
                         (SENDER, "@alice · Mal"))

    async def test_it_names_the_account_that_wrote_not_its_main(self):
        alt, main = 777777777777777777, "discord-888888888888888888"
        with patch.object(fritz_utils, "IDENTITY_LINKS", {f"discord-{alt}": main}):
            recipient = _member()
            await self.relay(self.interaction(user_id=alt, name="mallory", display="mallory"),
                             recipient=recipient)
            privacy.forget_relay(f"discord-{alt}")
            await self.cog.block_sender_from_message(self.menu(), recipient.send.return_value)
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [f"discord-{alt}"])
        self.assertNotIn(main, str(privacy.export_relay(RECIPIENT)))

    async def test_the_anchor_holds_nothing_else(self):
        await self.a_message_then_the_sender_forgets()
        [row] = self.rows()
        self.assertEqual(row["body"], "")
        for gone in ("recipient_account", "origin_id", "guild_id", "dm_channel_id", "error"):
            self.assertIsNone(row[gone], gone)
        for kept in ("sender_account", "shown_as", "dm_message_id", "recipient_id"):
            self.assertTrue(row[kept], kept)
        self.assertEqual(row["forgotten"], relay_store.FORGOTTEN_ANCHOR)
        self.assertTrue(row["closed_at"], "the sender forgot it; a reply must lapse")

    async def test_it_is_in_no_export_and_no_count(self):
        await self.a_message_then_the_sender_forgets()
        self.assertEqual(privacy.export_relay(SENDER)["sent"], [])
        self.assertEqual(privacy.export_relay(RECIPIENT)["received"], [])
        self.assertEqual(privacy.forget_relay(SENDER), 0)
        self.assertEqual(len(self.rows()), 1)                     # and a second forget keeps it

    async def test_it_outlives_the_hour_but_not_retention(self):
        await self.a_message_then_the_sender_forgets()
        self.backdate(minutes=61)
        privacy.forget_relay(SENDER)
        self.assertEqual(relay_store.purge_expired(7), 0)
        self.assertEqual(len(self.rows()), 1)
        self.backdate(days=8)
        self.assertEqual(relay_store.purge_expired(7), 1)

    async def test_a_sender_forgetting_later_cannot_revive_what_the_recipient_forgot(self):
        """The recipient's forget left no account on the row. Re-marking it as
        an anchor would offer the resolved sender_id in their block picker."""
        await self.relay()
        privacy.forget_relay(RECIPIENT)
        privacy.forget_relay(SENDER)
        [row] = self.rows()
        self.assertEqual(row["forgotten"], relay_store.FORGOTTEN_SCRUBBED)
        self.assertEqual(relay_store.recent_senders(RECIPIENT), [])

    async def test_the_recipient_forgetting_takes_it_too(self):
        await self.a_message_then_the_sender_forgets()
        privacy.forget_relay(RECIPIENT)
        [row] = self.rows()
        self.assertEqual((row["forgotten"], row["sender_account"], row["dm_message_id"]),
                         (relay_store.FORGOTTEN_SCRUBBED, None, None))
        self.assertEqual(relay_store.recent_senders(RECIPIENT), [])

    def test_one_still_in_flight_is_bound_when_it_lands(self):
        reservation = relay_store.reserve_send(SENDER, RECIPIENT, "in flight")
        privacy.forget_relay(SENDER)
        relay_store.mark_sent(reservation.id, 6161, 6262, shown_as="@alice")
        row = relay_store.get_by_dm_message(6161)
        self.assertEqual((row["sender_account"], row["shown_as"], row["body"]),
                         (SENDER, "@alice", ""))

    def test_one_that_never_lands_goes_with_the_scrubbed(self):
        reservation = relay_store.reserve_send(SENDER, RECIPIENT, "in flight")
        privacy.forget_relay(SENDER)
        relay_store.mark_failed(reservation.id, "HTTPException 500")
        self.backdate(minutes=61)
        self.assertEqual(relay_store.purge_expired(), 1)


class TestBlockingFromAMessageWhoseRowIsGone(RelayPrivacyTestCase):
    """Past retention, or forgotten by the recipient. "That is not a message I
    carried to you" would be untrue, and would send them looking for a bug."""

    def as_discord_has_it(self, recipient, author_id=BOT_ID):
        original = MagicMock(spec=discord.Message)
        original.id = recipient.send.return_value.id
        original.author = MagicMock()
        original.author.id = author_id
        original.embeds = [recipient.send.await_args.kwargs["embed"]]
        return original

    async def test_it_says_what_happened_and_what_still_works(self):
        recipient = _member()
        await self.relay(recipient=recipient)
        privacy.forget_relay(RECIPIENT)
        menu = self.menu()
        await self.cog.block_sender_from_message(menu, self.as_discord_has_it(recipient))
        self.assertEqual(menu.followup.send.await_args.args[0], bot_commands._BLOCK_GONE)
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [])

    async def test_a_lookalike_from_anyone_else_is_not_ours(self):
        recipient = _member()
        await self.relay(recipient=recipient)
        privacy.forget_relay(RECIPIENT)
        menu = self.menu()
        forged = self.as_discord_has_it(recipient, author_id=SENDER_SNOWFLAKE)
        await self.cog.block_sender_from_message(menu, forged)
        self.assertIn("not a message I carried", menu.followup.send.await_args.args[0])


class TestAForgetDuringASend(RelayPrivacyTestCase):
    """mark_sent used to write the routing key and author line back onto a row
    scrubbed while its DM was in flight."""

    def test_a_scrubbed_row_stays_scrubbed(self):
        reservation = relay_store.reserve_send(SENDER, RECIPIENT, "in flight")
        privacy.forget_relay(RECIPIENT)
        relay_store.mark_sent(reservation.id, 7171, 7272, shown_as="@alice")
        [row] = self.rows()
        self.assertEqual(row["status"], "delivered")                 # still charged
        self.assertEqual((row["dm_message_id"], row["dm_channel_id"], row["shown_as"]),
                         (None, None, None))
        self.assertIsNone(relay_store.get_by_dm_message(7171))

    def test_a_scrubbed_row_is_never_routed_even_holding_a_key(self):
        reservation = relay_store.reserve_send(SENDER, RECIPIENT, "hi")
        relay_store.mark_sent(reservation.id, 8181, 8282, shown_as="@alice")
        privacy.forget_relay(RECIPIENT)
        self.execute("UPDATE relay_messages SET dm_message_id = 8181")
        self.assertIsNone(relay_store.get_by_dm_message(8181))

    def test_a_block_token_for_a_scrubbed_row_names_no_one(self):
        """It would fall back to the resolved sender_id: the main behind an alt."""
        reservation = relay_store.reserve_send(SENDER, RECIPIENT, "hi")
        relay_store.mark_sent(reservation.id, 9191, 9292, shown_as="@alice")
        self.assertIsNotNone(bot_commands._block_target(reservation.id, RECIPIENT))
        privacy.forget_relay(RECIPIENT)
        self.assertIsNone(bot_commands._block_target(reservation.id, RECIPIENT))


# ─── /export ─────────────────────────────────────────────────────────────────

class TestExportRelay(RelayPrivacyTestCase):
    async def test_a_sender_sees_what_they_sent_addressed_as_they_addressed_it(self):
        """Not recipient_id: that is the identity after IDENTITY_LINKS, and it
        would tell a sender which main account the alt they wrote to is."""
        alt, main = "discord-777777777777777777", "discord-888888888888888888"
        with patch.object(fritz_utils, "IDENTITY_LINKS", {alt: main}):
            await self.relay(recipient=_member(uid=777777777777777777), message="hello alt")
        [sent] = privacy.export_relay(SENDER)["sent"]
        self.assertEqual(sent["to"], alt)
        self.assertEqual(sent["body"], "hello alt")
        self.assertEqual(sent["status"], "delivered")
        self.assertNotIn(main, str(privacy.export_relay(SENDER)))

    async def test_a_row_from_before_the_column_names_no_one(self):
        """Falling back to recipient_id would name the main behind an alt."""
        await self.relay()
        self.execute("UPDATE relay_messages SET recipient_account = NULL")
        [sent] = privacy.export_relay(SENDER)["sent"]
        self.assertIsNone(sent["to"])
        self.assertNotIn(RECIPIENT, str(sent))

    async def test_a_recipient_sees_what_reached_them_as_it_was_shown(self):
        await self.relay(self.interaction(display="Mal"), message="see you at 8")
        [got] = privacy.export_relay(RECIPIENT)["received"]
        self.assertEqual(got["from"], "@alice · Mal")
        self.assertEqual(got["body"], "see you at 8")

    async def test_a_message_from_an_alt_names_the_alt_as_it_was_shown(self):
        alt, main = 777777777777777777, "discord-888888888888888888"
        with patch.object(fritz_utils, "IDENTITY_LINKS", {f"discord-{alt}": main}):
            await self.relay(self.interaction(user_id=alt, name="mallory", display="mallory"),
                             message="guess who")
        exported = privacy.export_relay(RECIPIENT)
        self.assertEqual([r["from"] for r in exported["received"]], ["@mallory"])
        self.assertNotIn(main, str(exported))

    async def test_a_refused_attempt_is_not_shown_to_its_target(self):
        """It never reached them, and listing it would say that someone they
        blocked keeps trying."""
        relay_store.block(RECIPIENT, SENDER)
        await self.relay(message="let me in")
        self.assertEqual(privacy.export_relay(RECIPIENT)["received"], [])
        [sent] = privacy.export_relay(SENDER)["sent"]
        self.assertEqual(sent["status"], "refused")    # the sender's own row

    def test_only_what_was_delivered_counts_as_received(self):
        outcomes = {
            "reserved": lambda rid: None,
            "failed": lambda rid: relay_store.mark_failed(rid, "HTTPException 500"),
            "refused": relay_store.mark_refused,
            "rejected": lambda rid: relay_store.mark_rejected(rid, "HTTPException 400"),
            "delivered": lambda rid: relay_store.mark_sent(rid, 4242, 4343, shown_as="@carol"),
        }
        for status, settle in outcomes.items():
            reservation = relay_store.reserve_send(THIRD, RECIPIENT, status)
            settle(reservation.id)
        self.assertEqual(sorted(r["status"] for r in self.rows()), sorted(outcomes))
        received = privacy.export_relay(RECIPIENT)["received"]
        self.assertEqual([r["body"] for r in received], ["delivered"])

    async def test_a_block_and_closed_dms_export_identically_for_the_sender(self):
        relay_store.block(RECIPIENT, SENDER)
        await self.relay(message="one")
        relay_store.unblock(RECIPIENT, SENDER)
        closed = _member()
        from test_tell_command import _http_error
        closed.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
        await self.relay(recipient=closed, message="two")
        rows = privacy.export_relay(SENDER)["sent"]
        shape = [{k: v for k, v in r.items() if k not in ("id", "body", "created_at")} for r in rows]
        self.assertEqual(shape[0], shape[1])

    async def test_a_message_the_other_party_forgot_appears_nowhere(self):
        await self.relay()
        privacy.forget_relay(RECIPIENT)
        self.assertEqual(privacy.export_relay(SENDER)["sent"], [])

    async def test_your_blocks_are_yours_and_theirs_are_not(self):
        """Never the other party's opt-out state."""
        relay_store.block(SENDER, THIRD, label="<@333333333333333333>")
        relay_store.block(SENDER)
        relay_store.block(RECIPIENT, SENDER)        # someone blocked me
        blocks = privacy.export_relay(SENDER)["blocks"]
        self.assertEqual(sorted((b["everyone"], b["blocked"]) for b in blocks),
                         [(False, THIRD), (True, None)])
        self.assertNotIn(RECIPIENT, str(privacy.export_relay(SENDER)))

    async def test_export_user_data_carries_it(self):
        await self.relay()
        data = privacy.export_user_data(SENDER, None)
        self.assertEqual(len(data["relays"]["sent"]), 1)


class TestTheExportFile(RelayPrivacyTestCase):
    async def export(self, data):
        interaction = self.interaction()
        with patch.object(privacy, "export_user_data", return_value=data):
            await self.cog.export_slash.callback(self.cog, interaction)
        return interaction.followup.send.await_args

    async def test_what_people_typed_is_written_as_they_typed_it(self):
        """An emoji escaped as a surrogate pair is three times its UTF-8 size."""
        call = await self.export({"relays": {"sent": [{"body": "on my way \U0001F697"}]}})
        payload = call.kwargs["file"].fp.read()
        self.assertIn("on my way \U0001F697".encode("utf-8"), payload)

    async def test_too_large_says_what_made_it_so_and_what_trims_it(self):
        big = {"sent": [{"body": "x" * 4000}] * 50}
        with patch.object(bot_commands, "_EXPORT_MAX_BYTES", 10_000), \
             patch.object(fritz_utils, "RELAY_RETENTION_DAYS", 9):
            call = await self.export({"memories": [], "relays": big})
        text = call.args[0]
        self.assertIn("mostly your relays", text)
        self.assertIn("9 days", text)
        self.assertIn("/forget all", text)
        self.assertNotIn("file", call.kwargs)

    async def test_memories_are_pointed_at_their_own_command(self):
        with patch.object(bot_commands, "_EXPORT_MAX_BYTES", 1_000):
            call = await self.export({"memories": ["m" * 2000], "relays": {}})
        self.assertIn("mostly your memories", call.args[0])
        self.assertIn("/forget memories", call.args[0])


# ─── Reconcile ───────────────────────────────────────────────────────────────

class TestReconcile(RelayPrivacyTestCase):
    """A crash between reserving and settling leaves a row that will never
    route a reply and never tell its sender anything. on_ready runs again on
    every reconnect, so only a previous process's rows may be closed."""

    def status(self):
        return [r["status"] for r in self.rows()]

    def test_a_reservation_older_than_this_process_is_closed_as_failed(self):
        relay_store.reserve_send(SENDER, RECIPIENT, "hi")
        with patch.object(relay_store, "_PROCESS_STARTED", _ago(seconds=-5)):
            self.assertEqual(relay_store.reconcile_pending(), 1)
        self.assertEqual(self.status(), ["failed"])

    def test_however_young_it_is(self):
        """A grace period measured from now missed a restart inside a minute."""
        relay_store.reserve_send(SENDER, RECIPIENT, "hi")
        self.assertEqual(relay_store.reconcile_pending(before=_ago(seconds=-1)), 1)

    def test_a_reconnect_leaves_this_processes_reservations_alone(self):
        """Even one past the grace window: it may be a DM still in flight, and
        closing it made a delivered relay permanently lapsed."""
        with patch.object(relay_store, "_PROCESS_STARTED", _ago(hours=1)):
            relay_store.reserve_send(SENDER, RECIPIENT, "hi")
            self.execute("UPDATE relay_messages SET created_at = ?",
                         _ago(seconds=relay_store.RESERVATION_GRACE_SEC + 60))
            self.assertEqual(relay_store.reconcile_pending(), 0)
        self.assertEqual(self.status(), ["reserved"])

    async def test_settled_rows_are_never_touched(self):
        await self.relay()
        self.assertEqual(relay_store.reconcile_pending(before=_ago(seconds=-5)), 0)
        self.assertEqual(self.status(), ["delivered"])

    def test_a_closed_reservation_costs_the_sender_nothing(self):
        with patch.object(fritz_utils, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            relay_store.reserve_send(SENDER, RECIPIENT, "hi")
            relay_store.reconcile_pending(before=_ago(seconds=-5))
            self.assertTrue(relay_store.reserve_send(SENDER, THIRD, "again").ok)


# ─── Retention ───────────────────────────────────────────────────────────────

class TestRetention(RelayPrivacyTestCase):
    def age(self, body, *, days, answerable=False):
        """Age one message, its reply window with it — still open if asked."""
        expires = _ago(days=-1) if answerable else _ago(days=days - 1)
        self.execute("UPDATE relay_messages SET created_at = ?, expires_at = ? WHERE body = ?",
                     _ago(days=days), expires, body)

    def bodies(self):
        return {r["body"] for r in self.rows()}

    async def test_old_messages_go_and_recent_ones_stay(self):
        await self.relay(message="old")
        await self.relay(message="new")
        self.age("old", days=8)
        with patch.object(fritz_utils, "RELAY_RETENTION_DAYS", 30):
            self.assertEqual(relay_store.purge_expired(), 0)
        with patch.object(fritz_utils, "RELAY_RETENTION_DAYS", 7):
            self.assertEqual(relay_store.purge_expired(), 1)
        self.assertEqual(self.bodies(), {"new"})

    async def test_a_relay_that_can_still_be_answered_outlives_retention(self):
        """Deleting it turned the recipient's answer into a conversation with
        Fritz. The reply window wins, and validate_config says so."""
        await self.relay(message="still open")
        self.age("still open", days=3, answerable=True)
        self.assertEqual(relay_store.purge_expired(1), 0)
        self.execute("UPDATE relay_messages SET closed_at = ?", _ago(days=1))
        self.assertEqual(relay_store.purge_expired(1), 1)

    def test_only_a_delivered_relay_is_answerable(self):
        """A reservation nothing settled is unclosed and unexpired too — and
        can never be answered, so it must not outlive retention."""
        relay_store.reserve_send(SENDER, RECIPIENT, "never settled")
        self.age("never settled", days=3, answerable=True)
        [row] = self.rows()
        self.assertEqual((row["status"], row["closed_at"]), ("reserved", None))
        self.assertEqual(relay_store.purge_expired(1), 1)

    async def test_a_forgotten_message_goes_once_it_leaves_the_rate_window(self):
        await self.relay()
        privacy.forget_relay(RECIPIENT)
        self.assertEqual(relay_store.purge_expired(), 0)     # still counted by the caps
        self.backdate(minutes=61)
        self.assertEqual(relay_store.purge_expired(), 1)
        self.assertEqual(self.rows(), [])

    async def test_blocks_are_never_purged(self):
        """A block is a standing instruction, not a record; letting it expire
        would quietly re-arm whoever it was placed against."""
        relay_store.block(RECIPIENT, SENDER)
        self.execute("UPDATE relay_optouts SET created_at = '2000-01-01T00:00:00+00:00'")
        relay_store.purge_expired(1)
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [SENDER])

    async def test_a_zero_retention_still_keeps_a_day(self):
        await self.relay()
        # Closed, so it is not kept for being answerable instead.
        self.execute("UPDATE relay_messages SET closed_at = ?", _ago(minutes=1))
        self.assertEqual(relay_store.purge_expired(0), 0)
        self.age("hi", days=2)
        self.assertEqual(relay_store.purge_expired(0), 1)


class TestTheOperatorIsToldWhenTheWindowOutlastsRetention(unittest.TestCase):
    def check(self, window_min, retention_days):
        with patch.multiple(fritz_utils, DISCORD_BOT_TOKEN="token", ROOT_USER="discord-1",
                            RELAY_REPLY_WINDOW_MIN=window_min,
                            RELAY_RETENTION_DAYS=retention_days), \
             patch.object(logging.getLogger(fritz_utils.__name__), "warning") as warn:
            fritz_utils.validate_config()
        return [c.args[0] % c.args[1:] for c in warn.call_args_list
                if "RELAY_REPLY_WINDOW_MIN" in c.args[0]]

    def test_a_window_longer_than_retention_is_warned_about(self):
        [line] = self.check(2 * 1440 + 1, 2)
        self.assertIn("2 days", line)

    def test_a_window_inside_retention_is_not(self):
        self.assertEqual(self.check(1440, 1), [])


# ─── Housekeeping ────────────────────────────────────────────────────────────

class TestHousekeeping(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.reconcile = patch.object(relay_store, "reconcile_pending").start()
        self.purge = patch.object(relay_store, "purge_expired").start()
        self.addCleanup(patch.stopall)

    async def test_it_reconciles_purges_and_registers_the_purge(self):
        manager = MagicMock()
        await main_discord.relay_housekeeping(manager)
        self.reconcile.assert_called_once_with()          # this process's cutoff
        self.purge.assert_called_once_with()              # at boot, not only on the hour
        manager.scheduler.add_job.assert_called_once()
        args, kwargs = manager.scheduler.add_job.call_args
        self.assertIs(args[0], self.purge)
        self.assertEqual(kwargs["id"], "_internal_relay_retention")
        self.assertTrue(kwargs["replace_existing"])

    async def test_it_runs_hourly_and_is_never_skipped_as_late(self):
        """APScheduler's default misfire grace is one second: a purge due while
        the loop was busy was silently dropped until the next day."""
        manager = MagicMock()
        await main_discord.relay_housekeeping(manager)
        kwargs = manager.scheduler.add_job.call_args.kwargs
        fields = {f.name: str(f) for f in kwargs["trigger"].fields}
        self.assertIsInstance(kwargs["trigger"], CronTrigger)
        self.assertEqual((fields["hour"], fields["minute"]), ("*", "30"))
        self.assertIn("misfire_grace_time", kwargs)
        self.assertIsNone(kwargs["misfire_grace_time"])
        self.assertTrue(kwargs["coalesce"])

    async def test_a_real_scheduler_accepts_the_job(self):
        manager = MagicMock()
        manager.scheduler = AsyncIOScheduler()
        manager.scheduler.start(paused=True)
        try:
            await main_discord.relay_housekeeping(manager)
            job = manager.scheduler.get_job("_internal_relay_retention")
            self.assertIsNone(job.misfire_grace_time)
            self.assertTrue(job.coalesce)
            first = manager.scheduler.get_job("_internal_relay_retention")
            with self.assertNoLogs(main_discord.logger, logging.WARNING):
                await main_discord.relay_housekeeping(manager)   # replaced, not refused
            self.assertEqual(len(manager.scheduler.get_jobs()), 1)
            self.assertIsNot(manager.scheduler.get_job("_internal_relay_retention"), first)
        finally:
            manager.scheduler.shutdown(wait=False)

    async def test_no_step_failing_stops_the_others_or_the_bot(self):
        manager = MagicMock()
        self.reconcile.side_effect = RuntimeError("x")
        await main_discord.relay_housekeeping(manager)         # must not raise
        self.purge.assert_called_once()
        self.purge.side_effect = RuntimeError("y")
        await main_discord.relay_housekeeping(manager)
        self.assertEqual(manager.scheduler.add_job.call_count, 2)
        manager.scheduler.add_job.side_effect = RuntimeError("z")
        await main_discord.relay_housekeeping(manager)         # nor this

    def test_on_ready_awaits_it_after_the_scheduler_starts(self):
        src = (REPO / "main_discord.py").read_text(encoding="utf-8")
        body = src.split("async def on_ready(", 1)[1].split("\nasync def ", 1)[0]
        self.assertIn("await relay_housekeeping(schedule_manager)", body)
        self.assertLess(body.index("schedule_manager.start()"),
                        body.index("relay_housekeeping(schedule_manager)"))

    async def test_the_purge_is_never_listed_as_a_users_schedule(self):
        """Registered on a REAL scheduler, so a listing that read APScheduler's
        jobs rather than the schedules table would show it."""
        from test_scheduler import _make_manager
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        manager, _ = _make_manager(path)
        manager.scheduler = AsyncIOScheduler()
        manager.scheduler.start(paused=True)
        try:
            await main_discord.relay_housekeeping(manager)
            self.assertIsNotNone(manager.scheduler.get_job("_internal_relay_retention"))
            ids = {s["id"] for s in manager.list_all_schedules()}
        finally:
            manager.scheduler.shutdown(wait=False)
        self.assertNotIn("_internal_relay_retention", ids)


# ─── What /forget all says ───────────────────────────────────────────────────

class TestForgetAllSaysWhatItDoes(unittest.IsolatedAsyncioTestCase):
    """DECISIONS #22: /forget all no longer forgets literally everything, so it
    must say so, and name the one command that drops your blocks."""

    RESULT = {"memories": 101, "conversation_rows": 202, "schedules": 303,
              "workspace_dropped": True, "alias_dropped": True, "relays": 404}

    def test_every_key_is_rendered_against_its_own_label(self):
        """Distinct values, so two labels swapped cannot pass."""
        report = bot_commands._forget_all_report(self.RESULT)
        for line in ("memories: 101", "conversation rows: 202", "schedules: 303",
                     "workspace dropped: True", "the name I knew you by: forgotten",
                     "relayed messages, sent and received: 404"):
            self.assertIn(line, report)
        # Each flag flipped on its own, so rendering one from the other fails.
        no_alias = bot_commands._forget_all_report(dict(self.RESULT, alias_dropped=False))
        self.assertIn("the name I knew you by: none held", no_alias)
        self.assertIn("workspace dropped: True", no_alias)
        no_workspace = bot_commands._forget_all_report(dict(self.RESULT, workspace_dropped=False))
        self.assertIn("workspace dropped: False", no_workspace)
        self.assertIn("the name I knew you by: forgotten", no_workspace)

    def test_the_report_covers_whatever_forget_all_actually_returns(self):
        """Pinned against the real function, so a new store cannot be added to
        forget_all and quietly left out of what the user is told."""
        with patch.multiple(privacy, forget_memories=MagicMock(return_value=0),
                            forget_conversation=MagicMock(return_value=0),
                            forget_workspace=MagicMock(return_value=False),
                            forget_alias=MagicMock(return_value=False),
                            forget_relay=MagicMock(return_value=0)):
            result = privacy.forget_all("discord-1", MagicMock(remove_all_for_user=MagicMock(return_value=0)))
        bot_commands._forget_all_report(result)        # KeyError if a key is unrendered
        self.assertEqual(set(result), set(self.RESULT))

    def test_both_the_prompt_and_the_report_name_the_escape_hatch(self):
        report = bot_commands._forget_all_report(self.RESULT)
        self.assertIn("/relay forget-blocks", report)
        self.assertIn("/relay forget-blocks", bot_commands._BLOCKS_KEPT)
        self.assertIn("_BLOCKS_KEPT", self.prompt_source())

    def test_neither_claims_more_than_it_does(self):
        """Fritz cannot unsend a delivered DM, and it keeps a text-free audit
        line and the recipient's block anchor. Both the prompt and the report
        say so."""
        prompt = self.prompt_source()
        self.assertIn("I cannot unsend it", prompt)
        self.assertIn("_RELAY_KEPT", prompt)
        self.assertIn(bot_commands._RELAY_KEPT, bot_commands._forget_all_report(self.RESULT))
        for kept in ("audit log", "holds no words", "block you"):
            self.assertIn(kept, bot_commands._RELAY_KEPT)

    @staticmethod
    def prompt_source():
        src = (REPO / "bot_commands.py").read_text(encoding="utf-8")
        return src.split("async def forget_all_slash(", 1)[1].split("async def ", 1)[0]

    async def test_a_lapsed_confirmation_withdraws_its_buttons(self):
        origin = MagicMock()
        origin.edit_original_response = AsyncMock()
        view = bot_commands._ForgetConfirmView("discord-1", None, origin=origin)
        await view.on_timeout()
        kwargs = origin.edit_original_response.await_args.kwargs
        self.assertIsNone(kwargs["view"])
        self.assertIn("Nothing was deleted", kwargs["content"])

    async def test_cancelling_stops_the_view(self):
        view = bot_commands._ForgetConfirmView("discord-1", None, origin=MagicMock())
        press = MagicMock()
        press.response.edit_message = AsyncMock()
        await view.cancel.callback(press)
        self.assertTrue(view.is_finished(), "a lapse notice could follow a cancel")

    async def test_confirming_stops_the_view_and_reports(self):
        view = bot_commands._ForgetConfirmView("discord-1", None, origin=MagicMock())
        press = MagicMock()
        press.response.edit_message = AsyncMock()
        with patch.object(privacy, "forget_all", return_value=self.RESULT):
            await view.confirm.callback(press)
        self.assertTrue(view.is_finished(), "a lapse notice could follow a confirm")
        self.assertEqual(press.response.edit_message.await_args.kwargs["content"],
                         bot_commands._forget_all_report(self.RESULT))


class TestForgetAllIsWiredToItsOrigin(RelayPrivacyTestCase):
    async def test_the_view_can_withdraw_its_own_buttons(self):
        """Without the origin, on_timeout has nothing to edit and the buttons
        just stop working with nothing on screen to say so."""
        interaction = self.interaction()
        await self.cog.forget_all_slash.callback(self.cog, interaction)
        view = interaction.response.send_message.await_args.kwargs["view"]
        self.assertIs(view.origin, interaction)
        self.assertEqual(view.requester, SENDER)


class TestMigrationKnowsTheNewColumns(unittest.TestCase):
    def test_both_account_columns_are_registered(self):
        for column in ("sender_account", "recipient_account"):
            self.assertIn(("relay_messages", column), migrate_identity._SQLITE_TARGETS)


# ─── A reply to a relay whose row has gone ───────────────────────────────────

class TestAReplyToAGoneRelay(ReplyTestCase):
    """Forgotten by either party, purged, or reconciled: there is no row, but
    the DM is still Fritz's relay, and a reply to it must not become a
    conversation turn carrying an answer meant for someone else."""

    def setUp(self):
        super().setUp()
        self.client.user.id = BOT_ID

    @staticmethod
    def as_discord_has_it(embed, author_id=BOT_ID):
        original = MagicMock(spec=discord.Message)
        original.author = MagicMock()
        original.author.id = author_id
        original.embeds = [embed] if embed else []
        return original

    async def a_relay_then(self, forget_who):
        recipient = _member()
        dm = await self.relayed(recipient=recipient)
        embed = recipient.send.await_args.kwargs["embed"]
        forget_who()
        ctx = self.reply_to(dm, "sure, 8 works")
        ctx.reference.resolved = self.as_discord_has_it(embed)
        return ctx

    async def assert_lapsed(self, ctx):
        self.assertTrue(await self.route(ctx))
        self.assertEqual(self.said(ctx), relay_router.LAPSED)
        self.dm_channel.send.assert_not_awaited()
        self.client.create_dm.assert_not_awaited()

    async def test_after_the_recipient_forgot(self):
        await self.assert_lapsed(await self.a_relay_then(lambda: privacy.forget_relay(RECIPIENT)))

    async def test_after_the_sender_forgot(self):
        await self.assert_lapsed(await self.a_relay_then(lambda: privacy.forget_relay(SENDER)))

    async def test_after_the_purge(self):
        def purged():
            with sqlite3.connect(self.db) as conn:
                conn.execute("DELETE FROM relay_messages")
                conn.commit()
        await self.assert_lapsed(await self.a_relay_then(purged))

    async def test_a_carried_back_reply_is_recognised_too(self):
        dm = await self.relayed()
        await self.route(self.reply_to(dm, "yes"))
        back_embed = self.carried()
        privacy.forget_relay(SENDER)
        ctx = self.reply_to(self.sent_back, "great", author_id=SENDER_SNOWFLAKE)
        ctx.reference.resolved = self.as_discord_has_it(back_embed)
        self.dm_channel.send.reset_mock()
        self.client.create_dm.reset_mock()
        await self.assert_lapsed(ctx)

    async def test_a_reply_to_the_senders_own_copy_is_told_so(self):
        """The receipt was never routable. A reply to it used to reach the
        agent, carrying words meant for the other person."""
        interaction = self.interaction()
        await self.tell(interaction)
        receipt = interaction.user.send.await_args.kwargs["embed"]
        self.assertEqual(relay_format.relay_kind(self.as_discord_has_it(receipt), self.client.user),
                         "receipt")
        copy = MagicMock()
        copy.id = 616161616161616161
        ctx = self.reply_to(copy, "did you get this?", author_id=SENDER_SNOWFLAKE)
        ctx.reference.resolved = self.as_discord_has_it(receipt)
        self.assertTrue(await self.route(ctx))
        self.assertEqual(self.said(ctx), relay_router.OWN_COPY)
        self.client.create_dm.assert_not_awaited()

    async def test_the_same_footer_from_anyone_else_is_the_agents(self):
        """The footer text is forgeable; only Fritz's own message counts."""
        ctx = await self.a_relay_then(lambda: privacy.forget_relay(RECIPIENT))
        embed = ctx.reference.resolved.embeds[0]
        ctx.reference.resolved = self.as_discord_has_it(embed, author_id=RECIPIENT_SNOWFLAKE)
        self.assertFalse(await self.route(ctx))
        ctx.channel.send.assert_not_awaited()

    async def test_anything_else_of_fritzs_is_the_agents(self):
        for embed in (None, relay_format.relay_embed("x", shown_as="y", icon_url=None,
                                                     footer="Something else entirely")):
            with self.subTest(embed=embed):
                ctx = self.reply_to(MagicMock(id=626262626262626262), "what did you mean?")
                ctx.reference.resolved = self.as_discord_has_it(embed)
                self.assertFalse(await self.route(ctx))
                ctx.channel.send.assert_not_awaited()

    async def test_a_deleted_original_is_the_agents(self):
        ctx = self.reply_to(MagicMock(id=636363636363636363), "hello?")
        ctx.reference.resolved = MagicMock(spec=discord.DeletedReferencedMessage)
        self.assertFalse(await self.route(ctx))

    def test_every_footer_is_built_from_its_marker(self):
        """relay_kind reads these back off DMs already sent, so they are a
        format. Each sender must build its footer from the constant."""
        for module, marker in ((bot_commands, "FOOTER_RELAYED"), (bot_commands, "FOOTER_RECEIPT"),
                               (relay_router, "FOOTER_REPLY")):
            src = pathlib.Path(module.__file__).read_text(encoding="utf-8")
            self.assertIn(f"relay_format.{marker}", src)


class TestARoutedReplyLeavesNoTraceInTheAgent(ReplyTestCase):
    """PR 4's Definition of Done. The pool is a spy, so ask_stuff never really
    runs and the checkpoint and extraction checks alone could not fail: what
    catches a reply that fell through is the spy's record, and the steps
    on_message takes on the way to the agent — the identity roster, the
    message counter and the thinking placeholder."""

    async def test_no_checkpoint_and_no_memory_extraction(self):
        dm = await self.relayed()
        before = (privacy.count_conversation_checkpoints(SENDER),
                  privacy.count_conversation_checkpoints(RECIPIENT))
        seen = []

        async def spy(fn, *a, **kw):
            # Stands in for the pool, so a reply that wrongly fell through
            # would be RECORDED here rather than reaching a real model.
            seen.append(getattr(fn, "__name__", repr(fn)))
            return {"text": "x", "image_paths": []}

        import agent_tools
        import mister_fritz
        # Both names: mister_fritz does `from agent_tools import
        # extract_memories_background`, so patching agent_tools alone misses
        # the reference that would actually be called.
        with patch.object(agent_tools, "extract_memories_background") as extract_a, \
             patch.object(mister_fritz, "extract_memories_background") as extract_m, \
             patch.object(main_discord, "run_blocking", AsyncMock(side_effect=spy)), \
             patch.object(main_discord, "client", self.client), \
             patch.object(main_discord.identity_store, "record") as roster, \
             patch.object(main_discord.METRICS, "increment") as counted:
            ctx = self.reply_to(dm, "meet me at the old mill")
            await main_discord.on_message(ctx)
        self.assertEqual(seen, [], "on_message ran the agent's pool for a routed reply")
        roster.assert_not_called()
        self.assertNotIn("discord_messages", [c.args[0] for c in counted.call_args_list])
        sent = " ".join(str(c.args[0]) for c in ctx.channel.send.await_args_list if c.args)
        self.assertNotIn("thinking", sent)
        extract_a.assert_not_called()
        extract_m.assert_not_called()
        self.assertEqual(before, (privacy.count_conversation_checkpoints(SENDER),
                                  privacy.count_conversation_checkpoints(RECIPIENT)))
        self.assertEqual(self.carried().description, "meet me at the old mill")


if __name__ == "__main__":
    unittest.main()
