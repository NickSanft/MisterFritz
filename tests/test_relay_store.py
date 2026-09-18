"""
Tests for relay_store.py — the schema and admission gate for /tell.

The gate is the only thing between this feature and a DM-spam primitive, so
these lean hard on the cases that are invisible in normal use: two conditions
true at once, an identity arriving under an alias, and two senders racing.

Uses a temp DB, following tests/test_workspace_store.py.
"""
import importlib
import os
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


class RelayStoreTestCase(unittest.TestCase):
    """Temp DB, permissive caps. Subclasses tighten whichever cap they test."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db_path = self.tmp / "test_fritz.db"
        self._env_patcher = patch.dict(os.environ, {
            "SCHEDULE_DB": str(self.db_path),
            "DB_NAME": str(self.db_path),
        })
        self._env_patcher.start()

        import fritz_utils
        importlib.reload(fritz_utils)
        import relay_store
        importlib.reload(relay_store)
        self.fu = fritz_utils
        self.store = relay_store

        self._caps = patch.multiple(
            fritz_utils,
            RELAY_ENABLED=True,
            RELAY_MAX_BODY_CHARS=1000,
            RELAY_MAX_PER_SENDER_PER_HOUR=10,
            RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR=10,
            RELAY_REPLY_WINDOW_MIN=1440,
        )
        self._caps.start()

    def tearDown(self):
        self._caps.stop()
        self._env_patcher.stop()

    # -- helpers ---------------------------------------------------------
    def send(self, sender="discord-1", recipient="discord-2", body="hi", **kw):
        return self.store.reserve_send(sender, recipient, body, **kw)

    def deliver(self, sender="discord-1", recipient="discord-2", body="hi",
                dm_message_id=None):
        """A full successful round trip, so it counts against the caps."""
        res = self.send(sender, recipient, body)
        self.assertTrue(getattr(res, "ok", False), f"expected a Reservation, got {res}")
        self.store.mark_sent(res.id, dm_message_id or int(datetime.now().timestamp() * 1e6) % 10**15)
        return res

    def backdate(self, relay_id, **delta):
        when = datetime.now(timezone.utc) - timedelta(**delta)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE relay_messages SET created_at = ? WHERE id = ?",
                         (when.isoformat(), relay_id))
            conn.commit()


class TestSchema(RelayStoreTestCase):
    def test_tables_and_indexes_exist(self):
        self.store._init_db()
        with sqlite3.connect(self.db_path) as conn:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','index')")}
        for expected in ("relay_messages", "relay_optouts", "idx_relay_dm_message",
                         "idx_relay_sender", "idx_relay_recipient",
                         "idx_relay_open", "idx_relay_optouts_blocked"):
            self.assertIn(expected, names)

    def test_init_is_idempotent(self):
        self.store._init_db()
        self.store._INITIALISED = False  # force the CREATE IF NOT EXISTS path
        self.store._init_db()

    def test_many_rows_may_have_a_null_dm_message_id(self):
        """The unique index is partial; in-flight rows all sit at NULL."""
        a = self.send(body="one")
        b = self.send(body="two")
        self.assertIsNone(self.store.get(a.id)["dm_message_id"])
        self.assertIsNone(self.store.get(b.id)["dm_message_id"])

    def test_two_rows_cannot_share_a_dm_message_id(self):
        """A duplicate would route a reply to the wrong person."""
        a = self.deliver(body="one", dm_message_id=999)
        b = self.send(body="two")
        with self.assertRaises(self.store.RelayStoreError):
            self.store.mark_sent(b.id, 999)
        # and the loser is untouched, not half-written
        self.assertEqual(self.store.get(b.id)["status"], "reserved")
        self.assertEqual(self.store.get(a.id)["dm_message_id"], 999)


class TestSentinel(RelayStoreTestCase):
    def test_block_everyone_sentinel_cannot_collide_with_a_real_id(self):
        """'*' is safe as a sentinel only because no real id can ever be it."""
        with self.assertRaises(ValueError):
            self.fu.canonical_user_id("discord", self.store.BLOCK_EVERYONE)


class TestCheckOrder(RelayStoreTestCase):
    """The whole order, pinned by peeling one condition off at a time.

    Every condition below is true at the start. Each assertion says which one
    wins, so swapping any adjacent pair turns this red. The blocks are LAST:
    see TestBlockStateIsNotProbeable for why that is the property, not taste.
    """

    def test_order_is_exactly_as_specified(self):
        sender, recipient = "discord-1", "discord-2"
        for _ in range(10):                                # both caps full
            self.deliver(sender, recipient, "x")
        self.store.block(recipient, sender)                # blocked sender
        self.store.block(recipient)                        # blocked everyone
        long_body = "x" * 5000                             # over the cap

        def reason(body=long_body, **kw):
            return self.send(sender, recipient, body, **kw).reason

        # Facts anyone can see about the recipient come first, and Fritz
        # outranks the generic bot copy.
        self.assertEqual(reason(recipient_is_bot=True, recipient_is_fritz=True),
                         "recipient_is_fritz")
        self.assertEqual(reason(recipient_is_bot=True), "recipient_is_bot")
        self.assertEqual(self.send(sender, sender, long_body).reason,
                         "recipient_is_sender")

        # Then everything about the sender and the traffic...
        self.assertEqual(reason(), "body_too_long")
        self.assertEqual(reason("short"), "sender_rate_limited")
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 999):
            self.assertEqual(reason("short"), "recipient_rate_limited")
            with patch.object(self.fu, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 999):
                # ...and the recipient's choices last of all.
                self.assertEqual(reason("short"), "blocked_sender")
                self.store.unblock(recipient, sender)
                self.assertEqual(reason("short"), "blocked_everyone")
                self.store.unblock(recipient)
                self.assertTrue(self.send(sender, recipient, "short").ok)

    def test_disabled_relay_refuses_before_anything_else(self):
        with patch.object(self.fu, "RELAY_ENABLED", False):
            self.assertEqual(self.send().reason, "disabled")


class TestBlockStateIsNotProbeable(RelayStoreTestCase):
    """REFUSED must mean exactly what Discord's closed-DM 403 means.

    A 403 can only be discovered by sending, so it only ever arrives after
    every other check has passed. A block that is checked any earlier than
    that becomes a detector: the plan's original order returned REFUSED for
    an oversized message to someone who had blocked you and "too long" to
    someone who had not, which answers "has X blocked me" for free.
    """

    BLOCK_STATES = (None, "sender", "everyone")

    def _block(self, state, recipient, sender):
        if state == "sender":
            self.store.block(recipient, sender)
        elif state == "everyone":
            self.store.block(recipient)

    def test_an_oversized_message_reads_the_same_whether_or_not_you_are_blocked(self):
        seen = {}
        for n, state in enumerate(self.BLOCK_STATES):
            sender, recipient = f"discord-1{n}", f"discord-2{n}"
            self._block(state, recipient, sender)
            seen[state] = self.send(sender, recipient, "x" * 5000).reason
        self.assertEqual(len(set(seen.values())), 1, seen)

    def test_being_over_your_own_cap_reads_the_same_whether_or_not_you_are_blocked(self):
        seen = {}
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            for n, state in enumerate(self.BLOCK_STATES):
                sender, recipient = f"discord-1{n}", f"discord-2{n}"
                self.deliver(sender, "discord-999")           # spend the cap elsewhere
                self._block(state, recipient, sender)
                seen[state] = self.send(sender, recipient).reason
        self.assertEqual(len(set(seen.values())), 1, seen)

    def test_a_full_inbox_reads_the_same_whether_or_not_you_are_blocked(self):
        seen = {}
        with patch.object(self.fu, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 1):
            for n, state in enumerate(self.BLOCK_STATES):
                sender, recipient = f"discord-1{n}", f"discord-2{n}"
                self.deliver("discord-888", recipient)        # someone else fills it
                self._block(state, recipient, sender)
                seen[state] = self.send(sender, recipient).reason
        self.assertEqual(len(set(seen.values())), 1, seen)

    def _refused_row(self, sender, recipient):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM relay_messages WHERE sender_id = ? AND recipient_id = ?",
                (sender, recipient)).fetchone()
        return dict(row)

    def test_a_block_and_a_closed_dm_leave_identical_rows(self):
        """PR 6's /export shows senders their own rows."""
        self.store.block("discord-2", "discord-1")
        denial = self.send("discord-1", "discord-2")
        self.assertEqual(denial.reason, "blocked_sender")
        self.store.mark_refused(denial.relay_id)                 # the caller's half
        blocked = self._refused_row("discord-1", "discord-2")

        res = self.send("discord-3", "discord-4")               # Discord says 403
        self.store.mark_refused(res.id)
        closed = self._refused_row("discord-3", "discord-4")

        varying = {"id", "sender_id", "recipient_id", "created_at", "expires_at", "closed_at"}
        self.assertEqual({k: v for k, v in blocked.items() if k not in varying},
                         {k: v for k, v in closed.items() if k not in varying})
        for row in (blocked, closed):
            self.assertEqual(row["closed_at"], row["created_at"])
            self.assertEqual(row["status"], "refused")

    def test_a_block_and_a_closed_dm_cost_the_sender_the_same(self):
        """Otherwise the rate limit itself tells you which one blocked you."""
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 2):
            self.store.block("discord-2", "discord-1")
            for _ in range(2):
                self.send("discord-1", "discord-2")
            by_block = self.send("discord-1", "discord-2").reason

            for _ in range(2):
                self.store.mark_refused(self.send("discord-3", "discord-4").id)
            by_closed_dm = self.send("discord-3", "discord-4").reason

        self.assertEqual(by_block, "sender_rate_limited")
        self.assertEqual(by_closed_dm, by_block)

    def test_settled_refusals_never_fill_the_recipients_inbox(self):
        """Or a blocked harasser could lock out everyone else trying to reach them.

        Only once settled: until then a refusal is an in-flight reservation,
        on purpose — see the next test."""
        with patch.object(self.fu, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 1):
            self.store.block("discord-2", "discord-1")
            for _ in range(5):
                self.store.mark_refused(self.send("discord-1", "discord-2").relay_id)
            self.store.mark_refused(self.send("discord-3", "discord-2").id)
            self.assertTrue(self.send("discord-5", "discord-2").ok)

    def test_a_block_and_a_closed_dm_look_the_same_to_a_second_account_mid_flight(self):
        """The race the review found. A send heading for Discord's 403 sits
        `reserved` for its round trip and counts toward the recipient's inbox;
        a block used to go straight to `refused` and never did. Account B,
        sending while account A's attempt was in flight, saw "their inbox is
        full" only when A was NOT blocked."""
        seen = {}
        with patch.object(self.fu, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 1):
            for n, blocked in enumerate((False, True)):
                a, b, alice = f"discord-1{n}", f"discord-3{n}", f"discord-2{n}"
                if blocked:
                    self.store.block(alice, a)
                first = self.send(a, alice)                     # in flight, unsettled
                self.assertIsNotNone(first.id if first.ok else first.relay_id)
                seen[blocked] = getattr(self.send(b, alice), "reason", "allowed")  # B, mid-flight
        self.assertEqual(seen[False], seen[True])
        self.assertEqual(seen[False], "recipient_rate_limited")

    def test_only_a_recipient_refusal_leaves_a_reservation_open(self):
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            self.deliver("discord-1", "discord-9")
            self.assertIsNone(self.send("discord-1", "discord-2").relay_id)   # rate
        self.assertIsNone(self.send(body="x" * 5000).relay_id)               # size
        self.store.block("discord-2", "discord-1")
        self.assertIsNotNone(self.send().relay_id)                            # block


class TestRefusalsAreIndistinguishable(RelayStoreTestCase):
    def test_blanket_and_per_sender_blocks_read_identically(self):
        """Otherwise /tell is a reliable detector for "has X blocked me"."""
        self.store.block("discord-2", "discord-1")
        per_sender = self.send().message
        self.store.unblock("discord-2", "discord-1")
        self.store.block("discord-2")
        blanket = self.send().message

        # Asserting the two are equal, not that each matches a literal: the
        # equality IS the property, and a later copy edit to one branch is
        # exactly how it would be lost.
        self.assertEqual(per_sender, blanket)
        self.assertEqual(per_sender, self.store.REFUSED)

    def test_refusal_names_neither_the_block_nor_the_blocker(self):
        self.store.block("discord-2", "discord-1")
        message = self.send().message.lower()
        for leak in ("block", "discord-1", "discord-2", "opt"):
            self.assertNotIn(leak, message)


class TestIdentityResolution(RelayStoreTestCase):
    """A block must bind to the person, not to whichever alias they used."""

    def links(self, **mapping):
        return patch.object(self.fu, "IDENTITY_LINKS", dict(mapping))

    def test_block_on_canonical_id_binds_a_sender_arriving_as_an_alias(self):
        self.store.block("discord-2", "discord-1")
        with self.links(**{"web-alice": "discord-1"}):
            denial = self.send(sender="web-alice", recipient="discord-2")
        self.assertEqual(denial.reason, "blocked_sender")

    def test_block_binds_a_recipient_arriving_as_an_alias(self):
        """Resolving only the sender leaves this half a live bypass.

        The block is stored against the canonical id and the RECIPIENT is
        named by alias, so the lookup itself has to resolve. Aliasing at write
        time instead would prove nothing: the stored row would already be
        canonical and the read side would never be asked to do anything.
        """
        self.store.block("discord-2", "discord-1")
        with self.links(**{"web-bob": "discord-2"}):
            denial = self.send(sender="discord-1", recipient="web-bob")
        self.assertEqual(denial.reason, "blocked_sender")

    def test_a_recipients_inbound_cap_follows_them_across_aliases(self):
        """Otherwise the cap resets by addressing the same person differently."""
        with self.links(**{"web-bob": "discord-2"}), \
             patch.object(self.fu, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 1):
            self.deliver("discord-1", "discord-2")
            self.assertEqual(self.send("discord-3", "web-bob").reason,
                             "recipient_rate_limited")

    def test_alias_and_canonical_share_one_rate_limit_budget(self):
        with self.links(**{"web-alice": "discord-1"}), \
             patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            self.deliver("discord-1", "discord-2")
            self.assertEqual(self.send("web-alice", "discord-2").reason,
                             "sender_rate_limited")

    def test_stored_rows_record_the_resolved_identity(self):
        with self.links(**{"web-alice": "discord-1"}):
            res = self.send(sender="web-alice")
        self.assertEqual(res.sender_id, "discord-1")
        self.assertEqual(self.store.get(res.id)["sender_id"], "discord-1")


class TestQuotaAccounting(RelayStoreTestCase):
    def test_sender_cap_counts_delivered_messages(self):
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 2):
            self.deliver()
            self.deliver()
            self.assertEqual(self.send().reason, "sender_rate_limited")

    def test_recipient_cap_constrains_many_senders(self):
        """The brigade case: each sender is under their own cap."""
        with patch.object(self.fu, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 3):
            for n in range(3):
                self.deliver(sender=f"discord-1{n}", recipient="discord-9")
            denial = self.send(sender="discord-199", recipient="discord-9")
        self.assertEqual(denial.reason, "recipient_rate_limited")

    def test_failed_sends_do_not_consume_quota(self):
        """Discord being down is not the sender's fault."""
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            res = self.send()
            self.store.mark_failed(res.id, "Forbidden")
            self.assertTrue(self.send().ok)

    def test_a_crashed_reservation_stops_holding_quota(self):
        """Otherwise one crash costs a slot until the hour rolls over."""
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            res = self.send()                       # reserved, never marked
            self.assertEqual(self.send().reason, "sender_rate_limited")
            self.backdate(res.id, seconds=self.store.RESERVATION_GRACE_SEC + 5)
            self.assertTrue(self.send().ok)

    def test_a_fresh_reservation_does_hold_quota(self):
        """The in-flight window is what stops a double-send racing itself."""
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            self.send()
            self.assertEqual(self.send().reason, "sender_rate_limited")

    def test_messages_older_than_an_hour_fall_out_of_the_window(self):
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            res = self.deliver()
            self.backdate(res.id, hours=1, minutes=1)
            self.assertTrue(self.send().ok)

    def test_content_discord_rejects_is_charged_to_the_sender_only(self):
        """A body Discord reliably blocks (its harmful-link filter) must not be a
        free, unlimited supply of DM-channel opens."""
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 1), \
             patch.object(self.fu, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 1):
            self.store.mark_rejected(self.send("discord-1", "discord-2").id, "HTTPException 400")
            self.assertEqual(self.send("discord-1", "discord-3").reason, "sender_rate_limited")
            self.assertTrue(self.send("discord-4", "discord-2").ok)   # recipient untouched

    def test_a_full_inbox_does_not_date_someone_elses_message(self):
        """"Try again after 15:05" says a third party relayed to them at 14:05."""
        with patch.object(self.fu, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 1):
            self.deliver("discord-8", "discord-2")
            denial = self.send("discord-1", "discord-2")
        self.assertEqual(denial.reason, "recipient_rate_limited")
        self.assertIsNone(denial.retry_at)
        self.assertNotIn("UTC", denial.message)

    def test_rate_denial_says_when_to_come_back(self):
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            self.deliver()
            denial = self.send()
        self.assertIsNotNone(denial.retry_at)
        self.assertIn("UTC", denial.message)


class TestConcurrency(RelayStoreTestCase):
    def test_racing_senders_cannot_both_take_the_last_slot(self):
        """The reason the counts and the INSERT share one BEGIN IMMEDIATE.

        Counting in one transaction and inserting in another makes every cap
        advisory: both threads read "0 sent" and both write the first and
        second.
        """
        results, ready = [], threading.Barrier(2)
        lock = threading.Lock()

        def attempt():
            ready.wait()
            outcome = self.store.reserve_send("discord-1", "discord-2", "hi")
            with lock:
                results.append(outcome)

        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            threads = [threading.Thread(target=attempt) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        self.assertEqual(len(results), 2)
        allowed = [r for r in results if r.ok]
        self.assertEqual(len(allowed), 1, f"both got through: {results}")
        self.assertEqual(results[1 - results.index(allowed[0])].reason,
                         "sender_rate_limited")

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM relay_messages").fetchone()[0]
        self.assertEqual(rows, 1)


class TestMarking(RelayStoreTestCase):
    def test_mark_sent_records_the_routing_key(self):
        res = self.send()
        self.store.mark_sent(res.id, 555, dm_channel_id=777)
        row = self.store.get(res.id)
        self.assertEqual(row["status"], "delivered")
        self.assertEqual(row["dm_message_id"], 555)
        self.assertEqual(row["dm_channel_id"], 777)
        self.assertIsNotNone(row["delivered_at"])

    def test_mark_failed_records_the_reason(self):
        res = self.send()
        self.store.mark_failed(res.id, "Forbidden 50007")
        row = self.store.get(res.id)
        self.assertEqual(row["status"], "failed")
        self.assertIn("50007", row["error"])
        self.assertIsNotNone(row["closed_at"])

    def test_marking_an_unknown_id_raises_rather_than_passing_silently(self):
        """A swallowed write means telling the sender "delivered" with no row."""
        with self.assertRaises(self.store.RelayStoreError):
            self.store.mark_sent("deadbeef", 1)
        with self.assertRaises(self.store.RelayStoreError):
            self.store.mark_failed("deadbeef", "nope")

    def test_reservation_carries_the_reply_window(self):
        with patch.object(self.fu, "RELAY_REPLY_WINDOW_MIN", 60):
            res = self.send()
        delta = (datetime.fromisoformat(res.expires_at)
                 - datetime.fromisoformat(res.created_at))
        self.assertEqual(delta, timedelta(minutes=60))


class TestBlocks(RelayStoreTestCase):
    def test_block_is_idempotent(self):
        self.assertTrue(self.store.block("discord-2", "discord-1"))
        self.assertFalse(self.store.block("discord-2", "discord-1"))
        self.assertEqual(self.store.list_blocks("discord-2"), ["discord-1"])

    def test_unblock_reports_whether_it_removed_anything(self):
        self.store.block("discord-2", "discord-1")
        self.assertTrue(self.store.unblock("discord-2", "discord-1"))
        self.assertFalse(self.store.unblock("discord-2", "discord-1"))

    def test_blanket_and_per_sender_blocks_coexist(self):
        self.store.block("discord-2")
        self.store.block("discord-2", "discord-1")
        self.assertEqual(sorted(self.store.list_blocks("discord-2")),
                         ["*", "discord-1"])
        # Dropping the blanket block leaves the specific one standing.
        self.store.unblock("discord-2")
        self.assertEqual(self.store.list_blocks("discord-2"), ["discord-1"])
        self.assertEqual(self.send().reason, "blocked_sender")

    def test_blocks_are_per_recipient(self):
        self.store.block("discord-2", "discord-1")
        self.assertTrue(self.send(recipient="discord-3").ok)

    def test_cannot_block_yourself(self):
        with self.assertRaises(ValueError):
            self.store.block("discord-1", "discord-1")

    def test_list_blocks_is_empty_for_an_unknown_user(self):
        self.assertEqual(self.store.list_blocks("discord-404"), [])


class TestNoDiscordImport(unittest.TestCase):
    def test_the_gate_does_not_depend_on_discord(self):
        """It is a pure function of the DB and config, and stays testable."""
        source = Path(__file__).resolve().parents[1] / "relay_store.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("import discord", text)


if __name__ == "__main__":
    unittest.main()
