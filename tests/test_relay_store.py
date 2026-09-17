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
    wins, so swapping any adjacent pair turns this red.
    """

    def test_order_is_exactly_as_specified(self):
        sender, recipient = "discord-1", "discord-2"
        for _ in range(10):                                # both caps full
            self.deliver(sender, recipient, "x")
        self.store.block(recipient, sender)                # blocked sender
        self.store.block(recipient)                        # blocked everyone
        long_body = "x" * 5000                             # over the cap

        def reason(**kw):
            return self.send(sender, recipient, long_body, **kw).reason

        # A bot recipient outranks everything, and Fritz outranks the bot copy.
        self.assertEqual(reason(recipient_is_bot=True, recipient_is_fritz=True),
                         "recipient_is_fritz")
        self.assertEqual(reason(recipient_is_bot=True), "recipient_is_bot")
        self.assertEqual(self.send(sender, sender, long_body).reason,
                         "recipient_is_sender")

        self.assertEqual(reason(), "blocked_sender")
        self.store.unblock(recipient, sender)
        self.assertEqual(reason(), "blocked_everyone")
        self.store.unblock(recipient)
        self.assertEqual(reason(), "body_too_long")

        # Body now legal; the sender's own cap is next.
        self.assertEqual(self.send(sender, recipient, "short").reason,
                         "sender_rate_limited")
        with patch.object(self.fu, "RELAY_MAX_PER_SENDER_PER_HOUR", 999):
            self.assertEqual(self.send(sender, recipient, "short").reason,
                             "recipient_rate_limited")
            with patch.object(self.fu, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 999):
                self.assertTrue(self.send(sender, recipient, "short").ok)

    def test_disabled_relay_refuses_before_anything_else(self):
        with patch.object(self.fu, "RELAY_ENABLED", False):
            self.assertEqual(self.send().reason, "disabled")


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
