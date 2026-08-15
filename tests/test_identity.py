"""Tests for the single identity transformation in fritz_utils.

Everything downstream — the Chroma namespace, the LangGraph thread, the
schedules and workspaces tables, the admin gate — consumes canonical_user_id()
verbatim. These tests pin the shape of that output, because if it drifts the
write path and the delete path diverge again and /forget silently stops
deleting.
"""
import unittest
from unittest.mock import patch

import fritz_utils
from fritz_utils import (
    canonical_user_id,
    is_canonical_user_id,
    resolve_identity,
    safe_user_token,
    split_user_id,
    thread_id_for,
)


class TestCanonicalUserId(unittest.TestCase):
    def test_discord_snowflake(self):
        self.assertEqual(canonical_user_id("discord", 123456789), "discord-123456789")

    def test_platform_is_lowercased(self):
        self.assertEqual(canonical_user_id("DISCORD", 1), "discord-1")

    def test_web_name_is_stripped_of_punctuation(self):
        # `.` and `!` go; `_` and `-` survive, because they are legal in every
        # store this id reaches, including a Windows path.
        self.assertEqual(canonical_user_id("web", "Alice.Smith!"), "web-AliceSmith")
        self.assertEqual(canonical_user_id("web", "alice_smith-42"), "web-alice_smith-42")

    def test_case_of_the_id_half_is_preserved(self):
        # Discord names are case-sensitive and two users can differ only by
        # case; folding here would merge their memories.
        self.assertNotEqual(canonical_user_id("web", "Alice"),
                            canonical_user_id("web", "alice"))

    def test_empty_and_all_punctuation_raise(self):
        with self.assertRaises(ValueError):
            canonical_user_id("discord", "")
        with self.assertRaises(ValueError):
            canonical_user_id("discord", "!!!")
        with self.assertRaises(ValueError):
            canonical_user_id("discord", None)

    def test_long_ids_are_truncated(self):
        ident = canonical_user_id("web", "a" * 200)
        _platform, bare = split_user_id(ident)
        self.assertEqual(len(bare), 64)

    def test_no_colon_ever_appears(self):
        # DECISIONS #5. A colon is illegal in Windows filenames and silently
        # creates an NTFS alternate data stream on write.
        for raw in ("123", "alice", "a:b:c"):
            self.assertNotIn(":", canonical_user_id("discord", raw))


class TestSplitUserId(unittest.TestCase):
    def test_round_trips_a_canonical_id(self):
        self.assertEqual(split_user_id("discord-123"), ("discord", "123"))

    def test_legacy_bare_name_reports_no_platform(self):
        self.assertEqual(split_user_id("divora"), (None, "divora"))

    def test_unknown_prefix_is_not_split(self):
        # The allowlist is what keeps a legacy name containing a dash intact
        # instead of being misread as platform "jean".
        self.assertEqual(split_user_id("jean-luc"), (None, "jean-luc"))

    def test_dash_in_the_id_half_survives(self):
        # DECISIONS #5's worked example: partition takes the FIRST dash and no
        # platform name contains one.
        self.assertEqual(split_user_id("web-alice-bob"), ("web", "alice-bob"))

    def test_empty_input(self):
        self.assertEqual(split_user_id(""), (None, ""))
        self.assertEqual(split_user_id(None), (None, ""))

    def test_is_canonical(self):
        self.assertTrue(is_canonical_user_id("telegram-99"))
        self.assertFalse(is_canonical_user_id("divora"))
        self.assertFalse(is_canonical_user_id("jean-luc"))
        self.assertFalse(is_canonical_user_id(""))


class TestSafeUserToken(unittest.TestCase):
    def test_canonical_ids_pass_through(self):
        self.assertEqual(safe_user_token("discord-123"), "discord-123")

    def test_punctuation_becomes_underscores(self):
        self.assertEqual(safe_user_token("al.i/ce"), "al_i_ce")

    def test_empty_is_anonymous(self):
        self.assertEqual(safe_user_token(""), "anonymous")
        self.assertEqual(safe_user_token(None), "anonymous")

    def test_output_is_a_legal_windows_filename_component(self):
        for bad in ':*?"<>|/\\':
            self.assertNotIn(bad, safe_user_token(f"a{bad}b"))


class TestThreadIdFor(unittest.TestCase):
    def test_flag_off_ignores_the_channel(self):
        with patch.object(fritz_utils, "THREADS_PER_CHANNEL", False):
            self.assertEqual(thread_id_for("discord-1", "999"), "discord-1")

    def test_flag_on_scopes_by_channel(self):
        with patch.object(fritz_utils, "THREADS_PER_CHANNEL", True):
            self.assertEqual(thread_id_for("discord-1", "999"), "discord-1#999")

    def test_no_channel_key_always_yields_the_bare_identity(self):
        for flag in (True, False):
            with patch.object(fritz_utils, "THREADS_PER_CHANNEL", flag):
                self.assertEqual(thread_id_for("discord-1", None), "discord-1")

    def test_channel_suffix_cannot_be_confused_with_a_sibling(self):
        # `discord-1#999` must not be a prefix-match for `discord-10`, which is
        # what makes privacy.forget_conversation's LIKE sweep safe.
        with patch.object(fritz_utils, "THREADS_PER_CHANNEL", True):
            self.assertFalse(thread_id_for("discord-10", "1").startswith(
                thread_id_for("discord-1", "999")))


class TestResolveIdentity(unittest.TestCase):
    def test_link_is_followed(self):
        with patch.object(fritz_utils, "IDENTITY_LINKS", {"web-alice": "discord-1"}):
            self.assertEqual(resolve_identity("web-alice"), "discord-1")

    def test_unlinked_identity_is_returned_unchanged(self):
        with patch.object(fritz_utils, "IDENTITY_LINKS", {"web-alice": "discord-1"}):
            self.assertEqual(resolve_identity("discord-1"), "discord-1")

    def test_does_not_chain_a_second_hop(self):
        # One hop only. A chain would make a cycle in the config an infinite
        # loop on every single message.
        links = {"web-alice": "telegram-2", "telegram-2": "discord-1"}
        with patch.object(fritz_utils, "IDENTITY_LINKS", links):
            self.assertEqual(resolve_identity("web-alice"), "telegram-2")

    def test_a_self_referential_link_terminates(self):
        with patch.object(fritz_utils, "IDENTITY_LINKS", {"web-a": "web-a"}):
            self.assertEqual(resolve_identity("web-a"), "web-a")

    def test_empty_input(self):
        self.assertEqual(resolve_identity(""), "")


if __name__ == "__main__":
    unittest.main()
