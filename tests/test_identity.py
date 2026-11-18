"""Tests for the single identity transformation in fritz_utils.

Everything downstream — the Chroma namespace, the LangGraph thread, the
schedules and workspaces tables, the admin gate — consumes canonical_user_id()
verbatim. These tests pin the shape of that output, because if it drifts the
write path and the delete path diverge again and /forget silently stops
deleting.
"""
import contextlib
import json
import logging
import os
import pathlib
import subprocess
import sys
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

    def test_a_desktop_client_is_a_platform_of_its_own(self):
        """Plan 14 P2. The split is what KNOWN_PLATFORMS decides: without the
        entry, "desktop-nick-pc" reads as a legacy display name, and
        identity_store would record its platform as None."""
        ident = canonical_user_id("desktop", "nick-pc")
        self.assertEqual(ident, "desktop-nick-pc")
        self.assertEqual(split_user_id(ident), ("desktop", "nick-pc"))
        self.assertTrue(is_canonical_user_id(ident))

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
        # what makes privacy.forget_conversation's prefix sweep safe.
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


class TestReadingIdentityLinks(unittest.TestCase):
    """IDENTITY_LINKS as text. What parses is unchanged from the parser this
    replaced; what does not parse is now named instead of dropped in silence."""

    @staticmethod
    def read(raw):
        return fritz_utils._read_identity_links(raw)

    def test_pairs_are_split_on_commas_and_stripped(self):
        """An empty pair, even in the middle, ends nothing: what follows it is read."""
        links, misread = self.read(" web-alice = discord-1 ,, ,desktop-nick-pc=discord-1,")
        self.assertEqual(links, {"web-alice": "discord-1", "desktop-nick-pc": "discord-1"})
        self.assertEqual(misread, [])

    def test_a_pair_is_split_at_its_first_equals_only(self):
        """So a wrong separator is not a second link: it is one link to a
        primary nobody has, which identity_link_problems names."""
        for sep in (";", "\n", " "):
            with self.subTest(sep=sep):
                self.assertEqual(self.read(f"web-a=discord-1{sep}web-b=discord-2"),
                                 ({"web-a": f"discord-1{sep}web-b=discord-2"}, []))

    def test_nothing_set_is_no_links_and_nothing_to_say(self):
        for raw in ("", " ", ",", " , ,"):
            with self.subTest(raw=raw):
                self.assertEqual(self.read(raw), ({}, []))

    def test_a_pair_that_is_not_alias_equals_primary_links_nothing_and_is_named(self):
        for bad in ("web-alice", "web-alice=", "=discord-1", "=", "web-alice discord-1"):
            with self.subTest(bad=bad):
                links, misread = self.read(f"web-b=discord-2, {bad} ")
                self.assertEqual(links, {"web-b": "discord-2"})
                self.assertEqual(misread, [f"{bad!r} is not alias=primary, so it links nothing"])

    def test_an_alias_given_twice_keeps_the_last_and_is_named(self):
        links, misread = self.read("web-a=discord-1,web-b=discord-9,web-a=discord-2")
        self.assertEqual(links, {"web-a": "discord-2", "web-b": "discord-9"})
        self.assertEqual(misread, ["web-a is linked twice, to discord-1 and then to discord-2: "
                                   "the last is the one used"])

    def test_another_settings_line_run_on_is_read_whole_commas_and_all(self):
        """A password may hold a comma: what follows it is the password, not links."""
        for raw, links in (
                ("CHAT_PASSWORD=hunter2,CHAT_PASSWORD=hunter3", {"CHAT_PASSWORD": "hunter2,CHAT_PASSWORD=hunter3"}),
                ("web-a=discord-1 CHAT_PASSWORD=hunter2,tailpart9",
                 {"web-a": "discord-1 CHAT_PASSWORD=hunter2,tailpart9"}),
                ("web-a=discord-1 CHAT_PASSWORD = hunter2,tailpart9",
                 {"web-a": "discord-1 CHAT_PASSWORD = hunter2,tailpart9"}),
                # It ends at the first piece that is a link of its own, which a password never is.
                ("web-a=discord-1 CHAT_PASSWORD=hunter2,tail,web-b=discord-2",
                 {"web-a": "discord-1 CHAT_PASSWORD=hunter2,tail", "web-b": "discord-2"}),
                ("web-a=discord-1 CHAT_PASSWORD=hunter2,tail, web-b = discord-2 ",
                 {"web-a": "discord-1 CHAT_PASSWORD=hunter2,tail", "web-b": "discord-2"}),
                # A platform on the left or an identity on the right ends it too: a link
                # that parses is never taken for a password. Neither is password.
                ("CHAT_PASSWORD=hunter2,tail,web-x=alice", {"CHAT_PASSWORD": "hunter2,tail", "web-x": "alice"}),
                ("CHAT_PASSWORD=hunter2,tail,nick=discord-1", {"CHAT_PASSWORD": "hunter2,tail", "nick": "discord-1"}),
                ("CHAT_PASSWORD=hunter2,tail,divora=nick", {"CHAT_PASSWORD": "hunter2,tail,divora=nick"}),
                ("web-b=discord-2,desktop-nick-pc discord-1 CHAT_PASSWORD=hunter2,x=y",
                 {"web-b": "discord-2", "desktop-nick-pc discord-1 CHAT_PASSWORD": "hunter2,x=y"})):
            with self.subTest(raw=raw):
                self.assertEqual(self.read(raw), (links, []))

    def test_a_name_that_is_no_setting_reads_on(self):
        """NICK_PC=discord-1 links a name to an identity: the links after it are read."""
        for raw, links, misread in (
                ("NICK_PC=discord-1,web-b=discord-2", {"NICK_PC": "discord-1", "web-b": "discord-2"}, []),
                ("NICK_PC=discord-nick,web-alice=discord-1", {"NICK_PC": "discord-nick", "web-alice": "discord-1"}, []),
                ("web-a=discord-1;desktop-NICK_PC=discord-1,web-b=discord-3",
                 {"web-a": "discord-1;desktop-NICK_PC=discord-1", "web-b": "discord-3"}, []),
                ("desktop-nick-pc=web-NICK_PC,web-b=discord-2", {"desktop-nick-pc": "web-NICK_PC", "web-b": "discord-2"}, []),
                # Followed by a link that parses but is no link of its own, so that only
                # a false run-on, not where it would stop, could take it.
                ("desktop-nick-pc=web-NICK_PC,divora=nick", {"desktop-nick-pc": "web-NICK_PC", "divora": "nick"}, []),
                ("desktop-NICK_PC=discord-nick,divora=nick", {"desktop-NICK_PC": "discord-nick", "divora": "nick"}, []),
                ("web-a=discord-1,NICK_PC,divora=nick", {"web-a": "discord-1", "divora": "nick"},
                 ["'NICK_PC' is not alias=primary, so it links nothing"]),
                # A false run-on ends at the next piece with a platform on the left.
                ("NICK_PC=nick,web-alice=alice", {"NICK_PC": "nick", "web-alice": "alice"}, []),
                ("NICK_PC=discord-nick,Desktop-nick-pc=discord-1", {"NICK_PC": "discord-nick", "Desktop-nick-pc": "discord-1"}, []),
                ("desktop:NICK_PC=discord-1,web-b=discord-2", {"desktop:NICK_PC": "discord-1", "web-b": "discord-2"}, []),
                ("The DJ_2=discord-1,web-b=discord-2", {"The DJ_2": "discord-1", "web-b": "discord-2"}, []),
                ("web-a=discord-1,NICK_PC,web-b=discord-2", {"web-a": "discord-1", "web-b": "discord-2"},
                 ["'NICK_PC' is not alias=primary, so it links nothing"])):
            with self.subTest(raw=raw):
                self.assertEqual(self.read(raw), (links, misread))
        self.assertEqual(self.read("web-NICK_PC=discord-1,web-NICK_PC=discord-2")[1],
                         ["web-NICK_PC is linked twice, to discord-1 and then to discord-2: the last is the one used"])

    def test_a_pair_that_is_not_one_is_cut_short(self):
        for length, shown in ((200, repr("y" * 80 + "…")), (81, repr("y" * 80 + "…")), (80, repr("y" * 80))):
            with self.subTest(length=length):
                _links, misread = self.read("y" * length)
                self.assertEqual(misread, [shown + " is not alias=primary, so it links nothing"])

    def test_after_a_primary_not_shown_a_piece_with_no_equals_is_not_shown_either(self):
        """A lower-case setting's line run on: past a comma, its password goes on."""
        self.assertEqual(self.read("desktop-nick-pc=discord-1,chat_password=Zx9,Zx9"),
                         ({"desktop-nick-pc": "discord-1", "chat_password": "Zx9"},
                          ["a piece after chat_password=… is not alias=primary, so it links nothing"]))
        self.assertEqual(self.read("web-a=discord-1,oops")[1], ["'oops' is not alias=primary, so it links nothing"])

    def test_the_same_pair_twice_is_no_disagreement(self):
        self.assertEqual(self.read("web-a=discord-1,web-a=discord-1"), ({"web-a": "discord-1"}, []))

    def test_the_module_reads_its_own_environment_this_way(self):
        src = pathlib.Path(fritz_utils.__file__).read_text(encoding="utf-8")
        self.assertIn('IDENTITY_LINKS, _IDENTITY_LINKS_MISREAD = '
                      '_read_identity_links(os.environ.get("IDENTITY_LINKS", ""))', src)

    def test_the_test_run_starts_with_no_links(self):
        """conftest pins IDENTITY_LINKS empty: a developer's own link (their
        desk, linked to themselves) must not decide who a test's desk is."""
        self.assertEqual(os.environ.get("IDENTITY_LINKS"), "")
        conftest = pathlib.Path(__file__).with_name("conftest.py").read_text(encoding="utf-8")
        self.assertIn('os.environ["IDENTITY_LINKS"] = ""', conftest)


class TestSaidIdentity(unittest.TestCase):
    """How an identity, or what was given as one, goes into a log line."""

    said = staticmethod(fritz_utils.said_identity)

    def test_a_plain_one_is_said_as_it_is_up_to_80_characters(self):
        for value in ("discord-1", "web-Alice_B-c", "web-" + "a" * 76):
            with self.subTest(value=value):
                self.assertEqual(self.said(value), value)

    def test_anything_else_is_quoted_up_to_its_first_space_or_equals(self):
        """What follows may be another setting's line run on, a password say."""
        for value, said in (
                ("web-" + "a" * 77, repr("web-" + "a" * 36 + "…") + " (41 more characters not shown)"),
                ("discord-1 CHAT_PASSWORD=hunter2", "'discord-1…' (22 more characters not shown)"),
                ("discord-1\nWARNING forged", "'discord-1…' (15 more characters not shown)"),
                ("discord-1\tx", "'discord-1…' (2 more characters not shown)"),
                ("a=", "'a…' (1 more character not shown)"),
                ("=b", "'=…' (1 more character not shown)"),
                ("discord-1;web-b", "'discord-1;web-b'"),
                ("desktop-venn​", "'desktop-venn\\u200b'"),
                ("discord-1;" + "x" * 60, repr("discord-1;" + "x" * 30 + "…") + " (30 more characters not shown)")):
            with self.subTest(value=value):
                self.assertEqual(self.said(value), said)
                self.assertNotIn("\n", said)


class TestIdentityLinkProblems(unittest.TestCase):
    """A link is one dict lookup, character for character, so each of these
    used to start in silence and then quietly not work."""

    @staticmethod
    def problems(links, root="discord-1", admins=()):
        with patch.object(fritz_utils, "ROOT_USER", root), \
             patch.object(fritz_utils, "ADMIN_USERS", frozenset(admins)):
            return fritz_utils.identity_link_problems(links)

    def the_one(self, links, **who) -> str:
        found = self.problems(links, **who)
        self.assertEqual(len(found), 1, found)
        return found[0]

    def test_links_that_work_have_none(self):
        for links, who in (({}, {}),
                           ({"desktop-nick-pc": "discord-1"}, {}),
                           ({"web-alice": "discord-1", "telegram-99": "discord-1"}, {}),
                           ({"desktop-a": "web-owner"}, {}),
                           ({"desktop-" + "a" * 32: "discord-1"}, {}),
                           ({"web-Alice_B-c": "discord-123456789012345678"}, {}),
                           ({"web-" + "a" * 64: "discord-1"}, {}),
                           ({"desktop-b": "desktop-a"}, {}),
                           ({"web-NICK_PC": "discord-1"}, {}),
                           # A made-up primary two links share, or an admin's, is a hub.
                           ({"desktop-0": "local-me", "web-me": "local-me"}, {}),
                           ({"desktop-0": "local-me"}, {"root": "local-me"}),
                           ({"desktop-0": "local-me"}, {"admins": {"local-me"}}),
                           # The desk is the admin: the person joins it, rights and all.
                           ({"web-nick": "desktop-nick-pc"}, {"root": "desktop-nick-pc"}),
                           ({"web-nick": "desktop-nick-pc"}, {"admins": {"desktop-nick-pc"}})):
            with self.subTest(links=links, who=who):
                self.assertEqual(self.problems(links, **who), [])

    def test_with_no_links_given_it_reads_the_modules_own(self):
        self.assertEqual(self.problems(None), [])
        with patch.object(fritz_utils, "IDENTITY_LINKS", {"web-a": "web-a"}):
            self.assertEqual(len(self.problems(None)), 1)
        with patch.dict(fritz_utils.IDENTITY_LINKS, {"web-b": "web-b"}):
            self.assertEqual(len(self.problems(None)), 1)

    def test_an_alias_nothing_connects_as_is_never_followed(self):
        for alias, why in (
                ("Desktop-nick-pc", "is written desktop-nick-pc: the platform is lower case"),
                ("DISCORD-123", "is written discord-123: the platform is lower case"),
                ("desktop-Nick-PC", "is not a desk's identity"),
                ("desktop-nick_pc", "is not a desk's identity"),
                ("desktop--nick", "is not a desk's identity"),
                ("desktop-" + "a" * 33, "is not a desk's identity"),
                ("desktop-" + "a" * 70, "is not a desk's identity"),
                ("Desktop-Nick-PC", "is not a desk's identity"),
                ("discord-nick", "is not a Discord identity"),
                ("Discord-nick", "is not a Discord identity"),
                ("telegram-nick", "is not a Telegram identity"),
                ("TELEGRAM-nick", "is not a Telegram identity"),
                ("telegram-@nick", "has '@' in it"),
                ("web-a b", "has ' ' in it"),
                ("web-a b c d", "has ' ' in it"),
                ("web-" + "a" * 65, "is longer than an identity can be"),
                ("local-nick", "is local-<id>, a platform no surface uses (the web chat is web-<name>)"),
                ("Local-nick", "is local-<id>, a platform no surface uses"),
                ("local-a b", "has ' ' in it")):
            with self.subTest(alias=alias):
                said = self.the_one({alias: "discord-1"})
                self.assertTrue(said.startswith(
                    f"{fritz_utils.said_identity(alias)}=discord-1 is never followed: "
                    f"{fritz_utils.said_identity(alias)} "), said)
                self.assertIn(why, said)

    def test_one_with_no_platform_is_followed_only_for_what_was_stored_by_name(self):
        """A typo as a rule, but a store not yet migrated is keyed by names,
        and a schedule restored from one does follow the link."""
        for alias in ("divora", "jean-luc", "nick-pc", "desktop:nick-pc", "desktop-", "desktop", "nick_smith",
                      "NICK_PC"):
            with self.subTest(alias=alias):
                s = fritz_utils.said_identity(alias)
                self.assertEqual(
                    self.the_one({alias: "discord-1"}),
                    f"{s}=discord-1 is followed for nobody who talks to Fritz now: {s} is not "
                    "<platform>-<id> (the platforms: desktop, discord, telegram, web); only what was "
                    "stored by name before migrate_identity.py is keyed like that")

    def test_an_admin_still_written_by_name_is_taken_out_of_nothing(self):
        """No one is moved, and validate_config's own warning says how to mend ROOT_USER."""
        for name, who in (("divora", {"root": "divora"}), ("divora", {"admins": {"divora"}}),
                          ("BIG_BOSS", {"root": "BIG_BOSS"})):
            with self.subTest(who=who):
                said = self.the_one({name: "discord-111"}, **who)
                self.assertIn("is followed for nobody who talks to Fritz now", said)

    def test_a_primary_nothing_connects_as_is_named_as_what_the_alias_becomes(self):
        for primary, why in (
                ("discord-1;web-b=discord-2", "has ';=' in it"),
                ("discord-12x", "is not a Discord identity"),
                ("Discord-1", "is written discord-1: the platform is lower case"),
                ("discord:1", "is not <platform>-<id>"),
                ("divora", "is not <platform>-<id>"),
                ("desktop-Nick", "is not a desk's identity"),
                ("local-nick", "is local-<id>, a platform no surface uses"),
                ("Local-nick", "is local-<id>, a platform no surface uses"),
                ("local-me;web-b=discord-2", "has ';=' in it")):
            with self.subTest(primary=primary):
                said = self.the_one({"desktop-nick-pc": primary})
                self.assertTrue(said.startswith(
                    f"desktop-nick-pc={fritz_utils.said_identity(primary)} links desktop-nick-pc to an "
                    f"identity no surface makes: {fritz_utils.said_identity(primary)} "), said)
                self.assertIn(why, said)

    def test_a_local_hub_is_the_same_hub_whatever_the_platforms_case(self):
        """local-Me and local-me are two identities; Local-me is local-me mistyped."""
        self.assertEqual(len(self.problems({"desktop-0": "local-Me", "web-me": "local-me"})), 2)
        for who in ({}, {"root": "local-me"}):
            with self.subTest(who=who):
                links = {"desktop-0": "Local-me"} if who else {"desktop-0": "Local-me", "web-me": "local-me"}
                self.assertIn("is written local-me: the platform is lower case", self.the_one(links, **who))

    def test_a_link_wrong_on_both_sides_says_both_each_of_its_own_side(self):
        self.assertEqual(self.problems({"Desktop-nick-pc": "Discord-1"}), [
            "Desktop-nick-pc=Discord-1 is never followed: Desktop-nick-pc is written desktop-nick-pc: the "
            "platform is lower case; if Desktop-nick-pc is the PC's whole name, its desk is "
            "desktop-desktop-nick-pc",
            "Desktop-nick-pc=Discord-1 links Desktop-nick-pc to an identity no surface makes: Discord-1 "
            "is written discord-1: the platform is lower case"])

    def test_a_pc_name_pasted_as_windows_shows_it_is_named_as_the_desk_it_would_be(self):
        said = self.the_one({"DESKTOP-7G2K9QF": "discord-1"})
        self.assertTrue(said.endswith(
            "(NICK_PC is nick-pc); if DESKTOP-7G2K9QF is the PC's whole name, its desk is "
            "desktop-desktop-7g2k9qf"), said)
        # As client_id_from in desktop/src-tauri/src/settings.rs makes them (its
        # table's rows of plain characters: no other can reach this hint).
        for name, desk in (("DESKTOP-" + "A" * 40, "desktop-desktop-" + "a" * 24),
                           ("DESKTOP-NICK--PC", "desktop-desktop-nick-pc"),
                           ("DESKTOP-" + "A" * 23 + "-B", "desktop-desktop-" + "a" * 23),
                           ("Desktop-NICK_PC", "desktop-desktop-nick-pc")):
            with self.subTest(name=name):
                self.assertTrue(self.the_one({name: "discord-1"}).endswith(f"its desk is {desk}"))
        said = self.the_one({"DESKTOP-" + "A" * 100: "discord-1"})
        self.assertNotIn("A" * 41, said)
        self.assertIn("(68 more characters not shown) is the PC's whole name", said)
        # Typed in lower case, "desktop-" is the platform, as it should be.
        self.assertEqual(self.the_one({"desktop-NICK_PC": "discord-1"}),
                         "desktop-NICK_PC=discord-1 is never followed: desktop-NICK_PC is not a desk's identity: "
                         "that is desktop-, then lower-case letters, digits and dashes (32 at most), as the app "
                         "makes it from the PC's name (NICK_PC is nick-pc)")
        self.assertNotIn("whole name", self.the_one({"web-a": "Discord-2"}))

    def test_what_is_not_plain_is_quoted(self):
        said = self.the_one({"web-alice": "discord-1\nWARNING forged"})
        self.assertNotIn("\n", said)
        self.assertNotIn("forged", said)
        self.assertIn("'discord-1…' (15 more characters not shown)", said)

    def test_what_is_not_an_identity_is_cut_short(self):
        """It may be anything: the rest of a .env line, with a secret on it."""
        said = self.the_one({"web-alice": "discord-1 " + "z" * 200})
        self.assertIn("'discord-1…' (201 more characters not shown)", said)
        self.assertIn("has ' ' in it", said)
        self.assertNotIn("zz", said)
        token = "Zx9" * 14 + "q"
        for line in (f"discord-123456789012345678 COMPANION_TOKEN={token}",
                     f"discord-123456789012345678COMPANION_TOKEN={token}"):
            with self.subTest(line=line):
                self.assertNotIn(token[:8], self.the_one({"web-alice": line}))
        # Plain characters, but more of them than any identity has.
        said = self.the_one({"web-" + "a" * 200: "discord-1"})
        self.assertTrue(said.startswith(repr("web-" + "a" * 36 + "…") + " (164 more characters not shown)"))
        # And of the characters no identity has, the first few, never past
        # the space or "=" that ends what is shown.
        said = self.the_one({"web-alice": "discord-1!@#$%^&*()+"})
        self.assertIn("has '!@#$%^&*' in it", said)
        said = self.the_one({"web-alice": "discord-1 CHAT_PASSWORD=P@ss#w0rd!"})
        self.assertIn("has ' ' in it", said)
        for gone in ("@", "#", "!", "P@ss"):
            self.assertNotIn(gone, said)
        said = self.the_one({"web-alice": "discord-1CHAT_PASSWORD=#P@ssw0rd!"})
        self.assertIn("has '=' in it", said)
        for gone in ("#", "@", "!"):
            self.assertNotIn(gone, said)

    def test_another_setting_run_into_the_line_is_named_and_its_value_never_shown(self):
        self.assertEqual(
            self.the_one({"CHAT_PASSWORD": "hunter2"}),
            "CHAT_PASSWORD=… looks like another setting, not a link: it is never followed, and its "
            "value is not shown. Each setting goes on a line of its own")
        # A name with no underscore could be a legacy display name: no such guess.
        self.assertNotIn("another setting", self.the_one({"DIVORA": "discord-1"}))
        self.assertFalse(any("another setting" in said for said in self.problems({"DIVORA": "divora"})))
        # Nor a name linked to an identity: a PC's name, or one from before migrate_identity.py.
        said = self.the_one({"NICK_PC": "discord-1"})
        self.assertNotIn("another setting", said)
        self.assertIn("is not <platform>-<id>", said)
        for alias in ("telegram-@MaxP_2000", "desktop:NICK_PC", "The DJ_2"):
            with self.subTest(alias=alias):
                self.assertFalse(any("another setting" in said for said in self.problems({alias: "discord-1"})))
        # Run in without a space after a ":" typed for "=".
        said = self.the_one({"desktop-nick-pc:discord-1CHAT_PASSWORD": "hunter2-hunter2"})
        self.assertIn("ends in CHAT_PASSWORD", said)
        self.assertNotIn("hunter2", said)
        # A setting's value is hidden even when it starts like an identity.
        self.assertNotIn("hunter2", self.the_one({"CHAT_PASSWORD": "discord-hunter2"}))
        # A desk id mistyped with an underscore is that, whatever its primary.
        self.assertFalse(any("another setting" in said for said in self.problems({"desktop-NICK_PC": "nick"})))
        # And a hidden primary stays hidden in a chain's words.
        found = self.problems({"nick": "hunter2", "hunter2": "discord-1"})
        self.assertTrue(any(" stops at …" in said for said in found), found)
        self.assertFalse(any("=hunter2" in said or " hunter2" in said for said in found[:3]), found)
        # A setting's name in lower case looks like nothing: its value is not shown either.
        found = self.problems({"chat_password": "Zx9Zx9Zx9Zx9"})
        self.assertEqual(len(found), 2, found)
        self.assertFalse(any("Zx9" in said for said in found), found)
        self.assertTrue(found[1].startswith("chat_password=… links chat_password to an identity no surface "
                                            "makes: … is not <platform>-<id>"), found[1])
        # A mistake with a platform is shown: there is nothing to hide in discord-nick.
        self.assertIn("discord-nick is not a Discord identity", self.problems({"nick": "discord-nick"})[1])
        # A pair typed without its "=", then the next line: the setting's own "=" ends the alias.
        self.assertEqual(
            self.the_one({"desktop-nick-pc discord-1 CHAT_PASSWORD": "hunter2-hunter2"}),
            "'desktop-nick-pc…' (24 more characters not shown)=… ends in CHAT_PASSWORD, another setting whose "
            "line ran into this one after a pair with no \"=\" of its own: it is never followed, and its value "
            "is not shown. Each setting goes on a line of its own, and each link is alias=primary")

    def test_the_platforms_are_listed_for_one_that_is_not(self):
        self.assertIn("(the platforms: desktop, discord, telegram, web)", self.the_one({"nick-pc": "discord-1"}))

    def test_the_platforms_listed_are_the_ones_a_surface_makes(self):
        """Every canonical_user_id call in the app names one of them, and "local" none."""
        self.assertLess(fritz_utils._SURFACE_PLATFORMS, fritz_utils.KNOWN_PLATFORMS)
        self.assertNotIn("local", fritz_utils._SURFACE_PLATFORMS)
        import ast
        named = set()
        for path in pathlib.Path(fritz_utils.__file__).parent.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None))
                        == "canonical_user_id"):
                    first = node.args[0]
                    self.assertIsInstance(first, ast.Constant, f"{path.name}:{node.lineno}")
                    named.add(first.value)
        self.assertEqual(named, set(fritz_utils._SURFACE_PLATFORMS))

    def test_a_link_to_itself_does_nothing_and_says_only_that(self):
        self.assertEqual(self.the_one({"Web-a": "Web-a"}),
                         "Web-a=Web-a links an identity to itself, which does nothing")

    def test_a_chain_is_named_where_it_stops_and_where_it_ends(self):
        """What takes the second hop depends on the surface: the web chat
        resolves twice, and so does a reminder booked in conversation."""
        for alias, became in (
                ("web-alice", "web-alice ends up as two identities (discord-1 for all its web chat does but the "
                              "conversation itself, telegram-2 for that)"),
                ("discord-5", "discord-5 ends up as two identities (telegram-2 in conversation, discord-1 for a "
                              "reminder booked in conversation, when it fires)"),
                ("desktop-a", "desktop-a is telegram-2, not discord-1"),
                ("telegram-9", "telegram-9 is telegram-2, not discord-1")):
            with self.subTest(alias=alias):
                self.assertEqual(
                    self.the_one({alias: "telegram-2", "telegram-2": "discord-1"}),
                    f"{alias}=telegram-2 stops at telegram-2, though that is linked on to discord-1: links are "
                    f"not chained, so {became}. Link {alias} to discord-1 itself")
        found = self.problems({"web-a": "web-b", "web-b": "web-c", "web-c": "discord-1"})
        self.assertEqual(len(found), 2, found)
        self.assertTrue(found[0].endswith("(web-c for all its web chat does but the conversation itself, web-b "
                                          "for that). Link web-a to discord-1 itself"), found[0])
        self.assertTrue(found[1].endswith("Link web-b to discord-1 itself"), found[1])
        # Two aliases of one primary are no chain, and nor is a primary that
        # only links to itself, near or at the chain's end.
        self.assertEqual(self.problems({"web-a": "discord-1", "web-b": "discord-1"}), [])
        self.assertEqual(len(self.problems({"web-a": "discord-1", "discord-1": "discord-1"})), 1)
        found = self.problems({"web-a": "web-b", "web-b": "discord-1", "discord-1": "discord-1"})
        self.assertEqual(len(found), 2, found)
        self.assertTrue(found[0].endswith("Link web-a to discord-1 itself"), found[0])

    def test_a_link_never_followed_is_no_chain(self):
        """It moves nobody, so there is nothing for it to end up as."""
        found = self.problems({"Desktop-nick-pc": "discord-1", "discord-1": "desktop-nick-pc"})
        self.assertEqual(len(found), 3, found)
        self.assertFalse(any(" itself" in said or " stops at " in said for said in found), found)
        # Written both ways the same wrong way: the way back moves nobody either.
        for links, who in (({"Desktop-nick-pc": "discord-1", "discord-1": "Desktop-nick-pc"}, {}),
                           ({"local-me": "discord-2", "discord-2": "local-me"}, {"root": "local-me"})):
            with self.subTest(links=links):
                self.assertFalse(any("each other" in said for said in self.problems(links, **who)))

    def test_two_linked_to_each_other_are_told_to_keep_one(self):
        self.assertEqual(self.problems({"web-a": "discord-2", "discord-2": "web-a"}), [
            "web-a=discord-2 and discord-2=web-a link the two to each other, which swaps them in most places "
            "rather than making them one: keep one of the two"])

    def test_a_ring_has_no_end_to_link_to(self):
        def ring(a, p, o):
            return (f"{a}={p} stops at {p}, though that is linked on to {o}, and from there the links go round "
                    f"in a ring, so there is no end to link {a} to: take out the link that closes the ring")
        self.assertEqual(self.problems({"web-a": "web-b", "web-b": "web-c", "web-c": "web-a"}),
                         [ring("web-a", "web-b", "web-c"), ring("web-b", "web-c", "web-a"),
                          ring("web-c", "web-a", "web-b")])
        self.assertEqual(self.problems({"web-x": "web-a", "web-a": "discord-2", "discord-2": "web-a"}),
                         [ring("web-x", "web-a", "discord-2"),
                          "web-a=discord-2 and discord-2=web-a link the two to each other, which swaps them in "
                          "most places rather than making them one: keep one of the two"])

    def test_a_person_linked_to_a_desk_is_the_wrong_way_round(self):
        self.assertEqual(self.the_one({"discord-2": "desktop-nick-pc"}),
                         "discord-2=desktop-nick-pc makes discord-2 the desk, not the desk discord-2: "
                         "the desk goes on the left, desktop-nick-pc=discord-2")
        self.assertEqual(self.problems({"desktop-b": "desktop-a"}), [])

    def test_the_desk_on_the_left_is_never_advice_that_is_itself_warned_about(self):
        """Whoever is an admin, the pair it says to write starts clean."""
        for who in ({"root": "discord-9"}, {"root": "desktop-nick-pc"}, {"root": "web-nick"},
                    {"root": "web-nick", "admins": {"desktop-nick-pc"}}):
            with self.subTest(who=who):
                for said in self.problems({"web-nick": "desktop-nick-pc"}, **who):
                    if "the desk goes on the left, " in said:
                        alias, primary = said.rsplit("the desk goes on the left, ", 1)[1].split("=")
                        self.assertEqual(self.problems({alias: primary}, **who), [], said)
        # A link that moves nobody gives no advice: written so, it would make the desk nobody.
        self.assertFalse(any("the desk goes on the left" in said
                             for said in self.problems({"local-me": "desktop-nick-pc"})))
        # Both admins: the desk goes on the left, and nothing is lost by it.
        self.assertIn("the desk goes on the left, desktop-nick-pc=web-nick",
                      self.the_one({"web-nick": "desktop-nick-pc"}, root="web-nick", admins={"desktop-nick-pc"}))

    def test_an_admin_linked_to_somebody_who_is_not_one_is_named(self):
        self.assertEqual(self.the_one({"discord-1": "web-owner"}),
                         "discord-1=web-owner takes discord-1, ROOT_USER, out of its own identity: in "
                         "conversation it is web-owner, which is not an admin, with web-owner's memories")
        self.assertIn("takes discord-5, one of ADMIN_USERS, out of",
                      self.the_one({"discord-5": "web-owner"}, admins={"discord-5"}))
        # Admin to admin loses nothing, and nor does anyone linked TO one.
        self.assertEqual(self.problems({"discord-1": "discord-5"}, admins={"discord-5"}), [])
        self.assertEqual(self.problems({"discord-5": "discord-1"}, admins={"discord-5"}), [])
        self.assertEqual(self.problems({"web-owner": "discord-1"}), [])
        # With no ROOT_USER there is nobody to take out.
        self.assertEqual(self.problems({"discord-1": "web-owner"}, root=None), [])

    def test_the_owner_written_backwards_gets_both(self):
        found = self.problems({"discord-1": "desktop-nick-pc"})
        self.assertEqual(len(found), 2, found)
        self.assertIn("the desk goes on the left, desktop-nick-pc=discord-1", found[0])
        self.assertIn("takes discord-1, ROOT_USER, out of its own identity", found[1])

    def test_a_desks_id_is_the_protocols(self):
        """fritz_utils restates the pattern rather than import the protocol."""
        import companion_protocol
        self.assertEqual(fritz_utils._DESK_ID_RE.pattern,
                         companion_protocol.ClientId.__metadata__[0].pattern)
        self.assertLessEqual(set(fritz_utils._NUMBERED_PLATFORMS), fritz_utils.KNOWN_PLATFORMS)


class TestStartupSaysWhatTheLinksAre(unittest.TestCase):
    """validate_config: one INFO line for what was read, a WARNING for each
    thing wrong with it, and never a refusal."""

    def said(self, links, misread=(), root="discord-1", admins=()) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []

        class _Collect(logging.Handler):
            def emit(self, record):
                if record.getMessage().startswith("IDENTITY_LINKS"):
                    lines.append((record.levelname, record.getMessage()))

        log = logging.getLogger("fritz_utils")
        handler, level = _Collect(logging.INFO), log.level
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            with contextlib.ExitStack() as stack:
                for name, value in (("DISCORD_BOT_TOKEN", "x"), ("ROOT_USER", root),
                                    ("ADMIN_USERS", frozenset(admins)), ("IDENTITY_LINKS", links),
                                    ("_IDENTITY_LINKS_MISREAD", list(misread))):
                    stack.enter_context(patch.object(fritz_utils, name, value))
                fritz_utils.validate_config()
        finally:
            log.removeHandler(handler)
            log.setLevel(level)
        return lines

    def test_no_links_say_nothing(self):
        self.assertEqual(self.said({}), [])

    def test_only_a_pair_typed_wrong_is_still_said_and_nothing_is_listed(self):
        oops = "'desktop-nick-pc discord-1' is not alias=primary, so it links nothing"
        self.assertEqual(self.said({}, misread=[oops]), [("WARNING", f"IDENTITY_LINKS: {oops}.")])

    def test_links_that_work_are_listed_once_and_warn_about_nothing(self):
        self.assertEqual(
            self.said({"desktop-nick-pc": "discord-1", "web-a": "discord-5", "web-b": "discord-7"},
                      admins={"discord-5"}),
            [("INFO", "IDENTITY_LINKS: read as desktop-nick-pc=discord-1 (ROOT_USER); web-a=discord-5 "
                      "(one of ADMIN_USERS); web-b=discord-7.")])

    def test_a_hidden_primary_that_is_an_admin_is_still_marked(self):
        """A legacy alias linked to a legacy ROOT_USER: the mark is the one line
        that says the link hands out an admin's rights (validate_config's
        legacy warning already names ROOT_USER)."""
        self.assertEqual(self.said({"divora": "nicksanft"}, root="nicksanft")[-1],
                         ("INFO", "IDENTITY_LINKS: read as divora=… (ROOT_USER)."))

    def test_another_setting_is_listed_without_its_value_or_a_mark(self):
        """A mark would say what the value is: ROOT_USER's."""
        self.assertEqual(self.said({"CHAT_PASSWORD": "Sup3r_Secret"}, root="Sup3r_Secret")[-1],
                         ("INFO", "IDENTITY_LINKS: read as CHAT_PASSWORD=…."))

    def test_each_thing_wrong_is_a_warning_of_its_own_and_the_links_are_still_listed(self):
        said = self.said({"Desktop-nick-pc": "discord-1", "discord-1": "desktop-nick-pc"},
                         misread=["'oops' is not alias=primary, so it links nothing"])
        self.assertEqual([level for level, _ in said], ["WARNING"] * 4 + ["INFO"])
        for (_, line), words in zip(said, ("'oops' is not alias=primary",
                                           "Desktop-nick-pc=discord-1 is never followed",
                                           "the desk goes on the left",
                                           "takes discord-1, ROOT_USER, out of its own identity",
                                           "IDENTITY_LINKS: read as Desktop-nick-pc=discord-1 (ROOT_USER); "
                                           "discord-1=desktop-nick-pc.")):
            self.assertTrue(line.startswith("IDENTITY_LINKS: ") and line.endswith("."), line)
            self.assertIn(words, line)

    def test_no_line_carries_operator_text_raw(self):
        """Through the reader and validate_config, every alias and primary
        given as something that is not an identity."""
        h = "\nWARNING forged " + "s" * 100      # ends a line, and runs on past any cut
        token = "Zx9" * 14 + "q"
        raw = ",".join((f"a{h}=b{h}", f"a{h}=c{h}",          # linked twice
                        f"c{h}=d{h}",                        # a chain: a, c, d
                        f"e{h}=desktop-f{h}",                # the wrong way round
                        f"discord-1=g{h}",                   # an admin taken out
                        f"m{h}=n{h}", f"n{h}=m{h}",          # linked to each other
                        f"p{h}=q{h}", f"q{h}=r{h}", f"r{h}=p{h}",    # a ring
                        f"web-y=s{h}", f"s{h}=t{h}",         # a chain from the web chat
                        f"discord-7=u{h}", f"u{h}=v{h}",     # and one from Discord
                        f"web-x=discord-1 CHAT_PASSWORD={token},more,COMPANION_TOKEN={token}"))
        links, misread = fritz_utils._read_identity_links(raw)
        self.assertEqual(len(links), 14)
        said = self.said(links, misread=misread)
        for words in ("is linked twice", " stops at ", "the desk goes on the left", "out of its own identity",
                      "link the two to each other", "go round in a ring", " read as "):
            self.assertTrue(any(words in line for _, line in said), words)
        self.assertEqual(said[-1][0], "INFO")
        for _, line in said:
            self.assertNotIn("\n", line)
            self.assertNotIn("forged", line)
            self.assertNotIn(token[:6], line)

    def test_it_is_part_of_validate_config_whichever_brain_starts(self):
        with patch.object(fritz_utils, "DISCORD_BOT_TOKEN", None), \
             patch.object(fritz_utils, "ROOT_USER", "discord-1"), \
             patch.object(fritz_utils, "IDENTITY_LINKS", {"web-a": "web-a"}), \
             self.assertLogs("fritz_utils", level="WARNING") as logs:
            fritz_utils.validate_config(require_discord=False)
        self.assertIn("IDENTITY_LINKS: web-a=web-a links an identity to itself", "; ".join(logs.output))

    PROBE = ("import json, logging\n"
             "import fritz_utils as f\n"
             "said = []\n"
             "class H(logging.Handler):\n"
             "    def emit(self, r):\n"
             "        if r.getMessage().startswith('IDENTITY_LINKS'):\n"
             "            said.append([r.levelname, r.getMessage()])\n"
             "log = logging.getLogger('fritz_utils')\n"
             "log.addHandler(H(logging.INFO)); log.setLevel(logging.INFO)\n"
             "f.validate_config(require_discord=False)\n"
             "print(json.dumps({'links': f.IDENTITY_LINKS, 'said': said}))")

    def start(self, links: str) -> dict:
        """A brain's worth of fritz_utils, started on a real IDENTITY_LINKS."""
        repo = pathlib.Path(__file__).resolve().parents[1]
        env = {**os.environ, "ROOT_USER": "discord-1", "ADMIN_USERS": "", "IDENTITY_LINKS": links}
        proc = subprocess.run([sys.executable, "-c", self.PROBE], cwd=repo, env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_a_fresh_interpreter_reads_them_from_its_environment(self):
        """Every other test patches the parsed values; this one reads them, mistakes and all."""
        out = self.start(" Desktop-Nick-PC = discord-1 , oops ,web-a=discord-nick,desktop-nick-pc=discord-1")
        self.assertEqual(out["links"], {"Desktop-Nick-PC": "discord-1", "web-a": "discord-nick",
                                        "desktop-nick-pc": "discord-1"})
        levels = [level for level, _ in out["said"]]
        self.assertEqual(levels, ["WARNING", "WARNING", "WARNING", "INFO"], out["said"])
        for (_, line), words in zip(out["said"], ("'oops' is not alias=primary",
                                                  "Desktop-Nick-PC=discord-1 is never followed",
                                                  "links web-a to an identity no surface makes",
                                                  "read as Desktop-Nick-PC=discord-1 (ROOT_USER); "
                                                  "web-a=discord-nick; desktop-nick-pc=discord-1 (ROOT_USER).")):
            self.assertIn(words, line)

    def test_a_fresh_interpreter_with_only_a_pair_typed_wrong(self):
        out = self.start("desktop-nick-pc discord-1")
        self.assertEqual(out["links"], {})
        self.assertEqual(out["said"], [["WARNING", "IDENTITY_LINKS: 'desktop-nick-pc discord-1' is not "
                                                   "alias=primary, so it links nothing."]])

    def test_a_fresh_interpreter_with_another_setting_run_into_the_line(self):
        """A lost newline: no part of the next line's password is said, whatever came before it."""
        for links, secret in (("web-alice=discord-1 CHAT_PASSWORD=hunter2-hunter2-hunter2", "hunter2"),
                              ("web-alice=discord-1CHAT_PASSWORD=hunter2-hunter2-hunter2", "hunter2"),
                              ("desktop-nick-pc discord-1 CHAT_PASSWORD=hunter2-hunter2-hunter2", "hunter2"),
                              ("desktop-nick-pc:discord-1 ADMIN_PANEL_PASSWORD=Sup3rSecretPanelPass", "Sup3r"),
                              ("web-alice=discord-1 CHAT_PASSWORD=P@ss#w0rd,tailpart9", "tailpart9"),
                              ("web-alice=discord-1,CHAT_PASSWORD=P@ss#w0rd,tail=part9", "part9"),
                              ("web-alice=discord-1 CHAT_PASSWORD = P@ss#w0rd,tailpart9", "tailpart9"),
                              ("web-alice=discord-1 CHAT_PASSWORD = P@ss#w0rd,x=tailpart9", "tailpart9"),
                              ("desktop-nick-pc:discord-1CHAT_PASSWORD=hunter2-hunter2-hunter2", "hunter2"),
                              ("desktop-nick-pc=discord-1,chat_password=Zx9Zx9Zx9Zx9Zx9", "Zx9"),
                              ("desktop-nick-pc discord-1 chat_password=Zx9Zx9Zx9Zx9Zx9", "Zx9"),
                              ("desktop-nick-pc=discord-1,chat_password=Zx9Zx9,Zx9Zx9", "Zx9")):
            with self.subTest(links=links):
                said = json.dumps(self.start(links)["said"])
                for gone in (secret, "@", "#", "w0rd"):
                    self.assertNotIn(gone, said)


if __name__ == "__main__":
    unittest.main()
