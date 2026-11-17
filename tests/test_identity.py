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
        links, misread = self.read(" web-alice = discord-1 ,desktop-nick-pc=discord-1,")
        self.assertEqual(links, {"web-alice": "discord-1", "desktop-nick-pc": "discord-1"})
        self.assertEqual(misread, [])

    def test_a_pair_is_split_at_its_first_equals_only(self):
        """So a wrong separator is not a second link: it is one link to a
        primary nobody has, which identity_link_problems names."""
        links, misread = self.read("web-a=discord-1;web-b=discord-2")
        self.assertEqual(links, {"web-a": "discord-1;web-b=discord-2"})
        self.assertEqual(misread, [])

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

    def test_a_pair_that_is_not_one_is_cut_short(self):
        _links, misread = self.read("y" * 200)
        self.assertEqual(misread, [repr("y" * 80 + "…") + " is not alias=primary, so it links nothing"])
        _links, misread = self.read("y" * 80)
        self.assertEqual(misread, [repr("y" * 80) + " is not alias=primary, so it links nothing"])

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
        for links in ({},
                      {"desktop-nick-pc": "discord-1"},
                      {"web-alice": "discord-1", "telegram-99": "discord-1"},
                      {"desktop-a": "web-owner"},
                      {"desktop-0": "local-me"},
                      {"desktop-" + "a" * 32: "discord-1"},
                      {"web-Alice_B-c": "discord-123456789012345678"},
                      {"web-" + "a" * 64: "discord-1"},
                      {"desktop-b": "desktop-a"}):
            with self.subTest(links=links):
                self.assertEqual(self.problems(links), [])

    def test_with_no_links_given_it_reads_the_modules_own(self):
        self.assertEqual(self.problems(None), [])
        with patch.object(fritz_utils, "IDENTITY_LINKS", {"web-a": "web-a"}):
            self.assertEqual(len(self.problems(None)), 1)
        with patch.dict(fritz_utils.IDENTITY_LINKS, {"web-b": "web-b"}):
            self.assertEqual(len(self.problems(None)), 1)

    def test_an_alias_nobody_has_is_never_followed(self):
        for alias, why in (
                ("Desktop-nick-pc", "is written desktop-nick-pc: the platform is lower case"),
                ("DISCORD-123", "is written discord-123: the platform is lower case"),
                ("desktop-Nick-PC", "is not a desk's identity"),
                ("desktop-nick_pc", "is not a desk's identity"),
                ("desktop--nick", "is not a desk's identity"),
                ("desktop-" + "a" * 33, "is not a desk's identity"),
                ("Desktop-Nick-PC", "is not a desk's identity"),
                ("desktop:nick-pc", "is not <platform>-<id>"),
                ("nick-pc", "is not <platform>-<id>"),
                ("desktop-", "is not <platform>-<id>"),
                ("desktop", "is not <platform>-<id>"),
                ("discord-nick", "is not a Discord identity"),
                ("telegram-nick", "is not a Telegram identity"),
                ("telegram-@nick", "has '@' in it"),
                ("web-a b", "has ' ' in it"),
                ("web-" + "a" * 65, "is longer than an identity can be")):
            with self.subTest(alias=alias):
                said = self.the_one({alias: "discord-1"})
                self.assertIn("is never followed: ", said)
                self.assertIn(why, said)
                self.assertTrue(said.startswith(f"{fritz_utils._said(alias)}=discord-1 "), said)

    def test_a_primary_nobody_has_is_named_as_what_the_alias_becomes(self):
        for primary, why in (
                ("discord-1;web-b=discord-2", "has ';=' in it"),
                ("discord-12x", "is not a Discord identity"),
                ("Discord-1", "is written discord-1: the platform is lower case"),
                ("discord:1", "is not <platform>-<id>"),
                ("divora", "is not <platform>-<id>"),
                ("desktop-Nick", "is not a desk's identity")):
            with self.subTest(primary=primary):
                said = self.the_one({"desktop-nick-pc": primary})
                self.assertIn("makes desktop-nick-pc somebody nobody else is: ", said)
                self.assertIn(why, said)

    def test_what_is_not_plain_is_quoted(self):
        said = self.the_one({"web-alice": "discord-1\nWARNING forged"})
        self.assertNotIn("\n", said)
        self.assertIn(r"'discord-1\nWARNING forged'", said)

    def test_what_is_not_an_identity_is_cut_short(self):
        """It may be anything: the rest of a .env line, with a secret on it."""
        said = self.the_one({"web-alice": "discord-1 " + "z" * 200})
        self.assertIn(repr("discord-1 " + "z" * 70 + "…"), said)
        self.assertNotIn("z" * 71, said)
        # Plain characters, but more of them than any identity has.
        said = self.the_one({"web-" + "a" * 200: "discord-1"})
        self.assertIn(repr("web-" + "a" * 76 + "…"), said)
        self.assertNotIn("a" * 77, said)
        # And of the characters no identity has, the first few.
        said = self.the_one({"web-alice": "discord-1!@#$%^&*()+"})
        self.assertIn("has '!@#$%^&*' in it", said)
        self.assertNotIn("(", said.split(" in it")[0].split("has ")[1])

    def test_the_platforms_are_listed_for_one_that_is_not(self):
        self.assertIn("(the platforms: desktop, discord, local, telegram, web)",
                      self.the_one({"nick-pc": "discord-1"}))

    def test_a_link_to_itself_does_nothing_and_says_only_that(self):
        self.assertEqual(self.the_one({"Web-a": "Web-a"}),
                         "Web-a=Web-a links an identity to itself, which does nothing")

    def test_a_chain_is_named_where_it_stops(self):
        said = self.the_one({"web-alice": "telegram-2", "telegram-2": "discord-1"})
        self.assertEqual(said, "web-alice=telegram-2 stops at telegram-2, though that is linked on to "
                               "discord-1: links are not chained (only the web chat follows a second, "
                               "and only for memories). Link web-alice to discord-1 itself")
        # Two aliases of one primary are no chain, and nor is a primary that
        # only links to itself.
        self.assertEqual(self.problems({"web-a": "discord-1", "web-b": "discord-1"}), [])
        self.assertEqual(len(self.problems({"web-a": "discord-1", "discord-1": "discord-1"})), 1)

    def test_a_person_linked_to_a_desk_is_the_wrong_way_round(self):
        self.assertEqual(self.the_one({"discord-2": "desktop-nick-pc"}),
                         "discord-2=desktop-nick-pc makes discord-2 the desk, not the desk discord-2: "
                         "the desk goes on the left, desktop-nick-pc=discord-2")
        self.assertEqual(self.problems({"desktop-b": "desktop-a"}), [])

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

    def test_links_that_work_are_listed_once_and_warn_about_nothing(self):
        self.assertEqual(
            self.said({"desktop-nick-pc": "discord-1", "web-a": "discord-5", "web-b": "discord-7"},
                      admins={"discord-5"}),
            [("INFO", "IDENTITY_LINKS: desktop-nick-pc is treated as discord-1 (ROOT_USER); web-a is "
                      "treated as discord-5 (one of ADMIN_USERS); web-b is treated as discord-7.")])

    def test_each_thing_wrong_is_a_warning_of_its_own_and_the_links_are_still_listed(self):
        said = self.said({"Desktop-nick-pc": "discord-1", "discord-1": "desktop-nick-pc"},
                         misread=["'oops' is not alias=primary, so it links nothing"])
        self.assertEqual([level for level, _ in said], ["WARNING"] * 5 + ["INFO"])
        for (_, line), words in zip(said, ("'oops' is not alias=primary",
                                           "Desktop-nick-pc=discord-1 is never followed",
                                           "Desktop-nick-pc=discord-1 stops at discord-1",
                                           "the desk goes on the left",
                                           "takes discord-1, ROOT_USER, out of its own identity",
                                           "Desktop-nick-pc is treated as discord-1 (ROOT_USER); "
                                           "discord-1 is treated as desktop-nick-pc.")):
            self.assertTrue(line.startswith("IDENTITY_LINKS: ") and line.endswith("."), line)
            self.assertIn(words, line)

    def test_it_is_part_of_validate_config_whichever_brain_starts(self):
        with patch.object(fritz_utils, "DISCORD_BOT_TOKEN", None), \
             patch.object(fritz_utils, "ROOT_USER", "discord-1"), \
             patch.object(fritz_utils, "IDENTITY_LINKS", {"web-a": "web-a"}), \
             self.assertLogs("fritz_utils", level="WARNING") as logs:
            fritz_utils.validate_config(require_discord=False)
        self.assertIn("IDENTITY_LINKS: web-a=web-a links an identity to itself", "; ".join(logs.output))

    def test_a_fresh_interpreter_reads_them_from_its_environment(self):
        """Every other test patches the parsed values; this one starts a brain's
        worth of fritz_utils on a real IDENTITY_LINKS, mistakes and all."""
        repo = pathlib.Path(__file__).resolve().parents[1]
        probe = ("import json, logging\n"
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
        env = {**os.environ, "ROOT_USER": "discord-1", "ADMIN_USERS": "",
               "IDENTITY_LINKS": " Desktop-Nick-PC = discord-1 , oops ,web-a=discord-nick,"
                                 "desktop-nick-pc=discord-1"}
        proc = subprocess.run([sys.executable, "-c", probe], cwd=repo, env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(out["links"], {"Desktop-Nick-PC": "discord-1", "web-a": "discord-nick",
                                        "desktop-nick-pc": "discord-1"})
        levels = [level for level, _ in out["said"]]
        self.assertEqual(levels, ["WARNING", "WARNING", "WARNING", "INFO"], out["said"])
        for (_, line), words in zip(out["said"], ("'oops' is not alias=primary",
                                                  "Desktop-Nick-PC=discord-1 is never followed",
                                                  "makes web-a somebody nobody else is",
                                                  "desktop-nick-pc is treated as discord-1 (ROOT_USER).")):
            self.assertIn(words, line)


if __name__ == "__main__":
    unittest.main()
