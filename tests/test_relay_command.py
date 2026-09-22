"""
Tests for /relay and "Block this sender" — the recipient's side of /tell
(PR 3 of plans/12-direct-message-relay.md), and the ship blocker for it.

Built on the /tell harness so the property PR 3 exists for can be pinned end
to end through the real commands: however a recipient says no, the sender
reads the same thing. The other thread running through these: a person is
only ever shown what they typed or what they were shown — never a name
looked up afresh, never an id resolved through IDENTITY_LINKS.
"""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import discord
from discord.ext import commands

import bot_commands
import fritz_utils
import identity_store
import relay_store
from test_bot_commands import _fake_interaction, _make_cog
from test_tell_command import (
    RECIPIENT,
    RECIPIENT_SNOWFLAKE,
    SENDER,
    TellTestCase,
    _http_error,
    _member,
)

STRANGER_SNOWFLAKE = 123456789012345678
STRANGER = f"discord-{STRANGER_SNOWFLAKE}"
OTHER_SNOWFLAKE = 222222222222222222


class RelayTestCase(TellTestCase):
    """The recipient (555…) acting on relays from the sender (111…)."""

    def recipient(self, user_id=RECIPIENT_SNOWFLAKE):
        interaction = _fake_interaction("bob", user_id=user_id)
        # Post-defer state, as the real InteractionResponse reports it.
        interaction.response.is_done = MagicMock(return_value=True)
        return interaction

    async def run_cmd(self, name, interaction=None, **kw):
        interaction = interaction or self.recipient()
        await getattr(self.cog, name).callback(self.cog, interaction, **kw)
        return interaction

    def answer(self, interaction) -> str:
        """Deferred ephemerally first, then exactly one ephemeral followup."""
        interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
        interaction.response.send_message.assert_not_awaited()
        interaction.followup.send.assert_awaited_once()
        call = interaction.followup.send.await_args
        self.assertIs(call.kwargs.get("ephemeral"), True, "a /relay reply was not ephemeral")
        return call.args[0] if call.args else call.kwargs["content"]

    def blocks(self, user=RECIPIENT):
        return relay_store.list_blocks(user)

    async def delivered_message(self, sender_interaction=None, recipient=None):
        """Have the sender /tell the recipient; return the DM Discord 'sent'."""
        recipient = recipient or _member()
        await self.tell(sender_interaction, recipient=recipient)
        return recipient.send.return_value

    async def token_for(self, sender=SENDER):
        """The opaque autocomplete value for the latest relay from `sender`."""
        [row] = [r for r in relay_store.recent_senders(RECIPIENT) if r["account"] == sender]
        return row["relay_id"]

    def links(self, mapping):
        return patch.object(fritz_utils, "IDENTITY_LINKS", dict(mapping))


class TestBlock(RelayTestCase):
    async def test_block_by_the_token_autocomplete_offers(self):
        await self.delivered_message(self.interaction(display="Mal"))
        interaction = await self.run_cmd("relay_block", who=await self.token_for())
        self.assertEqual(self.blocks(), [SENDER])
        text = self.answer(interaction)
        self.assertIn("will not be told", text)
        self.assertIn("Mal", text)                          # as they were shown
        self.assert_no_mentions(interaction.followup.send.await_args.kwargs["allowed_mentions"])

    async def test_block_by_a_pasted_mention_or_id(self):
        for text in (f"<@{STRANGER_SNOWFLAKE}>", f"<@!{STRANGER_SNOWFLAKE}>",
                     f"  {STRANGER_SNOWFLAKE} ", STRANGER):
            with self.subTest(text):
                await self.run_cmd("relay_block", who=text)
                self.assertEqual(self.blocks(), [STRANGER])
                relay_store.forget_blocks(RECIPIENT)

    async def test_text_that_names_no_one_blocks_no_one(self):
        """Autocomplete only suggests; Discord submits whatever was typed."""
        for text in ("bob the builder", "*", "", "<@12>", "discord-",
                     f"Discord-{STRANGER_SNOWFLAKE}",           # would never bind
                     "web-alice", "telegram-99",                # would probe IDENTITY_LINKS
                     "١" * 18,                             # non-ASCII digits
                     "deadbeef", "b:1"):                        # not your relay / not a block
            with self.subTest(text):
                interaction = await self.run_cmd("relay_block", who=text)
                self.assertIn("could not tell who", self.answer(interaction))
        self.assertEqual(self.blocks(), [])

    async def test_someone_elses_relay_token_reads_like_gibberish(self):
        """A relay token counts only if that relay reached YOU."""
        await self.tell(recipient=_member(uid=OTHER_SNOWFLAKE))
        [row] = relay_store.recent_senders(f"discord-{OTHER_SNOWFLAKE}")
        interaction = await self.run_cmd("relay_block", who=row["relay_id"])
        self.assertIn("could not tell who", self.answer(interaction))
        self.assertEqual(self.blocks(), [])

    async def test_you_cannot_block_yourself(self):
        interaction = await self.run_cmd("relay_block", who=RECIPIENT)
        self.assertIn("cannot block yourself", self.answer(interaction))
        self.assertEqual(self.blocks(), [])

    async def test_blocking_twice_reads_the_same_as_once(self):
        """"You had already blocked X" says, for two ids resolving to one
        person, that they are linked."""
        first = self.answer(await self.run_cmd("relay_block", who=STRANGER))
        second = self.answer(await self.run_cmd("relay_block", who=STRANGER))
        self.assertEqual(first, second)

    async def test_blocks_can_be_placed_before_the_relay_is_switched_on(self):
        with patch.object(fritz_utils, "RELAY_ENABLED", False):
            await self.run_cmd("relay_block", who=STRANGER)
            await self.run_cmd("relay_block_everyone")
        self.assertEqual(sorted(self.blocks()), ["*", STRANGER])


class TestIdentityLinksStayPrivate(RelayTestCase):
    """IDENTITY_LINKS is the operator's to know. Blocks used to be stored
    resolved and shown back resolved, which answered "which Discord account
    is web-alice?" — or "is this alt that person?" — for anyone who asked."""

    ALT_SNOWFLAKE = 777777777777777777
    MAIN_SNOWFLAKE = 888888888888888888

    def ids(self):
        return f"discord-{self.ALT_SNOWFLAKE}", f"discord-{self.MAIN_SNOWFLAKE}"

    async def test_an_alias_cannot_be_typed_at_all(self):
        with self.links({"web-alice": STRANGER}):
            interaction = await self.run_cmd("relay_block", who="web-alice")
        self.assertIn("could not tell who", self.answer(interaction))

    async def test_blocking_an_alt_by_id_shows_only_the_id_you_typed(self):
        alt, main = self.ids()
        with self.links({alt: main}):
            await self.run_cmd("relay_block", who=str(self.ALT_SNOWFLAKE))
            status = self.answer(await self.run_cmd("relay_status"))
            choices = await self.cog._relay_unblock_who(self.recipient(), "")
        self.assertIn(f"<@{self.ALT_SNOWFLAKE}>", status)
        self.assertNotIn(str(self.MAIN_SNOWFLAKE), status)
        self.assertNotIn(str(self.MAIN_SNOWFLAKE), " ".join(c.name + c.value for c in choices))

    async def test_a_relay_from_an_alt_never_names_the_main_account(self):
        alt, main = self.ids()
        with self.links({alt: main}):
            dm = await self.delivered_message(
                self.interaction(user_id=self.ALT_SNOWFLAKE, name="alty", display="Alty"))
            block_choices = await self.cog._relay_block_who(self.recipient(), "")
            menu = self.recipient()
            await self.cog.block_sender_from_message(menu, dm)
            reply = self.answer(menu)
            status = self.answer(await self.run_cmd("relay_status"))
            unblock_choices = await self.cog._relay_unblock_who(self.recipient(), "")
        seen = " ".join([reply, status] + [c.name + c.value for c in block_choices + unblock_choices])
        self.assertNotIn(str(self.MAIN_SNOWFLAKE), seen)
        self.assertIn("@alty", reply)                          # how the relay named them
        self.assertEqual(self.blocks(), [alt])                 # the account that sent it...
        with self.links({alt: main}):
            interaction, _ = await self.tell(self.interaction(user_id=self.ALT_SNOWFLAKE))
            self.assertEqual(self.reply(interaction), relay_store.REFUSED)   # ...and it binds

    async def test_linked_accounts_are_separate_entries_in_block_autocomplete(self):
        """Merged into one entry, the list itself said they were one person."""
        alt, main = self.ids()
        with self.links({alt: main}):
            await self.delivered_message(self.interaction(
                user_id=self.MAIN_SNOWFLAKE, name="maine", display="Maine"))
            await self.delivered_message(self.interaction(
                user_id=self.ALT_SNOWFLAKE, name="alty", display="Alty"))
            names = sorted(c.name for c in await self.cog._relay_block_who(self.recipient(), ""))
        self.assertEqual(names, ["@alty \u00b7 Alty", "@maine \u00b7 Maine"])

    async def test_chained_links_block_the_real_sender_once(self):
        """Blocking the resolved id resolved it twice at the gate: under
        A->B, B->C the block caught C, and A kept getting through."""
        a, b, c = (f"discord-{n}" for n in (101010101010101010, 202020202020202020,
                                              303030303030303030))
        with self.links({a: b, b: c}):
            dm = await self.delivered_message(self.interaction(user_id=101010101010101010))
            await self.cog.block_sender_from_message(self.recipient(), dm)
            refused, _ = await self.tell(self.interaction(user_id=101010101010101010))
            self.assertEqual(self.reply(refused), relay_store.REFUSED)
            other, _ = await self.tell(self.interaction(user_id=202020202020202020))
            self.assertIn("Delivered", self.reply(other))

    async def test_a_chained_recipient_is_not_locked_out_of_their_own_relay(self):
        """The row holds the recipient already resolved. Resolving it again was
        a second hop, and the real recipient read "not a message I carried to you"."""
        r1, r2 = f"discord-{OTHER_SNOWFLAKE}", "discord-404040404040404040"
        with self.links({RECIPIENT: r1, r1: r2}):
            dm = await self.delivered_message()
            menu = self.recipient()
            await self.cog.block_sender_from_message(menu, dm)
            self.assertIn("will not be told", self.answer(menu))
            token = relay_store.recent_senders(RECIPIENT)[0]["relay_id"]
            typed = await self.run_cmd("relay_block", who=token)
            self.assertIn("will not be told", self.answer(typed))

    async def test_a_block_typed_as_an_alt_still_binds_the_main_account(self):
        alt, main = self.ids()
        with self.links({alt: main}):
            await self.run_cmd("relay_block", who=alt)
            interaction, _ = await self.tell(self.interaction(user_id=self.MAIN_SNOWFLAKE))
            self.assertEqual(self.reply(interaction), relay_store.REFUSED)


class TestBlockEveryone(RelayTestCase):
    async def test_blocks_everyone(self):
        interaction = await self.run_cmd("relay_block_everyone")
        self.assertIn("nothing to you from anyone", self.answer(interaction))
        await self.run_cmd("relay_block_everyone")
        self.assertEqual(self.blocks(), ["*"])


class TestUnblock(RelayTestCase):
    async def test_unblock_by_the_token_autocomplete_offers(self):
        await self.run_cmd("relay_block", who=STRANGER)
        [choice] = await self.cog._relay_unblock_who(self.recipient(), "")
        self.assertTrue(choice.value.startswith("b:"))
        interaction = await self.run_cmd("relay_unblock", who=choice.value)
        self.assertIn("may send you messages", self.answer(interaction))
        self.assertEqual(self.blocks(), [])

    async def test_unblock_by_a_pasted_id(self):
        await self.run_cmd("relay_block", who=STRANGER)
        interaction = await self.run_cmd("relay_unblock", who=f"<@{STRANGER_SNOWFLAKE}>")
        self.assertIn("may send you messages", self.answer(interaction))
        self.assertEqual(self.blocks(), [])

    async def test_unblocking_someone_you_never_blocked_says_so(self):
        interaction = await self.run_cmd("relay_unblock", who=STRANGER)
        self.assertIn("no block", self.answer(interaction))

    async def test_another_users_block_token_reads_like_gibberish(self):
        relay_store.block(SENDER, STRANGER, label="theirs")
        interaction = await self.run_cmd("relay_unblock",
                                         who=bot_commands._block_token(SENDER, STRANGER))
        self.assertIn("could not tell who", self.answer(interaction))
        self.assertEqual(relay_store.list_blocks(SENDER), [STRANGER])

    def test_tokens_are_keyed_and_bound_to_their_owner(self):
        """Without the host's secret a token must mean nothing, and two people
        blocking the same account must not hold the same token — that would
        say, to anyone comparing, that they had blocked the same person."""
        import hashlib
        mine = bot_commands._block_token(RECIPIENT, STRANGER)
        self.assertNotEqual(mine, bot_commands._block_token(SENDER, STRANGER))
        for unkeyed in (STRANGER, f"{RECIPIENT}\x00{STRANGER}"):
            self.assertNotEqual(mine[2:], hashlib.sha256(unkeyed.encode()).hexdigest()[:16])
        with patch.object(fritz_utils, "CHAT_COOKIE_SECRET", "another-host"):
            self.assertNotEqual(mine, bot_commands._block_token(RECIPIENT, STRANGER))

    async def test_tokens_have_no_order_and_a_stale_one_removes_nothing_else(self):
        """Rowid tokens counted everyone's blocks between two of yours, and a
        reused rowid let a stale token remove a different, newer block."""
        await self.run_cmd("relay_block", who=STRANGER)
        [stale] = await self.cog._relay_unblock_who(self.recipient(), "")
        await self.run_cmd("relay_unblock", who=stale.value)
        other = f"discord-{OTHER_SNOWFLAKE}"
        await self.run_cmd("relay_block", who=other)
        interaction = await self.run_cmd("relay_unblock", who=stale.value)
        self.assertIn("could not tell who", self.answer(interaction))
        self.assertEqual(self.blocks(), [other])
        [fresh] = await self.cog._relay_unblock_who(self.recipient(), "")
        self.assertRegex(fresh.value, r"^b:[0-9a-f]{16}$")
        self.assertNotEqual(fresh.value, stale.value)

    async def test_lifting_the_blanket_block_keeps_named_ones(self):
        await self.run_cmd("relay_block", who=STRANGER)
        await self.run_cmd("relay_block_everyone")
        interaction = await self.run_cmd("relay_unblock", who="*")
        self.assertIn("still blocked", self.answer(interaction))
        self.assertEqual(self.blocks(), [STRANGER])

    async def test_text_that_names_no_one_unblocks_no_one(self):
        await self.run_cmd("relay_block", who=STRANGER)
        interaction = await self.run_cmd("relay_unblock", who="whoever")
        self.assertIn("could not tell who", self.answer(interaction))
        self.assertEqual(self.blocks(), [STRANGER])


class TestStatus(RelayTestCase):
    async def test_nobody_blocked(self):
        text = self.answer(await self.run_cmd("relay_status"))
        self.assertIn("not blocking anyone", text)
        self.assertNotIn("switched off", text)              # the relay is on here

    async def test_everyone_and_named_blocks_are_both_shown(self):
        await self.run_cmd("relay_block", who=STRANGER)
        await self.run_cmd("relay_block_everyone")
        interaction = await self.run_cmd("relay_status")
        text = self.answer(interaction)
        self.assertIn("**everyone**", text)
        self.assertIn(f"<@{STRANGER_SNOWFLAKE}>", text)
        self.assert_no_mentions(interaction.followup.send.await_args.kwargs["allowed_mentions"])

    async def test_blocking_an_id_never_reveals_the_name_fritz_knows_it_by(self):
        """/relay block takes any ID. Answering with the recorded display name
        would make it a lookup service: block a stranger, read their name —
        and learn that they use Fritz at all — then unblock."""
        identity_store.record(STRANGER, "Secret Nickname", "discord")
        replies = [self.answer(await self.run_cmd("relay_block", who=STRANGER)),
                   self.answer(await self.run_cmd("relay_status"))]
        [choice] = await self.cog._relay_unblock_who(self.recipient(), "")
        replies.append(choice.name)
        replies.append(self.answer(await self.run_cmd("relay_unblock", who=STRANGER)))
        for text in replies:
            self.assertNotIn("Secret Nickname", text)

    async def test_a_long_list_still_fits_in_one_message(self):
        """Discord refuses content over 2000 characters, and /relay status
        failed outright once someone had blocked about eighty people."""
        for n in range(300):
            relay_store.block(RECIPIENT, f"discord-{STRANGER_SNOWFLAKE + n}",
                              label=f"<@{STRANGER_SNOWFLAKE + n}>")
        text = self.answer(await self.run_cmd("relay_status"))
        self.assertLessEqual(len(text), 2000)
        self.assertIn("more", text)

    async def test_says_when_the_relay_is_switched_off(self):
        with patch.object(fritz_utils, "RELAY_ENABLED", False):
            text = self.answer(await self.run_cmd("relay_status"))
        self.assertIn("switched off", text)


class TestForgetBlocks(RelayTestCase):
    """DECISIONS #22: /forget all keeps your blocks; this is the explicit way out."""

    async def open_view(self):
        interaction = await self.run_cmd("relay_forget_blocks")
        interaction.followup.send.assert_awaited_once()
        kwargs = interaction.followup.send.await_args.kwargs
        self.assertIs(kwargs.get("ephemeral"), True)
        return interaction, kwargs["view"]

    def press(self, user_id=RECIPIENT_SNOWFLAKE):
        press = MagicMock()
        press.user.id = user_id
        press.response.defer = AsyncMock()
        press.response.edit_message = AsyncMock()
        press.response.send_message = AsyncMock()
        press.edit_original_response = AsyncMock()
        return press

    async def test_nothing_to_forget(self):
        self.assertIn("no blocks", self.answer(await self.run_cmd("relay_forget_blocks")))

    async def test_confirm_removes_mine_and_never_theirs(self):
        await self.run_cmd("relay_block", who=STRANGER)
        await self.run_cmd("relay_block_everyone")
        relay_store.block(SENDER, RECIPIENT)                 # someone blocked me
        _, view = await self.open_view()
        press = self.press()
        await view.confirm.callback(press)
        press.response.defer.assert_awaited_once()           # answered in time
        self.assertEqual(self.blocks(), [])
        self.assertEqual(relay_store.list_blocks(SENDER), [RECIPIENT])
        kwargs = press.edit_original_response.await_args.kwargs
        self.assertIn("Removed 2 blocks", kwargs["content"])
        self.assertIsNone(kwargs["view"])
        self.assertTrue(view.is_finished(), "a lapse notice could still follow")

    async def test_a_failed_removal_still_answers_and_withdraws_the_buttons(self):
        await self.run_cmd("relay_block", who=STRANGER)
        _, view = await self.open_view()
        press = self.press()
        with patch.object(relay_store, "forget_blocks",
                          side_effect=relay_store.RelayStoreError("locked")):
            await view.confirm.callback(press)
        kwargs = press.edit_original_response.await_args.kwargs
        self.assertIn("could not remove", kwargs["content"])
        self.assertIsNone(kwargs["view"])
        self.assertTrue(view.is_finished())
        self.assertEqual(self.blocks(), [STRANGER])

    async def test_the_audit_log_never_names_whom_you_blocked_or_unblocked(self):
        """It outlives /relay forget-blocks. A record of the blocked party would
        outlive the block itself."""
        await self.run_cmd("relay_block", who=STRANGER)
        await self.run_cmd("relay_unblock", who=STRANGER)
        await self.run_cmd("relay_block", who=STRANGER)
        _, view = await self.open_view()
        await view.confirm.callback(self.press())
        events = {e["event"] for e in self.audit_events()}
        self.assertTrue({"relay_block", "relay_unblock", "relay_forget_blocks"} <= events)
        self.assertNotIn(str(STRANGER_SNOWFLAKE), self.audit_path.read_text(encoding="utf-8"))

    async def test_it_warns_that_it_re_arms_people(self):
        await self.run_cmd("relay_block", who=STRANGER)
        interaction, _ = await self.open_view()
        self.assertIn("bothering you", interaction.followup.send.await_args.args[0])

    async def test_keep_them_removes_nothing(self):
        await self.run_cmd("relay_block", who=STRANGER)
        _, view = await self.open_view()
        await view.cancel.callback(self.press())
        self.assertEqual(self.blocks(), [STRANGER])
        self.assertTrue(view.is_finished())

    async def test_nobody_else_can_press_it(self):
        await self.run_cmd("relay_block", who=STRANGER)
        _, view = await self.open_view()
        self.assertFalse(await view.interaction_check(self.press(user_id=STRANGER_SNOWFLAKE)))
        self.assertTrue(await view.interaction_check(self.press()))

    async def test_a_lapsed_confirmation_withdraws_its_buttons(self):
        """_ForgetConfirmView never did: after its timeout the buttons simply
        stopped working, with nothing on screen to say so."""
        await self.run_cmd("relay_block", who=STRANGER)
        interaction, view = await self.open_view()
        interaction.edit_original_response = AsyncMock()
        await view.on_timeout()
        kwargs = interaction.edit_original_response.await_args.kwargs
        self.assertIsNone(kwargs["view"])
        self.assertIn("lapsed", kwargs["content"])
        self.assertEqual(self.blocks(), [STRANGER])


class TestBlockFromMessage(RelayTestCase):
    async def test_blocks_whoever_sent_that_message_named_as_it_named_them(self):
        dm = await self.delivered_message(self.interaction(display="Mal"))
        interaction = self.recipient()
        await self.cog.block_sender_from_message(interaction, dm)
        self.assertEqual(self.blocks(), [SENDER])
        text = self.answer(interaction)
        self.assertIn("will not be told", text)
        self.assertIn("@alice", text)

    async def test_with_two_senders_each_message_blocks_only_its_own(self):
        """Anchored on the message, never on "whoever last relayed to you" —
        and in both directions, so neither "first" nor "latest" passes."""
        first = await self.delivered_message()
        second = await self.delivered_message(self.interaction(user_id=OTHER_SNOWFLAKE))
        await self.cog.block_sender_from_message(self.recipient(), second)
        self.assertEqual(self.blocks(), [f"discord-{OTHER_SNOWFLAKE}"])
        relay_store.forget_blocks(RECIPIENT)
        await self.cog.block_sender_from_message(self.recipient(), first)
        self.assertEqual(self.blocks(), [SENDER])

    async def test_a_message_that_is_not_a_relay_blocks_no_one(self):
        other = MagicMock()
        other.id = 42
        interaction = self.recipient()
        await self.cog.block_sender_from_message(interaction, other)
        self.assertIn("not a message I carried to you", self.answer(interaction))
        self.assertEqual(self.blocks(), [])

    async def test_only_the_recipient_can_act_on_a_relay(self):
        dm = await self.delivered_message()
        interaction = self.recipient(user_id=STRANGER_SNOWFLAKE)
        await self.cog.block_sender_from_message(interaction, dm)
        self.assertIn("not a message I carried to you", self.answer(interaction))
        self.assertEqual(relay_store.list_blocks(STRANGER), [])

    async def test_the_recipient_check_follows_the_recipient_across_aliases(self):
        """The relay row holds the recipient RESOLVED. Comparing it with the
        invoker's raw id would lock a linked recipient out of their own relay."""
        linked = f"discord-{OTHER_SNOWFLAKE}"
        with self.links({RECIPIENT: linked}):
            dm = await self.delivered_message()
            interaction = self.recipient()
            await self.cog.block_sender_from_message(interaction, dm)
            self.assertIn("will not be told", self.answer(interaction))
            self.assertEqual(relay_store.list_blocks(linked), [SENDER])


class TestALinkAddedAfterDelivery(RelayTestCase):
    """The row holds the recipient resolved as it was then. A link added since
    must not lock the real recipient out of blocking the message in their own
    inbox — the account it was sent to is them."""

    async def test_the_menu_and_the_picker_still_block(self):
        later = {RECIPIENT: "discord-555555555555555999"}
        dm = await self.delivered_message()
        rid = relay_store.get_by_dm_message(dm.id)["id"]
        with self.links(later):
            menu = self.recipient()
            await self.cog.block_sender_from_message(menu, dm)
            self.assertIn("will not be told", self.answer(menu))
            self.assertEqual(self.blocks(), [SENDER])     # held by who they are now
            self.assertEqual(bot_commands._block_target(rid, RECIPIENT)[0], SENDER)


class TestAutocomplete(RelayTestCase):
    async def test_block_offers_each_sender_as_you_saw_them(self):
        await self.delivered_message(self.interaction(display="Mallory"))
        await self.delivered_message(self.interaction(user_id=OTHER_SNOWFLAKE, name="dave", display="Dave"))
        choices = await self.cog._relay_block_who(self.recipient(), "")
        self.assertEqual(sorted(c.name for c in choices), ["@alice · Mallory", "@dave · Dave"])
        for c in choices:
            self.assertRegex(c.value, r"^[0-9a-f]{8}$")        # opaque, never an id

    async def test_a_later_name_from_somewhere_else_is_never_shown(self):
        """The recipient saw "Mal". A nickname the sender uses in some other
        server, or on the web chat, is not theirs to be shown."""
        await self.delivered_message(self.interaction(display="Mal"))
        identity_store.record(SENDER, "Mallory Realname - Acme Legal", "discord")
        [choice] = await self.cog._relay_block_who(self.recipient(), "")
        self.assertIn("Mal", choice.name)
        self.assertNotIn("Realname", choice.name)

    async def test_block_filters_on_what_is_typed(self):
        await self.delivered_message(self.interaction(display="Mallory"))
        await self.delivered_message(self.interaction(user_id=OTHER_SNOWFLAKE, name="dave"))
        choices = await self.cog._relay_block_who(self.recipient(), "mall")
        self.assertEqual([c.name for c in choices], ["@alice · Mallory"])

    async def test_an_older_sender_can_still_be_found_by_typing(self):
        """Filtered before the 25-choice cut: a brigade of recent senders must
        not bury the one the recipient is actually looking for."""
        await self.delivered_message(self.interaction(name="zed", display="zed"))
        with patch.object(fritz_utils, "RELAY_MAX_INBOUND_PER_RECIPIENT_PER_HOUR", 100):
            for n in range(30):
                await self.delivered_message(self.interaction(
                    user_id=STRANGER_SNOWFLAKE + n, name=f"b{n}", display=f"Brigade {n}"))
        choices = await self.cog._relay_block_who(self.recipient(), "zed")
        self.assertEqual([c.name for c in choices], ["@zed"])
        self.assertLessEqual(len(await self.cog._relay_block_who(self.recipient(), "")), 25)

    async def test_block_never_offers_someone_whose_message_did_not_arrive(self):
        relay_store.block(RECIPIENT, SENDER)
        await self.tell()                                   # refused
        self.assertEqual(await self.cog._relay_block_who(self.recipient(), ""), [])

    async def test_unblock_offers_your_blocks_including_everyone(self):
        await self.run_cmd("relay_block", who=STRANGER)
        await self.run_cmd("relay_block_everyone")
        names = {c.value: c.name for c in await self.cog._relay_unblock_who(self.recipient(), "")}
        self.assertIn("*", names)
        self.assertIn("Everyone", names["*"])
        [token] = [v for v in names if v != "*"]
        self.assertTrue(token.startswith("b:"))
        self.assertEqual(names[token], f"User ID {STRANGER_SNOWFLAKE}")

    async def test_unblock_names_a_relay_block_as_you_saw_them(self):
        dm = await self.delivered_message(self.interaction(display="Mallory"))
        await self.cog.block_sender_from_message(self.recipient(), dm)
        [choice] = await self.cog._relay_unblock_who(self.recipient(), "")
        self.assertEqual(choice.name, "@alice · Mallory")

    async def test_labels_fit_and_hide_nothing(self):
        # The hostile name travels on the sender's own /tell, as it would for
        # real: _identity records it, and would overwrite one set by hand.
        await self.delivered_message(self.interaction(display="‮evil" + "x" * 200))
        [choice] = await self.cog._relay_block_who(self.recipient(), "")
        self.assertLessEqual(len(choice.name), 100)
        self.assertNotIn("‮", choice.name)

    async def test_autocomplete_does_not_wait_on_the_worker_pool(self):
        """It cannot defer and has three seconds; long model turns can fill the
        pool for minutes."""
        await self.delivered_message()
        with patch.object(bot_commands, "run_blocking",
                          AsyncMock(side_effect=AssertionError("used the pool"))):
            self.assertEqual(len(await self.cog._relay_block_who(self.recipient(), "")), 1)
            await self.cog._relay_unblock_who(self.recipient(), "")


class TestEveryWayOfSayingNoReadsTheSame(RelayTestCase):
    """PR 3's pinned property, end to end through the real commands: a named
    block, a block on everyone, closed DMs, and the context menu all give the
    sender the same reply. Equality between the replies, not literals — a
    later "helpful error messages" pass is exactly how it would be lost."""

    async def sender_reads(self, recipient=None):
        interaction, _ = await self.tell(recipient=recipient)
        return self.reply(interaction)

    async def test_the_sender_cannot_tell_which_no_it_was(self):
        await self.delivered_message()
        await self.run_cmd("relay_block", who=await self.token_for())
        by_name = await self.sender_reads()
        relay_store.forget_blocks(RECIPIENT)

        await self.run_cmd("relay_block_everyone")
        by_everyone = await self.sender_reads()
        await self.run_cmd("relay_unblock", who="*")

        closed = _member()
        closed.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
        by_closed_dms = await self.sender_reads(closed)

        dm = await self.delivered_message()
        await self.cog.block_sender_from_message(self.recipient(), dm)
        by_menu = await self.sender_reads()

        self.assertEqual(by_name, by_everyone)
        self.assertEqual(by_everyone, by_closed_dms)
        self.assertEqual(by_closed_dms, by_menu)
        for word in ("block", "closed", "DM"):
            self.assertNotIn(word, by_name)


class TestTheAuditLogCannotTellABlockFromA403(RelayTestCase):
    """It outlives /relay forget-blocks. "relay_denied reason=blocked_sender"
    was a durable record of who blocked whom."""

    async def test_both_refusals_write_the_same_line(self):
        relay_store.block(RECIPIENT, SENDER)
        await self.tell()
        relay_store.unblock(RECIPIENT, SENDER)
        closed = _member()
        closed.send.side_effect = _http_error(discord.Forbidden, 403, 50007)
        await self.tell(recipient=closed)
        refused = [e for e in self.audit_events() if e["event"] == "relay_refused"]
        self.assertEqual(len(refused), 2)
        self.assertEqual(set(refused[0]), set(refused[1]))
        for event in refused:
            self.assertNotIn("reason", event)
            self.assertNotIn("discord_code", event)
        self.assertNotIn("blocked", self.audit_path.read_text(encoding="utf-8"))


class TestTheRelayedMessageNamesTheWayOut(RelayTestCase):
    async def test_the_footer_says_how_to_refuse(self):
        dm_recipient = _member()
        await self.tell(recipient=dm_recipient)
        footer = dm_recipient.send.await_args.kwargs["embed"].footer.text
        self.assertIn("/relay block", footer)
        self.assertIn("Block this sender", footer)


class TestContextMenuErrors(RelayTestCase):
    """A cog's error handler does not apply to a context menu, so its
    failures reach tree.on_error alone — and must be answered exactly once."""

    async def test_a_failing_menu_is_answered_once(self):
        bot = commands.Bot(command_prefix="$", intents=discord.Intents.default())
        cog = _make_cog()
        cog.bot = bot
        await bot.add_cog(cog)
        bot.tree.on_error = bot_commands.handle_tree_error
        menu = bot.tree.get_command("Block this sender", type=discord.AppCommandType.message)
        interaction = _fake_interaction("bob", user_id=RECIPIENT_SNOWFLAKE)
        interaction.command = menu
        answered = {"done": False}

        async def answer(*a, **kw):
            answered["done"] = True
        interaction.response.is_done = MagicMock(side_effect=lambda: answered["done"])
        interaction.response.send_message = AsyncMock(side_effect=answer)
        err = discord.app_commands.CommandInvokeError(menu, RuntimeError("x"))
        # app_commands/tree.py, the context-menu branch of _call, in order:
        if menu.on_error is not None:
            await menu.on_error(interaction, err)
        await bot.tree.on_error(interaction, err)
        self.assertEqual(interaction.response.send_message.await_count
                         + interaction.followup.send.await_count, 1)


class TestShape(RelayTestCase):
    async def test_relay_works_in_dms_and_servers_but_is_guild_installed(self):
        bot = commands.Bot(command_prefix="$", intents=discord.Intents.default())
        cog = _make_cog()
        cog.bot = bot
        await bot.add_cog(cog)
        payload = bot.tree.get_command("relay").to_dict(bot.tree)
        self.assertIs(payload["dm_permission"], True)
        self.assertEqual(payload["contexts"], [0, 1])
        self.assertEqual(payload["integration_types"], [0])

        menu = bot.tree.get_command("Block this sender", type=discord.AppCommandType.message)
        self.assertIsNotNone(menu, "cog_load did not register the context menu")
        menu_payload = menu.to_dict(bot.tree)
        self.assertEqual(menu_payload["contexts"], [1])            # bot DMs only
        self.assertEqual(menu_payload["integration_types"], [0])

        await bot.remove_cog("FritzCommands")
        self.assertIsNone(bot.tree.get_command("Block this sender",
                                               type=discord.AppCommandType.message))

    def test_no_relay_command_uses_a_user_picker(self):
        """In a DM with the bot the picker can reach only you and Fritz."""
        for command in bot_commands.FritzCommands.relay.commands:
            for param in command.parameters:
                self.assertIsNot(param.type, discord.AppCommandOptionType.user,
                                 f"/relay {command.name} {param.name}")

    async def test_every_relay_command_defers_first(self):
        """The pool can be full for minutes; three seconds is all Discord gives."""
        with patch.object(bot_commands, "run_blocking",
                          AsyncMock(side_effect=AssertionError("pool before defer"))):
            for name, kw in (("relay_block", {"who": STRANGER}), ("relay_block_everyone", {}),
                             ("relay_unblock", {"who": STRANGER}), ("relay_status", {}),
                             ("relay_forget_blocks", {})):
                with self.subTest(name):
                    interaction = self.recipient()
                    with self.assertRaises(AssertionError):
                        await self.run_cmd(name, interaction, **kw)
                    interaction.response.defer.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
