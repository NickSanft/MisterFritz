"""
Tests for PR 5 of plans/12-direct-message-relay.md: the relayed DM's buttons.

[Reply] opens a form whose submission is the reply; [Not now] takes the
buttons away and tells the sender nothing; [Block sender] is the one-press
opt-out DECISIONS #23 promises. All three are DynamicItems, so they keep
working on DMs sent before a restart, and every press is checked against the
row its message belongs to, since a custom_id is the presser's to send.

Three things here matter more than the rest, because each fails silently:
the classes are registered at import (a missed registration breaks every
relayed DM ever sent, and only after the next restart); one press runs one
callback; and a form submitted after a restart is still carried.
"""
import asyncio
import pathlib
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import discord
from discord.ext import commands

import bot_commands
import fritz_utils
import main_discord
import privacy
import relay_format
import relay_router
import relay_store
from test_relay_router import ReplyTestCase
from test_tell_command import (RECIPIENT, RECIPIENT_SNOWFLAKE, SENDER, SENDER_SNOWFLAKE,
                               _member)

REPO = pathlib.Path(__file__).resolve().parent.parent
BOT_ID = 999
STRANGER_SNOWFLAKE = 333333333333333333
KINDS = ("reply", "notnow", "block")
LINKED = "discord-111111111111111611"
CHAINED = "discord-111111111111111711"


class FakeResponse:
    """An InteractionResponse that remembers it has been used, as the real one
    does, and refuses a second use.

    A refusal is recorded as well as raised. The handlers swallow every
    exception when answering, so a raise alone proved nothing: a handler
    that answered on the wrong channel after its first response looked, to
    a count of calls, exactly like one that answered once. ButtonTestCase
    fails any test that leaves a refusal behind, and told() reads only what
    Discord would have accepted. `fail` makes one method raise as Discord
    would on a 5xx, leaving the response unused.
    """

    def __init__(self):
        self.done = False
        self.used = []
        self.refused = []
        self.fail = set()
        for name in ("send_modal", "send_message", "edit_message", "defer"):
            setattr(self, name, AsyncMock(side_effect=self._responder(name)))

    def _responder(self, name):
        async def respond(*args, **kwargs):
            if name in self.fail:
                raise discord.DiscordServerError(MagicMock(status=503, reason="x"), "down")
            if self.done:
                self.refused.append(name)
                raise RuntimeError(f"{name}: this interaction was already answered")
            self.done = True
            self.used.append((name, args, kwargs))
        return respond

    def is_done(self):
        return self.done


class ButtonTestCase(ReplyTestCase):
    def setUp(self):
        super().setUp()
        self.client.user.id = BOT_ID

    async def relayed_with_view(self, sender_interaction=None):
        """One /tell; the DM Discord 'delivered', its view, its row id, and
        the recipient mock (whose send kwargs hold the embed)."""
        recipient = _member()
        dm = await self.relayed(sender_interaction, recipient=recipient)
        view = recipient.send.await_args.kwargs["view"]
        return dm, view, relay_store.get_by_dm_message(dm.id)["id"], recipient

    def press_as(self, *, user_id=RECIPIENT_SNOWFLAKE, name="bob", display="Bob",
                 kind=discord.InteractionType.component, message=None, data=None):
        """A button press or form submission. Not self.interaction(), which is
        TellTestCase's /tell interaction and builds every sender here."""
        interaction = MagicMock()
        interaction.type = kind
        interaction.user.id = user_id
        interaction.user.name = name
        interaction.user.display_name = display
        interaction.user.display_avatar.url = "https://cdn.example/bob.png"
        interaction.message = message
        interaction.client = self.client
        interaction.data = data or {}
        interaction.response = FakeResponse()
        interaction.followup.send = AsyncMock()
        self.addCleanup(lambda: self.assertEqual(
            interaction.response.refused, [], "answered on a response already used"))
        return interaction

    async def press(self, kind, relay_id, message, **who):
        cls = {"reply": relay_router.ReplyButton, "notnow": relay_router.NotNowButton,
               "block": relay_router.BlockButton}[kind]
        interaction = self.press_as(message=message, **who)
        await cls(relay_id).callback(interaction)
        return interaction

    async def submit(self, relay_id, words, *, user_id=RECIPIENT_SNOWFLAKE, shape="row"):
        component = {"type": 4, "custom_id": "words", "value": words}
        components = ([{"type": 1, "components": [component]}] if shape == "row"
                      else [{"type": 18, "id": 1, "component": component}])
        interaction = self.press_as(
            user_id=user_id, kind=discord.InteractionType.modal_submit,
            data={"custom_id": f"fritz:relay:form:{relay_id}", "components": components})
        await relay_router.on_interaction(interaction)
        return interaction

    def told(self, interaction) -> str:
        """The one ephemeral thing the presser was told."""
        calls = ([(args, kwargs) for name, args, kwargs in interaction.response.used
                  if name == "send_message"]
                 + [(c.args, c.kwargs) for c in interaction.followup.send.await_args_list])
        self.assertEqual(len(calls), 1, f"expected one answer, got {calls}")
        args, kwargs = calls[0]
        self.assertIs(kwargs.get("ephemeral"), True, "a button answer was not ephemeral")
        self.assert_no_mentions(kwargs["allowed_mentions"])
        return args[0] if args else kwargs["content"]

    def withdrew(self, interaction) -> bool:
        """Did the buttons come off the message, as Discord would accept it?"""
        return ("edit_message", (), {"view": None}) in interaction.response.used

    def as_discord_has_it(self, recipient, author_id=BOT_ID):
        """The relayed DM as a real discord.Message would present it."""
        original = MagicMock(spec=discord.Message)
        original.id = recipient.send.return_value.id
        original.author = MagicMock()
        original.author.id = author_id
        original.embeds = [recipient.send.await_args.kwargs["embed"]]
        return original

    def linked(self, chain=False):
        """The recipient as an alt of another account — or, chained, of an
        account that is itself an alt. The stored recipient is resolved once;
        resolving it again would be the wrong person."""
        links = {RECIPIENT: LINKED}
        if chain:
            links[LINKED] = CHAINED
        return patch.object(fritz_utils, "IDENTITY_LINKS", links)

    async def relayed_with_view_linked(self, chain=False, legacy=False):
        """`legacy`: a row from before recipient_account existed, so only
        the resolved recipient can recognise who it was delivered to."""
        with self.linked(chain):
            relayed = await self.relayed_with_view()
        if legacy:
            with sqlite3.connect(self.db) as conn:
                conn.execute("UPDATE relay_messages SET recipient_account = NULL")
                conn.commit()
        return relayed

    def expire(self, relay_id):
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE relay_messages SET expires_at = ? WHERE id = ?",
                         ((datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                          relay_id))
            conn.commit()


# ─── What goes out ───────────────────────────────────────────────────────────

class TestTheButtonsGoOut(ButtonTestCase):
    async def test_a_relayed_dm_carries_all_three_bound_to_its_row(self):
        _, view, rid, _ = await self.relayed_with_view()
        self.assertEqual([type(c) for c in view.children], list(relay_router.RELAY_BUTTONS))
        self.assertEqual([c.custom_id for c in view.children],
                         [f"fritz:relay:{k}:{rid}" for k in KINDS])
        self.assertEqual([c.item.label for c in view.children],
                         ["Reply", "Not now", "Block sender"])
        for child in view.children:
            self.assertIsInstance(child, discord.ui.DynamicItem)
            self.assertLessEqual(len(child.custom_id), 100)   # Discord's; unchecked locally
            self.assertLessEqual(len(child.item.label), 80)
            self.assertEqual(child.row, 0)
        self.assertIsNone(view.timeout)

    async def test_the_view_is_never_stored(self):
        """Stored, a sent view registers its classes as a side effect, and a
        missed registration would then work until the first restart."""
        _, view, _, _ = await self.relayed_with_view()
        self.assertTrue(view.is_finished())

    async def test_the_senders_receipt_has_none(self):
        """Not routable, and a reply to it is OWN_COPY."""
        interaction = self.interaction()
        await self.tell(interaction)
        self.assertNotIn("view", interaction.user.send.await_args.kwargs)

    async def test_a_carried_reply_carries_buttons_for_its_own_row(self):
        """The reply is a relay in its own right, delivered to the original
        sender; its buttons must name ITS row, not the one it answers."""
        dm = await self.relayed()
        origin = relay_store.get_by_dm_message(dm.id)["id"]
        await self.route(self.reply_to(dm, "sure, 8 works"))
        view = self.dm_channel.send.await_args.kwargs["view"]
        back = relay_store.get_by_dm_message(self.sent_back.id)
        self.assertNotEqual(back["id"], origin)
        self.assertEqual(back["recipient_id"], SENDER)
        self.assertEqual([c.custom_id for c in view.children],
                         [f"fritz:relay:{k}:{back['id']}" for k in KINDS])

    async def test_each_custom_id_matches_exactly_one_class(self):
        """Two matching templates would BOTH run on one press."""
        _, view, _, _ = await self.relayed_with_view()
        for child in view.children:
            matching = [cls for cls in relay_router.RELAY_BUTTONS
                        if cls.__discord_ui_compiled_template__.fullmatch(child.custom_id)]
            self.assertEqual(matching, [type(child)], child.custom_id)

    async def test_a_press_rebuilds_the_same_button(self):
        _, view, rid, _ = await self.relayed_with_view()
        for child in view.children:
            cls = type(child)
            match = cls.__discord_ui_compiled_template__.fullmatch(child.custom_id)
            rebuilt = await cls.from_custom_id(None, None, match)
            self.assertEqual((rebuilt.custom_id, rebuilt.relay_id), (child.custom_id, rid))

    def test_a_malformed_id_builds_nothing(self):
        """The library builds with re.match and dispatches with re.fullmatch:
        without this, "abcdef12x" would build a button that never works."""
        for bad in ("abcdef12x", "ABCDEF12", "abcdef1", "", None, "abcdef1\n"):
            for cls in relay_router.RELAY_BUTTONS:
                with self.subTest(cls=cls.__name__, bad=bad), self.assertRaises(ValueError):
                    cls(bad)


# ─── Registration ────────────────────────────────────────────────────────────

class TestRegistration(unittest.TestCase):
    """A missed registration is invisible in-process and breaks every relayed
    DM ever sent after the next restart. So it is checked on the real client,
    which has sent nothing."""

    def test_importing_main_discord_registers_every_button(self):
        registered = main_discord.client._connection._view_store._dynamic_items
        for cls in relay_router.RELAY_BUTTONS:
            self.assertIs(registered.get(cls.__discord_ui_compiled_template__), cls,
                          cls.__name__)
        self.assertEqual(relay_router.missing_registrations(main_discord.client), [])

    def test_and_the_form_listener(self):
        self.assertIn(relay_router.on_interaction,
                      main_discord.client.extra_events.get("on_interaction", []))

    def test_registration_happens_at_import_not_in_on_ready(self):
        """on_ready loads TTS (~17 s) before add_cog, and a press in that
        window matched nothing and vanished; on a reconnect, add_cog raises
        before any line after it runs."""
        src = (REPO / "main_discord.py").read_text(encoding="utf-8")
        self.assertLess(src.index("relay_router.install(client)"),
                        src.index("async def on_ready("))

    def test_on_ready_checks_it(self):
        src = (REPO / "main_discord.py").read_text(encoding="utf-8")
        body = src.split("async def on_ready(", 1)[1].split("\nasync def ", 1)[0]
        self.assertLess(body.index("relay_router.ensure_installed(client)"),
                        body.index("await run_blocking(_load_tts)"))

    def test_on_readys_check_repairs_a_bare_client_loudly(self):
        bot = commands.Bot(command_prefix="$", intents=discord.Intents.default())
        with self.assertLogs(relay_router.logger, "ERROR") as logs:
            self.assertEqual(relay_router.ensure_installed(bot), list(relay_router.RELAY_BUTTONS))
        self.assertIn("were not registered", " ".join(logs.output))
        self.assertEqual(relay_router.missing_registrations(bot), [])
        with self.assertNoLogs(relay_router.logger, "ERROR"):
            self.assertEqual(relay_router.ensure_installed(bot), [])

    def test_a_bare_client_is_reported_and_install_repairs_it(self):
        bot = commands.Bot(command_prefix="$", intents=discord.Intents.default())
        self.assertEqual(relay_router.missing_registrations(bot),
                         list(relay_router.RELAY_BUTTONS))
        relay_router.install(bot)
        relay_router.install(bot)                      # idempotent
        self.assertEqual(relay_router.missing_registrations(bot), [])
        self.assertEqual(bot.extra_events["on_interaction"].count(relay_router.on_interaction), 1)


# ─── Through discord.py's own dispatcher ─────────────────────────────────────

def _user_payload(user_id, name):
    return {"id": str(user_id), "username": name, "discriminator": "0", "avatar": None,
            "global_name": name.title()}


def _interaction_payload(kind, data, *, user_id=RECIPIENT_SNOWFLAKE, message=None):
    payload = {"id": "1100000000000000001", "application_id": str(BOT_ID), "type": kind,
               "token": "token", "version": 1, "channel_id": "555",
               "channel": {"id": "555", "type": 1}, "user": _user_payload(user_id, "bob"),
               "data": data, "locale": "en-GB", "attachment_size_limit": 10485760,
               "app_permissions": "0", "entitlements": [], "context": 1,
               "authorizing_integration_owners": {"0": "1"}}
    if message is not None:
        payload["message"] = message
    return payload


class TestThroughTheRealDispatcher(ButtonTestCase):
    """Real payloads, parsed by discord.py's ConnectionState and dispatched by
    its ViewStore, into these classes. Only the HTTP calls are replaced."""

    def setUp(self):
        super().setUp()
        self.answers = []
        self.bot = commands.Bot(command_prefix="$", intents=discord.Intents.default())
        answers = self.answers

        def respond(kind):
            async def fake(response, *args, **kwargs):
                if response._response_type:
                    raise discord.InteractionResponded(response._parent)
                response._response_type = discord.InteractionResponseType.deferred_channel_message
                answers.append((kind, args, kwargs))
            return fake

        async def followup(webhook, *args, **kwargs):
            answers.append(("followup", args, kwargs))

        for name in ("send_modal", "send_message", "edit_message", "defer"):
            patcher = patch.object(discord.InteractionResponse, name, respond(name))
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(discord.Webhook, "send", followup)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def asyncSetUp(self):
        # Before login a Client has no loop, and dispatching an event raises.
        self.bot.loop = asyncio.get_running_loop()
        self.bot.create_dm = AsyncMock(return_value=self.dm_channel)

    def message_payload(self, dm_id, view):
        return {"id": str(dm_id), "channel_id": "555", "author": _user_payload(BOT_ID, "fritz"),
                "content": "", "timestamp": "2026-09-22T00:00:00+00:00",
                "edited_timestamp": None, "tts": False, "mention_everyone": False,
                "mentions": [], "mention_roles": [], "attachments": [], "pinned": False,
                "type": 0, "flags": 0, "components": view.to_components(),
                "embeds": [{"type": "rich", "description": "running late",
                            "footer": {"text": "Sent with /tell from X."}}]}

    async def dispatch(self, payload):
        self.bot._connection.parse_interaction_create(payload)
        for _ in range(50):
            await asyncio.sleep(0)
            if self.answers:
                await asyncio.sleep(0)
                break

    async def test_a_press_reaches_reply_and_opens_the_form(self):
        relay_router.install(self.bot)
        dm, view, rid, _ = await self.relayed_with_view()
        await self.dispatch(_interaction_payload(
            3, {"custom_id": f"fritz:relay:reply:{rid}", "component_type": 2},
            message=self.message_payload(dm.id, view)))
        [(kind, args, _)] = self.answers
        self.assertEqual(kind, "send_modal")
        self.assertEqual(args[0].custom_id, f"fritz:relay:form:{rid}")

    async def test_one_press_runs_one_callback(self):
        relay_router.install(self.bot)
        dm, view, rid, _ = await self.relayed_with_view()
        with patch.object(relay_router.NotNowButton, "pressed", AsyncMock()) as pressed:
            await self.dispatch(_interaction_payload(
                3, {"custom_id": f"fritz:relay:notnow:{rid}", "component_type": 2},
                message=self.message_payload(dm.id, view)))
            for _ in range(20):
                await asyncio.sleep(0)
        pressed.assert_awaited_once()
        self.assertEqual(self.answers, [], "the backstop answered a press that was handled")

    async def test_a_form_sent_after_a_restart_is_still_carried(self):
        """discord.py keeps an open form only in memory. This bot has never
        opened one, exactly as after a restart, and the reply still goes."""
        relay_router.install(self.bot)
        _, _, rid, _ = await self.relayed_with_view()
        await self.dispatch(_interaction_payload(5, {
            "custom_id": f"fritz:relay:form:{rid}",
            "components": [{"type": 1, "components": [
                {"type": 4, "custom_id": "words", "value": "see you at 8"}]}]}))
        for _ in range(200):          # mark_sent goes through the worker pool
            await asyncio.sleep(0.01)
            if any(kind == "followup" for kind, _, _ in self.answers):
                break
        self.assertEqual(self.carried().description, "see you at 8")
        self.assertEqual(self.answers[0][0], "defer")
        self.assertEqual(self.answers[0][2], {"ephemeral": True, "thinking": True})
        self.assertIn("Carried back", self.answers[-1][1][0])

    async def test_an_unregistered_button_is_answered_and_logged(self):
        """Without the backstop, a lost registration fails on every DM ever
        sent with nothing in the log to say why."""
        self.bot.add_listener(relay_router.on_interaction, "on_interaction")
        dm, view, rid, _ = await self.relayed_with_view()
        with self.assertLogs(relay_router.logger, "ERROR") as logs:
            await self.dispatch(_interaction_payload(
                3, {"custom_id": f"fritz:relay:reply:{rid}", "component_type": 2},
                message=self.message_payload(dm.id, view)))
        self.assertIn("no handler is registered", " ".join(logs.output))
        [(kind, args, kwargs)] = self.answers
        self.assertEqual((kind, args[0], kwargs["ephemeral"]),
                         ("send_message", relay_router.UNHANDLED, True))


# ─── [Reply] ─────────────────────────────────────────────────────────────────

class TestReplyOpensTheForm(ButtonTestCase):
    async def test_it_opens_the_form_and_nothing_else(self):
        dm, _, rid, _ = await self.relayed_with_view()
        interaction = await self.press("reply", rid, dm)
        interaction.response.send_modal.assert_awaited_once()
        interaction.response.defer.assert_not_awaited()
        [form] = interaction.response.send_modal.await_args.args
        spec = form.to_dict()
        self.assertEqual(spec["custom_id"], f"fritz:relay:form:{rid}")
        self.assertLessEqual(len(spec["title"]), 45)
        [row] = spec["components"]
        [text] = row["components"]
        self.assertEqual((text["custom_id"], text["style"], text["required"]),
                         ("words", discord.TextStyle.paragraph.value, True))
        self.assertLessEqual(len(text["label"]), 45)
        self.assertLessEqual(len(text["placeholder"]), 100)
        self.assertEqual(text["max_length"], 1000)          # the harness's body cap

    async def test_it_never_waits_on_the_worker_pool(self):
        """A form must be the press's FIRST response, so this cannot defer,
        and the pool can be full of model turns for minutes."""
        dm, _, rid, _ = await self.relayed_with_view()
        with patch.object(relay_router, "run_blocking",
                          AsyncMock(side_effect=AssertionError("used the pool"))):
            interaction = await self.press("reply", rid, dm)
        interaction.response.send_modal.assert_awaited_once()

    async def test_the_form_never_asks_for_more_than_discord_allows(self):
        dm, _, rid, _ = await self.relayed_with_view()
        with patch.object(fritz_utils, "RELAY_MAX_BODY_CHARS", 9000):
            interaction = await self.press("reply", rid, dm)
        [form] = interaction.response.send_modal.await_args.args
        self.assertEqual(form.to_dict()["components"][0]["components"][0]["max_length"], 4000)

    async def test_nobody_else_can_open_it(self):
        dm, _, rid, _ = await self.relayed_with_view()
        interaction = await self.press("reply", rid, dm, user_id=STRANGER_SNOWFLAKE)
        self.assertEqual(self.told(interaction), relay_router.NOT_YOURS)
        interaction.response.send_modal.assert_not_awaited()

    async def test_a_button_is_bound_to_its_own_message(self):
        """A custom_id is the presser's to send; the message is Discord's."""
        first, _, _, _ = await self.relayed_with_view()
        _, _, second_rid, _ = await self.relayed_with_view()
        interaction = await self.press("reply", second_rid, first)
        self.assertEqual(self.told(interaction), relay_router.NOT_YOURS)

    async def test_a_lapsed_relay_says_so_and_loses_its_buttons(self):
        dm, _, rid, _ = await self.relayed_with_view()
        self.expire(rid)
        interaction = await self.press("reply", rid, dm)
        self.assertTrue(self.withdrew(interaction))
        self.assertEqual(self.told(interaction), relay_router.LAPSED)
        interaction.response.send_modal.assert_not_awaited()

    async def test_a_forgotten_relay_says_so_too(self):
        _, _, rid, recipient = await self.relayed_with_view()
        privacy.forget_relay(RECIPIENT)
        interaction = await self.press("reply", rid, self.as_discord_has_it(recipient))
        self.assertTrue(self.withdrew(interaction))
        self.assertEqual(self.told(interaction), relay_router.LAPSED)

    async def test_while_the_relay_is_off(self):
        dm, _, rid, _ = await self.relayed_with_view()
        with patch.object(fritz_utils, "RELAY_ENABLED", False):
            interaction = await self.press("reply", rid, dm)
        self.assertEqual(self.told(interaction), relay_router.RELAY_OFF)
        interaction.response.send_modal.assert_not_awaited()

    async def test_a_sender_with_no_discord_account_is_named_before_any_typing(self):
        dm, _, rid, _ = await self.relayed_with_view()
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE relay_messages SET sender_id = 'web-alice', "
                         "sender_account = 'web-alice'")
            conn.commit()
        interaction = await self.press("reply", rid, dm)
        self.assertEqual(self.told(interaction), relay_router.UNREACHABLE)

    async def test_an_unforeseen_failure_is_still_answered(self):
        dm, _, rid, _ = await self.relayed_with_view()
        with patch.object(relay_store, "get_by_dm_message", side_effect=RuntimeError("x")), \
             self.assertLogs(relay_router.logger, "ERROR"):
            interaction = await self.press("reply", rid, dm)
        self.assertEqual(self.told(interaction), relay_router.REPLY_FAILED)
        # A native reply goes through the same store; offering it as the way
        # round would hand the words to the agent instead.
        self.assertNotIn("repl", relay_router.REPLY_FAILED.lower())

    async def test_a_press_before_the_dm_is_bound_opens_the_form(self):
        """mark_sent waits in the worker pool, and the DM can be pressed first.
        It used to say the relay had lapsed, and take its buttons away."""
        reservation = relay_store.reserve_send(SENDER, RECIPIENT, "just sent")
        message = MagicMock(id=515151515151515151)
        interaction = await self.press("reply", reservation.id, message)
        interaction.response.send_modal.assert_awaited_once()
        self.assertFalse(self.withdrew(interaction))

    async def test_but_only_for_a_row_still_in_flight(self):
        """A delivered row is bound to its own message; a button naming it on
        any other message is not about it."""
        _, _, rid, _ = await self.relayed_with_view()
        interaction = await self.press("reply", rid, MagicMock(id=525252525252525252))
        self.assertEqual(self.told(interaction), relay_router.NOT_YOURS)

    async def test_a_message_with_no_row_that_is_not_a_relay(self):
        """NOT_YOURS, and the buttons stay: it is not ours to say it lapsed."""
        interaction = await self.press("reply", "deadbeef", MagicMock(id=535353535353535353))
        self.assertEqual(self.told(interaction), relay_router.NOT_YOURS)
        self.assertFalse(self.withdrew(interaction))

    async def test_a_linked_recipient_can_open_it(self):
        dm, _, rid, _ = await self.relayed_with_view_linked()
        with self.linked():
            interaction = await self.press("reply", rid, dm)
        interaction.response.send_modal.assert_awaited_once()

    async def test_a_link_added_after_delivery_does_not_lock_them_out(self):
        dm, _, rid, _ = await self.relayed_with_view()
        with self.linked():
            interaction = await self.press("reply", rid, dm)
        interaction.response.send_modal.assert_awaited_once()

    async def test_a_linked_recipient_of_an_older_row_is_resolved_exactly_once(self):
        """With no recipient_account, only the resolved comparison recognises
        them: resolving `me` is required, and resolving the stored side again
        is a second hop that, chained, is somebody else."""
        for chain in (False, True):
            with self.subTest(chain=chain):
                dm, _, rid, _ = await self.relayed_with_view_linked(chain, legacy=True)
                with self.linked(chain):
                    interaction = await self.press("reply", rid, dm)
                interaction.response.send_modal.assert_awaited_once()


# ─── The form comes back ─────────────────────────────────────────────────────

class TestTheFormComesBack(ButtonTestCase):
    async def test_the_words_reach_the_sender_verbatim(self):
        _, _, rid, _ = await self.relayed_with_view()
        interaction = await self.submit(rid, "**yes** — see you at 8 @everyone")
        interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
        embed = self.carried()
        self.assertEqual(embed.description, "**yes** — see you at 8 @everyone")
        self.assertEqual(embed.author.name, "@bob · Bob")
        self.assert_no_mentions(self.dm_channel.send.await_args.kwargs["allowed_mentions"])
        self.assertIn("Carried back", self.told(interaction))
        self.assertEqual(self.client.create_dm.await_args.args[0].id, SENDER_SNOWFLAKE)

    async def test_it_goes_back_with_buttons_of_its_own(self):
        _, _, rid, _ = await self.relayed_with_view()
        await self.submit(rid, "yes")
        back = relay_store.get_by_dm_message(self.sent_back.id)
        self.assertEqual(back["origin_id"], rid)
        self.assertEqual([c.custom_id for c in self.dm_channel.send.await_args.kwargs["view"].children],
                         [f"fritz:relay:{k}:{back['id']}" for k in KINDS])

    async def test_both_shapes_of_form_are_read(self):
        _, _, rid, _ = await self.relayed_with_view()
        await self.submit(rid, "through a label", shape="label")
        self.assertEqual(self.carried().description, "through a label")

    async def test_whitespace_is_no_words(self):
        _, _, rid, _ = await self.relayed_with_view()
        interaction = await self.submit(rid, "   \n\t ")
        self.assertEqual(self.told(interaction), relay_router.NO_WORDS)
        self.dm_channel.send.assert_not_awaited()

    async def test_everything_is_checked_again_when_it_comes_back(self):
        """The form can sit open for minutes while the relay lapses."""
        _, _, rid, _ = await self.relayed_with_view()
        self.expire(rid)
        interaction = await self.submit(rid, "sorry, only just saw this")
        self.assertEqual(self.told(interaction), relay_router.LAPSED)
        self.dm_channel.send.assert_not_awaited()

    async def test_or_is_forgotten(self):
        _, _, rid, _ = await self.relayed_with_view()
        privacy.forget_relay(SENDER)
        interaction = await self.submit(rid, "hello?")
        self.assertEqual(self.told(interaction), relay_router.LAPSED)
        self.dm_channel.send.assert_not_awaited()

    async def test_a_forged_form_for_someone_elses_relay_carries_nothing(self):
        """The form's custom_id comes back from the client."""
        _, _, rid, _ = await self.relayed_with_view()
        with self.assertLogs(relay_router.logger, "WARNING"):
            forged = await self.submit(rid, "gotcha", user_id=STRANGER_SNOWFLAKE)
        self.dm_channel.send.assert_not_awaited()
        # ...and reads exactly like a guess at an id that does not exist, so
        # guessing says nothing about which relays there are.
        guessed = await self.submit("deadbeef", "gotcha", user_id=STRANGER_SNOWFLAKE)
        self.assertEqual(self.told(forged), self.told(guessed))

    async def test_a_linked_recipient_can_send_it(self):
        for chain in (False, True):
            with self.subTest(chain=chain):
                self.dm_channel.send.reset_mock()
                _, _, rid, _ = await self.relayed_with_view_linked(chain)
                with self.linked(chain):
                    interaction = await self.submit(rid, "yes")
                self.assertIn("Carried back", self.told(interaction))

    async def test_a_link_added_after_delivery_does_not_lock_them_out(self):
        _, _, rid, _ = await self.relayed_with_view()
        with self.linked():
            interaction = await self.submit(rid, "yes")
        self.assertIn("Carried back", self.told(interaction))

    async def test_a_linked_recipient_of_an_older_row_can_send_it(self):
        for chain in (False, True):
            with self.subTest(chain=chain):
                self.dm_channel.send.reset_mock()
                _, _, rid, _ = await self.relayed_with_view_linked(chain, legacy=True)
                with self.linked(chain):
                    interaction = await self.submit(rid, "yes")
                self.assertIn("Carried back", self.told(interaction))

    async def test_it_is_acknowledged_before_the_worker_pool(self):
        _, _, rid, _ = await self.relayed_with_view()
        interaction = await self.submit_watching_the_pool(rid, "yes")
        self.assertIn("Carried back", self.told(interaction))

    async def submit_watching_the_pool(self, rid, words):
        real = relay_router.run_blocking
        holder = {}

        async def watched(fn, *args, **kwargs):
            self.assertTrue(holder["interaction"].response.is_done(),
                            f"{getattr(fn, '__name__', fn)} ran before the answer")
            return await real(fn, *args, **kwargs)

        original_press_as = self.press_as

        def capture(**kw):
            holder["interaction"] = original_press_as(**kw)
            return holder["interaction"]

        with patch.object(relay_router, "run_blocking", watched), \
             patch.object(self, "press_as", capture):
            return await self.submit(rid, words)

    async def test_a_form_for_no_relay_at_all(self):
        interaction = await self.submit("deadbeef", "hello")
        self.assertEqual(self.told(interaction), relay_router.LAPSED)
        self.client.create_dm.assert_not_awaited()

    async def test_a_block_on_the_way_back_reads_exactly_like_closed_dms(self):
        _, _, rid, _ = await self.relayed_with_view()
        relay_store.block(SENDER, RECIPIENT)
        interaction = await self.submit(rid, "hello again")
        self.assertEqual(self.told(interaction), relay_store.REFUSED)
        self.dm_channel.send.assert_not_awaited()

    async def test_the_caps_still_apply(self):
        _, _, rid, _ = await self.relayed_with_view()
        with patch.object(fritz_utils, "RELAY_MAX_PER_SENDER_PER_HOUR", 1):
            await self.submit(rid, "one")
            second = await self.submit(rid, "two")
        self.assertIn("limit", self.told(second))
        self.assertEqual(self.dm_channel.send.await_count, 1)

    async def test_while_the_relay_is_off(self):
        _, _, rid, _ = await self.relayed_with_view()
        with patch.object(fritz_utils, "RELAY_ENABLED", False):
            interaction = await self.submit(rid, "hello")
        self.assertEqual(self.told(interaction), relay_router.RELAY_OFF)
        self.client.create_dm.assert_not_awaited()

    async def test_an_unforeseen_failure_is_still_answered(self):
        _, _, rid, _ = await self.relayed_with_view()
        with patch.object(relay_router, "_answer_relay", AsyncMock(side_effect=RuntimeError("x"))), \
             self.assertLogs(relay_router.logger, "ERROR"):
            interaction = await self.submit(rid, "hello")
        self.assertEqual(self.told(interaction), relay_router.SUBMIT_FAILED)

    async def test_other_forms_and_components_are_left_alone(self):
        """The prefix is the only thing between the backstop and every other
        button in the bot: /forget all's Confirm, /relay forget-blocks'."""
        self.client._connection._view_store._dynamic_items = (
            main_discord.client._connection._view_store._dynamic_items)
        for kind, custom_id in (
                (discord.InteractionType.modal_submit, "someone-elses-modal"),
                (discord.InteractionType.modal_submit, "fritz:relay:form:xyz"),
                (discord.InteractionType.modal_submit, "fritz:relay:form:"),
                (discord.InteractionType.component, "0123456789abcdef0123456789abcdef"),
                (discord.InteractionType.component, "someone-elses-button")):
            with self.subTest(kind=kind, custom_id=custom_id), \
                    self.assertNoLogs(relay_router.logger, "ERROR"):
                interaction = self.press_as(kind=kind,
                                            data={"custom_id": custom_id, "components": []})
                await relay_router.on_interaction(interaction)
                self.assertEqual(interaction.response.used, [])
                interaction.followup.send.assert_not_awaited()


# ─── [Not now] ───────────────────────────────────────────────────────────────

class TestNotNow(ButtonTestCase):
    async def test_the_buttons_go_and_the_sender_hears_nothing(self):
        dm, _, rid, _ = await self.relayed_with_view()
        before = (self.rows(), self.audit_events())
        interaction = await self.press("notnow", rid, dm)
        self.assertTrue(self.withdrew(interaction))
        self.assertEqual(self.told(interaction), relay_router.NOT_NOW)
        self.assertEqual((self.rows(), self.audit_events()), before)
        self.client.create_dm.assert_not_awaited()

    async def test_with_the_relay_off_it_does_not_suggest_replying(self):
        """A reply would reach no one: with the relay off it goes to the agent."""
        dm, _, rid, _ = await self.relayed_with_view()
        with patch.object(fritz_utils, "RELAY_ENABLED", False):
            interaction = await self.press("notnow", rid, dm)
        self.assertEqual(self.told(interaction), relay_router.NOT_NOW_OFF)
        self.assertNotIn("reply", relay_router.NOT_NOW_OFF.lower())
        self.assertTrue(self.withdrew(interaction))

    async def test_the_message_can_still_be_answered_afterwards(self):
        """Not now, not never."""
        dm, _, rid, _ = await self.relayed_with_view()
        await self.press("notnow", rid, dm)
        await self.route(self.reply_to(dm, "actually, yes"))
        self.assertEqual(self.carried().description, "actually, yes")


# ─── [Block sender] ──────────────────────────────────────────────────────────

class TestBlockSender(ButtonTestCase):
    async def test_one_press_blocks_them(self):
        dm, _, rid, _ = await self.relayed_with_view(self.interaction(display="Mal"))
        interaction = await self.press("block", rid, dm)
        self.assertTrue(self.withdrew(interaction))
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [SENDER])
        text = self.told(interaction)
        self.assertIn("will not be told", text)
        self.assertIn("@alice", text)
        self.assertEqual(relay_store.reserve_send(SENDER, RECIPIENT, "again").reason,
                         "blocked_sender")

    async def test_it_answers_exactly_as_the_context_menu_does(self):
        dm, _, rid, _ = await self.relayed_with_view()
        button = self.told(await self.press("block", rid, dm))
        relay_store.unblock(RECIPIENT, SENDER)
        menu = self.interaction(user_id=RECIPIENT_SNOWFLAKE, name="bob")
        await self.cog.block_sender_from_message(menu, dm)
        self.assertEqual(menu.followup.send.await_args.args[0], button)

    async def test_it_blocks_the_account_that_wrote_not_its_main(self):
        alt, main = 777777777777777777, "discord-888888888888888888"
        with patch.object(fritz_utils, "IDENTITY_LINKS", {f"discord-{alt}": main}):
            dm, _, rid, _ = await self.relayed_with_view(
                self.interaction(user_id=alt, name="mallory", display="mallory"))
            interaction = await self.press("block", rid, dm)
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [f"discord-{alt}"])
        self.assertNotIn("888888888888888888", self.told(interaction))

    async def test_it_works_while_the_relay_is_off(self):
        """People may refuse in advance."""
        dm, _, rid, _ = await self.relayed_with_view()
        with patch.object(fritz_utils, "RELAY_ENABLED", False):
            await self.press("block", rid, dm)
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [SENDER])

    async def test_nobody_else_can(self):
        dm, _, rid, _ = await self.relayed_with_view()
        interaction = await self.press("block", rid, dm, user_id=STRANGER_SNOWFLAKE)
        self.assertEqual(self.told(interaction), relay_router.NOT_YOURS)
        self.assertEqual(relay_store.list_blocks(f"discord-{STRANGER_SNOWFLAKE}"), [])
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [])

    async def test_a_relay_whose_row_has_gone(self):
        _, _, rid, recipient = await self.relayed_with_view()
        privacy.forget_relay(RECIPIENT)
        interaction = await self.press("block", rid, self.as_discord_has_it(recipient))
        self.assertEqual(self.told(interaction), relay_router.BLOCK_GONE)
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [])

    async def test_a_sender_who_forgot_can_still_be_blocked(self):
        """PR 6's anchor, reached from the button."""
        dm, _, rid, _ = await self.relayed_with_view()
        privacy.forget_relay(SENDER)
        await self.press("block", rid, dm)
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [SENDER])

    async def test_the_audit_log_says_who_blocked_never_whom(self):
        dm, _, rid, _ = await self.relayed_with_view()
        await self.press("block", rid, dm)
        [event] = [e for e in self.audit_events() if e["event"] == "relay_block"]
        self.assertEqual(event["user_id"], RECIPIENT)
        self.assertNotIn(str(SENDER_SNOWFLAKE), str(event))

    async def test_a_store_failure_is_still_answered(self):
        dm, _, rid, _ = await self.relayed_with_view()
        with patch.object(relay_store, "block", side_effect=RuntimeError("locked")), \
             self.assertLogs(relay_router.logger, "ERROR"):
            interaction = await self.press("block", rid, dm)
        self.assertEqual(self.told(interaction), relay_router.BLOCK_FAILED)

    async def test_a_press_before_the_dm_is_bound_still_blocks(self):
        """Named as the DM named them: the row has not recorded it yet."""
        reservation = relay_store.reserve_send(SENDER, RECIPIENT, "just sent")
        message = MagicMock(spec=discord.Message)
        message.id = 545454545454545454
        message.author = MagicMock(id=BOT_ID)
        message.embeds = [relay_format.relay_embed(
            "just sent", shown_as="@alice \u00b7 Mal", icon_url=None,
            footer=relay_format.FOOTER_RELAYED + " X.")]
        interaction = await self.press("block", reservation.id, message)
        self.assertIn("@alice", self.told(interaction))
        [entry] = relay_store.list_block_entries(RECIPIENT)
        self.assertEqual((entry["blocked_id"], entry["label"]), (SENDER, "@alice \u00b7 Mal"))

    async def test_a_message_with_no_row_that_is_not_a_relay(self):
        interaction = await self.press("block", "deadbeef", MagicMock(id=555555555555555001))
        self.assertEqual(self.told(interaction), relay_router.NOT_YOURS)
        self.assertEqual(relay_store.list_blocks(RECIPIENT), [])

    async def test_a_linked_recipient_can_block(self):
        await self.assert_a_linked_recipient_can_block(chain=False)

    async def test_a_chained_recipient_can_block(self):
        await self.assert_a_linked_recipient_can_block(chain=True)

    async def assert_a_linked_recipient_can_block(self, chain):
        dm, _, rid, _ = await self.relayed_with_view_linked(chain)
        with self.linked(chain):
            interaction = await self.press("block", rid, dm)
            self.assertEqual(relay_store.list_blocks(RECIPIENT), [SENDER])
        self.assertIn("will not be told", self.told(interaction))

    async def test_a_link_added_after_delivery_does_not_lock_them_out(self):
        dm, _, rid, _ = await self.relayed_with_view()
        with self.linked():
            interaction = await self.press("block", rid, dm)
        self.assertIn("will not be told", self.told(interaction))

    async def test_it_is_acknowledged_before_the_worker_pool(self):
        dm, _, rid, _ = await self.relayed_with_view()
        interaction = self.press_as(message=dm)
        real = relay_router.run_blocking

        async def watched(fn, *args, **kwargs):
            self.assertTrue(interaction.response.is_done(),
                            f"{getattr(fn, '__name__', fn)} ran before the answer")
            return await real(fn, *args, **kwargs)

        with patch.object(relay_router, "run_blocking", watched):
            await relay_router.BlockButton(rid).callback(interaction)
        self.assertIn("will not be told", self.told(interaction))

    async def test_if_the_buttons_cannot_be_removed_it_still_answers_in_time(self):
        dm, _, rid, _ = await self.relayed_with_view()
        interaction = self.press_as(message=dm)
        interaction.response.fail.add("edit_message")
        await relay_router.BlockButton(rid).callback(interaction)
        self.assertEqual(interaction.response.used[0][0], "defer")
        self.assertIn("will not be told", self.told(interaction))


class TestOneBlockCore(unittest.IsolatedAsyncioTestCase):
    def test_bot_commands_uses_the_routers_copy(self):
        """The menu, /relay block and the button must say the same thing."""
        self.assertIs(bot_commands._relay_sender, relay_router.relay_sender)
        self.assertIs(bot_commands._BLOCK_GONE, relay_router.BLOCK_GONE)
        self.assertIs(bot_commands._say, relay_format.block_label)
        src = (REPO / "bot_commands.py").read_text(encoding="utf-8")
        body = src.split("    async def _block(", 1)[1].split("\n    @", 1)[0]
        self.assertIn("relay_router.place_block(", body)


if __name__ == "__main__":
    unittest.main()
