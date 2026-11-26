"""His mood tag (plan 14 PR 11): asked for on the desk's turns, taken off
every reply wherever it goes, and told to whoever asked for the mood.

mood_tags itself (the tag read off a whole reply, and off one as it streams,
cut anywhere), _DeltaEmitter holding it back, and ask_stuff end to end: a
real ReAct agent over a scripted model, the streamed deltas, the mood
callback, the returned text and what was checkpointed.
"""
import itertools
import logging
import threading
import typing
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool

import companion_protocol
import mister_fritz
import mood_tags
from fritz_utils import MessageSource
from mister_fritz import INTERRUPTED_MARK
from mood_tags import ASK, MOODS, LeadingTag, strip_leading

_THREADS = itertools.count(1)

# (reply, its mood, what is shown of it)
CASES = [
    ("[mood:pleased] Very good, sir.", "pleased", "Very good, sir."),
    ("[mood:dry]\nGood evening.", "dry", "Good evening."),
    ("  \n[mood:weary]  \n\nAnother Monday.", "weary", "Another Monday."),
    ("[MOOD:Alarmed] Smoke!", "alarmed", "Smoke!"),
    ("[ mood : dry ] Quite.", "dry", "Quite."),
    ("[mood:dry]", "dry", ""),
    ("[mood:dry]\n\n", "dry", ""),
    # A tag naming no mood of his is dropped all the same, whatever its name.
    ("[mood:happy] Splendid.", None, "Splendid."),
    ("[mood:very-happy] Hi.", None, "Hi."),
    ("[mood:very happy] Hi.", None, "Hi."),
    ("[mood:dry_ish]\nHm.", None, "Hm."),
    # A Markdown link's text is not a tag; a bracket after a space is.
    ("[Mood: Chill](https://example.invalid/playlist) for tonight, sir.", None,
     "[Mood: Chill](https://example.invalid/playlist) for tonight, sir."),
    ("[mood:dry](https://example.invalid)", None, "[mood:dry](https://example.invalid)"),
    ("[mood:dry] (an aside)", "dry", "(an aside)"),
    # Only a leading one, and only one.
    ("Good evening. [mood:dry]", None, "Good evening. [mood:dry]"),
    ("[mood:dry][mood:pleased] Hm.", "dry", "[mood:pleased] Hm."),
    # Not tags.
    ("[1] The first source.", None, "[1] The first source."),
    ("[mood] is a word.", None, "[mood] is a word."),
    ("[mood:] Empty.", None, "[mood:] Empty."),
    ("[mood:2] Two.", None, "[mood:2] Two."),
    ("[mood:" + "x" * 17 + "] Too long.", None, "[mood:" + "x" * 17 + "] Too long."),
    ("Plain reply.", None, "Plain reply."),
    # A reply's own leading whitespace is its own business, without a tag.
    ("\nPlain.", None, "\nPlain."),
    ("   ", None, "   "),
    ("", None, ""),
    # A tag never finished is dropped.
    ("[mood:pleas", None, ""),
    ("[mo", None, ""),
]


class TestTheTag(unittest.TestCase):
    def test_his_moods_are_the_protocols(self):
        self.assertEqual(MOODS, typing.get_args(companion_protocol.MoodName))
        self.assertEqual(MOODS, ("dry", "pleased", "alarmed", "weary"))

    def test_the_ask_names_each_tag_once(self):
        for mood in MOODS:
            self.assertEqual(ASK.count(f"[mood:{mood}]"), 1, mood)
        self.assertEqual(ASK.count("[mood:"), len(MOODS))
        self.assertIn("never shown or spoken", ASK)
        # The wording gpt-oss was probed with (12 of 12 began with a tag).
        self.assertEqual(ASK, "Begin your reply with exactly one mood tag for your face, one of "
                              "[mood:dry], [mood:pleased], [mood:alarmed] or [mood:weary], then "
                              "your reply. The tag is never shown or spoken.")

    def test_a_whole_reply(self):
        for text, mood, shown in CASES:
            with self.subTest(text=text):
                self.assertEqual(strip_leading(text), (mood, shown))

    def test_streamed_cut_anywhere_it_is_the_same(self):
        """Every reply cut into two or three pieces at every place, and one
        character at a time: what is shown, and the mood, never change."""
        for text, mood, shown in CASES:
            cuts = [[text[:i], text[i:j], text[j:]] for i in range(len(text) + 1)
                    for j in range(i, len(text) + 1)]
            cuts.append(list(text))
            for pieces in cuts:
                with self.subTest(text=text, pieces=pieces):
                    tag = LeadingTag()
                    out = "".join(tag.feed(p) for p in pieces) + tag.end()
                    self.assertEqual((tag.mood, out), (mood, shown))

    def test_nothing_of_a_tag_is_let_go_on_its_way_past(self):
        tag = LeadingTag()
        for piece in ("[", "mo", "od:", "plea", "sed", "]", " \n"):
            self.assertEqual(tag.feed(piece), "", piece)
        self.assertEqual(tag.mood, "pleased")
        self.assertEqual(tag.feed("Very"), "Very")
        self.assertEqual(tag.feed(" good"), " good")
        self.assertEqual(tag.end(), "")

    def test_the_mood_is_known_at_the_character_after_the_tag(self):
        """Which says whether it was a link's text."""
        tag = LeadingTag()
        tag.feed("[mood:weary")
        tag.feed("]")
        self.assertIsNone(tag.mood)
        self.assertEqual(tag.feed(" "), "")
        self.assertEqual(tag.mood, "weary")

    def test_or_when_the_part_ends_on_it(self):
        tag = LeadingTag()
        self.assertEqual(tag.feed("[mood:weary]"), "")
        self.assertEqual(tag.end(), "")
        self.assertEqual(tag.mood, "weary")

    def test_text_that_is_plainly_no_tag_is_let_go_at_once(self):
        tag = LeadingTag()
        self.assertEqual(tag.feed("Good"), "Good")
        tag = LeadingTag()
        self.assertEqual(tag.feed("["), "")
        self.assertEqual(tag.feed("1] see"), "[1] see")
        tag = LeadingTag()
        self.assertEqual(tag.feed("[mood"), "")
        self.assertEqual(tag.feed("y"), "[moody")

    def test_what_is_held_is_bounded(self):
        tag = LeadingTag()
        self.assertEqual(tag.feed("[mood:" + "a" * 16), "")
        self.assertEqual(tag.feed("a"), "[mood:" + "a" * 17)
        # The longest that may yet be a tag, spaced as far as it may be: 34
        # characters, then its bracket and the character after it.
        longest = "[   mood   :   " + "a" * 16 + "   "
        self.assertEqual(len(longest), 34)
        tag = LeadingTag()
        self.assertEqual(tag.feed(longest), "")
        self.assertEqual(tag.feed("]"), "")
        self.assertEqual(tag.feed("("), longest + "](")
        tag = LeadingTag()
        self.assertEqual(tag.feed(longest), "")
        self.assertEqual(tag.feed("b"), longest + "b")

    def test_once_ended_the_rest_passes_as_it_is(self):
        tag = LeadingTag()
        tag.feed("[mood:pl")
        self.assertEqual(tag.end(), "")
        self.assertEqual(tag.feed("[mood:dry] I stopped."), "[mood:dry] I stopped.")
        self.assertIsNone(tag.mood)


class TestTheDeskAsksForIt(unittest.TestCase):
    def test_only_the_desks_turns_ask(self):
        for source in MessageSource:
            with self.subTest(source=source):
                line = mister_fritz.get_source_info(source, "erin", "desktop-erin-pc")
                asks = source in (MessageSource.DESKTOP_TEXT, MessageSource.DESKTOP_VOICE)
                self.assertEqual(line.endswith(ASK), asks, line)
                self.assertEqual("[mood:" in line, asks, line)

    def test_a_spoken_turn_is_still_asked_to_be_brief(self):
        line = mister_fritz.get_source_info(MessageSource.DESKTOP_VOICE, "erin", "desktop-erin-pc")
        self.assertIn("30 words or less. Begin your reply", line)


class TestTheEmitterHoldsItBack(unittest.TestCase):
    """_DeltaEmitter: a segment's leading tag never reaches the stream, and
    its mood is told before the segment's first emission."""

    def setUp(self):
        self.events = []

    def emitter(self, min_chars=1, streaming=True, on_mood=True):
        return mister_fritz._DeltaEmitter(
            (lambda d, a, r: self.events.append(("token", d, a, r))) if streaming else None,
            min_chars=min_chars,
            on_mood=(lambda m: self.events.append(("mood", m))) if on_mood else None)

    def test_a_tag_cut_across_chunks_is_never_emitted(self):
        em = self.emitter()
        for piece in ("[mo", "od:plea", "sed]", " ", "Very", " good"):
            em.feed(piece, segment_id="s")
        em.finish()
        self.assertEqual(self.events, [
            ("mood", "pleased"),
            ("token", "Very", "Very", True),
            ("token", " good", "Very good", False),
        ])

    def test_each_segment_is_read_for_its_own(self):
        em = self.emitter()
        em.feed("[mood:alarmed] Let me look.", segment_id="a")
        em.feed("\n[mood:pleased]\n\nFound it.", segment_id="b")
        em.finish()
        self.assertEqual(self.events, [
            ("mood", "alarmed"),
            ("token", "Let me look.", "Let me look.", True),
            ("mood", "pleased"),
            ("token", "Found it.", "Found it.", True),
        ])

    def test_a_segment_that_is_only_a_tag_emits_nothing(self):
        em = self.emitter()
        em.feed("[mood:weary]", segment_id="a")
        em.feed("Done.", segment_id="b")
        em.finish()
        self.assertEqual(self.events, [("mood", "weary"), ("token", "Done.", "Done.", True)])

    def test_a_last_segment_that_is_only_a_tag_tells_it_at_the_end(self):
        em = self.emitter()
        em.feed("Let me look.", segment_id="a")
        em.feed("[mood:pleased]", segment_id="b")
        self.assertEqual(self.events, [("token", "Let me look.", "Let me look.", True)])
        em.finish()
        self.assertEqual(self.events[-1], ("mood", "pleased"))

    def test_whitespace_held_at_a_segments_start_is_let_go_as_the_next_begins(self):
        em = self.emitter()
        em.feed("  ", segment_id="a")
        self.assertEqual(self.events, [])
        em.feed("Hi.", segment_id="b")
        self.assertEqual(self.events, [("token", "  ", "  ", True), ("token", "Hi.", "Hi.", True)])

    def test_the_mood_is_told_once_a_segment(self):
        em = self.emitter()
        for piece in ("[mood:dry]", " Quite", " so", "."):
            em.feed(piece, segment_id="s")
        self.assertEqual([e for e in self.events if e[0] == "mood"], [("mood", "dry")])

    def test_a_tag_naming_no_mood_is_dropped_and_tells_nothing(self):
        em = self.emitter()
        em.feed("[mood:happy] Splendid.", segment_id="s")
        self.assertEqual(self.events, [("token", "Splendid.", "Splendid.", True)])

    def test_buffered_it_still_comes_out_whole(self):
        em = self.emitter(min_chars=50)
        for piece in ("[mood:dry]\n", "Good", " evening", "."):
            em.feed(piece, segment_id="s")
        self.assertEqual(self.events, [("mood", "dry")])
        em.finish()
        self.assertEqual(self.events, [("mood", "dry"), ("token", "Good evening.", "Good evening.", True)])

    def test_finishing_drops_a_tag_never_finished(self):
        em = self.emitter()
        em.feed("[mood:pl", segment_id="s")
        em.finish()
        self.assertEqual(self.events, [])

    def test_finishing_lets_go_of_whitespace_alone(self):
        em = self.emitter()
        em.feed("  ", segment_id="s")
        self.assertEqual(self.events, [])
        em.finish()
        self.assertEqual(self.events, [("token", "  ", "  ", True)])

    def test_words_added_after_a_tag_never_finished_show_no_fragment(self):
        em = self.emitter()
        em.feed("[mood:pl", segment_id="s")
        em.end_with("I stopped.")
        self.assertEqual(self.events, [("token", "I stopped.", "I stopped.", True)])

    def test_words_added_after_a_reply_follow_what_was_shown(self):
        em = self.emitter()
        em.feed("[mood:dry] Let me try that again.", segment_id="s")
        em.end_with("I stopped.")
        self.assertEqual(self.events[-1],
                         ("token", "\n\nI stopped.", "Let me try that again.\n\nI stopped.", False))

    def test_words_added_after_only_a_tag_are_the_whole_of_it(self):
        em = self.emitter()
        em.feed("[mood:dry]\n", segment_id="s")
        em.end_with("I stopped.")
        self.assertEqual(self.events, [("mood", "dry"), ("token", "I stopped.", "I stopped.", True)])

    def test_the_mood_is_told_without_a_stream_to_show(self):
        em = self.emitter(streaming=False)
        em.feed("[mood:alarmed] Smoke!", segment_id="s")
        em.finish()
        self.assertEqual(self.events, [("mood", "alarmed")])

    def test_without_a_stream_nothing_piles_up(self):
        em = self.emitter(min_chars=1000, streaming=False)
        for _ in range(5):
            em.feed("[mood:dry] words words words", segment_id="s")
            em.flush()
            self.assertEqual(em._buffer, "")

    def test_a_broken_mood_consumer_does_not_stop_the_reply(self):
        def boom(_mood):
            raise RuntimeError("consumer exploded")
        tokens = []
        em = mister_fritz._DeltaEmitter(lambda d, a, r: tokens.append(d), min_chars=1, on_mood=boom)
        with self.assertLogs(mister_fritz.logger, logging.WARNING) as logs:
            em.feed("[mood:dry] Quite.", segment_id="s")
        self.assertEqual(tokens, ["Quite."])
        self.assertIn("mood callback raised", logs.output[0])

    def test_without_a_mood_consumer_the_tag_is_still_taken_off(self):
        em = self.emitter(on_mood=False)
        em.feed("[mood:dry] Quite.", segment_id="s")
        self.assertEqual(self.events, [("token", "Quite.", "Quite.", True)])


class _Record:
    def __init__(self):
        self.calls = 0


class _Scripted(BaseChatModel):
    """Answers call N with `turns[N]`: a list of chunks, and a tool call
    after them if `tool` is set for that call."""

    record: Any
    turns: list
    tool_on: tuple = ()
    cancel_after: int | None = None
    cancel_event: Any = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = ""
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            text += chunk.text
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        n = self.record.calls
        self.record.calls += 1
        for piece in self.turns[n]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
        if n in self.tool_on:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": "lookup", "args": "{}", "id": f"call_{n}", "index": 0, "type": "tool_call_chunk"},
            ]))


@tool
def lookup() -> str:
    """Look something up."""
    return "found"


class TestAskStuffEndToEnd(unittest.TestCase):
    """The real ask_stuff and executor over a real ReAct agent."""

    def setUp(self):
        self.thread = f"mood-test-{next(_THREADS)}"
        for name, value in (("search_memories_internal", MagicMock(return_value="{}")),
                            ("get_user_profile", MagicMock(return_value={})),
                            ("extract_memories_background", MagicMock()),
                            ("STREAM_MIN_CHARS", 1)):
            p = patch.object(mister_fritz, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.events = []

    def ask(self, model, tools=(), source=MessageSource.DESKTOP_TEXT, **kwargs):
        agent = create_agent(model, tools=list(tools), middleware=mister_fritz._agent_middleware())
        box = {}

        def run():
            box["result"] = mister_fritz.ask_stuff(
                "good evening", source, "local-moodtest", thread_id=self.thread,
                streaming_callback=lambda d, a, r: self.events.append(("token", d, r)),
                mood_callback=lambda m: self.events.append(("mood", m)), **kwargs)

        with patch.object(mister_fritz, "_get_conversation_agent", return_value=agent):
            t = threading.Thread(target=run, daemon=True)
            t.start()
            t.join(30)
        self.assertFalse(t.is_alive(), "ask_stuff did not return")
        return box["result"]

    def stored_reply(self):
        messages = mister_fritz.app.get_state(
            {"configurable": {"thread_id": self.thread}}).values.get("messages", [])
        return messages[-1].content

    def test_the_tag_is_told_and_never_shown_stored_or_returned(self):
        model = _Scripted(record=_Record(), turns=[["[mo", "od:plea", "sed]\n", "Good evening", ", sir."]])
        result = self.ask(model)
        self.assertEqual(self.events, [("mood", "pleased"), ("token", "Good evening", True),
                                       ("token", ", sir.", False)])
        self.assertEqual(result["text"], "Good evening, sir.")
        self.assertEqual(self.stored_reply(), "Good evening, sir.")

    def test_a_reply_in_parts_tells_each_parts_mood_before_it(self):
        model = _Scripted(record=_Record(), tool_on=(0,),
                          turns=[["[mood:alarmed] ", "Let me look."], ["\n[mood:pleased]\n\n", "Found it, sir."]])
        result = self.ask(model, tools=[lookup])
        self.assertEqual(self.events, [("mood", "alarmed"), ("token", "Let me look.", True),
                                       ("mood", "pleased"), ("token", "Found it, sir.", True)])
        self.assertEqual(result["text"], "Found it, sir.")
        self.assertEqual(self.stored_reply(), "Found it, sir.")

    def test_what_is_held_at_the_end_of_the_stream_is_settled(self):
        """Whitespace at a part's start is held, as a tag might follow; when
        the model's stream ends there, it is let go, as it was before tags."""
        model = _Scripted(record=_Record(), tool_on=(0,), turns=[["Let me look."], ["  "]])
        self.ask(model, tools=[lookup])
        self.assertEqual(self.events, [("token", "Let me look.", True), ("token", "  ", True)])

    def test_a_reply_without_one_is_as_it_was(self):
        model = _Scripted(record=_Record(), turns=[["Good evening", ", sir."]])
        result = self.ask(model)
        self.assertEqual(self.events, [("token", "Good evening", True), ("token", ", sir.", False)])
        self.assertEqual(result["text"], "Good evening, sir.")

    def test_any_source_loses_it(self):
        """Discord's turns are not asked for one, but a reply that has one
        anyway (the model copying what it has seen) loses it."""
        model = _Scripted(record=_Record(), turns=[["[mood:dry] ", "Quite."]])
        result = self.ask(model, source=MessageSource.DISCORD_TEXT)
        self.assertEqual(result["text"], "Quite.")
        self.assertEqual(self.stored_reply(), "Quite.")

    def test_without_a_stream_the_returned_and_stored_reply_lose_it(self):
        model = _Scripted(record=_Record(), turns=[["[mood:dry] ", "Quite."]])
        agent = create_agent(model, tools=[], middleware=mister_fritz._agent_middleware())
        with patch.object(mister_fritz, "_get_conversation_agent", return_value=agent):
            result = mister_fritz.ask_stuff("hi", MessageSource.DISCORD_TEXT, "local-moodtest",
                                            thread_id=self.thread)
        self.assertEqual(result["text"], "Quite.")
        self.assertEqual(self.stored_reply(), "Quite.")

    def test_an_interrupted_reply_keeps_what_was_shown(self):
        event = threading.Event()
        said = []

        def callback(delta, accumulated, restart):
            said.append(delta)
            if len(said) == 2:
                event.set()

        model = _Scripted(record=_Record(), tool_on=(0,),
                          turns=[["[mood:alarmed] ", "Let me look."],
                                 ["[mood:pleased] ", "Found", " it", ", sir."]])
        agent = create_agent(model, tools=[lookup], middleware=mister_fritz._agent_middleware())
        with patch.object(mister_fritz, "_get_conversation_agent", return_value=agent):
            result = mister_fritz.ask_stuff("hi", MessageSource.DESKTOP_TEXT, "local-moodtest",
                                            thread_id=self.thread, streaming_callback=callback,
                                            cancel_event=event)
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["text"], "Let me look.\n\nFound" + INTERRUPTED_MARK)
        self.assertEqual(self.stored_reply(), "Let me look.\n\nFound" + INTERRUPTED_MARK)

    def test_only_a_mood_consumer_still_streams(self):
        """The tag is read from the tokens, so a caller that wants the mood
        and not the words still has the agent stream them."""
        model = _Scripted(record=_Record(), turns=[["[mood:weary] ", "Monday."]])
        agent = create_agent(model, tools=[], middleware=mister_fritz._agent_middleware())
        with patch.object(mister_fritz, "_get_conversation_agent", return_value=agent):
            result = mister_fritz.ask_stuff("hi", MessageSource.DESKTOP_TEXT, "local-moodtest",
                                            thread_id=self.thread, mood_callback=self.events.append)
        self.assertEqual(self.events, ["weary"])
        self.assertEqual(result["text"], "Monday.")


if __name__ == "__main__":
    unittest.main()
