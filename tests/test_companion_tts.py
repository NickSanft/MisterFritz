"""companion_tts: a streaming reply cut into sentences for the desk to speak,
each let go once the next has begun, and said without what is for the eye."""
import unittest

from companion_tts import MIN_CHARS, SentenceChunker, speakable


def stream(deltas, restart_at=()):
    """Every sentence a chunker lets go of for `deltas`, flush included, and
    which came from flush."""
    chunker = SentenceChunker()
    said = []
    for i, delta in enumerate(deltas):
        said += chunker.feed(delta, restart=i in restart_at)
    tail = chunker.flush()
    return said, tail


class TestTheSentences(unittest.TestCase):
    def test_a_sentence_is_let_go_once_the_next_has_begun_and_the_last_at_the_end(self):
        chunker = SentenceChunker()
        self.assertEqual(chunker.feed("Good evening, sir. "), [], "the next has not begun: it may be the last")
        self.assertEqual(chunker.feed("It is "), ["Good evening, sir."])
        self.assertEqual(chunker.feed("half past nine."), [])
        self.assertEqual(chunker.flush(), ["It is half past nine."])
        self.assertEqual(chunker.flush(), [], "flushed is empty")

    def test_streamed_in_any_pieces_it_says_the_same(self):
        reply = "Good evening, sir. It is half past nine! Shall I draw the curtains? Very good."
        whole = stream([reply])
        for size in (1, 2, 3, 7, 13):
            pieces = [reply[i:i + size] for i in range(0, len(reply), size)]
            with self.subTest(size=size):
                self.assertEqual(stream(pieces), whole)
        said, tail = whole
        self.assertEqual(said, ["Good evening, sir.", "It is half past nine!", "Shall I draw the curtains?"])
        self.assertEqual(tail, ["Very good."])

    def test_a_short_sentence_waits_to_be_joined_to_the_next(self):
        self.assertEqual(MIN_CHARS, 12)
        said, tail = stream(["Yes. Mr. Fritz is here. Indeed it is so. Ok"])
        self.assertEqual(said, ["Yes. Mr. Fritz is here.", "Indeed it is so."])
        self.assertEqual(tail, ["Ok"], "the last is said whatever its length")
        self.assertEqual(stream(["Hm."]), ([], ["Hm."]))
        # Short ones held at the end are said with what is left.
        self.assertEqual(stream(["Indeed it is so. ", "Yes. ", "No"]), (["Indeed it is so."], ["Yes. No"]))

    def test_a_line_break_ends_a_sentence_too(self):
        said, tail = stream(["First line here\nSecond line here\n\n"])
        self.assertEqual((said, tail), (["First line here"], ["Second line here"]))

    def test_a_restart_drops_what_was_not_yet_let_go(self):
        """A new segment after a tool call, which the client clears. What was
        let go has been spoken and is not taken back."""
        said, tail = stream(["Let me look. ", "I shall check the", "It is nine o'clock. ", "Good."], restart_at=(2,))
        self.assertEqual(said, ["Let me look.", "It is nine o'clock."])
        self.assertEqual(tail, ["Good."])
        # A short one held for the next is dropped with it.
        said, tail = stream(["Hm. Ok", "It is nine o'clock now. Good."], restart_at=(1,))
        self.assertEqual((said, tail), (["It is nine o'clock now."], ["Good."]))

    def test_numbers_and_abbreviations_do_not_end_a_sentence(self):
        said, tail = stream(["It costs 3.50 pounds, e.g. at the shop. Indeed."])
        self.assertEqual(said, ["It costs 3.50 pounds, e.g. at the shop."])
        self.assertEqual(tail, ["Indeed."])
        said, tail = stream(["Dr. Watson and Mrs. Hudson are in. Vs. the rest."])
        self.assertEqual(said, ["Dr. Watson and Mrs. Hudson are in."])
        # Only the word before the stop counts, and a line break always ends one.
        self.assertEqual(stream(["I asked the good Mr.\nHe said yes, sir."]),
                         (["I asked the good Mr."], ["He said yes, sir."]))
        self.assertEqual(stream(["The answer is no. I won't, sir."]), (["The answer is no."], ["I won't, sir."]))

    def test_an_ellipsis_ends_one(self):
        said, tail = stream(["I suppose that I could… If you like, sir."])
        self.assertEqual((said, tail), (["I suppose that I could…"], ["If you like, sir."]))

    def test_a_numbered_item_is_one_sentence_said_without_its_number(self):
        reply = "Here are the steps:\n1. Open the door.\n2. Walk in.\n3. Sit down please.\n10. Rest."
        whole = stream([reply])
        self.assertEqual(whole, (["Here are the steps:", "Open the door.", "Walk in. Sit down please."], ["Rest."]))
        for size in (1, 2, 5):
            with self.subTest(size=size):
                self.assertEqual(stream([reply[i:i + size] for i in range(0, len(reply), size)]), whole)
        # Only at the start of a line: mid-line, a number's stop ends a sentence.
        said, tail = stream(["I counted them all, sir. 7. That was the score."])
        self.assertEqual(said, ["I counted them all, sir."])
        self.assertEqual(tail, ["7. That was the score."], "a number after a sentence is said")
        # A new segment, and a flushed chunker, begin a line again.
        said, tail = stream(["Let me look. Hm", "7. Open it now please."], restart_at=(1,))
        self.assertEqual((said, tail), (["Let me look."], ["Open it now please."]))
        chunker = SentenceChunker()
        chunker.feed("I counted them all, sir. Hm")
        chunker.flush()
        self.assertEqual(chunker.feed("7. Open it now please. And then sit."), ["Open it now please."])

    def test_nothing_to_say_says_nothing(self):
        self.assertEqual(stream(["", "   ", "\n\n", "**", "`"]), ([], []))

    def test_marks_left_at_the_end_keep_the_last_sentence_for_the_end(self):
        """A trailing fence or bold mark is nothing to say: the sentence before
        it is still the last, and comes from flush."""
        chunker = SentenceChunker()
        self.assertEqual(chunker.feed("Here it is, sir.\n```"), [])
        self.assertEqual(chunker.flush(), ["Here it is, sir."])


class TestWhatIsSpeakable(unittest.TestCase):
    def test_marks_links_headings_and_bullets_are_for_the_eye(self):
        for text, said in (
            ("**Indeed**, sir: see [the log](https://example.com/log).", "Indeed, sir: see the log."),
            ("Go to https://example.com/a?b=c now.", "Go to a link now."),
            ("<https://example.com>", "a link"),
            ("See https://bbc.co.uk/weather, sir.", "See a link, sir."),
            ("Visit https://example.com.", "Visit a link."),
            ("Check <https://example.com/x>. Then (https://x.y/z).", "Check a link. Then (a link)."),
            ("Did you see https://x.y/z?", "Did you see a link?"),
            ("Visit https://example.com… Then", "Visit a link… Then"),
            ('He said "go to https://example.com." Then', 'He said "go to a link." Then'),
            ("It is at https://x.y/z”, sir [or https://x.y/w]", "It is at a link”, sir [or a link]"),
            ("At https://bbc.co.uk/weather<br>now", "At a link<br>now"),
            # Brackets and quotes it holds are its own; one it didn't open is not.
            ("Ollama is at http://[::1]:11434. Shall I?", "Ollama is at a link. Shall I?"),
            ("Fetch https://x.y/items?filter[status]=open&ids[]=1 now.", "Fetch a link now."),
            ("Read https://en.wikipedia.org/wiki/Mercury_(planet)#Orbit now", "Read a link now"),
            ("(see https://en.wikipedia.org/wiki/Mercury_(planet)) and so on", "(see a link) and so on"),
            ("[see https://x.y/w]", "[see a link]"),
            ("Try https://x.y/it's-here, sir.", "Try a link, sir."),
            ("Visit **<https://example.com>** or `https://x.y/z.`", "Visit a link or a link."),
            ("The docs at https://example.com—they are good", "The docs at a link—they are good"),
            ("The table is at https://en.wikipedia.org/wiki/2023–24_Premier_League, sir.", "The table is at a link, sir."),
            ("Read https://github.com/ollama/…/README.md now.", "Read a link now."),
            ("See <https://en.wikipedia.org/wiki/A—B>, sir.", "See a link, sir."),
            ("See [Mercury](https://en.wikipedia.org/wiki/Mercury_(planet)), sir.", "See Mercury, sir."),
            ("```python\nprint(1)\n```", "print(1)"),
            ("~~~\nls\n~~~", "ls"),
            ("```ls``` lists them", "ls lists them"),
            ("## The plan", "The plan"),
            ("- buy the milk", "buy the milk"),
            ("1. first things", "first things"),
            ("> as he said", "as he said"),
            ("Run `ls -la` and ~~not~~ this", "Run ls -la and not this"),
            ("  lots   of\tspace  ", "lots of space"),
        ):
            with self.subTest(text=text):
                self.assertEqual(speakable(text), said)

    def test_streamed_markdown_is_said_plainly(self):
        said, tail = stream(["**Good** evening, ", "sir. See [the ", "notes](https://x.y/z). ", "`Done`."])
        self.assertEqual(said, ["Good evening, sir.", "See the notes."])
        self.assertEqual(tail, ["Done."])
        # A fence's language is not said; inline code with three ticks is.
        said, tail = stream(["Here is the code:\n```python\n", "print('hi there')\n```\n", "Use ```ls``` too."])
        self.assertEqual((said, tail), (["Here is the code:", "print('hi there')"], ["Use ls too."]))


if __name__ == "__main__":
    unittest.main()
