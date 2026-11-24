"""Sentences for the desk to speak (plan 14, PR 9).

A reply streams in as deltas, and a desk that speaks text itself (its hello's
`speak-local`, the body's Kokoro) is sent it a sentence at a time, so the
first words are spoken while the rest is still being written, not after the
whole reply. `SentenceChunker` cuts the stream into sentences: each is let go
once the next has begun, so the last of a reply always comes from `flush()`
and can be said to be the last. Short ones ("Yes.", "Mr.") are joined to the
one after. A numbered item's "1." is its marker, not a sentence. `speakable`
takes out what is for the eye only: Markdown's marks and fences, and a link's
address.

Pure, and without the TTS engine: COMPANION_TTS=local, the only mode built so
far, leaves the speaking to the desk.
"""
from __future__ import annotations

import re

# A sentence ends at . ! ? or … followed by space, or at a line break; but
# not at the stop of an abbreviation, unless a line break follows it, nor at
# the stop of a numbered item's marker ("1. Open the door").
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "e.g", "i.e", "cf", "approx"})
MIN_CHARS = 12

# A link's address may hold one level of brackets (Wikipedia's "_(planet)").
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^()\s]|\([^()\s]*\))+\)")
# An autolink's <> say where it ends, so it is "a link" whole.
_AUTOLINK = re.compile(r"<https?://[^\s<>]+>")
# A bare address runs to a space, a tag or an em dash (an en dash or an
# ellipsis can be part of one: "2023–24_Premier_League", ".../…/api.md");
# then, as a linkifier does, the stops, commas, quotes and marks at its end,
# and any bracket it didn't open, are left to be said (see _a_link).
_BARE_URL = re.compile(r"https?://[^\s<>—]+")
_TRAILING = frozenset(".,;:!?…*_~`'\"”’")
_OPENS = {")": "(", "]": "["}
# A code fence's line, its language ("```python") with it.
_FENCE = re.compile(r"^[ \t]*(?:```|~~~)[\w+#.-]*[ \t]*$", re.M)
_MARKS = re.compile(r"\*\*|__|\*|`+|~~")
_HEADING_OR_ITEM = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)")
_ITEM_NUMBER = re.compile(r"[ \t]*\d+\.")


def _a_link(address: re.Match) -> str:
    """A bare address said as "a link", with what ends the sentence or
    closes around the address still said after it."""
    url = address.group()
    # Brackets it closed but didn't open; only closers are trimmed, so the
    # openers' count stands.
    unopened = {closer: url.count(closer) - url.count(opener) for closer, opener in _OPENS.items()}
    end = len(url)
    while True:
        last = url[end - 1]
        if last in _TRAILING:
            end -= 1
        elif unopened.get(last, 0) > 0:
            unopened[last] -= 1
            end -= 1
        else:
            return "a link" + url[end:]


def speakable(text: str) -> str:
    """`text` as it should be said: a link's words without its address, a
    bare address as "a link", and no Markdown marks, fences, headings or
    bullets."""
    text = _LINK.sub(r"\1", text)
    text = _AUTOLINK.sub("a link", text)
    text = _BARE_URL.sub(_a_link, text)
    text = _FENCE.sub("", text)
    text = _HEADING_OR_ITEM.sub("", text)
    text = _MARKS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentence_end(text: str, line_start: bool = True) -> re.Match | None:
    """Where the first sentence in `text` ends, or None. `line_start`: `text`
    begins a line, so a number it begins with can be an item's marker."""
    for end in SENTENCE_END.finditer(text):
        if "\n" not in end.group():
            before = text[:end.start()]
            words = before.split()
            if words and words[-1].lower().rstrip(".") in ABBREVIATIONS:
                continue
            line = before.rfind("\n")
            if (line >= 0 or line_start) and _ITEM_NUMBER.fullmatch(before[line + 1:]):
                continue
        return end
    return None


class SentenceChunker:
    """Feed a reply's deltas as they stream; sentences come back as they are
    complete, and what is left, the last, from `flush()`.

    A sentence is complete once something to say has begun after its end, so
    the one that ends the reply waits for `flush()`. One shorter than
    `min_chars` waits to be joined to the next. `restart` (a new segment of
    the reply, after a tool call, which the client clears) drops what is not
    yet let go: sentences already let go have been spoken and can't be
    unsaid.
    """

    def __init__(self, min_chars: int = MIN_CHARS):
        self.min_chars = min_chars
        self._buf = ""
        self._held = ""
        # Whether _buf begins a line, as it does until a sentence ends mid-line.
        self._line_start = True

    def feed(self, delta: str, restart: bool = False) -> list[str]:
        if restart:
            self._buf = ""
            self._held = ""
            self._line_start = True
        self._buf += delta
        out: list[str] = []
        while True:
            end = _sentence_end(self._buf, self._line_start)
            if end is None or not speakable(self._buf[end.end():]):
                break
            sentence, self._buf = self._buf[:end.start()], self._buf[end.end():]
            self._line_start = "\n" in end.group()
            out += self._take(sentence)
        return out

    def flush(self) -> list[str]:
        """What is left, whatever its length, as the reply's last piece; none
        if nothing is left to say. The chunker is empty after."""
        lines = (speakable(line) for line in self._buf.split("\n"))
        text = " ".join(piece for piece in (self._held, *lines) if piece)
        self._buf = ""
        self._held = ""
        self._line_start = True
        return [text] if text else []

    def _take(self, sentence: str) -> list[str]:
        sentence = speakable(sentence)
        if not sentence:
            return []
        text = f"{self._held} {sentence}" if self._held else sentence
        if len(text) < self.min_chars:
            self._held = text
            return []
        self._held = ""
        return [text]
