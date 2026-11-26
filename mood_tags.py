"""His mood, as the model names it at the start of a reply: `[mood:pleased]`.

Plan 14 PR 11: the owner's call on Q3 was inline tags. A desk turn's context
asks the model to begin its reply with one (ASK, added by get_source_info).
Every reply from any source (the desk, Discord, the web chat) has a leading
tag taken off before anyone sees or hears it. Only a desk is told the mood,
in a `mood` frame (companion_api), because only a desk has a face to show it.

A tag counts only at the very start of a part of the reply: the model may
write, call a tool and write again, and each part may begin with one. Leading
whitespace before it is allowed, and the whitespace after it is dropped as
well (gpt-oss writes "[mood:dry]\\n" more often than not). A tag is matched
case-insensitively. Its name is a letter and then up to 15 more letters,
spaces, hyphens or underscores; one naming a mood outside MOODS ("[mood:very
happy]") is dropped all the same, telling nothing. One followed at once by
"(" is a Markdown link's text ("[Mood: Chill](url)"), not a tag, and is left
as it is.

As it streams, text that only starts like a tag is held until it is one or
plainly is not: at most 35 characters, the longest tag, after any leading
whitespace (held until something follows it). A whole tag waits for the
character after it, which says whether it is a link's text and is never
held itself. So a tag never shows on its way past.
"""
from __future__ import annotations

import re
import typing

from companion_protocol import MoodName

MOODS: tuple[str, ...] = typing.get_args(MoodName)

_TAGS = [f"[mood:{m}]" for m in MOODS]
ASK = (f"Begin your reply with exactly one mood tag for your face, one of "
       f"{', '.join(_TAGS[:-1])} or {_TAGS[-1]}, then your reply. The tag is "
       f"never shown or spoken.")

# A whole tag, and anything a tag could still become. Bounded, so text that
# merely starts with "[" is held for a few characters at most.
_NAME = r"[a-z][a-z _-]{0,15}"
_WHOLE = re.compile(r"\[\s{0,3}mood\s{0,3}:\s{0,3}(" + _NAME + r")\s{0,3}\]", re.IGNORECASE)
_BEGUN = re.compile(
    r"\[\s{0,3}(?:m(?:o(?:o(?:d(?:\s{0,3}(?::\s{0,3}(?:" + _NAME + r"\s{0,3})?)?)?)?)?)?)?\Z",
    re.IGNORECASE)

_START, _AFTER, _OPEN = "start", "after", "open"


class LeadingTag:
    """One part of a reply, fed as it streams, with a leading tag held back
    until it is whole and then dropped, along with the whitespace after it.

    feed(text) answers the text that may be shown now (often all of it);
    end() answers what is still held once the part is over. `mood` is the
    tag's mood once one is read, if it names one of MOODS.
    """

    def __init__(self) -> None:
        self._held = ""
        self._state = _START
        self.mood: str | None = None

    def feed(self, text: str) -> str:
        if self._state == _OPEN:
            return text
        self._held += text
        return self._settle()

    def end(self) -> str:
        """The part is over. A whole tag still held (nothing came after it)
        is read and dropped; one still being written is dropped, and so is
        the whitespace after a tag. Whitespace alone is given back: it was
        never a tag. Text fed after this passes as it is."""
        held, state = self._held, self._state
        self._held, self._state = "", _OPEN
        if state != _START:
            return ""
        lead = held.lstrip()
        if not lead:
            return held
        whole = _WHOLE.match(lead)
        if whole and whole.end() == len(lead):
            self._read(whole)
        return ""

    def _read(self, whole: re.Match) -> None:
        name = whole.group(1).strip().lower()
        self.mood = name if name in MOODS else None

    def _open(self) -> str:
        held, self._held, self._state = self._held, "", _OPEN
        return held

    def _settle(self) -> str:
        if self._state == _START:
            lead = self._held.lstrip()
            whole = _WHOLE.match(lead)
            if whole is None:
                if not lead or _BEGUN.match(lead):
                    return ""
                return self._open()
            rest = lead[whole.end():]
            if not rest:
                return ""      # the next character says whether it is a link's text
            if rest[0] == "(":
                return self._open()
            self._read(whole)
            self._held, self._state = rest, _AFTER
        rest = self._held.lstrip()
        if not rest:
            self._held = ""
            return ""
        self._held, self._state = "", _OPEN
        return rest


def strip_leading(text: str) -> tuple[str | None, str]:
    """A whole part of a reply without its leading tag, and the tag's mood."""
    tag = LeadingTag()
    shown = tag.feed(text) + tag.end()    # end() first: it reads a tag the text ends on
    return tag.mood, shown
