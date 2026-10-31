"""Small text helpers shared by the call flows."""
from __future__ import annotations

import re

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")
_EMOTICON = re.compile(r"(?<!\w)[:;=]-?[()DPp](?!\w)")
_MARKUP = re.compile(r"[*_#`~>|\[\]{}()]")

_SENTENCE_END = re.compile(r"[.!?…]+\s+")


def split_sentences(text: str) -> list[str]:
    parts, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        parts.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        parts.append(text[start:].strip())
    return [p for p in parts if p]


def take_finished_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a streaming buffer into sentences that are surely finished (followed
    by whitespace) and the unfinished tail still being written."""
    *done, tail = split_sentences(buffer + "\0") or ["\0"]
    return done, tail.rstrip("\0")


def clean_for_speech(text: str) -> str:
    """Drop what a voice can't say: emoji, ":)", markdown and brackets."""
    text = _EMOJI.sub("", text)
    text = _EMOTICON.sub("", text)
    text = _MARKUP.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
