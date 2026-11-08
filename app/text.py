"""Small text helpers shared by the call flows."""
from __future__ import annotations

import re

_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")
_EMOTICON = re.compile(r"(?<!\w)[:;=]-?[()DPp](?!\w)")
_MARKUP = re.compile(r"[*_#`~>|\[\]{}()]")
# "[הערת מערכת: …]" / "[נקטעתי]" / "(רציתי לומר: …)" are notes to her, never words to say.
_NOTE = re.compile(r"\[[^\]]*\]|\((?:רציתי|רצית) לומר[^)]*\)?|^\s*\(?(?:רציתי|רצית) לומר:\s*")

_SENTENCE_END = re.compile(r"[.!?…]+\s+")

# Gemini Lite now and then writes an Arabic letter inside a Hebrew word
# ("أرשום", "בשمחה" on a real call). The sounds match one to one, so swap them back.
_ARABIC_TO_HEBREW = str.maketrans({
    "ا": "א", "أ": "א", "إ": "א", "آ": "א", "ب": "ב", "ت": "ת", "ث": "ת", "ج": "ג", "ح": "ח", "خ": "ח",
    "د": "ד", "ذ": "ד", "ر": "ר", "ز": "ז", "س": "ס", "ش": "ש", "ص": "צ", "ض": "צ", "ط": "ט", "ظ": "ט",
    "ع": "ע", "غ": "ע", "ف": "פ", "ق": "ק", "ك": "כ", "ل": "ל", "م": "מ", "ن": "נ", "ه": "ה", "ة": "ה",
    "و": "ו", "ي": "י", "ى": "י", "ئ": "י", "ؤ": "ו", "ء": None, "ـ": None, "،": ",", "؟": "?",
    **{chr(c): None for c in range(0x064B, 0x0660)},  # Arabic vowel marks
})
_ARABIC = re.compile("[؀-ۿ]")
_FINAL = {"מ": "ם", "נ": "ן", "כ": "ך", "פ": "ף", "צ": "ץ"}
_HEBREW_LETTER = re.compile("[א-ת]")


def fix_arabic_letters(text: str) -> str:
    if not _ARABIC.search(text):
        return text
    out = []
    for i, ch in enumerate(text):
        heb = ch.translate(_ARABIC_TO_HEBREW)
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if heb != ch and heb in _FINAL and out and _HEBREW_LETTER.match(out[-1]) and nxt and not _HEBREW_LETTER.match(nxt):
            heb = _FINAL[heb]  # last letter of the word
        out.append(heb)
    return "".join(out)


_PHONE_RUN = re.compile(r"\+?\d[\d\s\-]{5,}\d")
_VALID_PHONE = re.compile(r"0(?:5\d|7\d)\d{7}|0[23489]\d{7}|1[5-9]00\d{6}")


def phone_digits(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    return "0" + digits[3:] if digits.startswith("972") else digits


def is_valid_phone(number: str) -> bool:
    """An Israeli number: mobile/07x = 10 digits, landline = 9, 1-800 style = 10."""
    return bool(_VALID_PHONE.fullmatch(phone_digits(number)))


def bad_phone_numbers(text: str) -> list[str]:
    """Digit runs that look like a phone number someone said, but can't be one
    ("054123456" — a digit went missing in recognition)."""
    found = []
    for m in _PHONE_RUN.finditer(text):
        digits = phone_digits(m.group())
        if len(digits) >= 7 and not is_valid_phone(digits):
            found.append(digits)
    return found


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
    text = _NOTE.sub("", text)
    text = _EMOJI.sub("", text)
    text = _EMOTICON.sub("", text)
    text = _MARKUP.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
