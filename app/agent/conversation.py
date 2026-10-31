"""One live phone conversation with the LLM, streamed token by token."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

from ..llm import END_CALL, LLM, CallerName
from ..settings import Office
from .prompt import build_system_prompt, now_in_israel

logger = logging.getLogger(__name__)

FALLBACK_REPLY = "סליחה, לא שמעתי טוב. אפשר לחזור על זה?"
ERROR_REPLY = "סליחה, יש לי תקלה קטנה. תוכלו להשאיר שם ומספר טלפון, ונחזור אליכם בהקדם?"
DEFAULT_GOODBYE = "תודה רבה, ההודעה הועברה. יום טוב!"

# A silent hang-up (end_call with no words) is only trusted once someone on
# the call already said goodbye — otherwise it's the model misfiring, and
# cutting a client off mid-message is the worst thing the secretary can do.
_FAREWELLS = ("להתראות", "ביי", "זהו", "זה הכל", "יום טוב", "המשך יום", "ערב טוב", "שבוע טוב", "שבת שלום", "תודה רבה")


def _says_goodbye(text: str) -> bool:
    return any(word in text for word in _FAREWELLS)


# Unmistakable closings — only these, at the very end of her reply, end the call.
_CLOSINGS = ("להתראות", "ביי", "יום טוב", "יום נעים", "ערב טוב", "ערב נעים", "שבוע טוב", "שבת שלום")


# Fallback for when the model doesn't call note_caller_name: "קוראים לי אבי מזרחי".
_NAME_RE = re.compile(r"(?:קוראים לי|שמי|השם שלי(?: הוא)?)\s+([א-ת'\-]+(?:\s+[א-ת'\-]+)?)")
_NOT_NAMES = {
    "זה", "לא", "כן", "רק", "ו", "אני", "של", "עם", "על", "גם", "בבקשה", "פה", "כאן",
    "ואני", "ואשמח", "וצריך", "ורוצה", "ומספר", "והמספר", "והטלפון", "במספר", "מהמספר",
    "בטלפון", "מהטלפון", "לגבי", "בקשר", "מדבר", "מדברת", "מתקשר", "מתקשרת",
}
MAX_BLOCKED_HANGUPS = 2  # after this many, let the call end even without a name
MAX_NAME_ASKS = 3        # after asking this many times, stop and let the lawyer confirm it
_YES = ("כן", "נכון", "בסדר", "אפשר", "בטח", "זה המספר", "למספר הזה", "אוקיי", "יאללה")
WRAP_UP_MINUTES = 3.5    # start steering the call to a close
MUST_FINISH_MINUTES = 5  # summarize and say goodbye now
_NAME_QUESTION = ("שמך", "השם", "שם המשפחה", "שם מלא", "שם פרטי", "איך קוראים", "באיזה שם", "לאיית")
ASK_NAME = "רגע, לפני שנסיים — איך קוראים לך? שם פרטי ושם משפחה, כדי שעורך הדין יידע מי חיפש אותו."
_REFUSALS = ("לא רוצה", "לא מעוניין", "לא מעוניינת", "מעדיף שלא", "מעדיפה שלא", "בלי שם",
             "לא חשוב", "לא משנה", "לא אגיד", "לא אמסור", "אין צורך", "לא צריך")


def name_in(text: str) -> str:
    m = _NAME_RE.search(text)
    if not m:
        return ""
    words = [w for w in m.group(1).split() if w not in _NOT_NAMES]
    return " ".join(words[:2])


def _ends_with_goodbye(text: str) -> bool:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return bool(sentences) and any(word in sentences[-1] for word in _CLOSINGS)


class _NoteFilter:
    """Drops "[...]" spans from a token stream as it's written. Models imitate the
    "[הערת מערכת: …]" notes they're given; a note spanning several streamed
    sentences would slip past any per-sentence cleanup and be read aloud."""

    def __init__(self) -> None:
        self.inside = False

    def feed(self, token: str) -> str:
        out = []
        for ch in token:
            if ch == "[":
                self.inside = True
            elif ch == "]" and self.inside:
                self.inside = False
            elif not self.inside:
                out.append(ch)
        return "".join(out)


@dataclass
class Turn:
    role: str  # "caller" | "assistant"
    text: str


@dataclass
class Conversation:
    office: Office
    llm: LLM | None
    caller_number: str | None = None
    started_at: datetime = field(default_factory=now_in_israel)
    turns: list[Turn] = field(default_factory=list)
    ended: bool = False
    caller_name: str = ""
    name_refused: bool = False
    blocked_hangups: int = 0
    callback_confirmed: bool = False
    # System notes for her next turn ("the caller talked over you…"). They go on
    # the caller's side: notes inside her own past turns get imitated and read aloud.
    _notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.system_prompt = build_system_prompt(self.office, self.caller_number, self.started_at)

    # ── history ──────────────────────────────────────────────────────────────

    def add_note(self, note: str) -> None:
        self._notes.append(note)

    def add_caller(self, text: str) -> None:
        last_assistant = next((t.text for t in reversed(self.turns) if t.role == "assistant"), "")
        if self._notes:
            text = " ".join(self._notes) + " " + text.strip()
            self._notes.clear()
        self.turns.append(Turn("caller", text.strip()))
        if not self.caller_name:
            self.caller_name = name_in(text)
        if not self.callback_confirmed and (
            "הקשתי במקלדת" in text
            or ("מספר" in last_assistant and "?" in last_assistant and any(w in text for w in _YES))
        ):
            self.callback_confirmed = True

    def name_asks(self) -> int:
        """How many times she has already asked for the caller's name."""
        return sum(
            1 for t in self.turns
            if t.role == "assistant" and "?" in t.text and any(w in t.text for w in _NAME_QUESTION)
        )

    def _all_done(self) -> bool:
        """Everything the lawyer needs is in hand: the name and a confirmed number."""
        return bool((self.caller_name or self.name_refused) and self.callback_confirmed)

    def may_hang_up(self) -> bool:
        """Only once we know who called (or they won't say, or recognition kept
        failing) — or we've already held the line open twice, so nobody gets
        trapped on the call."""
        return bool(
            self.caller_name or self.name_refused
            or self.name_asks() >= MAX_NAME_ASKS or self.blocked_hangups >= MAX_BLOCKED_HANGUPS
        )

    def _status(self) -> str:
        """Live checklist appended to the system prompt each turn."""
        callback = "\nמספר לחזרה: המתקשר כבר אישר — אל תשאלי עליו שוב." if self.callback_confirmed else ""
        return self._name_status() + callback + self._time_status()

    def _time_status(self) -> str:
        # A message-taking call shouldn't drag on: steer toward a close, never cut mid-sentence.
        minutes = (now_in_israel() - self.started_at).total_seconds() / 60
        if minutes >= MUST_FINISH_MINUTES:
            return (f"\nהשיחה נמשכת כבר {minutes:.0f} דקות — סיימי עכשיו: סכמי במשפט אחד את הפרטים "
                    "שיש לך, אמרי שעורך הדין יחזור, ונפרדי.")
        if minutes >= WRAP_UP_MINUTES:
            return (f"\nהשיחה נמשכת כבר {minutes:.1f} דקות — התחילי לכוון לסיום: חזרי בקצרה על השם, "
                    "המספר והנושא, שאלי אם יש עוד משהו חשוב, ונפרדי.")
        return ""

    def _name_status(self) -> str:
        if self.name_asks() >= MAX_NAME_ASKS and not self.name_refused:
            # Names are what phone recognition gets wrong most (one surname came out five
            # different ways in one call). Going round in circles makes callers hang up.
            heard = f' ("{self.caller_name}")' if self.caller_name else ""
            return (f"\n\nשם המתקשר: כבר שאלת עליו {self.name_asks()} פעמים{heard}. אל תשאלי שוב — "
                    "רשמי את מה שנקלט ואמרי שעורך הדין יוודא את הפרטים כשיחזור.")
        if self.caller_name:
            return f"\n\nשם המתקשר: {self.caller_name}."
        if self.name_refused:
            return "\n\nשם המתקשר: המתקשר לא רצה למסור שם — לא להתעקש."
        return ("\n\nשם המתקשר: עוד לא ידוע. לפני שנפרדים חובה לשאול לשמו — שם פרטי וגם שם משפחה, "
                "כדי שעורך הדין יידע מי חיפש אותו. אם הוא לא רוצה למסור, לא להתעקש.")

    def mark_interrupted(self, spoken: str, wanted: str = "") -> None:
        """The caller talked over us: only `spoken` was actually heard, so keep
        just that in history — otherwise the model believes it said things the
        caller never heard. `wanted` (the full line) lets it say the rest again."""
        heard = spoken.strip()
        if self.turns and self.turns[-1].role == "assistant":
            if heard:
                self.turns[-1].text = heard + "…"
            else:
                self.turns.pop()
        elif heard:
            self.turns.append(Turn("assistant", heard + "…"))
        note = f'[הערת מערכת: המתקשר דיבר מעלייך — הוא שמע רק "{heard}"' if heard else \
            "[הערת מערכת: המתקשר דיבר מעלייך לפני שהספקת לומר משהו"
        if wanted and wanted.strip() != heard:
            note += f'. רצית לומר: "{wanted.strip()}"'
        self.add_note(note + "]")

    def messages(self) -> list[dict]:
        """Turns → Messages API format. Consecutive same-role turns are merged
        (e.g. the caller spoke twice because our reply was cut off before any
        text), and the list must start with a user message."""
        out: list[dict] = []
        for turn in self.turns:
            role = "user" if turn.role == "caller" else "assistant"
            if out and out[-1]["role"] == role:
                out[-1]["content"] += "\n" + turn.text
            else:
                out.append({"role": role, "content": turn.text})
        while out and out[0]["role"] != "user":
            out.pop(0)
        return out

    def transcript(self) -> str:
        lines = [f"{self.office.assistant_name}: {self.office.greeting()}"]
        for turn in self.turns:
            who = "מתקשר" if turn.role == "caller" else self.office.assistant_name
            lines.append(f"{who}: {turn.text}")
        return "\n".join(lines)

    # ── generation ───────────────────────────────────────────────────────────

    def _goodbye_already_said(self) -> bool:
        last_caller = next((t.text for t in reversed(self.turns) if t.role == "caller"), "")
        last_assistant = next((t.text for t in reversed(self.turns) if t.role == "assistant"), "")
        return _says_goodbye(last_caller) or _says_goodbye(last_assistant)

    async def reply(self) -> AsyncIterator[str]:
        """Stream the next spoken reply. Records it in history once fully
        generated; if the caller interrupts, the task is cancelled and
        mark_interrupted() records what was actually heard instead."""
        parts: list[str] = []
        hang_up = False
        notes = _NoteFilter()
        try:
            if self.llm is None:
                raise RuntimeError("no LLM configured")
            use_tools = True
            for attempt in range(2):
                noted = False
                async for item in self.llm.stream(self.system_prompt + self._status(), self.messages(), tools=use_tools):
                    if item is END_CALL:
                        hang_up = True
                    elif isinstance(item, CallerName):
                        noted = True
                        if item.refused:
                            # Only an actual refusal counts — a plain "ביי" isn't one.
                            last = next((t.text for t in reversed(self.turns) if t.role == "caller"), "")
                            self.name_refused = any(cue in last for cue in _REFUSALS)
                        elif item.name:
                            self.caller_name = item.name
                    else:
                        item = notes.feed(item)
                        if item:
                            parts.append(item)
                            yield item
                silent = not "".join(parts).strip()
                if silent and self._all_done():
                    # Name and number are settled and the model has nothing more to
                    # say (it answered with tool calls only): a short goodbye ends it.
                    hang_up = True
                    break
                if noted and silent:
                    # Models often stop right after noting the name, with nothing said
                    # (sometimes even trying to hang up silently). The name is in the
                    # status line now — ask again, words only. A spoken goodbye still
                    # ends the call through the closing-words rule below.
                    use_tools, hang_up = False, False
                    continue
                if hang_up and silent and not self._goodbye_already_said():
                    logger.warning("ignoring a silent hang-up mid-conversation; asking again")
                    hang_up = False
                    continue
                break
            if not "".join(parts).strip() and self._all_done():
                hang_up = True  # the words-only retry came back empty too
        except Exception:
            if parts:
                raise
            logger.exception("LLM call failed")
            yield ERROR_REPLY
            self.turns.append(Turn("assistant", ERROR_REPLY))
            return

        if not hang_up and _ends_with_goodbye("".join(parts)):
            # She closed with "...goodbye" but the model forgot end_call: the
            # caller would be left on a silent open line.
            hang_up = True
        if hang_up and "".join(parts).strip().endswith("?"):
            # "Anything else before we finish?" + end_call: she asked, so she waits for the answer.
            logger.info("hang-up ignored: her reply ends with a question")
            hang_up = False
        if hang_up and not self.may_hang_up():
            # The lawyer needs to know who called: keep the line open — the
            # status line now tells her the name is still missing.
            self.blocked_hangups += 1
            hang_up = False
            logger.info("hang-up blocked: caller's name not known yet")
            if _ends_with_goodbye("".join(parts)):
                # She already said goodbye out loud: don't leave a silent open line.
                parts.append(" " + ASK_NAME)
                yield " " + ASK_NAME
        if hang_up:
            self.ended = True
            if not "".join(parts).strip():
                parts.append(DEFAULT_GOODBYE)
                yield DEFAULT_GOODBYE
        elif not "".join(parts).strip():
            parts.append(FALLBACK_REPLY)
            yield FALLBACK_REPLY

        self.turns.append(Turn("assistant", "".join(parts).strip()))
