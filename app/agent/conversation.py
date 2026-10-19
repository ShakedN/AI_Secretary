"""One live phone conversation with the LLM, streamed token by token."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

from ..llm import END_CALL, LLM
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

    def __post_init__(self) -> None:
        self.system_prompt = build_system_prompt(self.office, self.caller_number, self.started_at)

    # ── history ──────────────────────────────────────────────────────────────

    def add_caller(self, text: str) -> None:
        self.turns.append(Turn("caller", text.strip()))

    def mark_interrupted(self, spoken: str) -> None:
        """The caller talked over us: only `spoken` was actually heard, so keep
        just that in history — otherwise Claude believes it said things the
        caller never heard."""
        text = (spoken.strip() + " [נקטע]").strip()
        if self.turns and self.turns[-1].role == "assistant":
            self.turns[-1].text = text
        elif spoken.strip():
            self.turns.append(Turn("assistant", text))

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
        try:
            if self.llm is None:
                raise RuntimeError("no LLM configured")
            for attempt in range(2):
                async for item in self.llm.stream(self.system_prompt, self.messages()):
                    if item is END_CALL:
                        hang_up = True
                    else:
                        parts.append(item)
                        yield item
                if hang_up and not "".join(parts).strip() and not self._goodbye_already_said():
                    logger.warning("ignoring a silent hang-up mid-conversation; asking again")
                    hang_up = False
                    continue
                break
        except Exception:
            if parts:
                raise
            logger.exception("LLM call failed")
            yield ERROR_REPLY
            self.turns.append(Turn("assistant", ERROR_REPLY))
            return

        if hang_up:
            self.ended = True
            if not "".join(parts).strip():
                parts.append(DEFAULT_GOODBYE)
                yield DEFAULT_GOODBYE
        elif not "".join(parts).strip():
            parts.append(FALLBACK_REPLY)
            yield FALLBACK_REPLY

        self.turns.append(Turn("assistant", "".join(parts).strip()))
