"""Twilio ConversationRelay WebSocket protocol for one call.

Twilio does speech-to-text, text-to-speech and barge-in detection; we only
exchange text. Incoming: setup / prompt / interrupt / dtmf / error.
Outgoing: {"type": "text", "token", "last"} streamed as Claude generates,
and {"type": "end"} to hang up.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

from .agent.conversation import Conversation

logger = logging.getLogger(__name__)

Send = Callable[[dict], Awaitable[None]]
ConversationFactory = Callable[[str | None], Conversation]

# Hebrew TTS speaks roughly this fast; used to wait for the goodbye to finish
# playing before hanging up (Twilio's "end" cuts audio that is still playing).
_WORDS_PER_MINUTE = 150


def speak_seconds(text: str) -> float:
    return len(text.split()) / _WORDS_PER_MINUTE * 60


class RelaySession:
    def __init__(
        self,
        send: Send,
        make_conversation: ConversationFactory,
        is_allowed: Callable[[str], bool] = lambda call_sid: True,
        hangup_padding: float = 1.0,
    ) -> None:
        self._send = send
        self._make_conversation = make_conversation
        self._is_allowed = is_allowed
        self._hangup_padding = hangup_padding
        self._task: asyncio.Task | None = None
        self.call_sid: str | None = None
        self.caller_number: str | None = None
        self.conversation: Conversation | None = None
        self.closed = False

    async def handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "setup":
            await self._on_setup(msg)
        elif self.conversation is None:
            logger.warning("message before setup: %s", kind)
        elif kind == "prompt":
            await self._on_prompt(msg)
        elif kind == "interrupt" and not self.closed:
            await self._cancel_reply()
            self.conversation.mark_interrupted(msg.get("utteranceUntilInterrupt", ""))
            # Talking over the goodbye ("wait, one more thing") keeps the call going.
            self.conversation.ended = False
        elif kind == "error":
            logger.error("ConversationRelay error on %s: %s", self.call_sid, msg.get("description"))

    async def close(self) -> None:
        await self._cancel_reply()

    # ── handlers ────────────────────────────────────────────────────────────

    async def _on_setup(self, msg: dict) -> None:
        call_sid = msg.get("callSid", "")
        if not self._is_allowed(call_sid):
            logger.warning("rejecting unknown call %s", call_sid)
            self.closed = True
            return
        self.call_sid = call_sid
        number = msg.get("from") or ""
        # Withheld caller ID arrives as "anonymous" or a non-number placeholder.
        self.caller_number = number if number.lstrip("+").isdigit() and len(number) > 6 else None
        self.conversation = self._make_conversation(self.caller_number)
        logger.info("call %s from %s connected", call_sid, self.caller_number or "withheld")

    async def _on_prompt(self, msg: dict) -> None:
        text = (msg.get("voicePrompt") or "").strip()
        if not text or self.conversation.ended:
            return
        # A new utterance supersedes any reply still being generated.
        await self._cancel_reply()
        self.conversation.add_caller(text)
        self._task = asyncio.create_task(self._reply())

    async def _reply(self) -> None:
        try:
            await self._stream_reply()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("reply failed on %s", self.call_sid)

    async def _stream_reply(self) -> None:
        spoken: list[str] = []
        async for token in self.conversation.reply():
            spoken.append(token)
            await self._send({"type": "text", "token": token, "last": False})
        await self._send({"type": "text", "token": "", "last": True})

        if self.conversation.ended:
            await asyncio.sleep(speak_seconds("".join(spoken)) + self._hangup_padding)
            await self._send({"type": "end", "handoffData": json.dumps({"reason": "completed"})})
            self.closed = True

    async def _cancel_reply(self) -> None:
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
