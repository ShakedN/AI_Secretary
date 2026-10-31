"""A phone call over Twilio Media Streams: we do the listening and the talking.

Twilio sends the caller's raw audio (8 kHz mu-law) over a WebSocket and plays
whatever audio we send back. That gives us what <Gather> can't:

- Listening: Azure recognizes speech while the caller talks; a pause ends
  their turn (~0.5-1 s), so replies start much sooner.
- Speaking: each sentence is voiced as soon as the LLM writes it.
- Talking over her: only real words (not line noise) stop her. Because we
  send the audio ourselves we know exactly what was heard when she stopped:
  if she got through at least two thirds, the caller heard it and is
  answering; otherwise the LLM is told what was cut and what she meant to say,
  so the interruption isn't mistaken for an answer.

Incoming Twilio events: start, media, mark, dtmf, stop.
Outgoing: media (audio), mark (tells us when playback reached a point), clear
(drop any audio not played yet — used when the caller talks over her).
"""
from __future__ import annotations

import asyncio
import base64
import itertools
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .agent.conversation import Conversation, Turn
from .relay import caller_number
from .settings import Office
from .text import clean_for_speech, take_finished_sentences

logger = logging.getLogger(__name__)

BYTES_PER_SECOND = 8000      # 8 kHz, 1 byte per mu-law sample
CHUNK = 1600                 # 200 ms of audio per media message
BARGE_IN_WORDS = 1           # recognized words (not just noise) needed to stop her
GOODBYE_BARGE_IN_WORDS = 3   # ...while she says goodbye: "כן"/"ביי" back shouldn't stop the hang-up
GREETING_BARGE_IN_WORDS = 3  # ...and over the greeting: people say "הלו?" when a line connects
# Her own voice leaking back through the caller's phone gets "recognized" as the
# caller ("שלום." right as she says "שלום."). Words heard while she talks that
# are mostly hers are echo, not an interruption.
ECHO_SHARE = 0.6
ECHO_TAIL = 1.0              # echo can still arrive this long after her audio ends
_WORD = re.compile(r"[\u05D0-\u05EAa-zA-Z0-9]+")
_GREETINGS = {"הלו", "שלום", "היי", "אהלן", "כן", "בוקר", "טוב", "ערב"}
HEARD_ENOUGH = 2 / 3         # heard at least this much → it counts as said
# Below this, the LLM is told the words may be misheard. Measured on a real call:
# ordinary sentences 0.72-0.89; names 0.51-0.61, whether recognized right or not.
LOW_CONFIDENCE = 0.7
SILENCE_SECONDS = 8.0        # quiet after she finishes → "hello? I'm here"
MAX_SILENCES = 2

START_BUFFER_SECONDS = 2.0   # start a reply only with this much audio ready, so it doesn't stutter
PLAYBACK_DELAY = 0.4         # network + Twilio: the caller hears audio a bit after we send it
# Clear goodbyes only — "תודה רבה" alone also opens plenty of mid-call sentences.
_BYE = ("להתראות", "ביי", "יום טוב", "המשך יום", "ערב טוב", "שבוע טוב", "שבת שלום")

HELLO = "הלו? אני כאן, אפשר לדבר."
GOODBYE_SILENT = "לא שומעים אתכם. אפשר להתקשר שוב בכל עת. להתראות."
TROUBLE = "סליחה, יש לי תקלה בקו. אנא התקשרו שוב בעוד כמה דקות. להתראות."

Send = Callable[[dict], Awaitable[None]]
Synthesize = Callable[[str], Awaitable[bytes]]


@dataclass
class Segment:
    text: str
    seconds: float
    start: float  # clock time the caller starts hearing it


@dataclass
class Utterance:
    """One stretch of her talking: a reply, the greeting, or a stock line."""
    id: int
    is_greeting: bool = False
    in_history: bool = True        # replies are part of the conversation; stock lines aren't
    text: str = ""                 # everything written so far
    segments: list[Segment] = field(default_factory=list)
    done: bool = False             # all audio sent
    recorded: bool = False         # already in the conversation history

    def heard(self, now: float) -> tuple[str, float]:
        """What the caller has heard by `now`, and which share of the text that is."""
        parts: list[str] = []
        for seg in self.segments:
            played = min(max(now - seg.start, 0.0), seg.seconds)
            if played >= seg.seconds:
                parts.append(seg.text)
                continue
            if played > 0:
                words = seg.text.split()
                parts.append(" ".join(words[: int(len(words) * played / seg.seconds)]))
            break
        heard = " ".join(p for p in parts if p).strip()
        return heard, (len(heard) / len(self.text) if self.text else 0.0)


class CallStream:
    def __init__(
        self,
        *,
        send: Send,
        office: Office,
        llm,
        make_recognizer: Callable,
        synthesize: Synthesize,
        is_allowed: Callable[[str], bool] = lambda token: True,
        clock: Callable[[], float] = time.monotonic,
        record: bool = False,
    ) -> None:
        # The caller's raw audio, kept only when debugging recognition (RECORD_CALLS).
        self.recording: bytearray | None = bytearray() if record else None
        self._send_raw = send
        self._office = office
        self._llm = llm
        self._make_recognizer = make_recognizer
        self._synthesize = synthesize
        self._is_allowed = is_allowed
        self._clock = clock

        self.call_sid: str | None = None
        self.stream_sid: str | None = None
        self.conversation: Conversation | None = None
        self.closed = False

        self._recognizer = None
        self._ids = itertools.count(1)
        self._speaking: Utterance | None = None
        self._play_cursor = 0.0           # when the audio already sent will finish playing
        self._reply_task: asyncio.Task | None = None
        self._silence_task: asyncio.Task | None = None
        self._pending: list[str] = []     # caller's words not answered yet
        self._note = ""                   # "[cut off in the greeting…]" for the next turn
        self._silences = 0
        self._digits = ""
        self._hangup_after: int | None = None
        self._recent_said: list[tuple[float, str]] = []  # (when it finished playing, text) — for echo checks

    # ── Twilio events ────────────────────────────────────────────────────────

    async def handle(self, msg: dict) -> None:
        event = msg.get("event")
        if event == "start":
            await self._on_start(msg.get("start") or {})
        elif self.conversation is None:
            return
        elif event == "media":
            audio = base64.b64decode(msg["media"]["payload"])
            self._recognizer.push(audio)
            if self.recording is not None:
                self.recording += audio
        elif event == "mark":
            await self._on_mark(msg["mark"]["name"])
        elif event == "dtmf":
            await self._on_dtmf(msg["dtmf"]["digit"])
        elif event == "stop":
            self.closed = True

    async def close(self) -> None:
        for task in (self._reply_task, self._silence_task):
            if task and not task.done():
                task.cancel()
        if self._recognizer is not None:
            self._recognizer.close()

    async def _on_start(self, start: dict) -> None:
        params = start.get("customParameters") or {}
        if not self._is_allowed(params.get("token", "")):
            logger.warning("rejecting media stream without a valid token")
            self.closed = True
            return
        self.call_sid = start.get("callSid")
        self.stream_sid = start.get("streamSid")
        self.conversation = Conversation(office=self._office, llm=self._llm, caller_number=caller_number(params.get("from")))
        self._recognizer = self._make_recognizer(self.on_partial, self.on_final, self.on_error)
        logger.info("call %s streaming (caller %s)", self.call_sid, self.conversation.caller_number or "withheld")
        await self._say(self._office.greeting(), is_greeting=True)

    async def _on_mark(self, name: str) -> None:
        u = self._speaking
        if u is None or name != f"u{u.id}" or not u.done:
            return  # an older utterance (e.g. cleared by an interruption)
        self._speaking = None
        self._remember_said(u)
        if u.id == self._hangup_after:
            self.closed = True  # the goodbye finished playing — hang up
            return
        self._start_silence_watch()

    async def _on_dtmf(self, digit: str) -> None:
        self._cancel_silence_watch()
        if digit == "#":
            typed, self._digits = self._digits, ""
            if typed:
                await self.on_final(f"(הקשתי במקלדת) {' '.join(typed)}")
        else:
            self._digits += digit

    # ── Speech recognition ───────────────────────────────────────────────────

    def _playing(self) -> bool:
        return self._speaking is not None and self._play_cursor > self._clock()

    def _barge_in_words(self) -> int:
        # During her goodbye, "כן"/"תודה"/"ביי" are just the caller saying bye
        # back — only a real sentence ("רגע, עוד משהו") should stop the hang-up.
        if self._speaking is not None and self._speaking.id == self._hangup_after:
            return GOODBYE_BARGE_IN_WORDS
        if self._speaking is not None and self._speaking.is_greeting:
            return GREETING_BARGE_IN_WORDS
        return BARGE_IN_WORDS

    def _is_echo(self, text: str) -> bool:
        now = self._clock()
        said = [t for end, t in self._recent_said if now - end <= ECHO_TAIL]
        if self._speaking is not None:
            said.append(self._speaking.text)
        if not said:
            return False
        hers = set(_WORD.findall(" ".join(said)))
        words = _WORD.findall(text)
        return bool(words) and sum(w in hers for w in words) / len(words) >= ECHO_SHARE

    def _remember_said(self, u: Utterance) -> None:
        self._recent_said = [(end, t) for end, t in self._recent_said if self._clock() - end <= ECHO_TAIL]
        self._recent_said.append((max(self._clock(), self._play_cursor), u.text))

    async def on_partial(self, text: str) -> None:
        if not text.strip():
            return
        self._cancel_silence_watch()
        if self._playing():
            if len(text.split()) >= self._barge_in_words() and not self._is_echo(text):
                await self._interrupt()
        elif self._reply_task and not self._reply_task.done() and not self._speaking_has_audio():
            # They kept talking before she started answering: wait for their whole turn.
            await self._cancel_reply()

    async def on_final(self, text: str, confidence: float = 1.0) -> None:
        text = text.strip()
        if not text:
            return
        talking = self._playing() or bool(self._recent_said and self._clock() - self._recent_said[-1][0] <= ECHO_TAIL)
        if talking and self._is_echo(text):
            logger.info("call %s ignoring echo of her own voice: %s", self.call_sid, text)
            return
        words = len(text.split())
        if confidence < LOW_CONFIDENCE:
            # e.g. a name heard as "שקל מוזמ": she should check it, not repeat it as fact.
            text += " [זיהוי לא בטוח]"
        self._cancel_silence_watch()
        self._silences = 0
        if self._playing():
            if words < self._barge_in_words():
                if not (self._speaking.is_greeting and set(_WORD.findall(text)) <= _GREETINGS):
                    self._pending.append(text)  # e.g. "כן" over her goodbye — she keeps talking
                return
            await self._interrupt()
        logger.info("call %s caller (%.2f): %s", self.call_sid, confidence, text)
        if self._reply_task and not self._reply_task.done():
            await self._cancel_reply()
        self._pending.append(text)
        self._reply_task = asyncio.create_task(self._respond())

    async def on_error(self, details: str) -> None:
        logger.error("call %s speech recognition failed: %s", self.call_sid, details)
        await self._say(TROUBLE, end_call=True)

    # ── Talking over her ─────────────────────────────────────────────────────

    def _speaking_has_audio(self) -> bool:
        return self._speaking is not None and bool(self._speaking.segments)

    async def _interrupt(self) -> None:
        u, now = self._speaking, self._clock()
        await self._send({"event": "clear"})
        self._play_cursor = now
        self._speaking = None
        self._remember_said(u)
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()
            try:
                await self._reply_task
            except asyncio.CancelledError:
                pass

        heard, share = u.heard(now)
        logger.info("call %s caller talked over her after %.0f%%: %r", self.call_sid, share * 100, heard)
        if u.id == self._hangup_after:  # "wait, one more thing" over the goodbye
            self._hangup_after = None
            self.conversation.ended = False
        if share >= HEARD_ENOUGH:
            if u.in_history and not u.recorded:  # heard enough: it counts as said
                self.conversation.turns.append(Turn("assistant", u.text.strip()))
        elif u.is_greeting:
            self._note = f'[הערת מערכת: המתקשר דיבר מעל הפתיחה שלך — הוא שמע רק "{heard}"]'
        elif u.in_history:
            self.conversation.mark_interrupted(heard, wanted=u.text)

    # ── Talking ──────────────────────────────────────────────────────────────

    async def _send(self, msg: dict) -> None:
        await self._send_raw({**msg, "streamSid": self.stream_sid})

    async def _play(self, u: Utterance, text: str, audio: bytes) -> None:
        if self._speaking is not u or not audio:
            return
        start = max(self._clock() + PLAYBACK_DELAY, self._play_cursor)
        seconds = len(audio) / BYTES_PER_SECOND
        self._play_cursor = start + seconds
        u.segments.append(Segment(text, seconds, start))
        for i in range(0, len(audio), CHUNK):
            payload = base64.b64encode(audio[i:i + CHUNK]).decode("ascii")
            await self._send({"event": "media", "media": {"payload": payload}})

    async def _finish(self, u: Utterance) -> None:
        u.done = True
        await self._send({"event": "mark", "mark": {"name": f"u{u.id}"}})

    async def _voice(self, text: str) -> bytes:
        text = clean_for_speech(text)
        if not text:
            return b""
        try:
            return await self._synthesize(text)
        except Exception:
            logger.exception("call %s TTS failed for %r", self.call_sid, text)
            return b""

    async def _say(self, text: str, *, is_greeting: bool = False, end_call: bool = False) -> None:
        """A fixed line (greeting, "hello?", goodbye) — not part of the LLM conversation."""
        u = Utterance(next(self._ids), is_greeting=is_greeting, in_history=False, text=text)
        self._speaking = u
        if end_call:
            self._hangup_after = u.id
        logger.info("call %s says%s: %s", self.call_sid, " (hanging up)" if end_call else "", text)
        await self._play(u, text, await self._voice(text))
        if self._speaking is u:
            await self._finish(u)

    async def _respond(self) -> None:
        conv = self.conversation
        text = " ".join(self._pending).strip()
        self._pending.clear()
        if self._note:
            text, self._note = f"{self._note} {text}", ""
        conv.add_caller(text)
        caller_turn = conv.turns[-1]
        u = Utterance(next(self._ids))
        self._speaking = u
        voiced: asyncio.Queue = asyncio.Queue()
        player = asyncio.create_task(self._play_in_order(u, voiced))
        t0 = self._clock()
        try:
            tail = ""
            async for token in conv.reply():
                u.text += token
                done, tail = take_finished_sentences(tail + token)
                for sentence in done:
                    voiced.put_nowait((sentence, asyncio.create_task(self._voice(sentence))))
            if tail.strip():
                voiced.put_nowait((tail.strip(), asyncio.create_task(self._voice(tail.strip()))))
            voiced.put_nowait(None)
            await player
            u.recorded = True
            if not conv.ended and _is_bye(text) and _is_bye(u.text) and conv.may_hang_up():
                # Both sides said goodbye but the model forgot to hang up.
                conv.ended = True
            logger.info("call %s assistant%s (%.1fs): %s", self.call_sid,
                        " (hanging up)" if conv.ended else "", self._clock() - t0, u.text.strip())
            if self._speaking is u:
                if conv.ended:
                    self._hangup_after = u.id
                await self._finish(u)
        except asyncio.CancelledError:
            player.cancel()
            if not u.segments and caller_turn in conv.turns:
                # Nothing was said yet: give the caller's words back to be answered with what comes next.
                conv.turns.remove(caller_turn)
                self._pending.insert(0, text)
                if self._speaking is u:
                    self._speaking = None
            raise

    async def _play_in_order(self, u: Utterance, voiced: asyncio.Queue) -> None:
        """Send sentences in order — but only start once a couple of seconds are
        ready, so the next sentence is (almost always) voiced before the
        previous one finishes, instead of her speaking in fragments."""
        held: list[tuple[str, bytes]] = []
        while (item := await voiced.get()) is not None:
            sentence, audio_task = item
            audio = await audio_task
            if held is None:
                await self._play(u, sentence, audio)
                continue
            held.append((sentence, audio))
            if sum(len(a) for _, a in held) / BYTES_PER_SECOND >= START_BUFFER_SECONDS:
                for s, a in held:
                    await self._play(u, s, a)
                held = None
        for s, a in held or []:
            await self._play(u, s, a)

    async def _cancel_reply(self) -> None:
        task, self._reply_task = self._reply_task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ── Silence ──────────────────────────────────────────────────────────────

    def _start_silence_watch(self) -> None:
        self._cancel_silence_watch()
        self._silence_task = asyncio.create_task(self._silence_watch())

    def _cancel_silence_watch(self) -> None:
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()
        self._silence_task = None

    async def _silence_watch(self) -> None:
        await asyncio.sleep(SILENCE_SECONDS)
        # From here on it's just her talking: don't let the caller's next words
        # cancel this task halfway through sending the audio.
        self._silence_task = None
        if self._speaking or self._pending or (self._reply_task and not self._reply_task.done()):
            return
        self._silences += 1
        if self._silences > MAX_SILENCES:
            self.conversation.ended = True
            await self._say(GOODBYE_SILENT, end_call=True)
        else:
            await self._say(HELLO)


def _is_bye(text: str) -> bool:
    return any(word in text for word in _BYE)
