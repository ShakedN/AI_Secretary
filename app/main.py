"""FastAPI app: Twilio webhooks for the AI secretary.

Call flow (VOICE_MODE=gather, the default — Hebrew works here):
  office line, no answer → forwarded to our Twilio number
  → POST /voice/incoming  (TwiML: <Gather speech><Say greeting/></Gather>)
  → POST /voice/gather    (each caller utterance → LLM → <Gather><Say reply/></Gather>)
  → call ends             → summary → SQLite + email to the lawyer
  → POST /voice/status    (Twilio status callback: catches callers who hang up)

VOICE_MODE=relay uses ConversationRelay instead (POST /voice/incoming →
WS /relay → POST /voice/after) — streaming and faster, but Twilio doesn't
support Hebrew speech recognition there.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import Connect, Gather, Play, Say, VoiceResponse

from . import db, notify, settings, tts, vocabulary
from .stream import GOODBYE_SILENT, HELLO, TROUBLE, CallStream
from .text import clean_for_speech, split_sentences
from .agent.conversation import Conversation
from .agent.prompt import now_in_israel
from .agent.summary import CallSummary, summarize
from .llm import LLM, make_llm
from .relay import RelaySession, caller_number

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("secretary")

# Calls Twilio announced through the (signature-checked) /voice/incoming
# webhook. The WebSocket only serves these, so a stranger who finds the URL
# can't use it to talk to the LLM on our quota.
_expected_calls: dict[str, float] = {}
_EXPECTED_TTL = 3600
_background: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.office = settings.load_office()
    app.state.llm = make_llm()
    tts.set_pronunciation(app.state.office.pronunciation)
    if tts.enabled():
        # Pre-synthesize the greeting (and warm the Azure connection) so every
        # call opens instantly instead of waiting on the first TTS request.
        try:
            for line in (app.state.office.greeting(), _DIDNT_CATCH, "הלו? אני כאן, אפשר לדבר."):
                await tts.audio_id_for(line)
            for line in (app.state.office.greeting(), HELLO, GOODBYE_SILENT, TROUBLE):
                await tts.synthesize_mulaw(line)
        except Exception:
            logger.exception("could not pre-synthesize the greeting with Azure")
    if not settings.TWILIO_AUTH_TOKEN and settings.PUBLIC_BASE_URL.startswith("https://"):
        logger.warning(
            "TWILIO_AUTH_TOKEN is not set — webhooks are NOT verified, so anyone who finds "
            "this URL can start conversations on your API quota. Set it before going live."
        )
    yield


app = FastAPI(title="Law Office AI Secretary", lifespan=lifespan)


def _twiml(vr: VoiceResponse) -> Response:
    return Response(content=str(vr), media_type="application/xml")


async def _check_twilio_signature(request: Request) -> dict:
    form = dict(await request.form())
    if settings.TWILIO_AUTH_TOKEN:
        url = settings.PUBLIC_BASE_URL + request.url.path
        if request.url.query:
            url += "?" + request.url.query
        signature = request.headers.get("X-Twilio-Signature", "")
        if not RequestValidator(settings.TWILIO_AUTH_TOKEN).validate(url, form, signature):
            raise HTTPException(status_code=403, detail="invalid Twilio signature")
    return form


def build_connect_twiml(office: settings.Office) -> VoiceResponse:
    ws_url = settings.PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://") + "/relay"
    relay_attrs = dict(
        url=ws_url,
        welcome_greeting=office.greeting(),
        language=settings.LANGUAGE,
        tts_provider=settings.TTS_PROVIDER,
        voice=settings.TTS_VOICE,
        transcription_provider=settings.TRANSCRIPTION_PROVIDER,
        interruptible="any",
        welcome_greeting_interruptible="any",
    )
    if settings.SPEECH_MODEL:
        relay_attrs["speech_model"] = settings.SPEECH_MODEL
    vr = VoiceResponse()
    connect = Connect(action=settings.PUBLIC_BASE_URL + "/voice/after")
    connect.conversation_relay(**relay_attrs)
    vr.append(connect)
    return vr


@app.post("/voice/incoming")
async def voice_incoming(request: Request):
    form = await _check_twilio_signature(request)
    call_sid = form.get("CallSid", "")
    logger.info("incoming call %s from %s (%s)", call_sid, form.get("From"), settings.VOICE_MODE)
    if settings.VOICE_MODE == "stream" and tts.enabled():
        return _twiml(build_stream_twiml(form.get("From")))
    if settings.VOICE_MODE in ("gather", "stream"):
        return _twiml(await _start_gather_call(request.app, call_sid, form.get("From")))
    now = time.time()
    for sid, seen in list(_expected_calls.items()):
        if now - seen > _EXPECTED_TTL:
            del _expected_calls[sid]
    _expected_calls[call_sid] = now
    return _twiml(build_connect_twiml(request.app.state.office))


@app.post("/voice/after")
async def voice_after(request: Request):
    await _check_twilio_signature(request)
    vr = VoiceResponse()
    vr.hangup()
    return _twiml(vr)


# ── Gather mode: Twilio recognizes the speech, we answer turn by turn ───────

_gather_calls: dict[str, Conversation] = {}
_gather_silences: dict[str, int] = {}
_gather_noise: dict[str, int] = {}
_MAX_SILENCES = 2        # "hello? are you there?" twice, then say goodbye
_MAX_CALLER_TURNS = 25   # safety net against a call that never ends
# Below this, Twilio's recognition is often wrong (numbers especially), so the
# assistant is told to double-check instead of confidently repeating garbage.
_LOW_CONFIDENCE = 0.6
# Below this it's noise or echo (e.g. "תּוֹדָה" at 0.0 right after the greeting):
# don't feed it to the LLM, just ask again — instantly, from a cached clip.
_NOISE_CONFIDENCE = 0.15
_DIDNT_CATCH = "סליחה, לא שמעתי טוב. אפשר לחזור על זה?"


# ── Was she interrupted? ─────────────────────────────────────────────────────
# Twilio stops playback the moment the caller talks over it, but never says
# how much was heard. We know how long our clip is and when Twilio's answer
# arrived, so: heard ≈ time since we answered − fetch/start delay − the caller's
# own talking − Twilio's end-of-speech wait. Rough, but enough to tell "cut
# after the first words" from "heard it all".

_PLAYBACK_START = 0.6     # Twilio fetching the clip and starting to play
_END_OF_SPEECH_WAIT = 1.2  # speechTimeout="auto" silence before Twilio posts
_SECONDS_PER_WORD = 0.35   # the caller's own speaking time
_now = time.monotonic      # swapped out in tests


@dataclass
class Spoken:
    text: str
    seconds: float
    sent_at: float
    is_greeting: bool = False


_last_spoken: dict[str, Spoken] = {}


def _node_seconds(node: Play | Say) -> float:
    if isinstance(node, Play):
        seconds = tts.clip_seconds(node.value.rsplit("/", 1)[-1].removesuffix(".mp3"))
        if seconds is not None:
            return seconds
    return len(str(node.value).split()) / 2.5  # Twilio <Say>: ~2.5 Hebrew words a second


def _remember(call_sid: str, text: str, twiml: VoiceResponse, is_greeting: bool = False) -> VoiceResponse:
    """Note what we're about to say and how long it plays, for interruption checks."""
    nodes = [n for v in twiml.verbs for n in ([v] if isinstance(v, (Play, Say)) else getattr(v, "verbs", []))]
    seconds = sum(_node_seconds(n) for n in nodes if isinstance(n, (Play, Say)))
    _last_spoken[call_sid] = Spoken(text, seconds, _now(), is_greeting)
    return twiml


def heard_part(spoken: Spoken, caller_text: str) -> str | None:
    """The part of `spoken` the caller heard before talking over it, or None if
    they heard all of it."""
    elapsed = _now() - spoken.sent_at
    talking = len(caller_text.split()) * _SECONDS_PER_WORD + _END_OF_SPEECH_WAIT
    heard = elapsed - _PLAYBACK_START - talking
    if heard >= spoken.seconds - 0.3:
        return None
    share = max(0.0, heard) / spoken.seconds if spoken.seconds else 0.0
    words = spoken.text.split()
    return " ".join(words[: int(len(words) * share)])


def unheard_rest(spoken: Spoken, heard: str) -> str:
    """Everything from the sentence the caller cut into, to say it again."""
    sentences = split_sentences(spoken.text)
    said = 0
    for i, sentence in enumerate(sentences):
        said += len(sentence) + 1
        if said > len(heard):
            return " ".join(sentences[i:])
    return spoken.text


def _say(text: str) -> Say:
    return Say(text, language=settings.LANGUAGE, voice=settings.say_voice())


async def _speak(text: str) -> Play | Say:
    """Azure's voice when configured (played from /audio), else Twilio's <Say>."""
    text = clean_for_speech(text) or text
    if tts.enabled():
        try:
            audio_id = await tts.audio_id_for(text)
            return Play(f"{settings.PUBLIC_BASE_URL}/audio/{audio_id}.mp3")
        except Exception:
            logger.exception("Azure TTS failed — falling back to Twilio <Say>")
    return _say(text)


@app.get("/audio/{audio_id}.mp3")
async def audio(audio_id: str):
    # Ids are unguessable hashes of the text; clips live only in memory.
    data = tts.get_audio(audio_id)
    if data is None:
        raise HTTPException(status_code=404)
    return Response(content=data, media_type="audio/mpeg")


# Twilio's barge-in fires on line noise too (phantom "תּוֹדָה" at 0.0 confidence)
# and has no sensitivity setting, so she got cut off mid-sentence by nothing.
# Off by default; real talk-over handling needs our own audio (Media Streams).
BARGE_IN = False


async def gather_twiml(text: str | list | None, barge_in: bool = BARGE_IN) -> VoiceResponse:
    """Speak `text` (or ready <Play>/<Say> nodes; None = just listen) and
    listen for the answer."""
    if text is None:
        nodes = []
    else:
        nodes = text if isinstance(text, list) else [await _speak(text)]
    vr = VoiceResponse()
    gather = Gather(
        # Keypad too: phone numbers are much more reliable typed than spoken.
        input="dtmf speech",
        finish_on_key="#",
        action=settings.PUBLIC_BASE_URL + "/voice/gather",
        method="POST",
        language=settings.LANGUAGE,
        speech_timeout="auto",
        timeout=6,
        action_on_empty_result=True,
        barge_in=barge_in,
    )
    for node in nodes:
        gather.append(node)
    vr.append(gather)
    return vr


async def goodbye_twiml(text: str | list) -> VoiceResponse:
    nodes = text if isinstance(text, list) else [await _speak(text)]
    vr = VoiceResponse()
    for node in nodes:
        vr.append(node)
    vr.hangup()
    return vr


async def _start_gather_call(app: FastAPI, call_sid: str, from_number: str | None) -> VoiceResponse:
    office = app.state.office
    _gather_calls[call_sid] = Conversation(office=office, llm=app.state.llm, caller_number=caller_number(from_number))
    _gather_silences[call_sid] = 0
    return _remember(call_sid, office.greeting(), await gather_twiml(office.greeting()), is_greeting=True)


def _end_gather_call(app: FastAPI, call_sid: str) -> None:
    """Hand the finished conversation to finish_call — exactly once per call."""
    _gather_silences.pop(call_sid, None)
    _gather_noise.pop(call_sid, None)
    _last_spoken.pop(call_sid, None)
    conv = _gather_calls.pop(call_sid, None)
    if conv is not None:
        task = asyncio.create_task(finish_call(app.state.office, app.state.llm, call_sid, conv))
        _background.add(task)
        task.add_done_callback(_background.discard)


@app.post("/voice/gather")
async def voice_gather(request: Request):
    form = await _check_twilio_signature(request)
    call_sid = form.get("CallSid", "")
    speech = (form.get("SpeechResult") or "").strip()
    digits = (form.get("Digits") or "").strip("#* ")
    noise = False
    if digits:
        speech = f"(הקשתי במקלדת) {' '.join(digits)}"
    elif speech:
        try:
            confidence = float(form.get("Confidence") or 1)
        except ValueError:
            confidence = 1.0
        if confidence < _NOISE_CONFIDENCE:
            noise = True
        elif confidence < _LOW_CONFIDENCE:
            speech += " [זיהוי לא בטוח]"
    conv = _gather_calls.get(call_sid)
    if conv is None:  # e.g. the server restarted mid-call
        return _twiml(await goodbye_twiml("סליחה, הייתה תקלה בקו. אנא התקשרו שוב."))
    spoken = _last_spoken.get(call_sid)
    heard = heard_part(spoken, speech) if (spoken and speech) else None

    if noise:
        _gather_noise[call_sid] = _gather_noise.get(call_sid, 0) + 1
        logger.info("call %s ignoring noise (%s): %s", call_sid, form.get("Confidence"), speech)
        if heard is not None:
            # Noise cut her off: just pick up again from the sentence it cut into.
            rest = unheard_rest(spoken, heard)
            return _twiml(_remember(call_sid, rest, await gather_twiml(rest), spoken.is_greeting))
        if _gather_noise[call_sid] == 1:
            return _twiml(await gather_twiml(None))  # first phantom: keep listening, say nothing
        return _twiml(_remember(call_sid, _DIDNT_CATCH, await gather_twiml(_DIDNT_CATCH)))
    _gather_noise[call_sid] = 0

    if heard is not None:
        logger.info("call %s caller interrupted after: %r", call_sid, heard)
        if spoken.is_greeting:
            speech = f"[נקטעתי בפתיחה — המתקשר שמע רק: \"{heard}\" ואז דיבר] {speech}"
        else:
            conv.mark_interrupted(heard, wanted=spoken.text)

    if not speech:
        _gather_silences[call_sid] = _gather_silences.get(call_sid, 0) + 1
        if _gather_silences[call_sid] > _MAX_SILENCES:
            _end_gather_call(request.app, call_sid)
            return _twiml(await goodbye_twiml("לא שומעים אתכם. אפשר להתקשר שוב בכל עת. להתראות."))
        prompt = "הלו? אני כאן, אפשר לדבר."
        return _twiml(_remember(call_sid, prompt, await gather_twiml(prompt)))

    _gather_silences[call_sid] = 0
    logger.info("call %s caller (%s): %s", call_sid, form.get("Confidence"), speech)
    conv.add_caller(speech)
    # Voice each sentence as soon as the LLM finishes it, in parallel with the
    # rest of the reply still being written, instead of waiting for all of it.
    t0 = time.perf_counter()
    reply, pending, voiced = "", "", []
    async for token in conv.reply():
        reply += token
        pending += token
        *done, pending = split_sentences(pending + "\x00")  # \x00 keeps the unfinished tail separate
        pending = pending.rstrip("\x00")
        voiced += [asyncio.create_task(_speak(s)) for s in done]
    if pending.strip():
        voiced.append(asyncio.create_task(_speak(pending.strip())))
    llm_ms = (time.perf_counter() - t0) * 1000
    nodes = list(await asyncio.gather(*voiced)) or [await _speak(reply)]
    total_ms = (time.perf_counter() - t0) * 1000
    turns = sum(t.role == "caller" for t in conv.turns)
    logger.info("call %s assistant%s: %s", call_sid, " (hanging up)" if conv.ended else "", reply)
    # Our side of the wait (Twilio's own end-of-speech detection comes before this).
    logger.info("call %s timing: llm %.0f ms, ready to speak after %.0f ms", call_sid, llm_ms, total_ms)

    if conv.ended or turns >= _MAX_CALLER_TURNS:
        _end_gather_call(request.app, call_sid)
        return _twiml(await goodbye_twiml(nodes))
    return _twiml(_remember(call_sid, reply, await gather_twiml(nodes)))


@app.post("/voice/status")
async def voice_status(request: Request):
    """Twilio's call status callback — the only way we learn the caller hung up."""
    form = await _check_twilio_signature(request)
    if form.get("CallStatus") in ("completed", "busy", "failed", "no-answer", "canceled"):
        _end_gather_call(request.app, form.get("CallSid", ""))
    return {"ok": True}


# ── Stream mode: Twilio Media Streams, our own listening and talking ───────

# One-time tokens handed to Twilio in the TwiML and checked when the media
# WebSocket starts, so only calls that came through /voice/incoming get in.
_stream_tokens: dict[str, float] = {}


def build_stream_twiml(from_number: str | None) -> VoiceResponse:
    now = time.time()
    for token, issued in list(_stream_tokens.items()):
        if now - issued > _EXPECTED_TTL:
            del _stream_tokens[token]
    token = secrets.token_urlsafe(16)
    _stream_tokens[token] = now
    vr = VoiceResponse()
    connect = Connect()
    stream = connect.stream(url=settings.PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://") + "/media")
    stream.parameter(name="token", value=token)
    stream.parameter(name="from", value=from_number or "")
    vr.append(connect)
    vr.hangup()  # runs when we close the stream after the goodbye
    return vr


@app.websocket("/media")
async def media(ws: WebSocket):
    from .stt import AzureRecognizer

    await ws.accept()
    office, llm = ws.app.state.office, ws.app.state.llm
    session = CallStream(
        send=ws.send_json,
        office=office,
        llm=llm,
        make_recognizer=lambda on_partial, on_final, on_error: AzureRecognizer(
            on_partial, on_final, on_error,
            phrases=vocabulary.phrases(office), phrase_weight=vocabulary.PHRASE_WEIGHT,
        ),
        synthesize=tts.synthesize_mulaw,
        is_allowed=lambda token: _stream_tokens.pop(token, None) is not None,
        record=settings.RECORD_CALLS,
    )
    try:
        while not session.closed:
            await session.handle(await ws.receive_json())
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
        if session.recording:
            from . import recording

            path = recording.save(settings.DB_PATH.parent / "recordings", session.call_sid or "call", bytes(session.recording))
            logger.info("call %s audio saved to %s", session.call_sid, path)
        if session.conversation is not None:
            task = asyncio.create_task(finish_call(office, llm, session.call_sid, session.conversation))
            _background.add(task)
            task.add_done_callback(_background.discard)


# ── Voice samples: call the number and hear every Hebrew voice in turn ─────
# Point the Twilio number at /voice/samples (scripts/twilio_number.py) to pick
# a voice by ear on a real phone line, then put the winner in TTS_VOICE.

# Only WaveNet: the Standard voices are the robotic-sounding ones.
SAMPLE_VOICES = ["he-IL-Wavenet-A", "he-IL-Wavenet-B", "he-IL-Wavenet-C", "he-IL-Wavenet-D"]
_SAMPLE_TEXT = "שלום, הגעתם למשרד עורכי דין. כרגע אין מי שיענה, אבל אשמח לרשום את הפנייה ולדאוג שיחזרו אליכם. במה מדובר?"


def build_samples_twiml() -> VoiceResponse:
    vr = VoiceResponse()
    for i, voice in enumerate(SAMPLE_VOICES, 1):
        vr.say(f"קול מספר {i}. {_SAMPLE_TEXT}", language="he-IL", voice=f"Google.{voice}")
        vr.pause(length=1)
    vr.say("זה הכל. תודה!", language="he-IL", voice=f"Google.{SAMPLE_VOICES[-1]}")
    vr.hangup()
    return vr


@app.post("/voice/samples")
async def voice_samples(request: Request):
    await _check_twilio_signature(request)
    return _twiml(build_samples_twiml())


@app.get("/health")
async def health():
    return {"ok": True, "llm": settings.LLM_PROVIDER if app.state.llm is not None else None}


@app.get("/", response_class=HTMLResponse)
async def home():
    # A browser check that the server is up — this address is really for Twilio.
    ok = app.state.llm is not None
    status = "פעילה ועונה לשיחות" if ok else "השרת פועל, אבל אין חיבור למודל השפה"
    return HTMLResponse(
        '<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><title>מזכירה דיגיטלית</title>'
        '<body style="font-family:system-ui,Arial;max-width:520px;margin:15vh auto;padding:0 16px;text-align:center">'
        f'<div style="font-size:48px">{"✅" if ok else "⚠️"}</div><h1>המזכירה הדיגיטלית {status}</h1>'
        '<p style="color:#666">הכתובת הזו משמשת את מערכת הטלפון (Twilio). אין כאן מה לעשות בדפדפן.</p></body></html>'
    )


# ── Browser voice test (no phone, no Twilio) ────────────────────────────────
# /test plays the part of Twilio ConversationRelay: the browser does speech
# recognition + speech synthesis and talks to /relay with the same messages,
# so everything after it — conversation, summary, email — is the real path.

_TEST_PAGE = Path(__file__).parent / "static" / "test.html"


def _require_local_test_page(request: Request) -> None:
    # ngrok also connects from localhost, but adds X-Forwarded-For.
    is_local = request.client is not None and request.client.host in ("127.0.0.1", "::1", "testclient")
    if not (settings.TEST_PAGE and is_local and "x-forwarded-for" not in request.headers):
        raise HTTPException(status_code=404)


@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    _require_local_test_page(request)
    return HTMLResponse(_TEST_PAGE.read_text(encoding="utf-8"))


@app.post("/test/start")
async def test_start(request: Request):
    _require_local_test_page(request)
    call_sid = f"TEST-{uuid.uuid4().hex[:12]}"
    _expected_calls[call_sid] = time.time()
    office = request.app.state.office
    return {
        "call_sid": call_sid,
        "greeting": office.greeting(),
        "assistant_name": office.assistant_name,
        "llm": f"{settings.LLM_PROVIDER} / {settings.CHAT_MODEL}" if request.app.state.llm else None,
        "email": settings.EMAIL_TO_OVERRIDE if settings.SMTP_USER else None,
    }


@app.get("/test/result/{call_sid}")
async def test_result(call_sid: str, request: Request):
    _require_local_test_page(request)
    row = await asyncio.to_thread(db.get_call, settings.DB_PATH, call_sid)
    return row or {"pending": True}


@app.websocket("/relay")
async def relay(ws: WebSocket):
    await ws.accept()
    office, llm = ws.app.state.office, ws.app.state.llm

    def make_conversation(caller_number: str | None) -> Conversation:
        return Conversation(office=office, llm=llm, caller_number=caller_number)

    session = RelaySession(
        send=ws.send_json,
        make_conversation=make_conversation,
        is_allowed=lambda sid: sid in _expected_calls,
    )
    try:
        while not session.closed:
            await session.handle(await ws.receive_json())
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
        _expected_calls.pop(session.call_sid or "", None)
        if session.conversation is not None:
            task = asyncio.create_task(finish_call(office, llm, session.call_sid, session.conversation))
            _background.add(task)
            task.add_done_callback(_background.discard)


async def finish_call(office: settings.Office, llm: LLM | None, call_sid: str | None, conv: Conversation) -> None:
    """Summarize, store, and email — runs after the caller is gone."""
    try:
        if any(t.role == "caller" for t in conv.turns):
            summary = await summarize(llm, office, conv.transcript(), conv.caller_number)
        elif conv.caller_number:
            summary = CallSummary(callback_phone=conv.caller_number, summary="המתקשר ניתק לפני שהשאיר הודעה.")
        else:
            logger.info("call %s: withheld number hung up without a message — nothing to send", call_sid)
            return

        transcript = conv.transcript()
        ended_at = now_in_israel()
        to = notify.recipients(office, summary)
        msg = notify.build_email(
            office, summary, transcript, f"{conv.started_at:%d/%m/%Y %H:%M}",
            duration_seconds=int((ended_at - conv.started_at).total_seconds()),
            caller_turns=sum(t.role == "caller" for t in conv.turns),
        )
        sent = await asyncio.to_thread(notify.send_email, msg, to)
        await asyncio.to_thread(
            db.save_call,
            settings.DB_PATH,
            call_sid=call_sid or "",
            from_number=conv.caller_number,
            started_at=conv.started_at,
            ended_at=ended_at,
            summary=summary,
            transcript=transcript,
            emailed_to=to if sent else [],
        )
        logger.info("call %s done → %s", call_sid, ", ".join(to) if sent else "not emailed")
    except Exception:
        logger.exception("finishing call %s failed", call_sid)
