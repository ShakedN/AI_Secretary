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
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import Connect, Gather, Say, VoiceResponse

from . import db, notify, settings
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
    if settings.VOICE_MODE == "gather":
        return _twiml(_start_gather_call(request.app, call_sid, form.get("From")))
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
_MAX_SILENCES = 2        # "hello? are you there?" twice, then say goodbye
_MAX_CALLER_TURNS = 25   # safety net against a call that never ends
# Below this, Twilio's recognition is often wrong (numbers especially), so the
# assistant is told to double-check instead of confidently repeating garbage.
_LOW_CONFIDENCE = 0.6


def _say(text: str) -> Say:
    return Say(text, language=settings.LANGUAGE, voice=settings.say_voice())


def gather_twiml(text: str) -> VoiceResponse:
    """Speak `text` and listen for the caller's answer (they can talk over it)."""
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
    )
    gather.append(_say(text))
    vr.append(gather)
    return vr


def goodbye_twiml(text: str) -> VoiceResponse:
    vr = VoiceResponse()
    vr.append(_say(text))
    vr.hangup()
    return vr


def _start_gather_call(app: FastAPI, call_sid: str, from_number: str | None) -> VoiceResponse:
    office = app.state.office
    _gather_calls[call_sid] = Conversation(office=office, llm=app.state.llm, caller_number=caller_number(from_number))
    _gather_silences[call_sid] = 0
    return gather_twiml(office.greeting())


def _end_gather_call(app: FastAPI, call_sid: str) -> None:
    """Hand the finished conversation to finish_call — exactly once per call."""
    _gather_silences.pop(call_sid, None)
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
    if digits:
        speech = f"(הקשתי במקלדת) {' '.join(digits)}"
    else:
        try:
            if speech and float(form.get("Confidence") or 1) < _LOW_CONFIDENCE:
                speech += " [זיהוי לא בטוח]"
        except ValueError:
            pass
    conv = _gather_calls.get(call_sid)
    if conv is None:  # e.g. the server restarted mid-call
        return _twiml(goodbye_twiml("סליחה, הייתה תקלה בקו. אנא התקשרו שוב."))

    if not speech:
        _gather_silences[call_sid] = _gather_silences.get(call_sid, 0) + 1
        if _gather_silences[call_sid] > _MAX_SILENCES:
            _end_gather_call(request.app, call_sid)
            return _twiml(goodbye_twiml("לא שומעים אתכם. אפשר להתקשר שוב בכל עת. להתראות."))
        return _twiml(gather_twiml("הלו? אני כאן, אפשר לדבר."))

    _gather_silences[call_sid] = 0
    logger.info("call %s caller (%s): %s", call_sid, form.get("Confidence"), speech)
    conv.add_caller(speech)
    reply = "".join([token async for token in conv.reply()])
    turns = sum(t.role == "caller" for t in conv.turns)
    logger.info("call %s assistant%s: %s", call_sid, " (hanging up)" if conv.ended else "", reply)

    if conv.ended or turns >= _MAX_CALLER_TURNS:
        _end_gather_call(request.app, call_sid)
        return _twiml(goodbye_twiml(reply))
    return _twiml(gather_twiml(reply))


@app.post("/voice/status")
async def voice_status(request: Request):
    """Twilio's call status callback — the only way we learn the caller hung up."""
    form = await _check_twilio_signature(request)
    if form.get("CallStatus") in ("completed", "busy", "failed", "no-answer", "canceled"):
        _end_gather_call(request.app, form.get("CallSid", ""))
    return {"ok": True}


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
