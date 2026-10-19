"""FastAPI app: Twilio webhooks + the ConversationRelay WebSocket.

Call flow:
  office line, no answer → forwarded to our Twilio number
  → POST /voice/incoming  (TwiML: <Connect><ConversationRelay url=wss://…/relay>)
  → WS /relay             (live conversation with the LLM — Gemini or Claude)
  → call ends             → summary → SQLite + email to the lawyer
  → POST /voice/after     (TwiML: <Hangup/>)
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
from twilio.twiml.voice_response import Connect, VoiceResponse

from . import db, notify, settings
from .agent.conversation import Conversation
from .agent.prompt import now_in_israel
from .agent.summary import CallSummary, summarize
from .llm import LLM, make_llm
from .relay import RelaySession

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
    now = time.time()
    for sid, seen in list(_expected_calls.items()):
        if now - seen > _EXPECTED_TTL:
            del _expected_calls[sid]
    _expected_calls[call_sid] = now
    logger.info("incoming call %s from %s", call_sid, form.get("From"))
    return _twiml(build_connect_twiml(request.app.state.office))


@app.post("/voice/after")
async def voice_after(request: Request):
    await _check_twilio_signature(request)
    vr = VoiceResponse()
    vr.hangup()
    return _twiml(vr)


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
