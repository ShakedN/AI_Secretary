import asyncio
import time

from fastapi.testclient import TestClient

from app import db, main, notify, settings
from app.agent.conversation import Conversation
from app.agent.summary import CallSummary, summarize

from .conftest import FakeLLM

SUMMARY = {
    "caller_name": "משה",
    "callback_phone": "0501234567",
    "lawyer_requested": 'עו"ד דנה כהן',
    "topic": "גירושין",
    "summary": "משה מבקש ייעוץ בנושא גירושין.",
    "urgency": "high",
    "existing_client": "no",
}


def test_summarize_and_route(office):
    summary = asyncio.run(summarize(FakeLLM(summary=SUMMARY), office, "תמליל", "0501234567"))
    assert summary.caller_name == "משה" and summary.urgency == "high"
    assert notify.recipients(office, summary) == ["dana@example.com", "office@example.com"]

    msg = notify.build_email(office, summary, "תמליל", "01/01/2026 10:00")
    assert msg["Subject"].startswith("דחוף!") and "משה" in msg["Subject"]


def test_unknown_or_emailless_lawyer_goes_to_office(office):
    assert notify.recipients(office, CallSummary(lawyer_requested="המשרד")) == ["office@example.com"]
    assert notify.recipients(office, CallSummary(lawyer_requested='עו"ד יוסי לוי')) == ["office@example.com"]


def test_finish_call_saves_to_db(office, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "calls.db")
    monkeypatch.setattr(settings, "SMTP_HOST", "")
    client = FakeLLM(summary=SUMMARY)
    conv = Conversation(office=office, llm=client, caller_number="0501234567")
    conv.add_caller("שלום, אני משה")

    asyncio.run(main.finish_call(office, client, "CA1", conv))

    [row] = db.recent_calls(tmp_path / "calls.db")
    assert row["call_sid"] == "CA1" and row["caller_name"] == "משה"
    assert "אני משה" in row["transcript"]
    assert row["emailed_to"] == ""  # SMTP not configured


def test_incoming_webhook_returns_conversation_relay(monkeypatch):
    monkeypatch.setattr(settings, "TWILIO_AUTH_TOKEN", "")
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://example.ngrok.app")
    with TestClient(main.app) as http:
        resp = http.post("/voice/incoming", data={"CallSid": "CA9", "From": "+972501234567"})
    assert resp.status_code == 200
    assert "<ConversationRelay" in resp.text
    assert 'url="wss://example.ngrok.app/relay"' in resp.text
    assert 'language="he-IL"' in resp.text
    assert "CA9" in main._expected_calls


def test_websocket_call_end_to_end(monkeypatch):
    from app import relay

    monkeypatch.setattr(relay, "_WORDS_PER_MINUTE", 10**6)
    finished = []

    async def fake_finish(office, client, call_sid, conv):
        finished.append((call_sid, conv.transcript()))

    monkeypatch.setattr(main, "finish_call", fake_finish)
    with TestClient(main.app) as http:
        main.app.state.llm = FakeLLM([(["שלום משה"], False), (["להתראות"], True)])
        main._expected_calls["CA5"] = time.time()
        with http.websocket_connect("/relay") as ws:
            ws.send_json({"type": "setup", "callSid": "CA5", "from": "+972501234567"})
            ws.send_json({"type": "prompt", "voicePrompt": "היי זה משה", "last": True})
            assert ws.receive_json() == {"type": "text", "token": "שלום משה", "last": False}
            assert ws.receive_json()["last"] is True
            ws.send_json({"type": "prompt", "voicePrompt": "תודה ביי", "last": True})
            assert ws.receive_json()["token"] == "להתראות"
            assert ws.receive_json()["last"] is True
            assert ws.receive_json()["type"] == "end"
            ws.send_json({"type": "dtmf", "digit": "1"})  # unblocks the receive loop so it sees `closed`

    assert finished and finished[0][0] == "CA5"
    assert "היי זה משה" in finished[0][1]
    assert "CA5" not in main._expected_calls


def test_incoming_webhook_rejects_bad_signature(monkeypatch):
    monkeypatch.setattr(settings, "TWILIO_AUTH_TOKEN", "secret")
    with TestClient(main.app) as http:
        resp = http.post("/voice/incoming", data={"CallSid": "CA9"}, headers={"X-Twilio-Signature": "nope"})
    assert resp.status_code == 403


def test_email_override_sends_everything_to_tester(office, monkeypatch):
    monkeypatch.setattr(settings, "EMAIL_TO_OVERRIDE", "me@example.com")
    assert notify.recipients(office, CallSummary(lawyer_requested='עו"ד דנה כהן')) == ["me@example.com"]


def test_email_includes_how_the_call_went(office):
    summary = CallSummary(**SUMMARY, caller_mood="upset", call_review="השיחה זרמה", next_step="לחזור היום")
    msg = notify.build_email(office, summary, "תמליל", "01/01/2026 10:00", duration_seconds=95, caller_turns=4)
    text = msg.get_body(("plain",)).get_content()
    assert "1:35 דקות" in text and "נסער/ת" in text and "השיחה זרמה" in text and "לחזור היום" in text


def test_test_page_disabled_by_default(monkeypatch):
    monkeypatch.setattr(settings, "TEST_PAGE", False)
    with TestClient(main.app) as http:
        assert http.get("/test").status_code == 404
        assert http.post("/test/start").status_code == 404


def test_test_page_blocks_requests_through_ngrok(monkeypatch):
    monkeypatch.setattr(settings, "TEST_PAGE", True)
    with TestClient(main.app) as http:
        assert http.get("/test", headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 404


def test_test_page_flow(office, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "TEST_PAGE", True)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "calls.db")
    monkeypatch.setattr(settings, "SMTP_USER", "")
    with TestClient(main.app) as http:
        assert "בדיקת שיחה" in http.get("/test").text
        start = http.post("/test/start").json()
        sid = start["call_sid"]
        assert sid in main._expected_calls and start["greeting"]
        assert http.get(f"/test/result/{sid}").json() == {"pending": True}

        conv = Conversation(office=office, llm=FakeLLM(summary=SUMMARY), caller_number="0501234567")
        conv.add_caller("שלום")
        asyncio.run(main.finish_call(office, FakeLLM(summary=SUMMARY), sid, conv))
        result = http.get(f"/test/result/{sid}").json()
        assert result["caller_name"] == "משה" and result["duration_seconds"] >= 0
