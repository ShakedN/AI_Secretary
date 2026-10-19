"""VOICE_MODE=gather: Twilio <Gather> speech recognition + <Say>, turn by turn."""
import pytest
from fastapi.testclient import TestClient

from app import main, settings

from .conftest import FakeLLM


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_MODE", "gather")
    monkeypatch.setattr(settings, "TWILIO_AUTH_TOKEN", "")
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://example.ngrok.app")
    monkeypatch.setattr(settings, "TTS_PROVIDER", "Google")
    monkeypatch.setattr(settings, "TTS_VOICE", "he-IL-Wavenet-C")
    finished = []

    async def fake_finish(office, llm, call_sid, conv):
        finished.append((call_sid, conv))

    monkeypatch.setattr(main, "finish_call", fake_finish)
    with TestClient(main.app) as client:
        client.finished = finished
        yield client
    main._gather_calls.clear()
    main._gather_silences.clear()


def start(http, sid="CA1"):
    return http.post("/voice/incoming", data={"CallSid": sid, "From": "+972501234567"})


def test_incoming_greets_and_listens_in_hebrew(http):
    xml = start(http).text
    assert '<Gather action="https://example.ngrok.app/voice/gather"' in xml
    assert 'input="dtmf speech"' in xml and 'language="he-IL"' in xml
    assert 'voice="Google.he-IL-Wavenet-C"' in xml
    assert "שלום" in xml
    assert main._gather_calls["CA1"].caller_number == "+972501234567"


def test_turn_replies_with_llm_text(http):
    http.app.state.llm = FakeLLM([(["היי, ", "במה מדובר?"], False)])
    start(http)
    xml = http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "שלום", "Confidence": "0.9"}).text
    assert "היי, במה מדובר?" in xml and "<Gather" in xml and "<Hangup" not in xml
    assert main._gather_calls["CA1"].turns[0].text == "שלום"


def test_end_call_says_goodbye_hangs_up_and_finishes_once(http):
    http.app.state.llm = FakeLLM([(["להתראות!"], True)])
    start(http)
    xml = http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "תודה, ביי"}).text
    assert "להתראות!" in xml and "<Hangup" in xml and "<Gather" not in xml
    # Twilio's status callback after our hangup must not finish the call twice.
    http.post("/voice/status", data={"CallSid": "CA1", "CallStatus": "completed"})
    assert [sid for sid, _ in http.finished] == ["CA1"]


def test_silence_reprompts_then_gives_up(http):
    start(http)
    for _ in range(main._MAX_SILENCES):
        xml = http.post("/voice/gather", data={"CallSid": "CA1"}).text
        assert "<Gather" in xml
    xml = http.post("/voice/gather", data={"CallSid": "CA1"}).text
    assert "<Hangup" in xml
    assert [sid for sid, _ in http.finished] == ["CA1"]


def test_caller_hanging_up_mid_call_still_sends_summary(http):
    http.app.state.llm = FakeLLM([(["מה שמך?"], False)])
    start(http)
    http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "רציתי לדבר עם דנה"})
    http.post("/voice/status", data={"CallSid": "CA1", "CallStatus": "completed"})
    [(sid, conv)] = http.finished
    assert sid == "CA1" and conv.turns[0].text == "רציתי לדבר עם דנה"


def test_unknown_call_gets_polite_goodbye(http):
    xml = http.post("/voice/gather", data={"CallSid": "nope", "SpeechResult": "הלו"}).text
    assert "<Hangup" in xml


def test_keypad_digits_reach_the_llm_exactly(http):
    http.app.state.llm = FakeLLM([(["תודה"], False)])
    start(http)
    http.post("/voice/gather", data={"CallSid": "CA1", "Digits": "0587858000#"})
    assert main._gather_calls["CA1"].turns[0].text == "(הקשתי במקלדת) 0 5 8 7 8 5 8 0 0 0"


def test_low_confidence_speech_is_flagged(http):
    http.app.state.llm = FakeLLM([(["סליחה?"], False)])
    start(http)
    http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "058 7 8 58 076", "Confidence": "0.47"})
    assert main._gather_calls["CA1"].turns[0].text.endswith("[זיהוי לא בטוח]")


def test_gather_accepts_keypad(http):
    xml = start(http).text
    assert 'input="dtmf speech"' in xml and 'finishOnKey="#"' in xml
