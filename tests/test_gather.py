"""VOICE_MODE=gather: Twilio <Gather> speech recognition + <Say>, turn by turn."""
import pytest
from fastapi.testclient import TestClient

from app import main, settings

from .conftest import FakeLLM

FAKE_MP3 = b"ID3" + b"\x00" * 40_000  # ~10 s at 32 kbit/s


class FakeClock:
    def __init__(self, auto_step=0.0):
        self.t, self.auto_step = 1000.0, auto_step

    def __call__(self):
        self.t += self.auto_step
        return self.t


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_MODE", "gather")
    monkeypatch.setattr(settings, "TWILIO_AUTH_TOKEN", "")
    monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://example.ngrok.app")
    monkeypatch.setattr(settings, "TTS_PROVIDER", "Google")
    monkeypatch.setattr(settings, "TTS_VOICE", "he-IL-Wavenet-C")
    monkeypatch.setattr(settings, "AZURE_SPEECH_KEY", "")
    finished = []

    async def fake_finish(office, llm, call_sid, conv):
        finished.append((call_sid, conv))

    monkeypatch.setattr(main, "finish_call", fake_finish)
    monkeypatch.setattr(main, "_now", FakeClock(auto_step=120))  # callers always hear everything
    with TestClient(main.app) as client:
        client.finished = finished
        yield client
    main._gather_calls.clear()
    main._gather_silences.clear()
    main._last_spoken.clear()


def wait_finished(http, n=1):
    """finish_call runs as a background task on the app's loop — give it a moment."""
    import time
    for _ in range(200):
        if len(http.finished) >= n:
            return
        time.sleep(0.01)


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
    xml = http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "קוראים לי משה, תודה ביי"}).text
    assert "להתראות!" in xml and "<Hangup" in xml and "<Gather" not in xml
    # Twilio's status callback after our hangup must not finish the call twice.
    http.post("/voice/status", data={"CallSid": "CA1", "CallStatus": "completed"})
    wait_finished(http)
    assert [sid for sid, _ in http.finished] == ["CA1"]


def test_silence_reprompts_then_gives_up(http):
    start(http)
    for _ in range(main._MAX_SILENCES):
        xml = http.post("/voice/gather", data={"CallSid": "CA1"}).text
        assert "<Gather" in xml
    xml = http.post("/voice/gather", data={"CallSid": "CA1"}).text
    assert "<Hangup" in xml
    wait_finished(http)
    assert [sid for sid, _ in http.finished] == ["CA1"]


def test_caller_hanging_up_mid_call_still_sends_summary(http):
    http.app.state.llm = FakeLLM([(["מה שמך?"], False)])
    start(http)
    http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "רציתי לדבר עם דנה"})
    http.post("/voice/status", data={"CallSid": "CA1", "CallStatus": "completed"})
    wait_finished(http)
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


@pytest.fixture
def azure(monkeypatch):
    from app import tts
    monkeypatch.setattr(settings, "AZURE_SPEECH_KEY", "k")
    monkeypatch.setattr(settings, "AZURE_SPEECH_REGION", "westeurope")
    calls = []

    async def fake_synthesize(text):
        calls.append(text)
        return FAKE_MP3

    monkeypatch.setattr(tts, "_azure_synthesize", fake_synthesize)
    tts._cache.clear()
    return calls


def test_azure_voice_is_played_instead_of_say(http, azure):
    import re
    xml = start(http).text
    assert "<Say" not in xml
    url = re.search(r"<Play>([^<]+)</Play>", xml).group(1)
    assert url.startswith("https://example.ngrok.app/audio/")
    audio = http.get(url.replace("https://example.ngrok.app", ""))
    assert audio.status_code == 200 and audio.content == FAKE_MP3
    assert audio.headers["content-type"] == "audio/mpeg"


def test_repeated_lines_are_synthesized_once(http, azure):
    start(http, "CA1")
    start(http, "CA2")
    assert len(azure) == 1  # same greeting, cached


def test_azure_failure_falls_back_to_twilio_voice(http, monkeypatch):
    from app import tts
    monkeypatch.setattr(settings, "AZURE_SPEECH_KEY", "k")
    monkeypatch.setattr(settings, "AZURE_SPEECH_REGION", "westeurope")

    async def broken(text):
        raise RuntimeError("azure down")

    monkeypatch.setattr(tts, "_azure_synthesize", broken)
    tts._cache.clear()
    xml = start(http).text
    assert 'voice="Google.he-IL-Wavenet-C"' in xml and "<Play" not in xml


def test_unknown_audio_is_404(http):
    assert http.get("/audio/" + "0" * 32 + ".mp3").status_code == 404


def test_noise_cannot_cut_her_off(http):
    # Twilio's barge-in triggers on line noise, so it's off.
    xml = start(http).text
    assert 'bargeIn="false"' in xml


def test_replies_are_not_cut_by_noise_either(http):
    http.app.state.llm = FakeLLM([(["מה שמך?"], False)])
    start(http)
    xml = http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "שלום"}).text
    assert 'bargeIn="false"' in xml


def test_each_sentence_is_voiced_separately(http, azure):
    http.app.state.llm = FakeLLM([(["הבנתי. ", "איך ", "קוראים לך?"], False)])
    start(http)
    azure.clear()
    xml = http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "אני צריך חוזה"}).text
    assert azure == ["הבנתי.", "איך קוראים לך?"]
    assert xml.count("<Play>") == 2


def test_split_sentences():
    assert main.split_sentences("שלום. מה שמך? ") == ["שלום.", "מה שמך?"]
    assert main.split_sentences("בלי נקודה") == ["בלי נקודה"]
    assert main.split_sentences("רק לוודא — 0 5 8, 7 8 5? תודה.") == ["רק לוודא — 0 5 8, 7 8 5?", "תודה."]


def test_noise_is_not_sent_to_the_llm(http):
    http.app.state.llm = FakeLLM([])  # any LLM call would fail: no replies queued
    start(http)
    noise = {"CallSid": "CA1", "SpeechResult": "תּוֹדָה.", "Confidence": "0.0"}
    first = http.post("/voice/gather", data=noise).text
    assert "<Gather" in first and main._DIDNT_CATCH not in first  # silently keep listening
    second = http.post("/voice/gather", data=noise).text
    assert main._DIDNT_CATCH in second
    assert main._gather_calls["CA1"].turns == []


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(main, "_now", c)
    return c


def test_talking_over_the_greeting_is_flagged_for_the_llm(http, clock):
    llm = FakeLLM([(["כאן נועה מהמשרד. במה אפשר לעזור?"], False)])
    http.app.state.llm = llm
    start(http)
    greeting_seconds = main._last_spoken["CA1"].seconds
    assert greeting_seconds > 4
    clock.t += 3.5  # caller said "הלו? מי מדבר?" ~1 s into the greeting
    http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "הלו? מי מדבר?", "Confidence": "0.9"})
    caller = main._gather_calls["CA1"].turns[0].text
    assert caller.startswith("[נקטעתי בפתיחה") and caller.endswith("הלו? מי מדבר?")


def test_listening_to_the_end_is_not_an_interruption(http, clock):
    http.app.state.llm = FakeLLM([(["בשמחה."], False)])
    start(http)
    clock.t += main._last_spoken["CA1"].seconds + 4
    http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "אני רוצה חוזה", "Confidence": "0.9"})
    assert main._gather_calls["CA1"].turns[0].text == "אני רוצה חוזה"


def test_interrupted_reply_keeps_what_she_wanted_to_say(http, clock):
    long_reply = "הבנתי שמדובר בחוזה לדירה. כדי שעורך הדין יחזור אליך, איך קוראים לך ומה המספר שלך?"
    http.app.state.llm = FakeLLM([([long_reply], False), (["בטח."], False)])
    start(http)
    clock.t += 60
    http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "אני צריך חוזה", "Confidence": "0.9"})
    clock.t += 3.0  # talked over the reply almost immediately
    http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "רגע רגע", "Confidence": "0.9"})
    turns = main._gather_calls["CA1"].turns
    assert "[נקטעתי]" in turns[1].text and long_reply in turns[1].text
    assert turns[2].text == "רגע רגע"


def test_noise_over_the_greeting_resumes_it(http, clock, azure):
    start(http)
    clock.t += 2.5  # noise right at the start
    xml = http.post("/voice/gather", data={"CallSid": "CA1", "SpeechResult": "תּוֹדָה", "Confidence": "0.0"}).text
    assert "<Play>" in xml
    assert main._last_spoken["CA1"].is_greeting
    # Resumes from the sentence the noise cut into — here the very first one.
    assert main._last_spoken["CA1"].text.startswith("שלום")
    assert main._gather_calls["CA1"].turns == []
