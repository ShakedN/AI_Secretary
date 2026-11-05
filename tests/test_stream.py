"""Stream mode (Twilio Media Streams): listening, talking, and being talked over."""
import asyncio
import base64

from app import stream as stream_mod
from app.stream import CallStream

from .conftest import FakeLLM

SECONDS_PER_WORD = 0.5


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeRecognizer:
    def __init__(self):
        self.audio = b""
        self.closed = False

    def push(self, audio):
        self.audio += audio

    def close(self):
        self.closed = True


async def fake_synthesize(text):
    return b"\xff" * int(len(text.split()) * SECONDS_PER_WORD * stream_mod.BYTES_PER_SECOND)


def make(office, llm=None, allowed=True):
    sent, clock = [], Clock()

    async def send(msg):
        sent.append(msg)

    s = CallStream(
        send=send, office=office, llm=llm or FakeLLM(), clock=clock,
        make_recognizer=lambda *handlers: FakeRecognizer(),
        synthesize=fake_synthesize, is_allowed=lambda token: allowed,
    )
    return s, sent, clock


START = {"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1",
                                     "customParameters": {"token": "t", "from": "+972587858000"}}}


async def finish_speaking(s, clock):
    """Let everything sent so far play, and have Twilio report the mark."""
    u = s._speaking
    clock.t = s._play_cursor + 0.1
    await s.handle({"event": "mark", "mark": {"name": f"u{u.id}"}})


def kinds(sent):
    return [m["event"] for m in sent]


def test_start_greets_and_rejects_strangers(office):
    async def scenario():
        s, sent, _ = make(office)
        await s.handle(START)
        bad, _, _ = make(office, allowed=False)
        await bad.handle(START)
        return s, sent, bad

    s, sent, bad = asyncio.run(scenario())
    assert s.conversation.caller_number == "+972587858000"
    assert kinds(sent).count("media") > 1 and kinds(sent)[-1] == "mark"
    assert all(m["streamSid"] == "MZ1" for m in sent)
    assert bad.closed and bad.conversation is None


def test_caller_audio_goes_to_the_recognizer(office):
    async def scenario():
        s, _, _ = make(office)
        await s.handle(START)
        await s.handle({"event": "media", "media": {"payload": base64.b64encode(b"\x01\x02").decode()}})
        return s

    assert asyncio.run(scenario())._recognizer.audio == b"\x01\x02"


def test_a_turn_gets_a_spoken_reply(office):
    async def scenario():
        s, sent, clock = make(office, FakeLLM([(["בשמחה. ", "איך קוראים לך?"], False)]))
        await s.handle(START)
        await finish_speaking(s, clock)
        sent.clear()
        await s.on_final("אני צריך עורך דין")
        await s._reply_task
        return s, sent

    s, sent = asyncio.run(scenario())
    turns = s.conversation.turns
    assert [t.role for t in turns] == ["caller", "assistant"]
    assert turns[1].text == "בשמחה. איך קוראים לך?"
    assert kinds(sent)[-1] == "mark" and "media" in kinds(sent)


def test_talking_over_the_greeting_early_is_not_an_answer(office):
    async def scenario():
        s, sent, clock = make(office, FakeLLM([(["כאן נועה מהמשרד."], False)]))
        await s.handle(START)
        clock.t += 1.0  # a second into a long greeting
        await s.on_partial("הלו מי")
        await s.on_final("הלו? מי מדבר?")
        await s._reply_task
        return s, sent

    s, sent = asyncio.run(scenario())
    assert "clear" in kinds(sent)
    caller = s.conversation.turns[0].text
    assert caller.startswith("[הערת מערכת: המתקשר דיבר מעל הפתיחה") and caller.endswith("הלו? מי מדבר?")


def reply_then_interrupt(office, after_share):
    """She asks a 10-word question; the caller talks over it after `after_share` of it."""
    question = "כדי שעורך הדין יחזור אליך אני צריכה את השם ואת המספר"
    llm = FakeLLM([([question], False), (["בטח."], False)])

    async def scenario():
        s, sent, clock = make(office, llm)
        await s.handle(START)
        await finish_speaking(s, clock)
        await s.on_final("אני צריך חוזה")
        await s._reply_task
        clock.t = s._speaking.segments[0].start + s._speaking.segments[0].seconds * after_share
        await s.on_partial("רגע רגע")
        await s.on_final("רגע רגע זה דחוף")
        await s._reply_task
        return s, sent, question

    return asyncio.run(scenario())


def test_cut_off_early_she_knows_what_was_missed(office):
    s, sent, question = reply_then_interrupt(office, after_share=0.3)
    assert "clear" in kinds(sent)
    said, caller = s.conversation.turns[1].text, s.conversation.turns[2].text
    assert said.endswith("…") and "[" not in said  # her turn: only what was heard
    assert question in caller and caller.endswith("רגע רגע זה דחוף")  # the note: what she meant to say


def test_two_thirds_heard_counts_as_said(office):
    s, sent, question = reply_then_interrupt(office, after_share=0.9)
    assert "clear" in kinds(sent)
    assert s.conversation.turns[1].text == question  # treated as a normal answer


def test_caller_still_talking_before_she_answers_gets_one_reply(office):
    llm = FakeLLM([(["לא אמורה להישמע"], False), (["הבנתי, חוזה שכירות."], False)], delay=0.05)

    async def scenario():
        s, sent, clock = make(office, llm)
        await s.handle(START)
        await finish_speaking(s, clock)
        await s.on_final("אני צריך")              # a pause mid-thought ends Azure's segment...
        await s.on_partial("חוזה שכירות")         # ...but they keep talking before she starts
        await s.on_final("חוזה שכירות לדירה")
        await s._reply_task
        return s

    s = asyncio.run(scenario())
    turns = s.conversation.turns
    assert [t.role for t in turns] == ["caller", "assistant"]
    assert turns[0].text == "אני צריך חוזה שכירות לדירה"


def test_goodbye_hangs_up_once_it_has_played(office):
    async def scenario():
        s, sent, clock = make(office, FakeLLM([(["תודה, להתראות!"], True)]))
        await s.handle(START)
        await finish_speaking(s, clock)
        await s.on_final("קוראים לי משה, זהו תודה רבה")
        await s._reply_task
        assert not s.closed
        await finish_speaking(s, clock)
        return s

    assert asyncio.run(scenario()).closed


def test_keypad_number(office):
    async def scenario():
        s, _, clock = make(office, FakeLLM([(["תודה."], False)]))
        await s.handle(START)
        await finish_speaking(s, clock)
        for d in "0587858000#":
            await s.handle({"event": "dtmf", "dtmf": {"digit": d}})
        await s._reply_task
        return s

    assert asyncio.run(scenario()).conversation.turns[0].text == "(הקשתי במקלדת) 0 5 8 7 8 5 8 0 0 0"


def test_silence_prompts_then_says_goodbye(office, monkeypatch):
    monkeypatch.setattr(stream_mod, "SILENCE_SECONDS", 0.01)

    async def scenario():
        s, sent, clock = make(office)
        await s.handle(START)
        for _ in range(stream_mod.MAX_SILENCES + 1):
            await finish_speaking(s, clock)
            await asyncio.sleep(0.05)
        return s

    s = asyncio.run(scenario())
    assert s.conversation.ended and s._hangup_after == s._speaking.id


def test_time_limit_says_goodbye_and_hangs_up_whatever_the_caller_does(office, monkeypatch):
    monkeypatch.setattr(stream_mod, "MAX_CALL_SECONDS", 0.05)

    async def scenario():
        s, sent, clock = make(office, FakeLLM([(["רגע, אני בודקת. "], False)]))
        await s.handle(START)
        await finish_speaking(s, clock)
        await s.on_final("ועוד דבר אחד")
        await asyncio.sleep(0.1)
        await s.on_final("רגע, לא סיימתי")  # ignored: no reply, no interruption
        bye = s._speaking
        await finish_speaking(s, clock)
        return s, sent, bye

    s, sent, bye = asyncio.run(scenario())
    assert bye.text == stream_mod.TIME_UP
    assert s.conversation.ended and s.closed
    assert any(t.text == "ועוד דבר אחד" for t in s.conversation.turns)  # still in the summary
    assert not any("לא סיימתי" in t.text for t in s.conversation.turns)


def test_mutual_goodbye_hangs_up_even_if_the_model_forgets(office):
    async def scenario():
        s, _, clock = make(office, FakeLLM([(["בשמחה, להתראות!"], False)]))
        await s.handle(START)
        await finish_speaking(s, clock)
        await s.on_final("קוראים לי משה, תודה רבה, להתראות")
        await s._reply_task
        await finish_speaking(s, clock)
        return s

    assert asyncio.run(scenario()).closed


def test_reply_waits_for_enough_audio_so_it_does_not_stutter(office):
    sent_at = []

    async def scenario():
        s, sent, clock = make(office, FakeLLM([(["שלום. ", "מדובר בחוזה שכירות לדירה שצריך לחתום עליו מחר. ", "איך קוראים לך?"], False)]))
        await s.handle(START)
        await finish_speaking(s, clock)
        sent.clear()
        await s.on_final("אני צריך חוזה")
        await s._reply_task
        return s, sent

    s, sent = asyncio.run(scenario())
    u_segments = s.conversation  # reply recorded
    assert s.conversation.turns[-1].text.startswith("שלום.")
    # "שלום." alone (0.5 s) isn't sent by itself: it goes out together with the next sentence.
    assert kinds(sent).count("media") >= 2


def test_emoji_and_brackets_are_not_spoken():
    from app.text import clean_for_speech
    assert clean_for_speech("שמי נועה :) (למשל: נדל\"ן)") == "שמי נועה למשל: נדל\"ן"


def test_short_reply_over_the_goodbye_does_not_stop_the_hang_up(office):
    # Real call: she said goodbye, the caller said "כן" over it, and she started over.
    async def scenario():
        s, sent, clock = make(office, FakeLLM([(["רשמתי הכול. המשך ערב נעים ולהתראות."], True)]))
        await s.handle(START)
        await finish_speaking(s, clock)
        await s.on_final("קוראים לי משה, אה הא")
        await s._reply_task
        clock.t = s._speaking.segments[0].start + 0.2
        await s.on_partial("כן")
        await s.on_final("כן")
        await finish_speaking(s, clock)
        return s, sent

    s, sent = asyncio.run(scenario())
    assert "clear" not in kinds(sent)
    assert s.closed


def test_a_real_sentence_over_the_goodbye_keeps_the_call_open(office):
    async def scenario():
        s, sent, clock = make(office, FakeLLM([(["המשך ערב נעים ולהתראות."], True), (["כן, בטח. מה עוד?"], False)]))
        await s.handle(START)
        await finish_speaking(s, clock)
        await s.on_final("קוראים לי משה, זהו תודה")
        await s._reply_task
        clock.t = s._speaking.segments[0].start + 0.2
        await s.on_partial("רגע עוד משהו")
        await s.on_final("רגע עוד משהו קטן")
        await s._reply_task
        return s, sent

    s, sent = asyncio.run(scenario())
    assert "clear" in kinds(sent)
    assert not s.conversation.ended and not s.closed


def test_unsure_recognition_is_flagged_for_the_llm(office):
    async def scenario():
        s, _, clock = make(office, FakeLLM([(["רק לוודא, איך מאייתים?"], False)]))
        await s.handle(START)
        await finish_speaking(s, clock)
        await s.on_final("שקל מוזמ", confidence=0.41)
        await s._reply_task
        return s

    assert asyncio.run(scenario()).conversation.turns[0].text == "שקל מוזמ [זיהוי לא בטוח]"


def test_echo_of_her_own_voice_does_not_stop_her(office):
    # Real call: she said "שלום. מה מספר הטלפון…" and "the caller" said "שלום."
    async def scenario():
        s, sent, clock = make(office, FakeLLM([(["שלום. מה מספר הטלפון שאליו תרצה שעורך הדין יחזור?"], False)]))
        await s.handle(START)
        await finish_speaking(s, clock)
        await s.on_final("למספר אחר")
        await s._reply_task
        clock.t = s._speaking.segments[0].start + 0.3
        await s.on_partial("שלום")
        await s.on_final("שלום.", confidence=0.55)
        return s, sent

    s, sent = asyncio.run(scenario())
    assert "clear" not in kinds(sent)
    assert [t.text for t in s.conversation.turns if t.role == "caller"] == ["למספר אחר"]


def test_different_words_still_interrupt(office):
    async def scenario():
        s, sent, clock = make(office, FakeLLM([(["שלום. מה מספר הטלפון שאליו תרצה שעורך הדין יחזור?"], False), (["בטח."], False)]))
        await s.handle(START)
        await finish_speaking(s, clock)
        await s.on_final("למספר אחר")
        await s._reply_task
        clock.t = s._speaking.segments[0].start + 0.3
        await s.on_partial("רגע")
        return s, sent

    s, sent = asyncio.run(scenario())
    assert "clear" in kinds(sent)


def test_hello_over_the_greeting_is_just_hello(office):
    async def scenario():
        s, sent, clock = make(office)
        await s.handle(START)
        clock.t += 0.5
        await s.on_partial("הלו")
        await s.on_final("הלו?")
        return s, sent

    s, sent = asyncio.run(scenario())
    assert "clear" not in kinds(sent)
    assert s._pending == [] and s._speaking is not None and s._speaking.is_greeting
