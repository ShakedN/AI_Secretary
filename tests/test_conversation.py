import asyncio

from app.agent.conversation import DEFAULT_GOODBYE, ERROR_REPLY, FALLBACK_REPLY, Conversation
from app.agent.prompt import build_system_prompt

from .conftest import FakeLLM


def run(coro):
    return asyncio.run(coro)


async def collect(conv):
    return "".join([t async for t in conv.reply()])


def test_reply_streams_and_records(office):
    client = FakeLLM([(["שלום ", "דנה"], False)])
    conv = Conversation(office=office, llm=client, caller_number="0501234567")
    conv.add_caller("רציתי לדבר עם דנה")

    assert run(collect(conv)) == "שלום דנה"
    assert conv.turns[-1].role == "assistant" and conv.turns[-1].text == "שלום דנה"
    assert not conv.ended
    call = client.stream_calls[0]
    assert call["messages"] == [{"role": "user", "content": "רציתי לדבר עם דנה"}]
    assert "כהן ושות'" in call["system"]


def test_end_call_without_text_says_goodbye(office):
    conv = Conversation(office=office, llm=FakeLLM([([], True)]))
    conv.add_caller("קוראים לי משה, זהו, תודה")
    assert run(collect(conv)) == DEFAULT_GOODBYE
    assert conv.ended


def test_llm_failure_falls_back(office):
    class Broken:
        async def stream(self, system, messages, tools=True):
            raise RuntimeError("down")
            yield

    conv = Conversation(office=office, llm=Broken())
    conv.add_caller("הלו")
    assert run(collect(conv)) == ERROR_REPLY


def test_messages_merge_and_interrupt(office):
    conv = Conversation(office=office, llm=None)
    conv.add_caller("הלו")
    conv.add_caller("שומעים?")
    conv.mark_interrupted("כן, אני")  # cut off before the reply was recorded
    conv.add_caller("רגע")
    assert conv.messages() == [
        {"role": "user", "content": "הלו\nשומעים?"},
        {"role": "assistant", "content": "כן, אני [נקטעתי]"},
        {"role": "user", "content": "רגע"},
    ]


def test_prompt_mentions_office_and_withheld_number(office):
    prompt = build_system_prompt(office, None)
    assert "כהן ושות'" in prompt and 'עו"ד דנה כהן' in prompt
    assert "חסוי" in prompt
    assert "050-123-4567" in build_system_prompt(office, "0501234567")


def test_caller_number_shown_in_local_format(office):
    from app.agent.prompt import local_format
    assert local_format("+972587858000") == "058-785-8000"
    assert local_format("+97233822974") == "03-382-2974"
    assert "058-785-8000" in build_system_prompt(office, "+972587858000")


def test_silent_hangup_mid_conversation_is_ignored(office):
    # The model hangs up without a word right after the caller gave details:
    # retry once instead of cutting the client off.
    llm = FakeLLM([([], True), (["רק לוודא — המספר הוא 0 5 8?"], False)])
    conv = Conversation(office=office, llm=llm)
    conv.add_caller("קוראים לי אפיק, 058 7 8 58 076 [זיהוי לא בטוח]")
    assert run(collect(conv)) == "רק לוודא — המספר הוא 0 5 8?"
    assert not conv.ended


def test_silent_hangup_after_goodbye_is_honored(office):
    conv = Conversation(office=office, llm=FakeLLM([([], True)]))
    conv.add_caller("קוראים לי משה, תודה, ביי")
    assert run(collect(conv)) == DEFAULT_GOODBYE
    assert conv.ended


def test_twice_silent_hangup_falls_back_to_a_question(office):
    conv = Conversation(office=office, llm=FakeLLM([([], True), ([], True)]))
    conv.add_caller("אני רוצה חוזה")
    assert run(collect(conv)) == FALLBACK_REPLY
    assert not conv.ended


def test_closing_goodbye_hangs_up_even_without_end_call(office):
    # 17:46:52 on a real call: "...שיהיה המשך יום נעים, להתראות." — but no end_call.
    conv = Conversation(office=office, llm=FakeLLM([(["רשמתי את הפרטים. שיהיה המשך יום נעים, להתראות."], False)]))
    conv.add_caller("קוראים לי משה, תוכלי להתקשר למספר שהתקשרתי ממנו")
    run(collect(conv))
    assert conv.ended


def test_goodbye_word_mid_reply_does_not_hang_up(office):
    conv = Conversation(office=office, llm=FakeLLM([(["יום טוב גם לך! במה מדובר?"], False)]))
    conv.add_caller("יום טוב")
    run(collect(conv))
    assert not conv.ended


def test_pronunciation_dictionary_changes_only_the_audio():
    from app import tts
    tts.set_pronunciation({"ושות'": "ושותפים", 'עו"ד': "עורך דין", 'עו"ד דנה כהן': "עורכת הדין דנה כהן"})
    try:
        assert tts.spoken_form("משרד כהן ושות'") == "משרד כהן ושותפים"
        assert tts.spoken_form('עו"ד דנה כהן ועו"ד לוי') == "עורכת הדין דנה כהן ועורך דין לוי"
        assert "ושותפים" in tts._ssml("כהן ושות'")
    finally:
        tts.set_pronunciation({})


def test_office_knowledge_is_in_the_prompt(office):
    office.knowledge = "שעות פעילות: ראשון עד חמישי"
    prompt = build_system_prompt(office, None)
    assert "שעות פעילות: ראשון עד חמישי" in prompt and "אל תשלימי מהדמיון" in prompt


def test_recognition_phrases_stay_short_and_include_the_office(office):
    from app import vocabulary
    office.vocabulary = ["תמ\"א 38"]
    words = vocabulary.phrases(office)
    assert "חוזה בניין" in words and "דנה כהן" in words and "תמ\"א 38" in words
    assert len(words) < 40  # a long list dilutes the boost (tested on a real call)


def test_no_hang_up_before_the_callers_name(office):
    # Goodbye + end_call, but nobody asked who called: the line stays open.
    conv = Conversation(office=office, llm=FakeLLM([(["תודה, להתראות."], True)]))
    conv.add_caller("אני צריך חוזה שכירות, זה הכל")
    run(collect(conv))
    assert not conv.ended and conv.blocked_hangups == 1


def test_status_tells_her_the_name_is_missing(office):
    llm = FakeLLM([(["מה שמך?"], False)])
    conv = Conversation(office=office, llm=llm)
    conv.add_caller("אני צריך חוזה")
    run(collect(conv))
    assert "עוד לא ידוע" in llm.stream_calls[0]["system"]


def test_name_from_the_tool_or_from_speech_allows_hang_up(office):
    from app.llm import CallerName

    class NamingLLM(FakeLLM):
        async def stream(self, system, messages, tools=True):
            yield CallerName("אבי מזרחי")
            async for item in super().stream(system, messages, tools):
                yield item

    conv = Conversation(office=office, llm=NamingLLM([(["תודה אבי, להתראות."], True)]))
    conv.add_caller("אבי מזרחי, זה הכל")
    run(collect(conv))
    assert conv.caller_name == "אבי מזרחי" and conv.ended

    spoken = Conversation(office=office, llm=FakeLLM([(["תודה, להתראות."], True)]))
    spoken.add_caller("קוראים לי רונית כהן, זה הכל")
    run(collect(spoken))
    assert spoken.caller_name == "רונית כהן" and spoken.ended


def test_caller_who_refuses_a_name_is_not_trapped(office):
    from app.llm import CallerName

    class RefusedLLM(FakeLLM):
        async def stream(self, system, messages, tools=True):
            yield CallerName("", refused=True)
            async for item in super().stream(system, messages, tools):
                yield item

    conv = Conversation(office=office, llm=RefusedLLM([(["בסדר, להתראות."], True)]))
    conv.add_caller("לא רוצה להגיד את השם, ביי")
    run(collect(conv))
    assert conv.ended


def test_after_two_blocked_hang_ups_the_call_ends_anyway(office):
    conv = Conversation(office=office, llm=FakeLLM([(["להתראות."], True)] * 3))
    for _ in range(3):
        conv.add_caller("ביי")
        run(collect(conv))
    assert conv.ended and conv.blocked_hangups == 2


def test_tool_only_reply_asks_again_for_the_words(office):
    # Gemini noted the name and stopped — no text. She must still answer.
    from app.llm import CallerName

    class ToolThenText(FakeLLM):
        async def stream(self, system, messages, tools=True):
            self.stream_calls.append({"system": system, "tools": tools})
            if tools:
                yield CallerName("אבי מזרחי")
            else:
                yield "תודה אבי, רשמתי."

    llm = ToolThenText()
    conv = Conversation(office=office, llm=llm)
    conv.add_caller("אבי מזרחי")
    assert run(collect(conv)) == "תודה אבי, רשמתי."
    assert conv.caller_name == "אבי מזרחי"
    assert [c["tools"] for c in llm.stream_calls] == [True, False]
    assert "אבי מזרחי" in llm.stream_calls[1]["system"]


def test_bye_is_not_a_refusal_and_she_asks_for_the_name(office):
    from app.agent.conversation import ASK_NAME
    from app.llm import CallerName

    class ClaimsRefusal(FakeLLM):
        async def stream(self, system, messages, tools=True):
            if tools:
                yield CallerName("", refused=True)
            async for item in super().stream(system, messages, tools):
                yield item

    conv = Conversation(office=office, llm=ClaimsRefusal([(["רשמתי. המשך ערב נעים, להתראות."], False)]))
    conv.add_caller("זהו, תודה ביי")
    said = run(collect(conv))
    assert not conv.name_refused and not conv.ended
    assert said.endswith(ASK_NAME)


def test_a_real_refusal_counts(office):
    from app.llm import CallerName

    class Refuses(FakeLLM):
        async def stream(self, system, messages, tools=True):
            if tools:
                yield CallerName("", refused=True)
            async for item in super().stream(system, messages, tools):
                yield item

    conv = Conversation(office=office, llm=Refuses([(["בסדר גמור, להתראות."], False)]))
    conv.add_caller("אני לא רוצה להגיד את השם שלי, ביי")
    run(collect(conv))
    assert conv.name_refused and conv.ended


def test_she_stops_asking_for_the_name_after_three_tries(office):
    from app.agent.conversation import MAX_NAME_ASKS
    asks = [(["מה שמך המלא?"], False)] * MAX_NAME_ASKS
    llm = FakeLLM(asks + [(["רשמתי, עורך הדין יוודא. להתראות."], True)])
    conv = Conversation(office=office, llm=llm)
    for heard in ["אני רוצה לקנות דירה", "עושה בל", "קלאסית"]:
        conv.add_caller(heard)
        run(collect(conv))
    assert conv.name_asks() == MAX_NAME_ASKS
    conv.add_caller("נכ")
    run(collect(conv))
    assert "אל תשאלי שוב" in llm.stream_calls[-1]["system"]
    assert conv.ended  # no name, but she may hang up instead of looping


def test_long_calls_are_steered_to_a_close(office):
    from datetime import timedelta

    def system_after(minutes):
        llm = FakeLLM([(["בסדר."], False)])
        conv = Conversation(office=office, llm=llm)
        conv.started_at -= timedelta(minutes=minutes)
        conv.add_caller("עוד משהו")
        run(collect(conv))
        return llm.stream_calls[0]["system"]

    assert "לכוון לסיום" not in system_after(1)
    assert "לכוון לסיום" in system_after(3.6)
    assert "סיימי עכשיו" in system_after(5.5)


def test_no_hang_up_right_after_asking_a_question(office):
    conv = Conversation(office=office, llm=FakeLLM([(["רשמתי. האם יש עוד משהו שתרצה להוסיף?"], True)]))
    conv.add_caller("קוראים לי משה, זה בערך הכל")
    run(collect(conv))
    assert not conv.ended
