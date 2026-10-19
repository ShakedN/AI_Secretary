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
    conv.add_caller("זהו, תודה")
    assert run(collect(conv)) == DEFAULT_GOODBYE
    assert conv.ended


def test_llm_failure_falls_back(office):
    class Broken:
        async def stream(self, system, messages):
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
        {"role": "assistant", "content": "כן, אני [נקטע]"},
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
    conv.add_caller("תודה, ביי")
    assert run(collect(conv)) == DEFAULT_GOODBYE
    assert conv.ended


def test_twice_silent_hangup_falls_back_to_a_question(office):
    conv = Conversation(office=office, llm=FakeLLM([([], True), ([], True)]))
    conv.add_caller("אני רוצה חוזה")
    assert run(collect(conv)) == FALLBACK_REPLY
    assert not conv.ended
