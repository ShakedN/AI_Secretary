import asyncio
import json

from app import relay
from app.agent.conversation import Conversation
from app.relay import RelaySession

from .conftest import FakeLLM

SETUP = {"type": "setup", "callSid": "CA1", "from": "+972501234567"}


def make_session(office, client, sent, **kw):
    async def send(msg):
        sent.append(msg)

    return RelaySession(
        send=send,
        make_conversation=lambda number: Conversation(office=office, llm=client, caller_number=number),
        hangup_padding=0,
        **kw,
    )


def test_prompt_streams_tokens_then_last(office):
    async def scenario():
        sent = []
        s = make_session(office, FakeLLM([(["היי ", "שם"], False)]), sent)
        await s.handle(SETUP)
        await s.handle({"type": "prompt", "voicePrompt": "שלום", "last": True})
        await s._task
        return s, sent

    s, sent = asyncio.run(scenario())
    assert s.caller_number == "+972501234567"
    assert sent == [
        {"type": "text", "token": "היי ", "last": False},
        {"type": "text", "token": "שם", "last": False},
        {"type": "text", "token": "", "last": True},
    ]


def test_end_call_sends_end(office, monkeypatch):
    monkeypatch.setattr(relay, "_WORDS_PER_MINUTE", 10**6)

    async def scenario():
        sent = []
        s = make_session(office, FakeLLM([(["להתראות"], True)]), sent)
        await s.handle(SETUP)
        await s.handle({"type": "prompt", "voicePrompt": "תודה, ביי", "last": True})
        await s._task
        return s, sent

    s, sent = asyncio.run(scenario())
    assert sent[-1]["type"] == "end"
    assert json.loads(sent[-1]["handoffData"]) == {"reason": "completed"}
    assert s.closed


def test_interrupt_cancels_reply_and_keeps_only_heard_text(office):
    async def scenario():
        sent = []
        client = FakeLLM([(["מילה "] * 50, False)], delay=0.01)
        s = make_session(office, client, sent)
        await s.handle(SETUP)
        await s.handle({"type": "prompt", "voicePrompt": "שלום", "last": True})
        await asyncio.sleep(0.03)
        await s.handle({"type": "interrupt", "utteranceUntilInterrupt": "מילה מילה"})
        return s, sent

    s, sent = asyncio.run(scenario())
    assert s._task is None
    assert not any(m.get("last") for m in sent)
    assert s.conversation.turns[-1].text == "מילה מילה [נקטע]"


def test_unknown_call_rejected(office):
    async def scenario():
        s = make_session(office, FakeLLM(), [], is_allowed=lambda sid: False)
        await s.handle(SETUP)
        return s

    s = asyncio.run(scenario())
    assert s.closed and s.conversation is None


def test_withheld_number(office):
    async def scenario():
        s = make_session(office, FakeLLM(), [])
        await s.handle({**SETUP, "from": "anonymous"})
        return s

    assert asyncio.run(scenario()).caller_number is None
