"""The provider adapters translate correctly to/from each SDK (SDK clients faked)."""
import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from app import llm as llm_mod
from app import settings
from app.llm import END_CALL, AnthropicLLM, GeminiLLM

MESSAGES = [
    {"role": "user", "content": "שלום"},
    {"role": "assistant", "content": "היי, במה אפשר לעזור?"},
    {"role": "user", "content": "תודה, ביי"},
]


async def collect(agen):
    return [x async for x in agen]


# ── Gemini ─────────────────────────────────────────────────────────────────

def _chunk(*parts):
    return NS(candidates=[NS(content=NS(parts=list(parts)))])


def _part(text=None, thought=None, call=None):
    return NS(text=text, thought=thought, function_call=NS(name=call) if call else None)


class FakeGeminiModels:
    def __init__(self, chunks, json_text="{}"):
        self._chunks, self._json_text = chunks, json_text
        self.calls = []

    async def generate_content_stream(self, **kw):
        self.calls.append(kw)

        async def gen():
            for c in self._chunks:
                yield c

        return gen()

    async def generate_content(self, **kw):
        self.calls.append(kw)
        return NS(text=self._json_text)


def make_gemini(models):
    g = GeminiLLM("test-key", "chat-model", "summary-model", thinking_level="low")
    g._client = NS(aio=NS(models=models))
    return g


def test_gemini_stream_text_skips_thoughts_and_detects_end_call():
    models = FakeGeminiModels([
        _chunk(_part(text="חושבת...", thought=True)),
        _chunk(_part(text="תודה, ")),
        _chunk(_part(text="להתראות"), _part(call="end_call")),
    ])
    out = asyncio.run(collect(make_gemini(models).stream("SYS", MESSAGES)))
    assert out == ["תודה, ", "להתראות", END_CALL]

    call = models.calls[0]
    assert call["model"] == "chat-model"
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    assert call["contents"][1].parts[0].text == "היי, במה אפשר לעזור?"
    assert call["config"].system_instruction == "SYS"
    assert call["config"].tools[0].function_declarations[0].name == "end_call"


def test_gemini_stream_without_end_call():
    models = FakeGeminiModels([_chunk(_part(text="שלום"))])
    assert asyncio.run(collect(make_gemini(models).stream("SYS", MESSAGES))) == ["שלום"]


def test_gemini_extract_uses_json_schema():
    models = FakeGeminiModels([], json_text=json.dumps({"caller_name": "משה"}))
    schema = {"type": "object", "properties": {"caller_name": {"type": "string"}}}
    out = asyncio.run(make_gemini(models).extract("SYS", "תמליל", "record_message", schema))
    assert out == {"caller_name": "משה"}
    config = models.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema
    assert models.calls[0]["model"] == "summary-model"


# ── Anthropic ──────────────────────────────────────────────────────────────

class FakeAnthropicStream:
    def __init__(self, tokens, end_call):
        self._tokens, self._end_call = tokens, end_call
        self.text_stream = self._gen()

    async def _gen(self):
        for t in self._tokens:
            yield t

    async def get_final_message(self):
        content = [NS(type="text", text="".join(self._tokens))]
        if self._end_call:
            content.append(NS(type="tool_use", name="end_call", input={}))
        return NS(content=content)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def make_anthropic(tokens=(), end_call=False, tool_input=None):
    calls = []

    def stream(**kw):
        calls.append(kw)
        return FakeAnthropicStream(list(tokens), end_call)

    async def create(**kw):
        calls.append(kw)
        return NS(content=[NS(type="tool_use", name=kw["tool_choice"]["name"], input=tool_input)])

    a = AnthropicLLM.__new__(AnthropicLLM)
    a._client = NS(messages=NS(stream=stream, create=create))
    a._chat_model, a._summary_model = "chat-model", "summary-model"
    return a, calls


def test_anthropic_stream_and_end_call():
    a, calls = make_anthropic(["ביי"], end_call=True)
    assert asyncio.run(collect(a.stream("SYS", MESSAGES))) == ["ביי", END_CALL]
    assert calls[0]["messages"] == MESSAGES and calls[0]["system"] == "SYS"


def test_anthropic_extract_forces_tool():
    a, calls = make_anthropic(tool_input={"caller_name": "משה"})
    out = asyncio.run(a.extract("SYS", "תמליל", "record_message", {"type": "object"}))
    assert out == {"caller_name": "משה"}
    assert calls[0]["tool_choice"] == {"type": "tool", "name": "record_message"}


# ── provider selection ─────────────────────────────────────────────────────

def test_make_llm_picks_provider(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    assert llm_mod.make_llm() is None
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "k")
    assert isinstance(llm_mod.make_llm(), GeminiLLM)

    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "k")
    assert isinstance(llm_mod.make_llm(), AnthropicLLM)

    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    with pytest.raises(ValueError):
        llm_mod.make_llm()
