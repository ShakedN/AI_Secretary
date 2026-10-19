from __future__ import annotations

import asyncio

import pytest

from app import settings
from app.llm import END_CALL
from app.settings import Lawyer, Office


class FakeLLM:
    """Stands in for GeminiLLM / AnthropicLLM (see app/llm.py)."""

    def __init__(self, replies=(), summary=None, delay=0.0):
        self._replies = list(replies)  # each: (tokens, end_call)
        self._summary = summary or {}
        self._delay = delay
        self.stream_calls: list[dict] = []

    async def stream(self, system, messages):
        self.stream_calls.append({"system": system, "messages": messages})
        tokens, end_call = self._replies.pop(0)
        for t in tokens:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield t
        if end_call:
            yield END_CALL

    async def extract(self, system, text, name, schema):
        return self._summary


@pytest.fixture(autouse=True)
def isolate_from_dotenv(monkeypatch):
    """Tests must not depend on the developer's .env — and must never send real email."""
    monkeypatch.setattr(settings, "EMAIL_TO_OVERRIDE", "")
    monkeypatch.setattr(settings, "SMTP_HOST", "")
    monkeypatch.setattr(settings, "SMTP_USER", "")
    monkeypatch.setattr(settings, "TEST_PAGE", False)


@pytest.fixture
def office() -> Office:
    return Office(
        name="כהן ושות'",
        assistant_name="נועה",
        general_email="office@example.com",
        lawyers=[
            Lawyer(name='עו"ד דנה כהן', email="dana@example.com", aliases=["דנה"]),
            Lawyer(name='עו"ד יוסי לוי', email="", aliases=["יוסי"]),
        ],
    )
