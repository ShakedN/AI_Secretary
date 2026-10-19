"""LLM providers behind one small interface, so switching Gemini ↔ Claude is
one line in .env (LLM_PROVIDER).

    stream(system, messages)          → async iterator of text chunks, plus
                                        END_CALL if the model hung up
    extract(system, text, name, schema) → dict matching the JSON schema

`messages` is [{"role": "user"|"assistant", "content": str}, ...].
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Protocol

from . import settings

logger = logging.getLogger(__name__)


class _EndCall:
    def __repr__(self) -> str:
        return "END_CALL"


END_CALL = _EndCall()

END_CALL_DESCRIPTION = (
    "Hang up the call. Use only after you already said a short goodbye in this same reply."
)


class LLM(Protocol):
    def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str | _EndCall]: ...

    async def extract(self, system: str, text: str, name: str, schema: dict) -> dict: ...


class AnthropicLLM:
    def __init__(self, api_key: str, chat_model: str, summary_model: str) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._chat_model, self._summary_model = chat_model, summary_model

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str | _EndCall]:
        async with self._client.messages.stream(
            model=self._chat_model,
            max_tokens=400,
            system=system,
            messages=messages,
            tools=[{"name": "end_call", "description": END_CALL_DESCRIPTION, "input_schema": {"type": "object", "properties": {}}}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        if any(b.type == "tool_use" and b.name == "end_call" for b in final.content):
            yield END_CALL

    async def extract(self, system: str, text: str, name: str, schema: dict) -> dict:
        message = await self._client.messages.create(
            model=self._summary_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": text}],
            tools=[{"name": name, "description": "Record the result.", "input_schema": schema}],
            tool_choice={"type": "tool", "name": name},
        )
        return next(b.input for b in message.content if b.type == "tool_use")


class GeminiLLM:
    def __init__(self, api_key: str, chat_model: str, summary_model: str, thinking_level: str = "") -> None:
        from google import genai
        from google.genai import types

        self._types = types
        # The SDK's default retries silently back off on rate limits for up to
        # a minute; on a live call it's better to fail fast and say so.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=20_000, retry_options=types.HttpRetryOptions(attempts=2)),
        )
        self._chat_model, self._summary_model = chat_model, summary_model
        # Thinking adds delay before the first word; on a phone call keep it low.
        self._thinking = types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None

    def _contents(self, messages: list[dict]) -> list:
        t = self._types
        return [
            t.Content(role="user" if m["role"] == "user" else "model", parts=[t.Part(text=m["content"])])
            for m in messages
        ]

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str | _EndCall]:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=400,
            tools=[t.Tool(function_declarations=[t.FunctionDeclaration(name="end_call", description=END_CALL_DESCRIPTION)])],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=self._thinking,
        )
        ended = False
        stream = await self._client.aio.models.generate_content_stream(
            model=self._chat_model, contents=self._contents(messages), config=config
        )
        async for chunk in stream:
            for candidate in chunk.candidates or []:
                for part in (candidate.content.parts if candidate.content else None) or []:
                    if part.text and not part.thought:
                        yield part.text
                    if part.function_call and part.function_call.name == "end_call":
                        ended = True
        if ended:
            yield END_CALL

    async def extract(self, system: str, text: str, name: str, schema: dict) -> dict:
        t = self._types
        response = await self._client.aio.models.generate_content(
            model=self._summary_model,
            contents=text,
            config=t.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
        return json.loads(response.text)


def make_llm() -> LLM | None:
    """Build the provider chosen in .env, or None if its API key is missing."""
    if settings.LLM_PROVIDER == "anthropic":
        if settings.ANTHROPIC_API_KEY:
            return AnthropicLLM(settings.ANTHROPIC_API_KEY, settings.CHAT_MODEL, settings.SUMMARY_MODEL)
    elif settings.LLM_PROVIDER == "gemini":
        if settings.GEMINI_API_KEY:
            return GeminiLLM(settings.GEMINI_API_KEY, settings.CHAT_MODEL, settings.SUMMARY_MODEL, settings.GEMINI_THINKING_LEVEL)
    else:
        raise ValueError(f"unknown LLM_PROVIDER: {settings.LLM_PROVIDER!r} (use 'gemini' or 'anthropic')")
    logger.warning("no API key for LLM_PROVIDER=%s — the assistant cannot talk", settings.LLM_PROVIDER)
    return None
