"""Our own text-to-speech for the Gather flow: Azure neural voices (e.g. Hila).

Twilio's <Say> voices sound robotic in Hebrew, so instead we synthesize the
reply with Azure, keep the audio in memory, and let Twilio <Play> it from
/audio/<id>.mp3. Repeated lines (the greeting, "hello? are you there?") are
synthesized once and reused.
"""
from __future__ import annotations

import hashlib
import html
import logging
from collections import OrderedDict

import httpx

from . import settings

logger = logging.getLogger(__name__)

_MAX_CACHED = 300  # a few minutes of calls; old clips are dropped first
_cache: OrderedDict[str, bytes] = OrderedDict()


_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    # One long-lived client: reusing the TLS connection to Azure saves several
    # hundred ms per reply compared with connecting from scratch each time.
    global _client
    if _client is None:
        # httpx drops idle connections after 5 s by default — far shorter than
        # the gap between two replies in a call — so keep them for 5 minutes.
        _client = httpx.AsyncClient(timeout=10, limits=httpx.Limits(keepalive_expiry=300))
    return _client


def enabled() -> bool:
    return bool(settings.AZURE_SPEECH_KEY and settings.AZURE_SPEECH_REGION)


# Pronunciation dictionary from config.yaml: written form → how to say it.
# Longest first, so 'עו"ד דנה כהן' wins over 'עו"ד'. Only the audio changes.
_pronunciation: list[tuple[str, str]] = []


def set_pronunciation(mapping: dict[str, str]) -> None:
    global _pronunciation
    _pronunciation = sorted(((k, v) for k, v in mapping.items() if k), key=lambda kv: -len(kv[0]))


def spoken_form(text: str) -> str:
    for written, spoken in _pronunciation:
        text = text.replace(written, spoken)
    return text


def _ssml(text: str) -> str:
    text = spoken_form(text)
    return (
        f"<speak version='1.0' xml:lang='{settings.LANGUAGE}'>"
        f"<voice name='{settings.AZURE_VOICE}'>"
        f"<prosody rate='{settings.AZURE_RATE}'>{html.escape(text)}</prosody>"
        "</voice></speak>"
    )


async def _azure_request(text: str, output_format: str) -> bytes:
    url = f"https://{settings.AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": settings.AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": output_format,
        "User-Agent": "law-office-secretary",
    }
    response = await _http().post(url, headers=headers, content=_ssml(text).encode("utf-8"))
    response.raise_for_status()
    return response.content


async def _azure_synthesize(text: str) -> bytes:
    # For <Play> (Gather mode). Phone audio is 8 kHz anyway; a small mp3 keeps Twilio's fetch fast.
    return await _azure_request(text, "audio-16khz-32kbitrate-mono-mp3")


async def _azure_synthesize_mulaw(text: str) -> bytes:
    # For Media Streams: raw 8 kHz mu-law, the exact format Twilio plays — no headers, no conversion.
    return await _azure_request(text, "raw-8khz-8bit-mono-mulaw")


async def audio_id_for(text: str) -> str:
    """Synthesize `text` (or reuse a cached clip) and return its id."""
    audio_id = hashlib.sha256(f"{settings.AZURE_VOICE}|{settings.AZURE_RATE}|{text}".encode()).hexdigest()[:32]
    if audio_id in _cache:
        _cache.move_to_end(audio_id)
        return audio_id
    _cache[audio_id] = await _azure_synthesize(text)
    while len(_cache) > _MAX_CACHED:
        _cache.popitem(last=False)
    return audio_id


def get_audio(audio_id: str) -> bytes | None:
    return _cache.get(audio_id)


def clip_seconds(audio_id: str) -> float | None:
    """Playing time of a cached clip (32 kbit/s mp3, see the output format above)."""
    data = _cache.get(audio_id)
    return len(data) * 8 / 32_000 if data else None


_mulaw_cache: OrderedDict[str, bytes] = OrderedDict()


async def synthesize_mulaw(text: str) -> bytes:
    """Speech for Media Streams (8000 bytes = 1 second), cached like the mp3 clips."""
    key = hashlib.sha256(f"{settings.AZURE_VOICE}|{settings.AZURE_RATE}|{text}".encode()).hexdigest()
    if key in _mulaw_cache:
        _mulaw_cache.move_to_end(key)
        return _mulaw_cache[key]
    audio = await _azure_synthesize_mulaw(text)
    _mulaw_cache[key] = audio
    while len(_mulaw_cache) > _MAX_CACHED:
        _mulaw_cache.popitem(last=False)
    return audio
