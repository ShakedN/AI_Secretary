"""Azure streaming speech recognition for Twilio Media Streams (8 kHz mu-law).

Audio is pushed in as it arrives from the phone; Azure answers with partial
results while the caller is still talking (used to notice them talking over
the assistant) and a final result once they pause (their turn is done).
The SDK calls back on its own threads, so results are handed to the event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

from . import settings

logger = logging.getLogger(__name__)

Handler = Callable[[str], Awaitable[None]]


class AzureRecognizer:
    def __init__(
        self, on_partial: Handler, on_final: Handler, on_error: Handler,
        phrases: list[str] = (), phrase_weight: float = 1.0,
    ):
        import azure.cognitiveservices.speech as sdk

        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._handlers = {"partial": on_partial, "final": on_final, "error": on_error}

        fmt = sdk.audio.AudioStreamFormat(
            samples_per_second=8000, bits_per_sample=8, channels=1,
            wave_stream_format=sdk.AudioStreamWaveFormat.MULAW,
        )
        self._push = sdk.audio.PushAudioInputStream(stream_format=fmt)
        config = sdk.SpeechConfig(subscription=settings.AZURE_SPEECH_KEY, region=settings.AZURE_SPEECH_REGION)
        config.speech_recognition_language = settings.LANGUAGE
        # How long a pause ends the caller's sentence.
        config.set_property(sdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, str(settings.STT_SILENCE_MS))
        # Detailed results carry a confidence score, so unsure words (names!) get double-checked.
        config.output_format = sdk.OutputFormat.Detailed
        self._recognizer = sdk.SpeechRecognizer(speech_config=config, audio_config=sdk.audio.AudioConfig(stream=self._push))

        # Names the office expects to hear (lawyers, office name) get recognized better.
        if phrases:
            phrase_list = sdk.PhraseListGrammar.from_recognizer(self._recognizer)
            for phrase in phrases:
                phrase_list.addPhrase(phrase)
            phrase_list.setWeight(phrase_weight)

        def post(kind: str, *args) -> None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (kind, args))

        def recognized(evt) -> None:
            post("final", evt.result.text, _confidence(evt.result))

        def canceled(evt) -> None:
            details = evt.cancellation_details
            if details.reason == sdk.CancellationReason.Error:
                post("error", f"{details.error_code}: {details.error_details}")

        self._recognizer.recognizing.connect(lambda evt: post("partial", evt.result.text))
        self._recognizer.recognized.connect(recognized)
        self._recognizer.canceled.connect(canceled)
        self._recognizer.start_continuous_recognition_async()
        self._pump = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            kind, args = await self._queue.get()
            try:
                await self._handlers[kind](*args)
            except Exception:
                logger.exception("handling %s result failed", kind)

    def push(self, audio: bytes) -> None:
        self._push.write(audio)

    def close(self) -> None:
        self._push.close()
        self._recognizer.stop_continuous_recognition_async()
        self._pump.cancel()


def _confidence(result) -> float:
    """Azure's confidence (0-1) for the best guess; 1.0 if it didn't say."""
    try:
        best = json.loads(result.json).get("NBest") or [{}]
        return float(best[0].get("Confidence", 1.0))
    except (ValueError, TypeError, AttributeError):
        return 1.0
