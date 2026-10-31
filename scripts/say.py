"""Hear how the assistant says something — for tuning the pronunciation dictionary.

    .venv/Scripts/python -m scripts.say "שלום, הגעתם למשרד כהן ושות'"

Applies `pronunciation` from config.yaml, synthesizes with the Azure voice,
saves data/say.mp3 and opens it.
"""
from __future__ import annotations

import asyncio
import os
import sys

from app import settings, tts


async def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('usage: python -m scripts.say "text to say"')
    if not tts.enabled():
        sys.exit("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env first.")
    text = " ".join(sys.argv[1:])
    tts.set_pronunciation(settings.load_office().pronunciation)
    print("says:", tts.spoken_form(text))
    audio_id = await tts.audio_id_for(text)
    path = settings.DB_PATH.parent / "say.mp3"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tts.get_audio(audio_id))
    print("saved", path)
    if sys.platform == "win32":
        os.startfile(path)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
