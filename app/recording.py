"""Save a call's incoming audio as a playable .wav — for debugging recognition only.

Turned on with RECORD_CALLS=true. Recordings of real clients are sensitive:
keep this off in production.
"""
from __future__ import annotations

import struct
from pathlib import Path


def mulaw_wav(audio: bytes, sample_rate: int = 8000) -> bytes:
    """Wrap raw 8-bit mu-law (Twilio's format) in a WAV header (format tag 7)."""
    fmt = struct.pack("<HHIIHHH", 7, 1, sample_rate, sample_rate, 1, 8, 0)
    fact = struct.pack("<I", len(audio))
    body = (
        b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"fact" + struct.pack("<I", len(fact)) + fact
        + b"data" + struct.pack("<I", len(audio)) + audio
        + (b"\x00" if len(audio) % 2 else b"")
    )
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def save(folder: Path, call_sid: str, audio: bytes) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{call_sid}.wav"
    path.write_bytes(mulaw_wav(audio))
    return path
