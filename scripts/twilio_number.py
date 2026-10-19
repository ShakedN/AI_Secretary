"""Point the Twilio number at this server.

    .venv/Scripts/python -m scripts.twilio_number              # show current setup
    .venv/Scripts/python -m scripts.twilio_number secretary    # calls → the AI secretary
    .venv/Scripts/python -m scripts.twilio_number samples      # calls → hear every Hebrew voice

Uses TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / PUBLIC_BASE_URL from .env, and the
account's only number (or TWILIO_PHONE_NUMBER if there are several).
"""
from __future__ import annotations

import os
import sys

from twilio.rest import Client

from app import settings

MODES = {"secretary": "/voice/incoming", "samples": "/voice/samples"}


def main() -> None:
    sid, token = os.getenv("TWILIO_ACCOUNT_SID", ""), settings.TWILIO_AUTH_TOKEN
    if not (sid and token):
        sys.exit("Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env first.")
    client = Client(sid, token)

    numbers = client.incoming_phone_numbers.list(limit=20)
    wanted = os.getenv("TWILIO_PHONE_NUMBER", "")
    if wanted:
        numbers = [n for n in numbers if n.phone_number == wanted]
    if not numbers:
        sys.exit("No phone number on this Twilio account yet — buy one in the Twilio Console first.")
    if len(numbers) > 1:
        sys.exit("Several numbers found — set TWILIO_PHONE_NUMBER in .env to pick one: "
                 + ", ".join(n.phone_number for n in numbers))
    number = numbers[0]

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode not in MODES:
            sys.exit(f"Unknown mode {mode!r}. Use: {', '.join(MODES)}")
        if settings.PUBLIC_BASE_URL.startswith("http://localhost"):
            sys.exit("PUBLIC_BASE_URL in .env must be the public https URL (e.g. your ngrok URL).")
        number = number.update(
            voice_url=settings.PUBLIC_BASE_URL + MODES[mode],
            voice_method="POST",
            # Tells us when the caller hangs up, so the summary is sent right away.
            status_callback=settings.PUBLIC_BASE_URL + "/voice/status",
            status_callback_method="POST",
        )

    print(f"{number.phone_number} -> {number.voice_url or '(no webhook)'}")


if __name__ == "__main__":
    main()
