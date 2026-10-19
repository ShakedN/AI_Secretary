"""Talk to the assistant in the terminal — same prompt and model as a real call,
no phone needed. At the end prints the summary/email the lawyer would get
(without sending it).

    .venv/Scripts/python -m scripts.chat [--from 0501234567]
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from app import notify, settings
from app.agent.conversation import Conversation
from app.agent.summary import summarize
from app.llm import make_llm


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="caller", default="0501234567", help="caller ID ('' = withheld)")
    args = parser.parse_args()

    llm = make_llm()
    if llm is None:
        key = "GEMINI_API_KEY" if settings.LLM_PROVIDER == "gemini" else "ANTHROPIC_API_KEY"
        sys.exit(f"Set {key} in .env first.")
    office = settings.load_office()
    conv = Conversation(office=office, llm=llm, caller_number=args.caller or None)

    print(f"[{settings.LLM_PROVIDER} / {settings.CHAT_MODEL}]")
    print(f"{office.assistant_name}: {office.greeting()}")
    while not conv.ended:
        try:
            line = input("\nאתה: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        conv.add_caller(line)
        print(f"{office.assistant_name}: ", end="", flush=True)
        async for token in conv.reply():
            print(token, end="", flush=True)
        print()

    if not any(t.role == "caller" for t in conv.turns):
        return
    print("\n--- ההודעה לעורך הדין ---")
    summary = await summarize(llm, office, conv.transcript(), conv.caller_number)
    msg = notify.build_email(office, summary, conv.transcript(), f"{conv.started_at:%d/%m/%Y %H:%M}")
    print("אל:", ", ".join(notify.recipients(office, summary)))
    print("נושא:", msg["Subject"])
    print(msg.get_body(("plain",)).get_content())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    asyncio.run(main())
