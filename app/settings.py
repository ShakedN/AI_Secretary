"""Environment settings + the office profile (config.yaml)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# "gemini" or "anthropic" — see app/llm.py
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_THINKING_LEVEL: str = os.getenv("GEMINI_THINKING_LEVEL", "low")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# A fast model for the live conversation (latency matters most on a phone
# call), optionally a stronger one for the summary written after the call.
_DEFAULT_MODELS = {
    # Flash-Lite: ~0.6s to first word on the free tier, vs. up to a minute
    # for gemini-3.8-flash when the free tier is busy.
    "gemini": ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),
    "anthropic": ("claude-haiku-5-5", "claude-sonnet-5-5"),
}
_chat_default, _summary_default = _DEFAULT_MODELS.get(LLM_PROVIDER, ("", ""))
CHAT_MODEL: str = os.getenv("CHAT_MODEL") or _chat_default
SUMMARY_MODEL: str = os.getenv("SUMMARY_MODEL") or _summary_default

TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
# Public https URL Twilio reaches us on (e.g. an ngrok URL), no trailing slash.
# On Render it's provided automatically as RENDER_EXTERNAL_URL.
PUBLIC_BASE_URL: str = (
    os.getenv("PUBLIC_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL") or "http://localhost:8000"
).rstrip("/")

# ConversationRelay speech settings — Hebrew voice quality differs a lot
# between providers, so these are meant to be tried and tuned on real calls.
LANGUAGE: str = os.getenv("LANGUAGE", "he-IL")
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "Google")
TTS_VOICE: str = os.getenv("TTS_VOICE", "he-IL-Wavenet-A")
TRANSCRIPTION_PROVIDER: str = os.getenv("TRANSCRIPTION_PROVIDER", "Google")
SPEECH_MODEL: str = os.getenv("SPEECH_MODEL", "")

SMTP_HOST: str = os.getenv("SMTP_HOST", "")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER: str = os.getenv("SMTP_USER", "")
# Google shows app passwords in groups of four ("abcd efgh ..."); spaces aren't part of it.
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "").replace(" ", "")
EMAIL_FROM: str = os.getenv("EMAIL_FROM", "") or SMTP_USER
# While testing: send every summary to this address instead of the lawyers.
EMAIL_TO_OVERRIDE: str = os.getenv("EMAIL_TO_OVERRIDE", "")

# Browser voice-test page at /test (local only). Keep off in production.
TEST_PAGE: bool = os.getenv("TEST_PAGE", "").strip().lower() in ("1", "true", "yes")

DB_PATH: Path = ROOT / os.getenv("DB_PATH", "data/calls.db")
OFFICE_CONFIG_PATH: Path = ROOT / os.getenv("OFFICE_CONFIG", "config.yaml")


@dataclass
class Lawyer:
    name: str
    email: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class Office:
    name: str
    assistant_name: str
    general_email: str
    lawyers: list[Lawyer]
    practice_areas: list[str] = field(default_factory=list)
    extra_instructions: str = ""
    welcome_greeting: str = ""
    cc_general: bool = True

    def greeting(self) -> str:
        return self.welcome_greeting or (
            f"שלום, הגעתם למשרד {self.name}. "
            f"מדברת {self.assistant_name}, העוזרת הדיגיטלית של המשרד. "
            "כרגע אין מי שיענה, אבל אשמח לרשום את הפנייה ולדאוג שיחזרו אליכם. במה מדובר?"
        )

    def find_lawyer(self, name: str | None) -> Lawyer | None:
        if not name:
            return None
        needle = name.strip()
        for lawyer in self.lawyers:
            if needle == lawyer.name or needle in lawyer.aliases:
                return lawyer
        return None


def load_office(path: Path = OFFICE_CONFIG_PATH) -> Office:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Office(
        name=data["office_name"],
        assistant_name=data.get("assistant_name", "נועה"),
        general_email=data.get("general_email", ""),
        lawyers=[Lawyer(**lw) for lw in data.get("lawyers", [])],
        practice_areas=data.get("practice_areas", []),
        extra_instructions=data.get("extra_instructions", ""),
        welcome_greeting=data.get("welcome_greeting", ""),
        cc_general=data.get("cc_general", True),
    )
