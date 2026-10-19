"""Email the call summary to the lawyer who was asked for."""
from __future__ import annotations

import html
import logging
import smtplib
from email.message import EmailMessage

from . import settings
from .agent.summary import CallSummary
from .settings import Office

logger = logging.getLogger(__name__)

_URGENCY = {"high": "דחוף", "normal": "רגיל", "low": "לא דחוף"}
_CLIENT = {"yes": "כן", "no": "לא", "unknown": "לא ידוע"}
_MOOD = {"calm": "רגוע/ה", "stressed": "לחוץ/ה", "upset": "נסער/ת", "angry": "כועס/ת"}


def recipients(office: Office, summary: CallSummary) -> list[str]:
    # While testing, every email goes to one address instead of the lawyers.
    if settings.EMAIL_TO_OVERRIDE:
        return [settings.EMAIL_TO_OVERRIDE]
    lawyer = office.find_lawyer(summary.lawyer_requested)
    to = [lawyer.email] if lawyer and lawyer.email else []
    if office.general_email and (office.cc_general or not to):
        to.append(office.general_email)
    return list(dict.fromkeys(to))


def _duration(seconds: int | None) -> str:
    if seconds is None:
        return "לא ידוע"
    minutes, secs = divmod(max(0, seconds), 60)
    return f"{minutes}:{secs:02d} דקות"


def build_email(
    office: Office,
    summary: CallSummary,
    transcript: str,
    when: str,
    duration_seconds: int | None = None,
    caller_turns: int | None = None,
) -> EmailMessage:
    who = summary.caller_name or "מתקשר/ת"
    subject = f"שיחה שלא נענתה: {who}"
    if summary.callback_phone:
        subject += f" ({summary.callback_phone})"
    if summary.topic:
        subject += f" — {summary.topic}"
    if summary.urgency == "high":
        subject = "דחוף! " + subject

    details = [
        ("מתי", when),
        ("שם", summary.caller_name or "לא נמסר"),
        ("טלפון לחזרה", summary.callback_phone or "לא נמסר"),
        ("חיפש/ה את", summary.lawyer_requested),
        ("נושא", summary.topic),
        ("דחיפות", _URGENCY.get(summary.urgency, summary.urgency)),
        ("לקוח קיים", _CLIENT.get(summary.existing_client, summary.existing_client)),
        ("זמן נוח לחזרה", summary.callback_time or "לא צוין"),
    ]
    call_info = [
        ("משך השיחה", _duration(duration_seconds)),
        ("מצב רוח המתקשר", _MOOD.get(summary.caller_mood, summary.caller_mood or "לא ידוע")),
    ]
    if caller_turns is not None:
        call_info.append(("מספר תשובות של המתקשר", str(caller_turns)))
    if summary.missing_info:
        call_info.append(("מה חסר", summary.missing_info))
    if summary.call_review:
        call_info.append(("הערכת השיחה", summary.call_review))

    def lines(rows):
        return "\n".join(f"{k}: {v}" for k, v in rows)

    text = (
        f"{lines(details)}\n\nסיכום:\n{summary.summary}\n"
        + (f"\nהצעד הבא: {summary.next_step}\n" if summary.next_step else "")
        + f"\n--- איך הייתה השיחה ---\n{lines(call_info)}\n\n--- תמליל ---\n{transcript}"
    )

    e = html.escape

    def table(rows):
        return "<table cellpadding='4'>" + "".join(
            f"<tr><td style='color:#555;white-space:nowrap'><b>{e(k)}</b></td><td>{e(v)}</td></tr>" for k, v in rows
        ) + "</table>"

    body = (
        '<div dir="rtl" style="font-family:Arial,sans-serif;font-size:15px;line-height:1.5">'
        + table(details)
        + f"<h3>סיכום</h3><p>{e(summary.summary)}</p>"
        + (f"<p><b>הצעד הבא:</b> {e(summary.next_step)}</p>" if summary.next_step else "")
        + "<h3>איך הייתה השיחה</h3>" + table(call_info)
        + f"<h3>תמליל</h3><pre style='white-space:pre-wrap;font-family:inherit'>{e(transcript)}</pre>"
        + "</div>"
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.EMAIL_FROM
    msg.set_content(text)
    msg.add_alternative(body, subtype="html")
    return msg


def send_email(msg: EmailMessage, to: list[str]) -> bool:
    """Blocking — call via asyncio.to_thread."""
    if not (settings.SMTP_HOST and settings.SMTP_USER and to):
        logger.warning("email not sent (SMTP not configured or no recipients): %s", msg["Subject"])
        return False
    msg["To"] = ", ".join(to)
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception:
        logger.exception("sending email failed")
        return False
