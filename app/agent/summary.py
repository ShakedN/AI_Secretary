"""After the call: turn the transcript into a structured message for the lawyer."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..llm import LLM
from ..settings import Office

logger = logging.getLogger(__name__)

GENERAL = "המשרד"


@dataclass
class CallSummary:
    caller_name: str = ""
    callback_phone: str = ""
    lawyer_requested: str = GENERAL
    topic: str = ""
    summary: str = ""
    urgency: str = "normal"  # low | normal | high
    existing_client: str = "unknown"  # yes | no | unknown
    callback_time: str = ""
    next_step: str = ""
    caller_mood: str = ""  # calm | stressed | upset | angry
    missing_info: str = ""
    call_review: str = ""


def _schema(office: Office) -> dict:
    return {
        "type": "object",
        "properties": {
            "caller_name": {"type": "string", "description": "שם המתקשר, ריק אם לא נמסר"},
            "callback_phone": {"type": "string", "description": "מספר לחזרה, ספרות בלבד"},
            "lawyer_requested": {"type": "string", "enum": [lw.name for lw in office.lawyers] + [GENERAL]},
            "topic": {"type": "string", "description": "נושא הפנייה בכמה מילים"},
            "summary": {
                "type": "string",
                "description": "סיכום של 2-4 משפטים בעברית: מה המתקשר צריך, ופרטים חשובים שמסר",
            },
            "urgency": {"type": "string", "enum": ["low", "normal", "high"]},
            "existing_client": {"type": "string", "enum": ["yes", "no", "unknown"]},
            "callback_time": {"type": "string", "description": "מתי נוח לחזור, ריק אם לא צוין"},
            "next_step": {"type": "string", "description": "הפעולה המומלצת לעורך הדין, במשפט אחד"},
            "caller_mood": {"type": "string", "enum": ["calm", "stressed", "upset", "angry"]},
            "missing_info": {"type": "string", "description": "פרטים חשובים שלא הושגו בשיחה, ריק אם הכול נמסר"},
            "call_review": {
                "type": "string",
                "description": (
                    "הערכה קצרה וכנה (1-3 משפטים) של איך העוזרת ניהלה את השיחה: "
                    "מה עבד, מה היה מסורבל, חוזר או לא טבעי, ואי-הבנות אם היו"
                ),
            },
        },
        "required": [
            "caller_name", "callback_phone", "lawyer_requested", "topic", "summary", "urgency",
            "existing_client", "next_step", "caller_mood", "call_review",
        ],
    }


async def summarize(llm: LLM | None, office: Office, transcript: str, caller_number: str | None) -> CallSummary:
    fallback = CallSummary(callback_phone=caller_number or "", summary="לא ניתן היה לסכם אוטומטית — ראו תמליל.")
    if llm is None:
        return fallback
    system = (
        "את מסכמת שיחת טלפון שנכנסה למשרד עורכי דין ונענתה על ידי עוזרת דיגיטלית. "
        "כתבי הודעה מדויקת לעורך הדין, רק לפי מה שנאמר בשיחה."
        + (f" מספר המתקשר לפי זיהוי שיחה: {caller_number}." if caller_number else "")
    )
    try:
        data = await llm.extract(system, transcript, "record_message", _schema(office))
    except Exception:
        logger.exception("summary failed")
        return fallback

    fields = CallSummary.__dataclass_fields__
    result = CallSummary(**{k: str(v) for k, v in data.items() if k in fields})
    if not result.callback_phone and caller_number:
        result.callback_phone = caller_number
    return result
