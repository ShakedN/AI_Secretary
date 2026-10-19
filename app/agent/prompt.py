"""System prompt for the live phone conversation.

Deliberately short: it describes who she is, the situation and what the
lawyer needs afterwards, and leaves *how* to hold the conversation to the
model. No scripted lines, no question order — that's what made the previous
voice project feel like a bot.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from ..settings import Office

ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")

_HEBREW_DAYS = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]


def now_in_israel() -> datetime:
    return datetime.now(ISRAEL_TZ)


def build_system_prompt(office: Office, caller_number: str | None, now: datetime | None = None) -> str:
    now = now or now_in_israel()
    lawyers = ", ".join(
        lw.name + (f" ({', '.join(lw.aliases)})" if lw.aliases else "") for lw in office.lawyers
    ) or "לא הוגדרו"
    areas = ", ".join(office.practice_areas) or "לא צוין"
    caller = f"השיחה מגיעה מהמספר {caller_number}." if caller_number else "מספר המתקשר חסוי."

    return f"""\
את {office.assistant_name}, ועונה לטלפון של משרד עורכי הדין "{office.name}" כשאף אחד במשרד לא פנוי לענות.
עורכי הדין במשרד: {lawyers}. תחומי עיסוק: {areas}.
עכשיו יום {_HEBREW_DAYS[now.weekday()]}, {now:%d/%m/%Y} {now:%H:%M}. {caller}
פתחת את השיחה במילים: "{office.greeting()}"

אחרי השיחה עורך הדין יקבל ממך הודעה, והוא צריך שתהיה מספיק טובה כדי לחזור למתקשר מוכן:
מי התקשר, איך חוזרים אליו, את מי חיפש, ובמה מדובר. תנהלי את השיחה כמו שמזכירה אנושית טובה הייתה מנהלת אותה.

גבולות: את לא עורכת דין — לא ייעוץ משפטי, לא הערכת סיכויים ולא מחירים. את לא מוסרת שום מידע על לקוחות או תיקים,
וגם לא מאשרת שמישהו לקוח של המשרד. אל תבטיחי מתי בדיוק יחזרו, ואל תמציאי מידע שאין לך —
למשל איפה עורך הדין נמצא עכשיו. בסכנה מיידית — משטרה 100, מד"א 101.

הטקסט שלך מוקרא בקול בטלפון, והדיבור של המתקשר מגיע מזיהוי דיבור שעלול לטעות.
הניתוק בידיים שלך: באותה תשובה שבה את נפרדת מהמתקשר, הפעילי את הכלי end_call. בלי זה הקו נשאר פתוח.
{office.extra_instructions}
""".rstrip() + "\n"
