"""Words speech recognition should expect on a law office's phone.

Azure gets these as a phrase list: when the audio is unclear it leans toward
them — e.g. "חוזה בניין" instead of "חולה בניין". Tested on a recorded call:
a short list with extra weight fixes it, a long list (~60 terms) dilutes the
boost and doesn't. So keep it to the words callers really say here; add
office-specific ones in config.yaml under `vocabulary`.
"""
from __future__ import annotations

from .settings import Office

PHRASE_WEIGHT = 1.5

CORE_TERMS = [
    "חוזה", "לעשות חוזה", "חוזה בניין", "חוזה שכירות", "חוזה מכר",
    "צוואה", "ירושה", "ייפוי כוח מתמשך", "מקרקעין", "משכנתא", "טאבו",
    "עורך דין", "עורכת דין",
]


def phrases(office: Office) -> list[str]:
    words = [office.name, *CORE_TERMS, *office.vocabulary]
    for lawyer in office.lawyers:
        name = lawyer.name
        for title in ('עו"ד', "עורכת דין", "עורך דין"):
            name = name.replace(title, "")
        words += [name.strip(), *lawyer.aliases]
    return list(dict.fromkeys(w.strip() for w in words if w and w.strip()))
