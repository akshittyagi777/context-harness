"""Lab safety patterns: flags, risk rules, emergency path, endorsement phrases."""
from __future__ import annotations

import re
from typing import Any

from harness.memory import MemoryRecord

PRE_PATTERNS = {
    "reagent_question": [r"acrylamide", r"\bdmf\b", r"mercaptoethanol", r"\bbme\b",
                         r"pour my own gel", r"what reagent"],
}

# Each pattern needs a body part or a person, so "the gel burned" or "samples were exposed to room
# temp" stay quiet. The text is lowercased before matching.
_BODY = r"(?:arm|hand|hands|skin|face|eyes?|leg)"
EMERGENCY_PATTERNS = [
    rf"\b(?:splash|spill)(?:ed|es|s|ing)?\b(?:\s+\w+){{0,4}}?\s+(?:on|in)\s+(?:my|your)\s+{_BODY}\b",
    r"\bin (?:my|your) eyes?\b",
    r"\binhaled\b",
    r"\bi was exposed\b|\bgot exposed\b|\bexposed my\b",
    r"\bburn(?:ed|t|s|ing)?\b(?:\s+\w+){0,3}?\s+(?:(?:my|your)\s+)?(?:skin|hands?|arms?|fingers?|face|legs?|wrists?)\b",
    r"\b(?:got|had)\b.{0,20}\bon my skin\b",
]

EMERGENCY_MESSAGE = (
    "Stop and get to the eyewash or safety shower now if anything touched your eyes or skin, then call "
    "the lab safety officer. I've put together what I know about your current work for them. "
    "Tell me when you're safe."
)

ENDORSEMENT_PATTERNS = [r"go for it", r"should be fine", r"sounds good", r"push through it"]

ESCALATION_CONTACT = "lab safety officer"

_TEMP = re.compile(r"\b(\d{2})\s*(?:°\s*)?c\b", re.I)
_PUSH = re.compile(r"push through|just run it anyway", re.I)
_NEGATION = re.compile(r"\b(?:not|never)\b|n't\b|\bno longer\b", re.I)


def _negated(text: str, start: int) -> bool:
    """A negation within the four words before `start` ("I will NOT push through", "won't go to 37C")."""
    words = re.findall(r"[\w'\u2019]+", text[:start])[-4:]
    return _NEGATION.search(" ".join(words).replace("\u2019", "'")) is not None


def temperature_rule(text: str, pinned: list[MemoryRecord], sc: Any = None) -> tuple[str, str] | None:
    temps = [int(m.group(1)) for m in _TEMP.finditer(text) if not _negated(text, m.start())]
    for rec in pinned:
        limit = rec.extra.get("max_induction_c")
        if limit is not None and any(t > limit for t in temps):
            hot = max(temps)
            return ("protocol_risk", f"message proposes {hot}°C, above the {limit}°C limit in pinned record {rec.id}")
    return None


def push_through_rule(text: str, pinned: list[MemoryRecord], sc: Any = None) -> tuple[str, str] | None:
    constraints = [r for r in pinned if r.type == "constraint"]
    if constraints and any(not _negated(text, m.start()) for m in _PUSH.finditer(text)):
        named = [r.id for r in constraints if r.sensitivity != "high"]   # a sensitive record is never named
        where = f" ({', '.join(named)})" if named else ""
        return ("protocol_risk", f"'push through' language while pinned constraints exist{where}")
    return None


RISK_FLAG_RULES = [temperature_rule, push_through_rule]
