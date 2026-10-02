"""Deterministic safety gates around the model call.

Pre-check   Emergency patterns win outright and the model is not called. Otherwise
            domain patterns and domain risk rules raise flags; any flag makes the
            level `elevated`.
Post-check  Rules only, so they cannot be prompt-injected: no endorsement of a risky
            plan while a risk flag is up, every tension addressed, no sentence that
            recommends a forbidden term from a shipped sensitive record (naming it while
            steering away is fine), every number grounded in the prompt, and a length cap.
Fallback    A templated reply for when the answer cannot be trusted.

Seam: in production the pre-check would be a small classifier with these rules as the
floor, and the post-check would add a judge call. The rules stay as the backstop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable

from .clip import clip
from .domain import Domain
from .memory import MemoryRecord

MAX_WORDS = 200


@dataclass
class PreCheck:
    level: str                                   # none | elevated | emergency
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)   # flags raised by risk rules or triggers
    risk_reasons: list[str] = field(default_factory=list)

    def add_risk(self, flag: str, reason: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)
        if flag not in self.risk_flags:
            self.risk_flags.append(flag)
        self.risk_reasons.append(reason)
        self.reasons.append(f"{flag}: {reason}")
        self.level = "elevated"


def pre_check(domain: Domain, text: str, pinned: list[MemoryRecord], sc: Any = None) -> PreCheck:
    low = text.lower()
    if any(re.search(p, low) for p in domain.emergency_patterns) or (
            domain.emergency_check is not None and domain.emergency_check(text)):
        return PreCheck("emergency", ["emergency"], ["emergency pattern matched; deterministic path, model skipped"])
    pre = PreCheck("none")
    for flag, patterns in domain.pre_patterns.items():
        hit = next((p for p in patterns if re.search(p, low)), None)
        if hit:
            pre.flags.append(flag)
            pre.reasons.append(f"{flag}: matched /{hit}/")
    for rule in domain.risk_flag_rules:
        out = rule(text, pinned, sc)
        if out:
            pre.add_risk(*out)
    pre.level = "elevated" if pre.flags else "none"
    return pre


@dataclass
class PostCheck:
    passed: bool
    checks: dict[str, bool]
    notes: list[str]


def _has_word(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text.lower()) is not None


AVOIDANCE_CUES = ("avoid", "avoiding", "instead", "don't", "do not", "skip", "steer away", "not recommend",
                  "wouldn't", "would not", "rather than", "alternative", "precast", "swap", "replace")
_AVOIDANCE = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(c) for c in AVOIDANCE_CUES) + r")(?!\w)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.replace("\u2019", "'")) if s.strip()]


def forbidden_recommendations(response: str, forbid: list[str]) -> list[str]:
    """Sentences that contain a forbidden term and no avoidance cue. A reply may name a
    forbidden reagent while steering away from it; it may not recommend it."""
    return [s.strip() for s in _sentences(response)
            if any(_has_word(s, t) for t in forbid) and not _AVOIDANCE.search(s.lower())]


def _numbers(text: str) -> set[str]:
    """Numeric tokens, normalised so 09 and 9, or 0.20 and 0.2, compare equal."""
    return {format(Decimal(n).normalize(), "f") for n in _NUMBER.findall(text)}


def ungrounded_numbers(response: str, *sources: str) -> list[str]:
    """Numbers in the response that appear in none of the sources (the prompt as sent)."""
    known = _numbers("\n".join(sources))
    return sorted(_numbers(response) - known, key=float)


def post_check(domain: Domain, response: str, pre: PreCheck, must_mention: list[list[str]],
               forbid: list[str], sources: tuple[str, ...] | None = None) -> PostCheck:
    """`sources` is the prompt text as sent (system, user). When it is None the grounding
    check has nothing to compare against and passes."""
    low = response.lower()
    endorsed = any(re.search(p, low) for p in domain.endorsement_patterns)
    checks = {
        "no_risky_endorsement": not (pre.risk_flags and endorsed),
        "tensions_addressed": all(any(_has_word(response, w) for w in group) for group in must_mention),
        "no_forbidden_recommendation": not forbidden_recommendations(response, forbid),
        "numbers_grounded": sources is None or not ungrounded_numbers(response, *sources),
        "length_ok": len(response.split()) <= MAX_WORDS,
    }
    notes = [k for k, v in checks.items() if not v]
    return PostCheck(all(checks.values()), checks, notes)


def templated_fallback(incoming_text: str, redact: Callable[[str], str] | None = None) -> str:
    msg = " ".join(incoming_text.split())
    if len(msg) > 140:
        msg = clip(msg, 140, redact, word=True) + "..."      # redacted before it is cut: a clip first can split a denylisted word
    return (f'You said: "{msg}". I can\'t advise on this one right now. '
            f"What is the one thing you most want help with here?")


def escalation_summary(domain: Domain, pre: PreCheck, pinned: list[MemoryRecord],
                       plans: list[MemoryRecord], values: dict[str, Any], tensions: list[str]) -> dict[str, Any]:
    """What a human responder would need, built from the same resolved context the model
    would have seen. Sensitive pinned records contribute only their safe constraint text.
    Seam: delivery (to whom, over what channel) is product work and is not built here."""
    safe = [r.extra.get("constraint", "restricted record on file") if r.sensitivity == "high" else r.content
            for r in pinned]
    return {
        "trigger": pre.level, "flags": pre.flags, "pinned_safe_text": safe,
        "active_plans": [p.content for p in plans], "computed_values": values,
        "open_tensions": tensions, "contact": domain.escalation_contact,
    }
