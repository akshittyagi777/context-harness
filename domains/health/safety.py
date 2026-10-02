"""Health pre-check patterns, the emergency path, and risk rules.

Emergency detection is code, not just regexes, so negation can be handled properly: a symptom phrase
is ignored when a negation ("no", "not", "never", "without", "n't") sits in the clause before it.
Patterns need person-context (first person) where the phrase is ambiguous.
"""
from __future__ import annotations

import re

from harness.domain import Scenario
from harness.memory import MemoryRecord

EMERGENCY_MESSAGE = (
    "This could be a medical emergency. Call 911 now (in the US); if you are thinking about suicide or "
    "self-harm, call or text 988. This tool cannot help in an emergency. I have not looked at your data."
)

_EXERTION = (r"(?:run\w*|jog\w*|exercis\w*|work\w* out|workout|climb\w*|walk\w*|lift\w*|exert\w*|effort|sprint\w*|"
             r"train\w*|ride|riding|cycl\w*|stairs)")
_CHEST = r"(?P<kw>chest (?:pain|pressure|tightness|discomfort))"
_I = r"\b(?:i|i'm|im|i've|i am)\b"
EMERGENCY_BASE: list[str] = [
    # chest pain with exertion or radiating
    _CHEST + r"[^.?!]{0,60}?(?:radiat\w*|spread\w*|shoot\w*|(?:down|into|to) (?:my |the )?(?:left )?(?:arm|jaw|shoulder)|"
             r"(?:while|when|during|after|with|on) (?:i'?m |i |my |a |the )?" + _EXERTION + ")",
    r"(?:while|when|during|after) (?:i'?m |i |my |a |the )?" + _EXERTION + r"[^.?!]{0,40}?" + _CHEST,
    # trouble breathing
    r"\b(?:i|i'm|im|i've|my)\b[^.?!]{0,40}?(?P<kw>(?:trouble|difficulty|struggling|hard time)\s+(?:to\s+)?breath\w*|"
    r"can'?t\s+(?:catch my\s+)?breath\w*|cannot breath\w*|short(?:ness)? of breath|gasping for (?:air|breath))",
    # stroke signs
    r"\b(?:my|i)\b[^.?!]{0,30}?(?P<kw>face (?:is )?(?:droop\w*|numb)|(?:left |right )?arm(?: is| feels)? (?:weak|numb|limp)|"
    r"weakness in (?:my )?(?:left |right )?arm|slurr\w* (?:my )?(?:speech|words)|speech (?:is )?slurred|slurred speech)",
    # suicidal intent, self-harm, overdose
    r"(?P<kw>kill myself|end my life|take my own life|hurt myself|harm myself|want to die|don'?t want to (?:live|be alive))",
    _I + r"[^.?!]{0,30}?(?P<kw>suicidal|self[- ]harm)",
    r"(?P<kw>thinking (?:of|about) (?:suicide|killing myself))",
    _I + r"[^.?!]{0,30}?(?P<kw>overdos\w*|took too many (?:pills|tablets|meds))",
]
_EMERG = [re.compile(p) for p in EMERGENCY_BASE]
_NEGATION = re.compile(r"\b(?:no|not|never|without|none|denies|neither)\b|n't\b")


def _norm(text: str) -> str:
    return text.lower().replace("’", "'")


def _negated(text: str, pos: int) -> bool:
    """A negation within the four words before `pos`, in the same clause."""
    before = re.split(r"[.,;!?]| but | however | although ", text[:pos])[-1]
    words = re.findall(r"[a-z']+", before)[-4:]
    return _NEGATION.search(" ".join(words)) is not None


def emergency_check(text: str) -> bool:
    low = _norm(text)
    return any(not _negated(low, m.start("kw")) for rx in _EMERG for m in rx.finditer(low))


EMERGENCY_PATTERNS: list[str] = []      # all emergency detection goes through emergency_check

MED_WORDS = (r"meds?|medications?|medicine|pills?|dose|dosage|prescriptions?|supplements?|melatonin|magnesium|"
             r"ashwagandha|nsaids?|ibuprofen|tylenol|sleeping pills?|sleep aids?|antihistamines?")
PRE_PATTERNS: dict[str, list[str]] = {
    "medication_question": [
        rf"\bcan i (?:take|have|use|mix)\b(?!\s+(?:a |an |the |some )?(?:rest|day|break|nap|walk|run|shower|week|look|breather))",
        rf"\b(?:stop|stopping|start|starting|skip|skipping|double|doubling|miss(?:ed)?|change|changing|adjust|adjusting)\b[^.?!]{{0,30}}\b(?:{MED_WORDS})\b",
        r"\binteract(?:s|ion|ions|ing)?\b", r"\b(?:dose|dosage)\b",
        r"\b(?:melatonin|magnesium|ashwagandha|nsaids?|ibuprofen|tylenol|sleeping pills?|sleep aids?|supplements?)\b",
        r"\[a medication on file\]",      # the egress scrub's placeholder: the user named a medication from their own record
    ],
    "lab_question": [
        r"\[a lab result on file\]",      # the egress scrub's placeholder: the user named a lab result from their own record
    ],
    "ehr_lookup": [
        r"\bmy (?:meds|medications?|prescriptions?|allergies|allergy|labs?|lab results|diagnos[ei]s|conditions|problem list)\b",
        r"\b(?:last|latest|recent) (?:a1c|hba1c|ldl|cholesterol|lab|labs|blood work)\b", r"\bwhat (?:am i|meds am i|medications am i) (?:taking|on)\b",
        r"\bwhat meds\b", r"\b(?:a1c|hba1c)\b", r"\bwhat are my (?:allergies|meds|medications|labs)\b",
    ],
    "fitness_claim": [
        r"\bresting (?:hr|heart rate)\b[^.?!]{0,50}\b(?:down|lower|dropped|fell|improv\w*|better)\b",
        r"\b(?:getting|got) (?:fitter|more fit|in shape)\b", r"\bfitness gains?\b",
    ],
}
ENDORSEMENT_PATTERNS = [r"go for it", r"should be fine", r"sounds good", r"push through it", r"green light",
                        r"you(?:'re| are) good to go", r"yes,? (?:you(?:'re| are)|that(?:'s| is)) (?:getting )?fitter",
                        r"great (?:fitness )?(?:news|progress)"]
ESCALATION_CONTACT = "emergency services (911, or 988 for suicidal thoughts in the US)"

_TRAIN = re.compile(r"push(?:ing)? through|go(?:ing)? hard|max(?:imum|imal)? effort|all[- ]out|full send", re.I)
_NEG = re.compile(r"\b(?:not|never|no longer|won't|don't|wouldn't|shouldn't|avoid|skip)\b", re.I)


def is_medication_question(text: str) -> bool:
    low = _norm(text)
    return any(re.search(p, low) for p in PRE_PATTERNS["medication_question"])


def _train_negated(text: str, start: int) -> bool:
    """A negation within the six words before `start`, inside the same clause."""
    clause = re.split(r"[.,;!?]|\bbut\b|\bhowever\b", text[:start].replace("\u2019", "'"))[-1]
    words = re.findall(r"[\w']+", clause)[-6:]
    return _NEG.search(" ".join(words)) is not None


def training_load_rule(text: str, pinned: list[MemoryRecord], sc: Scenario | None):
    """Training-load language while recovery is under 33% or a baseline divergence is active."""
    hits = [m for m in _TRAIN.finditer(text) if not _train_negated(text, m.start())]
    if not hits or sc is None:
        return None
    from .rules import compute_values            # local import: rules imports this module's patterns
    v = compute_values(sc)
    rec = v.get("recovery")
    if rec and rec["value"] < 33:
        return ("training_load_risk", f"training-load language while recovery is {rec['value']:g}% (under 33%)")
    active = [s for s, d in v.get("divergence", {}).items() if d["active"]]
    if active:
        return ("training_load_risk", f"training-load language while a baseline divergence is active ({', '.join(active)})")
    return None


RISK_FLAG_RULES = [training_load_rule]
