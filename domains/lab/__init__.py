"""The lab domain: a research assistant for one scientist in a protein-biochemistry lab."""
from pathlib import Path

from harness.domain import Domain

from .rules import computed_values, measured_beats_recorded, same_day_conflict
from .safety import (EMERGENCY_MESSAGE, EMERGENCY_PATTERNS, ENDORSEMENT_PATTERNS, ESCALATION_CONTACT,
                     PRE_PATTERNS, RISK_FLAG_RULES)
from .scenario import build_scenario
from .signals import candidates
from .triggers import TRIGGERS

SYSTEM_PROMPT = (
    "You are a research assistant for one scientist in a protein-biochemistry lab. Ground every claim "
    "in the context below; if the context does not support a claim, do not make it. Be direct and brief, "
    "no lectures. Never recommend a protocol that conflicts with a pinned constraint. If a PLAN AMBIGUITY "
    "note is present, name the change and ask which plan is live rather than assuming. Respond in under "
    "180 words."
)

# Small lookup so a user's word reaches the records that use a neighbouring word.
SYNONYMS = {
    "induction": {"induced", "induce", "express", "expression"},
    "37": {"shaker", "setpoint", "hot"},
    "temperature": {"shaker", "setpoint", "induction", "aggregation"},
    "aggregation": {"aggregates", "aggregated", "aggregation", "void"},
    "gel": {"gels", "page"},
    "gels": {"gel", "page"},
    "dtt": {"reagent", "reducing"},
}

DOMAIN = Domain(
    name="lab",
    system_prompt=SYSTEM_PROMPT,
    build_scenario=build_scenario,
    seed_path=Path(__file__).parent / "seed.json",
    rules=[measured_beats_recorded, same_day_conflict, computed_values],
    candidates=candidates,
    pre_patterns=PRE_PATTERNS,
    emergency_patterns=EMERGENCY_PATTERNS,
    emergency_message=EMERGENCY_MESSAGE,
    risk_flag_rules=RISK_FLAG_RULES,
    endorsement_patterns=ENDORSEMENT_PATTERNS,
    triggers=TRIGGERS,
    query_synonyms=SYNONYMS,
    escalation_contact=ESCALATION_CONTACT,
)
