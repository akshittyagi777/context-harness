"""The Domain protocol: everything the core needs to know about a subject area.

The core never imports from `domains/`. A domain is a `Domain` instance that supplies
a system prompt, a scenario builder, a memory seed, resolution rules, extra context
candidates, safety patterns, triggers and a few lookup tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .memory import MemoryOp, MemoryRecord, MemoryStore

if TYPE_CHECKING:  # avoid import cycles; only needed for annotations
    from .context import Candidate
    from .resolve import Resolved


@dataclass
class Turn:
    ts: str
    role: str                     # user | assistant
    text: str
    session: int


@dataclass
class Scenario:
    user_id: str
    profile: dict[str, Any]
    incoming: Turn
    history: list[Turn]
    series: dict[str, list[dict[str, Any]]]
    session_summaries: list[dict[str, Any]]      # {session, date, text}
    now: datetime


@dataclass
class Trigger:
    kind: str                     # scheduled | signal
    name: str
    ts: str
    text: str                     # what the model sees in place of an incoming message
    query: str                    # what memory is ranked against
    facts: str = ""               # the data sentence a reply should lead with
    flags: list[str] = field(default_factory=list)
    why: str = ""


@dataclass
class Resolution:
    topic: str
    outcome: str
    rule: str
    evidence: str
    winner_id: str | None = None             # None means a note about how to read the data
    loser_ids: list[str] = field(default_factory=list)
    kind: str = "record"                     # record | measured_vs_recorded | later_wins
    about_incoming: bool = False             # the note concerns what the user just said


@dataclass
class Tension:
    key: str
    text: str                                # goes into the prompt
    must_mention: list[list[str]]            # the response needs one word from each group
    why: str
    old: MemoryRecord | None = None
    new: MemoryRecord | None = None
    hits: list[str] = field(default_factory=list)   # the user's own overlapping words


@dataclass
class RuleResult:
    resolutions: list[Resolution] = field(default_factory=list)
    tensions: list[Tension] = field(default_factory=list)
    excluded: set[str] = field(default_factory=set)      # day keys left out of trends
    values: dict[str, Any] = field(default_factory=dict)  # computed values for context and escalation
    ops: list[MemoryOp] = field(default_factory=list)     # e.g. reinforce requests


@dataclass
class LocalReply:
    """A reply produced by deterministic local code; the model is not called."""
    text: str
    note: str                     # one line for the trace, e.g. "answered locally from EHR; no model call"
    kind: str = "local"


Rule = Callable[[Scenario, MemoryStore], RuleResult]
RiskRule = Callable[[str, list[MemoryRecord], Scenario], "tuple[str, str] | None"]
LocalResponder = Callable[[str, Any, Scenario], "LocalReply | None"]   # (text, PreCheck, scenario)


@dataclass
class Domain:
    name: str
    system_prompt: str
    build_scenario: Callable[["datetime | None"], Scenario]
    seed_path: Path
    rules: list[Rule]
    candidates: Callable[[Scenario, "Resolved"], "list[Candidate]"]
    pre_patterns: dict[str, list[str]]
    emergency_patterns: list[str]
    emergency_message: str
    risk_flag_rules: list[RiskRule]
    endorsement_patterns: list[str]
    triggers: dict[str, Callable[[Scenario, datetime], "Trigger | None"]]
    query_synonyms: dict[str, set[str]]
    escalation_contact: str
    # Optional hooks (the emergency short-circuit generalised): a pre-check flag can route the turn
    # to a local responder instead of the model; a domain can supply its own stub reply and add
    # post-check requirements. A responder under the key "*" is offered every reactive turn, before
    # the flag-specific ones, and returns None to pass.
    local_responders: dict[str, LocalResponder] = field(default_factory=dict)
    stub_reply: Callable[[Any], str] | None = None
    post_requirements: Callable[[Any, Any], list[list[str]]] | None = None   # (PreCheck, Resolved) -> must_mention groups
    emergency_check: Callable[[str], bool] | None = None   # code-level emergency detection (e.g. with negation handling)
