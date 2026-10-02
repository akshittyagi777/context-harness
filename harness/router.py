"""Routing: which model and how much effort a turn deserves.

  small  claude-haiku-4-5   no thinking, no effort; acknowledgement-class turns
  mid    claude-sonnet-5    effort low; judgment or planning asked, scheduled or signal turns
  large  claude-opus-5      effort medium; any risk flag or tension

A failed post-check escalates one tier (or raises effort at the top) for one retry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TIERS = {
    "small": {"model": "claude-haiku-4-5", "effort": None},
    "mid": {"model": "claude-sonnet-5", "effort": "low"},
    "large": {"model": "claude-opus-5", "effort": "medium"},
}
ORDER = ["small", "mid", "large"]

JUDGMENT = (r"\bthoughts\?|\bshould i\b|\bis it (?:ok|okay|safe)\b|\bwhat do you think\b"
            r"|\banything i should\b|\bright\?")
PLAN_ASKED = r"\bhow should\b|\bwhat should\b|\bplan\b"


@dataclass
class Route:
    tier: str
    model: str
    effort: str | None
    reasons: list[str] = field(default_factory=list)


def _at_least(tier: str, floor: str) -> str:
    return max(tier, floor, key=ORDER.index)


def _make(tier: str, reasons: list[str]) -> Route:
    t = TIERS[tier]
    return Route(tier, t["model"], t["effort"], reasons)


def route(message: str, flags: list[str], n_tensions: int, n_open_loops: int = 0,
          trigger_kind: str | None = None) -> Route:
    tier, reasons = "small", []
    low = message.lower()
    if flags:
        tier = "large"
        reasons.append(f"flags {flags} require the large tier")
    if n_tensions:
        tier = "large"
        reasons.append(f"{n_tensions} unresolved tension(s) must be surfaced carefully")
    if re.search(JUDGMENT, low):
        tier = _at_least(tier, "mid")
        reasons.append("judgment asked: at least mid")
    elif re.search(PLAN_ASKED, low):
        tier = _at_least(tier, "mid")
        reasons.append("plan asked: at least mid")
    if trigger_kind == "scheduled" and n_open_loops:
        tier = _at_least(tier, "mid")
        reasons.append(f"scheduled trigger carrying {n_open_loops} open loop(s): at least mid")
    if trigger_kind == "signal":
        tier = _at_least(tier, "mid")
        reasons.append("signal trigger: the assistant is interrupting, so at least mid")
    if not reasons:
        reasons.append("no flags, no tension, no judgment or plan asked: small tier")
    return _make(tier, reasons)


def emergency_route() -> Route:
    return Route("large", TIERS["large"]["model"], "high",
                 ["emergency path: deterministic reply, model not called"])


def escalate(r: Route, why: str) -> Route:
    i = ORDER.index(r.tier)
    if i == len(ORDER) - 1:
        return Route(r.tier, r.model, "high", r.reasons + [f"already at top tier; effort raised to high: {why}"])
    nxt = ORDER[i + 1]
    return _make(nxt, r.reasons + [f"escalated from {r.tier}: {why}"])
