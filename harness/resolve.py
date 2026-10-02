"""Settle stale and contradictory records before anything reaches the prompt.

Core rule: when two active records share a topic, the later one wins and the earlier
one is superseded by a read-time repair op. If the incoming message touches a losing
*plan* (two or more of the user's own words reach it), the change is not applied
silently: it becomes a Tension the model must name and ask about.

Then each domain rule runs and its resolutions, tensions, excluded day keys, computed
values and memory ops are merged in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .domain import Domain, Resolution, Scenario, Tension
from .memory import MemoryOp, MemoryRecord, MemoryStore, expand, overlap_count, tokens


@dataclass
class Resolved:
    resolutions: list[Resolution]
    tensions: list[Tension]
    memory_ops: list[MemoryOp]               # repairs plus domain-requested ops
    plans_active: list[MemoryRecord]
    prefs_active: list[MemoryRecord]
    excluded: set[str] = field(default_factory=set)
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def losers(self) -> set[str]:
        return {l for r in self.resolutions for l in r.loser_ids}

    @property
    def notes(self) -> list[Resolution]:
        return [r for r in self.resolutions if r.winner_id is None]


def resolve(domain: Domain, sc: Scenario, store: MemoryStore, reactive: bool = True) -> Resolved:
    syn = domain.query_synonyms
    resolutions: list[Resolution] = []
    tensions: list[Tension] = []
    ops: list[MemoryOp] = []
    user_words = tokens(sc.incoming.text)

    for topic, recs in store.conflicts().items():
        winner = max(recs, key=lambda r: r.updated_at)
        losers = [r for r in recs if r.id != winner.id]
        for l in losers:
            ops.append(MemoryOp("supersede", None, l.id,
                                f"read-time repair: {winner.id} is later on topic {topic}", by=winner.id))
        resolutions.append(Resolution(
            topic, f"{winner.id} wins; {[l.id for l in losers]} superseded",
            "latest record on a topic wins",
            f"{winner.day} > " + ", ".join(l.day for l in losers), winner.id, [l.id for l in losers]))
        if not reactive:
            continue
        for l in losers:
            loser_words = tokens(l.content)
            if l.type != "plan" or overlap_count(user_words, loser_words, syn) < 2:
                continue
            target = expand(loser_words, syn)
            hits = sorted(w for w in user_words if expand({w}, syn) & target)
            overlap = expand(user_words, syn) & target
            tensions.append(Tension(
                key=f"tension:{topic}",
                text=(f"PLAN AMBIGUITY. On {winner.day} the user replaced '{l.content.rstrip('.')}' with "
                      f"'{winner.content.rstrip('.')}'. Today's message uses words from the old plan "
                      f"({', '.join(hits)}). Do not assume either plan; name the change and ask "
                      f"which plan is live now."),
                must_mention=[sorted(set(hits) | overlap), ["plan", "changed", "switched", "decided", "which"]],
                why=f"incoming message overlaps superseded {l.id}; the user may have changed their mind again",
                old=l, new=winner, hits=hits))

    excluded: set[str] = set()
    values: dict[str, Any] = {}
    for rule in domain.rules:
        rr = rule(sc, store)
        resolutions += rr.resolutions
        tensions += rr.tensions
        ops += rr.ops
        excluded |= rr.excluded
        values.update(rr.values)

    gone = {l for r in resolutions for l in r.loser_ids}
    plans = [r for r in store.active() if r.type == "plan" and r.id not in gone]
    prefs = [r for r in store.active() if r.type == "preference" and r.id not in gone]
    return Resolved(resolutions, tensions, ops, plans, prefs, excluded, values)
