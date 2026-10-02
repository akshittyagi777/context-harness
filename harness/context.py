"""Context assembly under a hard token budget.

Every piece of context is a Candidate: a tier (0 = must ship, 5 = nice to have), an
ordered list of variants (richest first) and a required flag. The assembler walks the
tiers. For each candidate it keeps the richest variant that still leaves room for the
cheapest variant of every required candidate not yet placed. Required candidates
degrade but never drop; optional ones drop with the reason and every variant's cost
recorded. Within a tier, optional candidates go in by score per token, so a cheap
low-value item cannot slip in after a valuable one failed to fit.

A sensitive record is gated whether or not it is pinned. With its gate closed only its
`extra.constraint` can ship (a pinned one always does; any other is an optional tier-5
line, dropped first); with the gate open it is an ordinary record.

Section headers count against the budget, so the rendered prompt stays under it.
If a relevant memory was dropped, one line saying so (with ids) is added and also
counted, which pairs with the `recall` tool.

Origin. Every candidate carries an `origin`. In `remote` mode (any model API) candidates of
origin `ehr` are removed before assembly: never truncated, degraded or placed. They are
recorded in `Assembly.excluded` (key and token count only) so the trace shows what was held back.
In `local` mode every origin may be rendered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .domain import Domain, Scenario
from .resolve import Resolved
from .safety import PreCheck
from .tokens import count_tokens

SECTION_ORDER = ["system", "pinned", "tensions", "plans", "notes", "data", "memory", "history",
                 "note", "incoming"]
SECTION_TITLES = {
    "pinned": "Pinned constraints", "tensions": "Unresolved tensions (must address)",
    "plans": "Current plans and preferences", "notes": "How to read the data",
    "data": "Measurements", "memory": "Relevant records", "history": "Conversation history",
    "incoming": "Incoming message",
}
GATED_TIER = 5   # a closed-gate, non-pinned record is the first thing to go when room runs out
TIER3_NOTE = "{n} relevant record(s) were left out for budget; call recall if you need them."


class BudgetTooSmall(ValueError):
    """The budget cannot hold the required context even at its cheapest variants."""

    def __init__(self, budget: int, floor: int):
        super().__init__(f"budget {budget} cannot fit the required context ({floor} tokens at cheapest)")
        self.budget, self.floor = budget, floor


@dataclass
class Candidate:
    key: str
    section: str
    tier: int
    variants: list[tuple[str, str]]          # (name, text), richest first
    required: bool = False
    why: str = ""
    score: float = 0.0                       # used to order optional candidates in a tier
    meta: dict[str, Any] = field(default_factory=dict)
    origin: str = "user"

    def costs(self) -> dict[str, int]:
        return {name: count_tokens(text) for name, text in self.variants}

    def cheapest(self) -> int:
        return min(self.costs().values())

    def density(self) -> float:
        return self.score / max(1, self.costs()[self.variants[0][0]])


@dataclass
class Placed:
    candidate: Candidate
    variant: str
    text: str
    tokens: int
    why: str


@dataclass
class Assembly:
    budget: int
    placed: list[Placed]
    decisions: list[dict[str, Any]]
    used: int
    note: str = ""
    note_tokens: int = 0
    gates: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "local"

    @property
    def headroom(self) -> int:
        return self.budget - self.used

    def placed_ids(self) -> set[str]:
        return {i for p in self.placed for i in p.candidate.meta.get("ids", [])}

    def forbid_terms(self) -> list[str]:
        return sorted({t for p in self.placed for t in p.candidate.meta.get("forbid", [])})


def _hdr(section: str) -> int:
    title = SECTION_TITLES.get(section)
    return count_tokens(f"\n\n## {title}\n") if title else 0


# ------------------------------------------------------------------ candidates

def build_candidates(domain: Domain, sc: Scenario, ranked: list[dict[str, Any]], res: Resolved,
                     pre: PreCheck) -> list[Candidate]:
    c: list[Candidate] = []
    losers = res.losers

    c.append(Candidate("system", "system", 0, [("verbatim", domain.system_prompt)], required=True,
                       why="fixed instructions; stable prefix for caching"))
    c.append(Candidate("incoming", "incoming", 0,
                       [("verbatim", f"[{sc.incoming.ts[:16]}] {sc.incoming.text}")], required=True,
                       why="the turn itself"))

    # Pinned records. A sensitive one is gated: the constraint always ships, the content
    # only when a flag listed in extra.reveal_on is up.
    for m in ranked:
        rec = m["record"]
        if not rec.pinned or rec.id in losers:
            continue
        if rec.sensitivity == "high":
            constraint = rec.extra.get("constraint", "A restricted fact applies; do not assume anything about it.")
            reveal = sorted(set(rec.extra.get("reveal_on", [])) & set(pre.flags))
            meta = {"ids": [rec.id], "sensitive": True, "forbid": list(rec.extra.get("forbid", [])),
                    "constraint": constraint,
                    "safe_note": rec.extra.get("safe_note", "")}
            if reveal:
                meta["gate"] = "open"
                c.append(Candidate(f"pinned:{rec.id}", "pinned", 0,
                                   [("full", f"- {rec.content}"), ("constraint", f"- {constraint}")],
                                   required=True, meta=meta, origin=rec.origin,
                                   why=f"pinned + sensitive; gate OPEN (flag {reveal} in reveal_on): full content ships"))
            else:
                meta["gate"] = "closed"
                c.append(Candidate(f"pinned:{rec.id}", "pinned", 0, [("constraint", f"- {constraint}")],
                                   required=True, meta=meta, origin=rec.origin,
                                   why="pinned + sensitive; gate CLOSED: name withheld, constraint ships"))
        else:
            c.append(Candidate(f"pinned:{rec.id}", "pinned", 0, [("full", f"- {rec.content}")],
                               required=True, meta={"ids": [rec.id], "sensitive": False}, origin=rec.origin,
                               why="pinned; always ships"))

    for t in res.tensions:
        if t.old is not None and t.new is not None:
            short = (f"PLAN AMBIGUITY: '{t.old.content}' vs '{t.new.content}' (changed {t.new.day}). "
                     f"Ask which plan is live.")
        else:
            short = t.text[:160]
        ids = [r.id for r in (t.old, t.new) if r is not None]
        c.append(Candidate(t.key, "tensions", 1, [("full", t.text), ("short", short)], required=True,
                           meta={"ids": ids}, why=t.why))

    plans = "\n".join(f"- {p.content} (stated {p.day})" for p in res.plans_active) or "- none recorded"
    prefs = "\n".join(f"- {p.content}" for p in res.prefs_active) or "- none recorded"
    compact = (f"Plan: {'; '.join(p.content for p in res.plans_active) or 'none'}. "
               f"Prefs: {'; '.join(p.content for p in res.prefs_active) or 'none'}")
    c.append(Candidate("plans_prefs", "plans", 1,
                       [("full", f"Current plan:\n{plans}\nPreferences:\n{prefs}"), ("compact", compact)],
                       required=True,
                       meta={"ids": [r.id for r in res.plans_active + res.prefs_active]},
                       why=f"{len(losers)} superseded record(s) excluded by the resolver"))

    if res.notes:
        full = "\n".join(f"- {r.topic}: {r.outcome}" for r in res.notes)
        short = "; ".join(r.topic.split(":")[0] + " resolved" for r in res.notes)
        c.append(Candidate("resolution_notes", "notes", 2, [("full", full), ("short", short)],
                           score=0.7, why="resolver output that changes how the data should be read"))

    k = 0
    for m in ranked:
        rec = m["record"]
        if rec.pinned or rec.id in losers or rec.type in ("plan", "preference"):
            continue
        k += 1
        tier = 3 if m["score"] >= 0.45 else 5
        text, variant = rec.content, "full"
        meta: dict[str, Any] = {"ids": [rec.id]}
        why = (f"score {m['score']:.2f} = 0.45*rel {m['relevance']} + 0.25*rec {m['recency']} "
               f"+ 0.30*prio {m['type_priority']}, x conf {rec.confidence}; rank {k}")
        if rec.sensitivity == "high":
            reveal = sorted(set(rec.extra.get("reveal_on", [])) & set(pre.flags))
            meta |= {"sensitive": True, "forbid": list(rec.extra.get("forbid", []))}
            if reveal:
                meta["gate"] = "open"
                why = f"sensitive; gate OPEN (flag {reveal} in reveal_on), ranks as a normal record. {why}"
            elif rec.extra.get("constraint"):
                meta["gate"] = "closed"
                text, variant, tier = rec.extra["constraint"], "constraint", GATED_TIER
                why = f"sensitive, not pinned; gate CLOSED: constraint text only, optional. {why}"
            else:
                continue   # no safe text to ship; the recall tool still returns a withheld marker
        c.append(Candidate(f"memory:{rec.id}", "memory", tier,
                           [(variant, f"- ({rec.type}, {rec.day}) {text}")], score=m["score"],
                           meta=meta, why=why, origin=rec.origin))

    sessions = sorted({t.session for t in sc.history})
    if sessions:
        fmt = lambda ts: "\n".join(f"[{t.ts[5:16].replace('T', ' ')}] {t.role}: {t.text}" for t in ts)
        last2 = [t for t in sc.history if t.session in sessions[-2:]]
        last1 = [t for t in sc.history if t.session == sessions[-1]]
        c.append(Candidate("recent_history", "history", 3,
                           [("last_2_sessions", fmt(last2)), ("last_session", fmt(last1)),
                            ("last_user_turn", fmt(last1[-1:]))], score=0.6,
                           why="verbatim recency; older sessions are summarized instead"))
        older = [s for s in sc.session_summaries if s["session"] not in sessions[-2:]]
        if older:
            line = lambda s: f"- S{s['session']} {s['date'][5:]}: {s['text']}"
            head = "Earlier sessions (summaries):\n"
            c.append(Candidate("older_history", "history", 4,
                               [("summaries", head + "\n".join(line(s) for s in older)),
                                ("last_two_only", head + "\n".join(line(s) for s in older[-2:]))], score=0.3,
                               why="summaries written at session close"))

    c.extend(domain.candidates(sc, res))
    return c


# ------------------------------------------------------------------- assembly

def _order(cands: list[Candidate]) -> list[Candidate]:
    idx = {id(c): i for i, c in enumerate(cands)}
    return sorted(cands, key=lambda c: (c.tier, 0 if c.required else 1,
                                        0.0 if c.required else -c.density(), idx[id(c)]))


def required_floor(cands: list[Candidate]) -> int:
    """Smallest budget that fits every required candidate at its cheapest variant."""
    req = [c for c in cands if c.required]
    return sum(c.cheapest() for c in req) + sum(_hdr(s) for s in {c.section for c in req})


def remove_ehr_for_remote(cands: list[Candidate], mode: str) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Remote mode: drop every ehr-origin candidate before any budget arithmetic, required or not.
    The record names only the key and the token count of its cheapest variant."""
    if mode != "remote":
        return cands, []
    kept = [c for c in cands if c.origin != "ehr"]
    gone = [{"key": c.key, "tokens": c.cheapest(), "why": "excluded: remote mode, origin ehr"}
            for c in cands if c.origin == "ehr"]
    return kept, gone


def assemble(cands: list[Candidate], budget: int, mode: str = "local") -> Assembly:
    if mode not in ("local", "remote"):
        raise ValueError(f"unknown assembly mode {mode!r}")
    cands, excluded = remove_ehr_for_remote(cands, mode)
    floor = required_floor(cands)
    if floor > budget:
        raise BudgetTooSmall(budget, floor)
    order = _order(cands)
    pending_hdr = {c.section for c in order if c.required}
    reserved = sum(c.cheapest() for c in order if c.required)
    opened: set[str] = set()
    placed: list[Placed] = []
    decisions: list[dict[str, Any]] = []
    used = 0
    for cand in order:
        costs = cand.costs()
        if cand.required:
            reserved -= cand.cheapest()
        hdr_now = 0 if cand.section in opened else _hdr(cand.section)
        held = sum(_hdr(s) for s in pending_hdr) - (_hdr(cand.section) if cand.section in pending_hdr else 0)
        room = budget - used - reserved - held
        chosen = next(((n, t) for n, t in cand.variants if costs[n] + hdr_now <= room), None)
        base = {"tier": cand.tier, "key": cand.key, "variant_costs": costs, "required": cand.required}
        if chosen is None:
            if cand.required:
                raise RuntimeError(f"required candidate {cand.key} did not fit")  # unreachable
            decisions.append({**base, "kept": "DROPPED", "tokens": 0,
                              "why": (f"cheapest variant {cand.cheapest() + hdr_now} > {room} left after "
                                      f"reserving required context; score/token {cand.density():.3f}. {cand.why}")})
            continue
        name, text = chosen
        used += costs[name] + hdr_now
        opened.add(cand.section)
        pending_hdr.discard(cand.section)
        richest = cand.variants[0][0]
        how = ("richest variant" if name == richest else
               f"degraded from '{richest}' ({costs[richest]}) to fit {room} left")
        placed.append(Placed(cand, name, text, costs[name], how))
        decisions.append({**base, "kept": name, "tokens": costs[name], "why": f"{how}. {cand.why}"})
    gates = [{"id": p.candidate.meta["ids"][0], "gate": p.candidate.meta["gate"], "shipped": p.variant}
             for p in placed if p.candidate.meta.get("gate")]
    return Assembly(budget, placed, decisions, used, gates=gates, excluded=excluded, mode=mode)


def _missing_note(asm: Assembly) -> str:
    ids = [d["key"].split(":", 1)[1] for d in asm.decisions
           if d["kept"] == "DROPPED" and d["key"].startswith("memory:") and d["tier"] <= 3]
    if not ids:
        return ""
    return TIER3_NOTE.format(n=len(ids)) + f" Ids: {', '.join(ids)}."


def assemble_with_note(cands: list[Candidate], budget: int, mode: str = "local") -> Assembly:
    """Assemble, then add the missing-context line if a relevant memory was dropped.
    The line is counted: the budget is reduced by its cost and the assembly re-run
    until the line fits the room that was set aside for it. When the budget is under the
    floor, the error carries the smallest budget that does work (the line included)."""
    try:
        return _assemble_with_note(cands, budget, mode)
    except BudgetTooSmall as exc:
        for b in range(exc.floor, exc.floor + 500):
            try:
                _assemble_with_note(cands, b, mode)
            except (BudgetTooSmall, RuntimeError):
                continue
            raise BudgetTooSmall(budget, b) from None
        raise


def _assemble_with_note(cands: list[Candidate], budget: int, mode: str = "local") -> Assembly:
    reserve = 0
    for _ in range(6):
        asm = assemble(cands, budget - reserve, mode)
        note = _missing_note(asm)
        cost = count_tokens(note) + 1 if note else 0
        if cost <= reserve:
            asm.budget, asm.note, asm.note_tokens = budget, note, cost
            asm.used += cost
            return asm
        reserve = cost
    raise RuntimeError("missing-context note did not converge")


def render(asm: Assembly) -> tuple[str, str]:
    """Return (system, user prompt). Stable content first."""
    by: dict[str, list[Placed]] = {}
    for p in asm.placed:
        by.setdefault(p.candidate.section, []).append(p)
    system = "\n".join(p.text for p in by.get("system", []))
    parts = []
    for sec in SECTION_ORDER[1:]:
        if sec == "note":
            if asm.note:
                parts.append(asm.note)
        elif sec in by:
            parts.append(f"## {SECTION_TITLES[sec]}\n" + "\n".join(p.text for p in by[sec]))
    return system, "\n\n".join(parts)
