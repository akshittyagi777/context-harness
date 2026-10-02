"""Domain resolution rules for the lab.

1. Measured beats recorded: a concentration the user stated versus the instrument
   readout for that day.
2. Same-day conflict: a positive and a negative statement about the same run on one day.
3. Computed values: the numbers the context summary is built from, and a reinforce
   request for the yield pattern when new data agrees with it.
All of them read only the scenario, which is already filtered to rows dated before today.
"""
from __future__ import annotations

import re

from harness.domain import Resolution, RuleResult, Scenario
from harness.memory import MemoryOp, MemoryStore, tokens

CLAIM = re.compile(r"(\d+(?:\.\d+)?)\s*mg\s*/\s*ml", re.I)
POSITIVE = re.compile(r"\b(clean|great|fine)\b", re.I)
NEGATIVE = re.compile(r"\b(aggregated|void|awful|failed)\b", re.I)
BAD_RESULT = re.compile(r"\b(rough|bad|failed|aggregat\w*|poor|smeary|degraded|low yield|awful)\b", re.I)
PATTERN_TOPIC = "pattern:yield-vs-ratio"
GENERIC = {"run", "looks", "had", "the"}


def measured_beats_recorded(sc: Scenario, store: MemoryStore) -> RuleResult:
    rows = sc.series.get("readouts", [])
    by_day = {r["date"]: r for r in rows}
    claims = [(t.ts[:10], t.text, False) for t in sc.history if t.role == "user"]
    if rows:   # a claim made today is checked against the latest reading
        claims.append((rows[-1]["date"], sc.incoming.text, True))
    out = RuleResult()
    for day, text, today in claims:
        m = CLAIM.search(text)
        row = by_day.get(day)
        if not m or row is None:
            continue
        said, measured = float(m.group(1)), row["conc_mg_ml"]
        if abs(said - measured) / measured <= 0.5:
            continue
        out.resolutions.append(Resolution(
            f"concentration:{day}",
            (f"the instrument read {measured:g} mg/mL on {day}, but the user's statement said about "
             f"{said:g} mg/mL; the measurement stands and the statement is kept as what was recorded"),
            "measured beats recorded when they differ by more than 50%",
            f"readout {measured:g} vs statement {said:g} mg/mL", kind="measured_vs_recorded",
            about_incoming=today))
    return out


def _conflict_days(sc: Scenario) -> dict[str, tuple]:
    """Days on which an earlier positive and a later negative statement concern the same run."""
    by_day: dict[str, list] = {}
    for t in sc.history:
        if t.role == "user":
            by_day.setdefault(t.ts[:10], []).append(t)
    found = {}
    for day, turns in by_day.items():
        for i, early in enumerate(turns):
            for late in turns[i + 1:]:
                shared = (tokens(early.text) & tokens(late.text)) - GENERIC
                if (POSITIVE.search(early.text) and not NEGATIVE.search(early.text)
                        and NEGATIVE.search(late.text) and shared):
                    found[day] = (early, late)
    return found


def same_day_conflict(sc: Scenario, store: MemoryStore) -> RuleResult:
    out = RuleResult()
    for day, (early, late) in _conflict_days(sc).items():
        out.excluded.add(day)
        out.resolutions.append(Resolution(
            f"run:{day}",
            (f"the {late.ts[11:16]} statement overrides the {early.ts[11:16]} one for the same run; "
             f"{day} is left out of the yield trend"),
            "later statement wins on the same day", f"{early.ts[:16]} vs {late.ts[:16]}", kind="later_wins"))
    return out


def computed_values(sc: Scenario, store: MemoryStore) -> RuleResult:
    out = RuleResult()
    ro, sh, yl = (sc.series.get(k, []) for k in ("readouts", "shaker", "yield"))
    if not ro:
        return out
    excluded = set(_conflict_days(sc))
    first, last = ro[:5], ro[-5:]
    avg = lambda rows: round(sum(r["conc_mg_ml"] for r in rows) / len(rows), 2)
    v = {
        "conc_first_avg": avg(first), "conc_first_from": first[0]["date"], "conc_first_to": first[-1]["date"],
        "conc_last_avg": avg(last), "conc_last_from": last[0]["date"], "conc_last_to": last[-1]["date"],
        "a260_latest": ro[-1]["a260_280"], "a260_date": ro[-1]["date"],
        "a260_clean_max": sc.profile["a260_280_clean_max"],
        "typical_yield": sc.profile["typical_yield_mg_per_l"],
        "shaker_deviations": [r["date"] for r in sh if abs(r["actual_c"] - r["setpoint_c"]) > 2],
        "yield_excluded": sorted(excluded & {r["date"] for r in yl}),
    }
    kept = [r for r in yl if r["date"] not in excluded]
    if kept:
        v.update(yield_first=kept[0]["mg_per_l"], yield_last=kept[-1]["mg_per_l"], yield_n=len(kept),
                 yield_from=kept[0]["date"], yield_to=kept[-1]["date"])
    out.values = v
    pattern = next((r for r in store.active() if r.topic == PATTERN_TOPIC), None)
    if pattern and v["a260_latest"] > 1.0 and BAD_RESULT.search(sc.incoming.text):
        out.ops.append(MemoryOp(
            "reinforce", None, pattern.id,
            f"latest A260/280 is {v['a260_latest']} (> 1.0) and the user reports a bad result; "
            f"new data agrees with the pattern"))
    return out
