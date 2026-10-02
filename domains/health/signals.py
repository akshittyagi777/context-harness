"""Context candidates: the wearable summary with source attribution, disagreement notes, raw rows, and
(local mode only) EHR candidates.

Every number a reply may quote appears in the `facts` sentence, which both the full and the one-line
variants carry, so a degraded summary still grounds the reply.
"""
from __future__ import annotations

from typing import Any, Callable

from harness.context import Candidate
from harness.domain import Scenario
from harness.resolve import Resolved

from .canonical import SOURCE_LABEL, hm


def facts(v: dict[str, Any]) -> str:
    """One sentence of last-night facts, each with its source."""
    last = v.get("last", {})
    parts = []
    if "sleep_duration" in last:
        parts.append(f"sleep {hm(last['sleep_duration']['value'])} ({last['sleep_duration']['label']})")
    if "resting_hr" in last:
        parts.append(f"resting HR {last['resting_hr']['value']:g} bpm ({last['resting_hr']['label']})")
    rec = v.get("recovery")
    if rec:
        parts.append(f"recovery {rec['value']:g}% ({rec['label']})")
    return f"Last night ({v['as_of']}): " + ", ".join(parts) + "." if parts else f"Last night ({v['as_of']}): no data."


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{abs(x):.0f}%"


def summary_text(v: dict[str, Any]) -> tuple[str, str]:
    """(full, one_line). The facts sentence leads both."""
    lead = v["facts"]
    others = []
    sl = v["last"].get("sleep_duration")
    for src, s in v["sources"].items():
        if sl and src != sl["source"] and s["asleep_min"] is not None:
            others.append(f"{s['label']} shows sleep {hm(s['asleep_min'])}")
    hrv = []
    for src, h in v["hrv"].items():
        med = "" if h["median"] is None else f" against its own 30-day median of {h['median']:.0f} ms"
        d = f", {h['direction']} baseline" if h["direction"] else ""
        hrv.append(f"{h['label']} HRV {h['value']:.0f} ms ({h['method']}){med}{d}")
    full = [lead]
    if others:
        full.append("; ".join(others) + ".")
    if hrv:
        full.append("HRV is not one signal; each device is read against its own baseline and the two are never averaged: "
                    + "; ".join(hrv) + ".")
    for sig, name in (("sleep_duration", "sleep"), ("resting_hr", "resting HR")):
        b = v["baseline"].get(sig)
        if not b or b["rolling"] is None or b["anchored"] is None:
            continue
        fmt = hm if sig == "sleep_duration" else (lambda x: f"{x:.0f} bpm")
        line = (f"{b['label']} {name} 30-day median {fmt(b['rolling'])} versus anchored (first 90 days) {fmt(b['anchored'])}")
        d = v["divergence"][sig]
        if d["active"]:
            line += f"; {_pct(d['delta_pct'])} {'below' if d['delta'] < 0 else 'above'} anchored for {d['run_days']} days, divergence active"
        full.append(line + ".")
    if v.get("clinical_regime"):
        full.append(f"Resting HR and HRV level changed from {v['clinical_regime']['month']} because of a clinical factor, not fitness.")
    dis = v["disagreement_14d"]
    if dis["nights_compared"]:
        a, b = (SOURCE_LABEL.get(s, s) for s in dis["sources"])
        full.append(f"Last {dis['days']} days, {a} versus {b}: sleep differs on average by {dis['mean_sleep_diff_min']:+.0f} minutes "
                    f"({a} minus {b}); {len(dis['sleep_over_nights'])} night(s) differ by more than 45 minutes; "
                    f"HRV direction disagrees on {len(dis['hrv_mismatch_nights'])} night(s).")
    one = lead + (f" Divergence active on {', '.join(s.replace('_', ' ') for s, d in v['divergence'].items() if d['active'])}."
                  if any(d["active"] for d in v["divergence"].values()) else "")
    return " ".join(full), one


def candidates(sc: Scenario, res: Resolved, ehr_candidates: Callable[[], list[Candidate]] | None = None) -> list[Candidate]:
    v = res.values
    if not v:
        return []
    full, one = summary_text(v)
    out = [Candidate("wearable_summary", "data", 2, [("full", full), ("one_line", one)], required=True, origin="wearable",
                     why="computed after resolution; names the source of every number")]
    rows = []
    for d in sorted({r["date"] for r in sc.series["nights"]})[-7:]:
        parts = [d[5:]]
        for src in ("whoop", "garmin"):
            r = next((x for x in sc.series["nights"] if x["date"] == d and x["source"] == src), None)
            parts.append("-" if r is None or r["asleep_min"] is None else f"{r['asleep_min']}")
        rows.append("  ".join(parts))
    out.append(Candidate("raw_table", "data", 4, [("7_nights", "Raw nights (asleep minutes):\ndate  WHOOP  Garmin\n" + "\n".join(rows)),
                                                   ("3_nights", "Raw nights (asleep minutes):\ndate  WHOOP  Garmin\n" + "\n".join(rows[-3:]))],
                         score=0.2, origin="wearable", why="raw rows are evidence, not analysis; digits tokenize expensively"))
    if ehr_candidates is not None:
        out.extend(ehr_candidates())
    return out
