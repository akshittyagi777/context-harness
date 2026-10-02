"""Derived constraints: the only bridge from EHR content to a remote model.

EHR content is mapped, by a local deterministic table, to template ids. Each template is a fixed
sentence from the allowlist below. A sentence contains no identifier, drug name, condition name, code,
lab value, or date more specific than a month. The derived records are origin `derived_local`, pinned,
and may ship in remote mode; the EHR content they came from never does.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from harness.memory import MemoryOp, MemoryRecord, MemoryStore

from ..baselines import add_regime
from .classes import INTERACTS_WITH_SUPPLEMENTS, classify
from .model import EHR, date10
from .trends import flagged_trends

TEMPLATES: dict[str, str] = {
    "rhr_clinical_factor": ("A clinical factor lowers resting heart rate and may blunt heart-rate response from "
                            "{month}; do not read a lower resting heart rate since then as a fitness gain."),
    "supp_sleep_check": "Do not suggest supplements or sleep aids without checking with the prescriber.",
    "lab_trend_flag": "A lab trend is flagged for discussion with the clinician; do not speculate about it.",
    "no_specific_meds": "Do not suggest specific medications.",
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
_ALLOWED = [re.compile("^" + re.escape(t).replace(re.escape("{month}"), r"(?:%s) \d{4}" % "|".join(MONTHS)) + "$")
            for t in TEMPLATES.values()]


def is_allowlisted(sentence: str) -> bool:
    return any(p.match(sentence) for p in _ALLOWED)


def month_label(iso_date: str) -> str:
    d = date.fromisoformat(date10(iso_date))
    return f"{MONTHS[d.month - 1]} {d.year}"


@dataclass
class Derived:
    template_id: str
    text: str
    topic: str
    regime: dict[str, str] | None = None


def derive_constraints(ehr: EHR) -> list[Derived]:
    out: dict[str, Derived] = {}

    def put(tid: str, month_from: str | None = None, regime: dict | None = None) -> None:
        if tid in out:
            return
        text = TEMPLATES[tid].format(month=month_label(month_from)) if "{month}" in TEMPLATES[tid] else TEMPLATES[tid]
        out[tid] = Derived(tid, text, f"ehr-derived:{tid}", regime)

    for med in ehr.active_medications():
        d = {"display": med.display, "text": med.text, "codes": med.codes}
        cls = classify(d)
        if cls == "beta_blocker" and med.authored_on:
            start = date.fromisoformat(date10(med.authored_on))
            end_day = (date(start.year + (start.month == 12), start.month % 12 + 1, 1) - date.resolution)
            put("rhr_clinical_factor", med.authored_on,
                {"start": start.replace(day=1).isoformat(), "end": end_day.isoformat(), "kind": "clinical_change",
                 "label": "clinical factor affecting heart rate (derived locally)"})
        if cls is None or cls in INTERACTS_WITH_SUPPLEMENTS:       # unknown: the conservative default
            put("supp_sleep_check")
    if flagged_trends(ehr):
        put("lab_trend_flag")
    if any(a.category in ("medication", "") and a.status == "active" for a in ehr.allergies):
        put("no_specific_meds")
    return list(out.values())


def apply_to_memory(store: MemoryStore, derived: list[Derived], now: datetime) -> list[dict[str, Any]]:
    """Make the store's derived constraints match `derived`: add new, leave identical, supersede stale."""
    ts = now.isoformat()
    wanted = {d.topic: d for d in derived}
    ops: list[MemoryOp] = []
    n = sum(1 for r in store.records if r.id.startswith("dl-")) + 1     # own id space: never collides with r01..
    for rec in store.active():
        if rec.origin == "derived_local" and rec.topic.startswith("ehr-derived:"):
            d = wanted.get(rec.topic)
            if d is None or d.text != rec.content:
                ops.append(MemoryOp("supersede", None, rec.id, "EHR-derived constraint no longer applies"))
            else:
                wanted.pop(rec.topic)
    for d in wanted.values():
        ops.append(MemoryOp("add", MemoryRecord(
            id=f"dl-{n:02d}", type="constraint", content=d.text, created_at=ts, updated_at=ts,
            source="document", confidence=0.95, topic=d.topic, pinned=True, origin="derived_local",
            extra={"template": d.template_id}), None, f"derived locally from EHR (template {d.template_id})"))
        n += 1
    return store.apply(ops, now)


def run_derive(data_dir: Path, vault, now: datetime | None = None) -> dict[str, Any]:
    """Read the vault (logged as purpose `derive`), refresh derived constraints and the clinical regime."""
    now = now or datetime.now()
    data = vault.read("derive", ["medications", "allergies", "labs", "conditions"])
    ehr = EHR(data)
    derived = derive_constraints(ehr)
    mp = Path(data_dir) / "memory.json"
    store = MemoryStore.load(mp) if mp.exists() else MemoryStore([])
    applied = apply_to_memory(store, derived, now)
    store.save(mp)
    for d in derived:
        if d.regime:
            add_regime(data_dir, d.regime["start"], d.regime["end"], d.regime["kind"], d.regime["label"],
                       origin="derived_local")
    return {"templates": [d.template_id for d in derived], "applied": [a["op"] for a in applied]}
