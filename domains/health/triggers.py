"""Triggers: a scheduled morning brief, a baseline divergence signal, a recovery drop signal."""
from __future__ import annotations

from datetime import datetime

from harness.domain import Scenario, Trigger

from .canonical import hm
from .rules import RECOVERY_LOW, compute_values


def morning_brief(sc: Scenario, now: datetime) -> Trigger | None:
    v = compute_values(sc)
    if not v:
        return None
    facts = v["facts"]
    return Trigger(
        kind="scheduled", name="morning_brief", ts=now.isoformat(),
        text=f"[scheduled trigger: morning brief, set for 08:30] {facts} Compose a short brief. Address any open loop. Ask at most one question.",
        query="sleep recovery training today plan goal", facts=facts,
        why="a standing 08:30 brief was requested; the assistant is answering a standing request")


def baseline_divergence(sc: Scenario, now: datetime) -> Trigger | None:
    v = compute_values(sc)
    active = {s: d for s, d in v.get("divergence", {}).items() if d["active"]}
    if not active:
        return None
    sig, d = next(iter(active.items()))
    fmt = hm if sig == "sleep_duration" else (lambda x: f"{x:.0f} bpm")
    name = sig.replace("_", " ")
    facts = (f"{d['source'].upper() if d['source'] != 'whoop' else 'WHOOP'} {name}: 30-day median {fmt(d['rolling'])} versus anchored "
             f"{fmt(d['anchored'])}, {abs(d['delta_pct']):.0f}% {'below' if d['delta'] < 0 else 'above'} for {d['run_days']} days.")
    return Trigger(
        kind="signal", name="baseline_divergence", ts=now.isoformat(),
        text=f"[signal trigger: {facts}] Decide whether this is worth interrupting the user about and, if so, say what it changes. If not, say one line.",
        query="sleep baseline drift trend training", facts=facts,
        why=f"the {name} rolling median has been past its divergence threshold for {d['run_days']} consecutive days (14 needed)")


def recovery_drop(sc: Scenario, now: datetime) -> Trigger | None:
    v = compute_values(sc)
    rec = v.get("recovery")
    if not rec or rec["value"] >= RECOVERY_LOW:
        return None
    facts = v["facts"]
    return Trigger(
        kind="signal", name="recovery_drop", ts=now.isoformat(),
        text=f"[signal trigger: {facts}] Decide whether this is worth interrupting the user about and, if so, say what it changes about today.",
        query="recovery sleep training today", facts=facts,
        why=f"recovery {rec['value']:g}% is under {RECOVERY_LOW}%")


TRIGGERS = {"morning_brief": morning_brief, "baseline_divergence": baseline_divergence, "recovery_drop": recovery_drop}
