"""The personal-health Domain: plugs wearables, rules, safety and the local EHR into the core."""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.domain import Domain, Scenario, Turn
from harness.model import StubInputs

from .baselines import load_regimes
from .canonical import WearableStore
from .config import load_config, resolve_data_dir
from .ehr.answer import ehr_candidates, make_lab_responder, make_responder
from .precedence import load_precedence
from .rules import RULES
from .safety import (EMERGENCY_MESSAGE, EMERGENCY_PATTERNS, ENDORSEMENT_PATTERNS, ESCALATION_CONTACT, PRE_PATTERNS,
                     RISK_FLAG_RULES, emergency_check)
from .signals import candidates
from .triggers import TRIGGERS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TS = "2026-09-29T08:10:00"
DEFAULT_MESSAGE = "Thinking of going hard on a long run this morning. Thoughts?"
DEFAULT_BUDGET = 1250

SYSTEM_PROMPT = (
    "You are a personal health-data assistant for one person's own wearable data (WHOOP and Garmin). Ground every "
    "claim in the context below; if the context does not support a claim, do not make it. Name the source device for "
    "every number. HRV from two devices is not comparable as values: read each against its own baseline. Be direct and "
    "brief. You are not a clinician: do not diagnose or give medical, medication, supplement or dosing advice; for "
    "those, point to the prescriber, pharmacist or doctor. Treat every pinned constraint as binding and never speculate "
    "about what is behind it. Respond in under 120 words."
)
SYNONYMS = {
    "sleep": {"slept", "asleep", "night", "tired", "rested"}, "hrv": {"variability", "recovery"},
    "race": {"racing", "half", "marathon"}, "run": {"running", "runs", "workout", "training"},
    "hard": {"intensity", "effort", "push"}, "heart": {"hr", "rhr", "resting"},
}


# ------------------------------------------------------------------ conversation history (plain JSON)

def history_path(data_dir: Path) -> Path:
    return Path(data_dir) / "history.json"


def load_history(data_dir: Path) -> list[Turn]:
    p = history_path(data_dir)
    if not p.exists():
        return []
    return [Turn(**t) for t in json.loads(p.read_text())]


def append_history(data_dir: Path, turns: list[Turn]) -> None:
    hist = load_history(data_dir) + turns
    p = history_path(data_dir)
    p.write_text(json.dumps([asdict(t) for t in hist], indent=1))
    p.chmod(0o600)


# ------------------------------------------------------------------ scenario

def _visible(series: dict[str, list[dict[str, Any]]], now: datetime) -> dict[str, list[dict[str, Any]]]:
    """What had happened by `now`. A night is visible once its sleep has ended; a day's totals once the day is over
    (today keeps only the morning recovery score); a workout once it has started."""
    today, stamp = now.date().isoformat(), now.strftime("%Y-%m-%dT%H:%M:%S")
    nights = [r for r in series["nights"] if r["date"] < today or (r["date"] == today and (r["sleep_end"] or "")[:19] <= stamp)]
    have_today = any(r["date"] == today for r in nights)
    daily = []
    for r in series["daily"]:
        if r["date"] < today:
            daily.append(r)
        elif r["date"] == today and have_today and r.get("recovery_pct") is not None:
            daily.append({**r, "steps": None, "strain": None, "active_kcal": None, "body_battery_max": None, "body_battery_min": None,
                          "stress_avg": None, "intensity_minutes": None, "training_load_acute": None})
    workouts = [w for w in series["workouts"] if w["start"][:19] < stamp]
    return {"nights": nights, "daily": daily, "workouts": workouts}


def build_domain(data_dir: str | Path | None = None) -> Domain:
    dd = resolve_data_dir(data_dir)

    def build_scenario(now: datetime | None = None) -> Scenario:
        now = now or datetime.fromisoformat(DEFAULT_TS)
        cfg = load_config(dd)
        series = _visible(WearableStore.load(dd).series(), now)
        history = [t for t in load_history(dd) if t.ts < now.isoformat()]
        session = max((t.session for t in history), default=0) + 1
        profile = {"timezone": cfg["home_tz"], "precedence": load_precedence(dd), "regimes": load_regimes(dd), "cfg": cfg}
        return Scenario("me", profile, Turn(now.isoformat(), "user", DEFAULT_MESSAGE, session), history, series, [], now)

    return Domain(
        name="health", system_prompt=SYSTEM_PROMPT, build_scenario=build_scenario, seed_path=ROOT / "mock_exports" / "memory_seed.json",
        rules=RULES, candidates=lambda sc, res: candidates(sc, res, lambda: ehr_candidates(dd)),
        pre_patterns=PRE_PATTERNS, emergency_patterns=EMERGENCY_PATTERNS, emergency_message=EMERGENCY_MESSAGE,
        risk_flag_rules=RISK_FLAG_RULES, endorsement_patterns=ENDORSEMENT_PATTERNS, triggers=TRIGGERS,
        query_synonyms=SYNONYMS, escalation_contact=ESCALATION_CONTACT,
        local_responders={"*": make_lab_responder(dd), "ehr_lookup": make_responder(dd)}, stub_reply=stub_reply, post_requirements=post_requirements,
        emergency_check=emergency_check)


def post_requirements(pre: Any, res: Any) -> list[list[str]]:
    need: list[list[str]] = []
    if "medication_question" in pre.flags:
        need.append(["prescriber", "pharmacist", "doctor", "clinician"])      # the answer must point somewhere safe
    if "lab_question" in pre.flags:
        need.append(["clinician", "doctor"])                       # a lab result is read by the person who ordered it
    if "fitness_claim" in pre.flags:
        need.append(["clinical"])
    return need


# ------------------------------------------------------------------ stub reply

def stub_reply(inp: StubInputs) -> str:
    """A plain reply built from the turn's structured inputs. It names the clinical-factor constraint
    when one shipped, never the cause, and never quotes anything that is not in the prompt."""
    v, flags = inp.values, inp.flags
    shipped = [p.text.strip() for p in inp.pinned if not p.sensitive]
    parts: list[str] = []
    facts = (inp.trigger.facts if inp.trigger is not None and inp.trigger.facts else v.get("facts", ""))

    if "medication_question" in flags and "[a medication on file]" in inp.incoming:
        # the user named a medication from their own record; in remote mode the stub sees only the placeholder
        parts.append("That's a fair question about whether a medication you take could be behind how you feel.")
        parts.append("Medication questions are best answered by your prescriber or pharmacist, and I can note this to raise at your next visit if you'd like.")
    elif "lab_question" in flags and "[a lab result on file]" in inp.incoming:
        parts.append("That is a fair question about a lab result.")
        parts.append("I can't interpret a lab result from here, so please take it to your clinician or doctor, and I can note it to raise at your next visit if you'd like.")
    elif "medication_question" in flags:
        parts.append("I can't advise on starting, stopping or combining medications or supplements. Please check with your "
                     "prescriber or pharmacist before you take it.")
        on_file = [s.lstrip("- ") for s in shipped if s.lower().startswith("do not suggest")]
        if on_file:
            parts.append("On file: " + " ".join(on_file))
    elif "fitness_claim" in flags:
        clin = next((s for s in shipped if "lowers resting heart rate" in s), None)
        m = re.search(r"from ([A-Z][a-z]+ \d{4})", clin or "")
        if clin:
            parts.append(f"I can't read that as a fitness gain: a clinical factor on file lowers resting heart rate"
                         f"{' from ' + m.group(1) if m else ''}, so compare it only with levels after that change.")
        else:
            parts.append("A lower resting heart rate can come from fitness or from other clinical factors, and one number can't tell "
                         "which; check with your clinician before reading it as a fitness gain.")
    else:
        if facts:
            parts.append(facts)
        if "training_load_risk" in flags:
            rec = v.get("recovery")
            if rec and rec["value"] < 33:
                parts.append("Recovery is low, so keeping today easy or taking a rest day is the safer call than going hard.")
            else:
                parts.append("Your sleep baseline has been drifting down for weeks, so an easy effort is the safer call than going hard today.")
        for n in inp.notes:
            if n.kind == "measured_vs_recorded" and n.about_incoming:
                parts.append(f"On the data: {n.outcome}.")
            elif n.kind in ("provisional",) and n.about_incoming:
                parts.append(f"Noted, but ({n.evidence}) I'm treating that change as provisional and will ask you to confirm it on a later day.")
        for n in inp.notes:
            if n.topic.startswith(("sleep-disagreement", "hrv-direction", "missing-source")) and not any(n.outcome in p for p in parts):
                parts.append(n.outcome + ".")
        if inp.trigger is not None and inp.trigger.kind == "scheduled" and inp.open_loops:
            parts.append("Still open: " + inp.open_loops[0].content.replace(" Unanswered.", "") + " Which is it?")
    return " ".join(parts) or "Noted. Anything you want me to keep an eye on?"
