"""Domain resolution rules for personal health.

1. Sensor beats self-report for measured facts, with the source named.
2. Source disagreement notes: reported, never averaged; precedence is the user's setting.
3. Provisional changes on volatile days: a goal or preference change made on a day with a same-day
   contradiction, crash language, or recovery under 33% wins for the turn but does not supersede; it
   writes an open loop "Confirm: ..." and supersedes only when the user confirms on a later day.
4. Regime-aware: days inside a marked regime (travel, illness, a clinical change) are excluded from
   baselines and trend statements.
Rules read only the scenario, which is already filtered to what had happened by `now`.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from harness.clip import clip
from harness.domain import Resolution, RuleResult, Scenario
from harness.memory import MemoryOp, MemoryRecord, MemoryStore

from .baselines import Baselines, regime_days
from .canonical import SOURCE_LABEL, hm
from .ehr.derive import month_label
from .precedence import WearView, disagreement_summary, night_disagreement
from .safety import is_medication_question
from .signals import facts as facts_text

RECOVERY_LOW = 33
CHANGE = re.compile(
    r"\b(?:no more|not (?:going to|gonna|doing)|(?:i'm|im|i am) (?:done|giving up|quitting)|give up on|"
    r"forget (?:about )?(?:the|my)|cancel(?:ing|ed|led)?|scrap|drop(?:ping)? the|taking it easy|no (?:more )?rac(?:e|ing)|"
    r"switch(?:ing)? (?:to|my goal)|change(?:d)? (?:my )?(?:plan|goal)|new goal)\b", re.I)
CONFIRM = re.compile(r"\b(?:confirm(?:ed)?|still the plan|still true|sticking with|stand by|that's right|yes,? (?:i|that)|"
                     r"yes,? still)\b", re.I)
CRASH = re.compile(r"\b(?:wrecked|exhausted|crashed|terrible|awful|dead tired|burnt out|burned out|sick|shattered|"
                   r"drained|miserable)\b", re.I)
POSITIVE = re.compile(r"\b(?:great|strong|fresh|energi[sz]ed|fantastic|amazing|good)\b", re.I)
TRIGGER_TEXT = re.compile(r"^\[(?:scheduled|signal) trigger")
SELF_SLEEP = re.compile(r"\bslept\s+(?:about\s+|around\s+|only\s+|just\s+)?(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hours?)\b", re.I)


def _ctx(sc: Scenario) -> tuple[WearView, Baselines]:
    return (WearView(sc.series, sc.profile["precedence"]),
            Baselines(sc.series.get("nights", []), sc.profile["regimes"], sc.profile["cfg"]))


def compute_values(sc: Scenario) -> dict[str, Any]:
    """Everything the context summary, triggers and stub are built from."""
    view, base = _ctx(sc)
    as_of = view.last_night()
    if as_of is None:
        return {}
    cfg = sc.profile["cfg"]
    v: dict[str, Any] = {"as_of": as_of}
    last: dict[str, Any] = {}
    for sig in ("sleep_duration", "resting_hr", "hrv", "spo2"):
        p = view.pick(sig, as_of)
        if p:
            last[sig] = {"source": p.source, "label": p.label, "value": p.value, "skipped": p.skipped, "preferred": p.preferred}
    v["last"] = last
    stages = view.pick("sleep_stages", as_of)
    if stages:
        v["stages"] = {"label": stages.label, **{k: stages.row.get(k) for k in ("deep_min", "rem_min", "light_min", "awake_min")}}
    rec = view.pick("recovery", as_of)
    if rec:
        v["recovery"] = {"value": rec.value, "source": rec.source, "label": rec.label}
    # each source against its own baseline: HRV values are never compared across sources
    hrv: dict[str, Any] = {}
    for src in sorted({s for s, d in view.nights if d == as_of}):
        row = view.nights[(src, as_of)]
        if row.get("hrv_ms") is not None:
            hrv[src] = {"label": SOURCE_LABEL.get(src, src), "value": row["hrv_ms"], "method": row.get("hrv_method"),
                        "median": base.rolling(src, "hrv_ms", as_of, include_as_of=False),
                        "direction": base.direction(src, "hrv_ms", as_of)}
    v["hrv"] = hrv
    v["sources"] = {src: {"label": SOURCE_LABEL.get(src, src), "asleep_min": view.nights[(src, as_of)].get("asleep_min"),
                          "resting_hr": view.nights[(src, as_of)].get("resting_hr_bpm")}
                    for src in sorted({s for s, d in view.nights if d == as_of})}
    baseline, divergence = {}, {}
    for sig in ("sleep_duration", "resting_hr"):
        if sig in last:
            src = last[sig]["source"]
            d = base.divergence(src, sig, as_of)
            divergence[sig] = d
            baseline[sig] = {"source": src, "label": last[sig]["label"], "rolling": d["rolling"], "anchored": d["anchored"],
                             "delta": d["delta"], "delta_pct": d["delta_pct"]}
    v["baseline"], v["divergence"] = baseline, divergence
    v["disagreement_last"] = night_disagreement(view, base, as_of, cfg)
    v["disagreement_14d"] = disagreement_summary(view, base, as_of, cfg)
    regs = sc.profile["regimes"]
    v["regimes"] = regs
    clin = next((r for r in regs if r["kind"] == "clinical_change"), None)
    v["clinical_regime"] = None if clin is None else {"start": clin["start"], "end": clin["end"], "month": month_label(clin["start"])}
    v["regime_days_with_data"] = sorted(regime_days(regs) & {d for _, d in view.nights})
    v["facts"] = facts_text(v)
    return v


def computed_values(sc: Scenario, store: MemoryStore) -> RuleResult:
    v = compute_values(sc)
    return RuleResult(values=v, excluded=set(v.get("regime_days_with_data", [])))


def source_disagreement(sc: Scenario, store: MemoryStore) -> RuleResult:
    out = RuleResult()
    v = compute_values(sc)
    if not v:
        return out
    day = v["as_of"]
    for sig, info in v["last"].items():
        if info["skipped"]:
            names = ", ".join(SOURCE_LABEL.get(s, s) for s in info["skipped"])
            out.resolutions.append(Resolution(
                f"missing-source:{sig}:{day}", f"{names} has no {sig.replace('_', ' ')} for last night; using {info['label']} per your settings",
                "precedence falls through to the next source with a value", f"{day}", kind="note"))
    d = v["disagreement_last"]
    if d:
        a, b = (SOURCE_LABEL.get(s, s) for s in d["sources"])
        winner = SOURCE_LABEL.get(v["last"].get("sleep_duration", {}).get("source", d["sources"][0]))
        if d.get("sleep_over"):
            ra, rb = (v["sources"][s]["asleep_min"] for s in d["sources"])
            out.resolutions.append(Resolution(
                f"sleep-disagreement:{day}",
                f"Sources disagree on last night's sleep by {abs(d['sleep_diff_min'])} minutes; using {winner} per your settings",
                "overlapping signals: disagreement is reported, never averaged; precedence is the user's setting",
                f"{a} {hm(ra)} vs {b} {hm(rb)}", kind="note"))
        if d.get("rhr_over"):
            out.resolutions.append(Resolution(
                f"rhr-disagreement:{day}", f"Sources disagree on last night's resting heart rate by {abs(d['rhr_diff']):g} bpm; "
                f"using {SOURCE_LABEL.get(v['last'].get('resting_hr', {}).get('source', d['sources'][0]))} per your settings",
                "overlapping signals: disagreement is reported, never averaged", f"{a} vs {b}", kind="note"))
        if d.get("hrv_mismatch"):
            dirs = d["hrv_direction"]
            out.resolutions.append(Resolution(
                f"hrv-direction:{day}",
                f"HRV disagrees in direction against each device's own baseline ({a} {dirs[d['sources'][0]]}, {b} {dirs[d['sources'][1]]}); "
                f"the two HRV measures are not averaged or compared as values",
                "HRV is measured differently by each device; only direction against its own baseline is comparable", f"{a} vs {b}", kind="note"))
    return out


def sensor_beats_self_report(sc: Scenario, store: MemoryStore) -> RuleResult:
    out = RuleResult()
    v = compute_values(sc)
    m = SELF_SLEEP.search(sc.incoming.text)
    if not m or "sleep_duration" not in v.get("last", {}):
        return out
    said = float(m.group(1)) * 60
    info = v["last"]["sleep_duration"]
    if abs(said - info["value"]) <= 60:
        return out
    out.resolutions.append(Resolution(
        f"self-report-sleep:{v['as_of']}",
        f"{info['label']} measured {hm(info['value'])} for last night but you said about {hm(said)}; the measurement stands and "
        f"your statement is kept as what you recorded",
        "sensor beats self-report for measured facts", f"{info['label']} {hm(info['value'])} vs stated {hm(said)}",
        kind="measured_vs_recorded", about_incoming=True))
    return out


def _volatile_reasons(sc: Scenario, v: dict[str, Any]) -> list[str]:
    reasons = []
    text, today = sc.incoming.text, sc.now.date().isoformat()
    if CRASH.search(text):
        reasons.append("crash language in the message")
    rec = v.get("recovery")
    if rec and rec["value"] < RECOVERY_LOW and v.get("as_of") == today:
        reasons.append(f"recovery {rec['value']:g}% today")
    earlier = [t for t in sc.history if t.role == "user" and t.ts[:10] == today]
    if any(POSITIVE.search(t.text) for t in earlier) and CRASH.search(text):
        reasons.append("same-day contradiction")
    return reasons


def _short(text: str, n: int = 120, redact=None) -> str:
    """`text` cut to `n` characters at a word boundary, redacted first when it has to be cut (see harness.clip)."""
    return clip(" ".join(text.split()).rstrip("."), n, redact, word=True)


def provisional_changes(sc: Scenario, store: MemoryStore) -> RuleResult:
    out = RuleResult()
    redact = sc.profile.get("redact")       # set by the boundary; None when nothing is redacted
    v = compute_values(sc)
    today = sc.now.date().isoformat()
    loops = [r for r in store.active() if r.type == "open_loop" and r.topic.startswith("open:confirm:")]
    pending = []
    for r in store.active():
        old_id = r.extra.get("provisional_for")
        old = store.get(old_id) if old_id else None
        if old is not None and old.status == "active" and loops:
            pending.append((r, old))
    for new, old in pending:
        confirmed = (not TRIGGER_TEXT.match(sc.incoming.text) and new.day < today and CONFIRM.search(sc.incoming.text))
        if confirmed:
            out.ops.append(MemoryOp("supersede", None, old.id, f"user confirmed {new.id} on a later day ({today})", by=new.id))
            for lp in loops:
                out.ops.append(MemoryOp("supersede", None, lp.id, "the provisional change was confirmed"))
            out.resolutions.append(Resolution(
                f"provisional-confirmed:{new.id}", f"you confirmed '{_short(new.content, redact=redact)}' on a later day; it now supersedes "
                f"'{_short(old.content, redact=redact)}'", "a provisional change supersedes only on confirmation on a later day",
                f"stated {month_label(new.day)}, confirmed on a later day", None, [old.id], kind="later_wins"))
        else:
            out.resolutions.append(Resolution(
                f"provisional:{new.topic}",
                f"'{_short(new.content, redact=redact)}' (stated {month_label(new.day)}) wins for this turn but was made on a volatile day, so it is "
                f"provisional; '{_short(old.content, redact=redact)}' stays on file until you confirm on a later day",
                "provisional change on a volatile day: wins for the turn, supersedes only on later confirmation",
                f"{new.id} provisional for {old.id}", None, [old.id], kind="provisional"))
    # a new change in today's message
    text = sc.incoming.text
    if TRIGGER_TEXT.match(text) or not CHANGE.search(text) or pending:
        return out
    plans = [r for r in store.active() if r.type == "plan" and not r.extra.get("provisional_for")]
    if not plans:
        return out
    old = max(plans, key=lambda r: r.updated_at)
    stamp = f"{sc.now:%Y%m%d-%H%M}"
    new_text = _short(text, 160, redact) + "."
    reasons = _volatile_reasons(sc, v)
    ts = sc.now.isoformat()
    if reasons:
        nid = f"prov-{stamp}"
        out.ops.append(MemoryOp("add", MemoryRecord(
            id=nid, type="plan", content=new_text, created_at=ts, updated_at=ts, source="user_stated", confidence=0.7,
            topic=f"{old.topic}:provisional", extra={"provisional_for": old.id, "volatile_because": reasons}), None,
            f"change stated on a volatile day ({'; '.join(reasons)}): provisional, does not supersede"))
        out.ops.append(MemoryOp("add", MemoryRecord(
            id=f"loop-{stamp}", type="open_loop", topic=f"open:confirm:{old.topic}", created_at=ts, updated_at=ts,
            source="inferred", confidence=0.8,
            content=f"Confirm: you changed '{_short(old.content, redact=redact)}' to '{_short(text, 100, redact)}' on {today}. Unanswered."), None,
            "ask for confirmation on a later day"))
        out.resolutions.append(Resolution(
            f"provisional:{old.topic}", f"your change ('{_short(text, 100, redact)}') wins for this turn but was made on a volatile day "
            f"({'; '.join(reasons)}), so it is provisional; '{_short(old.content, redact=redact)}' stays on file until you confirm on a later day",
            "provisional change on a volatile day: wins for the turn, supersedes only on later confirmation",
            "; ".join(reasons), None, [old.id], kind="provisional", about_incoming=True))
    else:
        nid = f"chg-{stamp}"
        out.ops.append(MemoryOp("add", MemoryRecord(
            id=nid, type="plan", content=new_text, created_at=ts, updated_at=ts, source="user_stated", confidence=0.85,
            topic=old.topic), None, "plan change on an ordinary day"))
        out.ops.append(MemoryOp("supersede", None, old.id, "replaced by the user's later statement", by=nid))
    return out


def medication_notes(sc: Scenario, store: MemoryStore) -> RuleResult:
    out = RuleResult()
    redact = sc.profile.get("redact")
    if not is_medication_question(sc.incoming.text) or TRIGGER_TEXT.match(sc.incoming.text):
        return out
    pref = next((r for r in store.active() if r.type == "preference" and "see-a-doctor" in r.content.lower().replace(" ", "-")
                 or r.topic == "pref:no-reflexive-referral"), None)
    if pref is not None:
        out.resolutions.append(Resolution(
            "preference-override:medication", f"preference '{_short(pref.content, redact=redact)}' is overridden for medication questions: "
            f"the reply must point to the prescriber or pharmacist",
            "by design, medication questions override the no-reflexive-referral preference",
            f"{pref.id} vs medication question", kind="note"))
    return out


def regime_note(sc: Scenario, store: MemoryStore) -> RuleResult:
    out = RuleResult()
    v = compute_values(sc)
    if not v:
        return out
    clin = v.get("clinical_regime")
    if clin:
        out.resolutions.append(Resolution(
            "regime:clinical", f"since {clin['month']} resting heart rate and HRV sit at a new level because of a clinical factor; "
            f"compare only with levels after the change and do not read the shift as a fitness change",
            "regime break: the clinical change month is excluded from baselines and trends", f"regime {clin['start']}..{clin['end']}",
            kind="note"))
    for r in v["regimes"]:
        if r["kind"] != "clinical_change" and r["end"] >= _days_before(v["as_of"], 90):
            out.resolutions.append(Resolution(
                f"regime:{r['kind']}:{r['start']}", f"{r['start']} to {r['end']} is marked as a {r['kind']} regime ({r['label']}); "
                f"those days are excluded from baselines and trend statements", "regime break", f"{r['start']}..{r['end']}", kind="note"))
    return out


def _days_before(day: str, n: int) -> str:
    from datetime import date, timedelta
    return (date.fromisoformat(day) - timedelta(days=n)).isoformat()


RULES = [computed_values, source_disagreement, sensor_beats_self_report, provisional_changes, medication_notes, regime_note]
