"""Deterministic mock exports for one invented person, "Jordan Rivera" (all data is made up).

  python3 health_cli.py mock --out mock_exports

Writes WHOOP export CSVs, Garmin Connect export JSON, WHOOP API-shaped JSON for September 2026, Epic
FHIR R4 searchset bundles plus an equivalent C-CDA CCD, and a user-memory seed. Formats follow
health_formats.md; anything that file marks UNCONFIRMED is marked here too.

One latent "truth" per night is generated first (sleep, resting HR, HRV, workouts); each device then
reports it with its own bias and noise, so the sources disagree the way two real wearables do.
The seeded realities (each has an eval case):
  1 WHOOP reports ~22 min more sleep than Garmin; ~8% of nights differ by more than 45 min
  2 strap-off gaps: WHOOP misses 6 nights in March 2026, Garmin 3 nights elsewhere
  3 travel: New York 2026-05-10..05-17, red-eye SFO-JFK on the night ending 2026-05-10
  4 clinical change: hypertension 2026-02-03, metoprolol succinate 2026-02-10, resting HR -6 bpm after
  5 lab trend: HbA1c 5.4, 5.6, 5.8; LDL twice; one basic metabolic panel
  6 drug allergy: penicillin (hives)
  7 slow decline: sleep drifts down ~35 min over the last 60 days
  8 acute night 2026-09-29: WHOOP 4h40, Garmin 4h05, recovery 21%
  9 user memory seed (user-stated, not EHR), including a provisional change on a volatile day
Ids, codes and values are invented. Codes marked `# illustrative code` are not guaranteed real.
"""
from __future__ import annotations

import base64
import csv
import json
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape
from zoneinfo import ZoneInfo

SEED = 20261001
GARMIN_START, WHOOP_START, END = date(2024, 8, 1), date(2025, 10, 1), date(2026, 9, 30)
LA, NY = "America/Los_Angeles", "America/New_York"
TRAVEL = (date(2026, 5, 10), date(2026, 5, 17))
WHOOP_GAP = [date(2026, 3, d) for d in range(8, 14)]                       # 6 nights, strap off
GARMIN_GAP = [date(2025, 6, 14), date(2025, 12, 2), date(2026, 7, 21)]    # 3 nights, charging
CHANGE_START = date(2026, 2, 10)       # metoprolol succinate started
FORCED_DIFF = {date(2026, 9, 10): 55}  # WHOOP minus Garmin minutes, over the 45 min threshold
ACUTE = date(2026, 9, 29)
USER_ID = "123456789"
GARMIN_ROOT = "3f1c9a52-8b7e-4d1a-9c63-0a5d2e7b4f10_1"
UTC = timezone.utc

EHR_FACTS = {   # shared by the FHIR bundles and the CCD; every value here is invented
    "patient": {"given": ["Jordan", "A."], "family": "Rivera", "birth": "1990-04-12", "gender": "other",
                "phone": "(415) 555-0142", "email": "jordan.rivera@example.com",
                "address": {"line": "1450 Valencia Street", "city": "San Francisco", "state": "CA", "postal": "94110"},
                "mrn": "Z0048213", "epi": "E3318420"},
    "condition": {"snomed": "59621000", "icd10": "I10", "display": "Essential hypertension", "text": "Hypertension",
                  "onset": "2026-02-03"},
    "meds": [
        {"rxnorm": "866924", "display": "metoprolol succinate 50 MG 24 HR Extended Release Oral Tablet",  # illustrative code
         "text": "metoprolol succinate ER 50 mg tablet", "status": "active", "authored": "2026-02-10T17:40:00Z",
         "dosage": "Take 1 tablet (50 mg) by mouth once daily"},
        {"rxnorm": "849574", "display": "naproxen sodium 220 MG Oral Tablet",  # illustrative code
         "text": "naproxen 220 mg tablet", "status": "stopped", "authored": "2025-08-19T16:00:00Z",
         "dosage": "Take 1 tablet by mouth as needed"},
    ],
    "allergy": {"rxnorm": "7980", "display": "Penicillin G", "text": "Penicillins", "reaction": "Hives",
                "severity": "moderate", "recorded": "2012-05-01"},
    "labs": [   # (loinc, display, text, value, unit, effective, ref_high, ref_text)
        ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "Hemoglobin A1c", 5.4, "%", "2025-03-18T16:10:00Z", 5.6, "<5.7 %"),
        ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "Hemoglobin A1c", 5.6, "%", "2025-11-12T17:05:00Z", 5.6, "<5.7 %"),
        ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "Hemoglobin A1c", 5.8, "%", "2026-06-09T16:20:00Z", 5.6, "<5.7 %"),
        ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation", "LDL cholesterol", 118, "mg/dL", "2025-11-12T17:05:00Z", None, ""),
        ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation", "LDL cholesterol", 104, "mg/dL", "2026-06-09T16:20:00Z", None, ""),
        ("2951-2", "Sodium [Moles/volume] in Serum or Plasma", "Sodium", 140, "mmol/L", "2026-06-09T16:20:00Z", None, ""),
        ("2823-3", "Potassium [Moles/volume] in Serum or Plasma", "Potassium", 4.2, "mmol/L", "2026-06-09T16:20:00Z", None, ""),
        ("2075-0", "Chloride [Moles/volume] in Serum or Plasma", "Chloride", 102, "mmol/L", "2026-06-09T16:20:00Z", None, ""),
        ("2028-9", "Carbon dioxide, total [Moles/volume] in Serum or Plasma", "Carbon dioxide", 25, "mmol/L", "2026-06-09T16:20:00Z", None, ""),
        ("3094-0", "Urea nitrogen [Mass/volume] in Serum or Plasma", "Blood urea nitrogen", 14, "mg/dL", "2026-06-09T16:20:00Z", None, ""),
        ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", "Creatinine", 0.9, "mg/dL", "2026-06-09T16:20:00Z", None, ""),
        ("2345-7", "Glucose [Mass/volume] in Serum or Plasma", "Glucose", 96, "mg/dL", "2026-06-09T16:20:00Z", None, ""),
        ("17861-6", "Calcium [Mass/volume] in Serum or Plasma", "Calcium", 9.4, "mg/dL", "2026-06-09T16:20:00Z", None, ""),
    ],
    "bp": [("2026-02-03T19:05:00Z", 148, 94), ("2026-02-10T17:35:00Z", 142, 90), ("2026-06-09T16:05:00Z", 126, 80)],
    "weight": [("2026-02-03T19:05:00Z", 74.2), ("2026-06-09T16:05:00Z", 72.9)],
    "epic_local_oid": "urn:oid:1.2.840.114350.1.13.999.2.7.2.768282",  # illustrative code
}


@dataclass
class Truth:
    nights: dict[str, dict[str, Any]] = field(default_factory=dict)       # date -> per-night truth
    counts: dict[str, int] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)


def daterange(a: date, b: date):
    for i in range((b - a).days + 1):
        yield a + timedelta(days=i)


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ramp(d: date) -> float:
    """0 before the clinical change, 1 ten days after it."""
    return clip((d - CHANGE_START).days / 10.0, 0.0, 1.0)


def _off_min(zone: str, naive_local: datetime) -> int:
    return int(naive_local.replace(tzinfo=ZoneInfo(zone)).utcoffset().total_seconds() // 60)


def _fmt_off(m: int) -> str:
    return f"{'-' if m < 0 else '+'}{abs(m) // 60:02d}:{abs(m) % 60:02d}"


# ------------------------------------------------------------------------- truth model

def night_truth(d: date) -> dict[str, Any]:
    rng = random.Random(f"{SEED}:night:{d}")
    zone = NY if TRAVEL[0] <= d <= TRAVEL[1] else LA
    days_from_start = (d - WHOOP_START).days
    decline = -min(42.0, 42.0 * max(0, days_from_start) / 300.0)                    # gentle drift to July
    last60 = -35.0 * clip((d - date(2026, 8, 1)).days / 59.0, 0.0, 1.0)             # the slow decline
    weekend = d.weekday() in (5, 6)
    true_sleep = 450 + 6 * math.sin(d.toordinal() / 58.0) + decline + last60 + (22 if weekend else 0) + rng.gauss(0, 22)
    hrv_true = 62 + 6 * math.sin(d.toordinal() / 41.0) + 0.08 * (true_sleep - 450) + 7 * ramp(d) + rng.gauss(0, 6)
    rhr_true = 53.5 - 0.02 * (true_sleep - 450) - 6 * ramp(d) + rng.gauss(0, 1.2)
    awake_w, awake_g = round(rng.gauss(48, 10)), round(rng.gauss(40, 10))
    w_asleep = round(true_sleep + rng.gauss(22, 12.2))
    g_asleep = round(true_sleep + rng.gauss(0, 12.2))
    if d in FORCED_DIFF:
        g_asleep = w_asleep - FORCED_DIFF[d]
    hrv_adj = 0.0
    if d == date(2026, 2, 3):
        hrv_adj = -8.0
    if d == ACUTE:
        w_asleep, g_asleep, hrv_true, rhr_true = 280, 245, 36.0, rhr_true + 5
    wake_local = datetime.combine(d, time(6, 50)) + timedelta(minutes=rng.gauss(0, 28) + (55 if weekend else 0))
    if d == date(2026, 5, 10):                                                      # the red-eye: asleep SFO->JFK
        w_asleep, g_asleep = 233, 214
        wake_utc = datetime(2026, 5, 10, 11, 10, 0)                                  # 07:10 EDT, before landing
        zone = NY
        in_bed = max(w_asleep + awake_w, g_asleep + awake_g)
        onset_utc = datetime(2026, 5, 10, 6, 50, 0)                                  # 23:50 PDT on May 9
        in_bed = int((wake_utc - onset_utc).total_seconds() // 60)
        awake_w, awake_g = in_bed - w_asleep, in_bed - g_asleep
    else:
        wake_utc = wake_local.replace(tzinfo=ZoneInfo(zone)).astimezone(UTC).replace(tzinfo=None)
        in_bed = max(w_asleep + awake_w, g_asleep + awake_g)
        onset_utc = wake_utc - timedelta(minutes=in_bed)
    off = _off_min(zone, wake_utc.replace(tzinfo=UTC).astimezone(ZoneInfo(zone)).replace(tzinfo=None))
    rec = 62 + (hrv_true + hrv_adj - (62 + 7 * ramp(d))) * 1.0 - 2.0 * (rhr_true - (53.5 - 6 * ramp(d))) \
        + 0.12 * (w_asleep - 450) + rng.gauss(0, 9)
    recovery = int(clip(round(rec), 1, 99))
    recovery = {date(2026, 2, 3): 28, date(2026, 2, 4): 31, ACUTE: 21}.get(d, recovery)
    return {"date": d, "zone": zone, "off_min": off, "onset_utc": onset_utc, "wake_utc": wake_utc, "rng_seed": f"{SEED}:dev:{d}",
            "true_sleep": true_sleep, "hrv_true": hrv_true + hrv_adj, "rhr_true": rhr_true, "recovery": recovery,
            "w_asleep": w_asleep, "g_asleep": g_asleep, "awake_w": awake_w, "awake_g": awake_g, "weekend": weekend}


def device_values(t: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """WHOOP and Garmin renderings of one night's truth."""
    rng = random.Random(t["rng_seed"])

    def stages(asleep: int) -> tuple[int, int, int]:
        deep = round(asleep * clip(rng.gauss(0.17, 0.025), 0.08, 0.28))
        rem = round(asleep * clip(rng.gauss(0.22, 0.03), 0.10, 0.34))
        return deep, rem, asleep - deep - rem

    wd, wr, wl = stages(t["w_asleep"])
    gd, gr, gl = stages(t["g_asleep"])
    jit = lambda: timedelta(seconds=rng.randint(-150, 150))
    w = {"asleep": t["w_asleep"], "deep": wd, "rem": wr, "light": wl, "awake": t["awake_w"],
         "rhr": round(t["rhr_true"] + rng.gauss(0, 0.8)), "hrv": round(t["hrv_true"] + rng.gauss(0, 3), 2),
         "resp": round(rng.gauss(14.8, 0.7), 1), "skin": round(rng.gauss(33.5, 0.35), 2),
         "spo2": round(rng.gauss(95.4, 0.9), 2), "onset": t["onset_utc"] + jit(), "wake": t["wake_utc"] + jit(),
         "recovery": t["recovery"], "consistency": int(clip(rng.gauss(78, 9), 40, 99))}
    g = {"asleep": t["g_asleep"], "deep": gd, "rem": gr, "light": gl, "awake": t["awake_g"],
         "rhr": round(t["rhr_true"] + 1.2 + rng.gauss(0, 1.2)), "hrv": round(0.88 * t["hrv_true"] + rng.gauss(0, 4.5), 1),
         "resp": round(rng.gauss(14.2, 0.7), 1), "skin": round(rng.gauss(34.0, 0.3), 2),
         "spo2": round(rng.gauss(95.0, 1.1), 1), "onset": (t["onset_utc"] + jit()).replace(second=0),
         "wake": (t["wake_utc"] + jit()).replace(second=0), "score": int(clip(rng.gauss(78, 8), 40, 98)),
         "bb_high": int(clip(50 + 0.5 * t["recovery"] + rng.gauss(0, 5), 25, 100)), "bb_low": rng.randint(5, 28),
         "stress": round(rng.gauss(30, 5), 0)}
    return w, g


def day_truth(d: date) -> dict[str, Any]:
    rng = random.Random(f"{SEED}:day:{d}")
    workouts = []
    if rng.random() < 0.58:   # about four sessions a week
        kind = rng.choices(["Running", "Cycling", "Weightlifting"], [0.65, 0.15, 0.20])[0]
        start_local = datetime.combine(d, time(17, 40)) + timedelta(minutes=rng.randint(-90, 60))
        dur = rng.randint(38, 78)
        speed = rng.uniform(2.7, 3.2) if kind == "Running" else (rng.uniform(6.0, 8.0) if kind == "Cycling" else 0.0)
        workouts.append({"sport": kind, "start_local": start_local, "dur_min": dur, "avg_hr": rng.randint(128, 158),
                         "max_hr": rng.randint(160, 184), "strain": round(rng.uniform(7.5, 15.5), 1),
                         "distance_m": round(speed * dur * 60, 1) if speed else None, "te": round(rng.uniform(2.2, 4.4), 1)})
    zone = NY if TRAVEL[0] <= d <= TRAVEL[1] else LA
    steps = int(clip(rng.gauss(9200, 2600), 2500, 24000)) + (4000 if zone == NY else 0)
    return {"workouts": workouts, "steps": steps, "strain": round(clip(rng.uniform(5.5, 9.5) + (workouts[0]["strain"] * 0.45 if workouts else 0), 4.0, 19.5), 1),
            "stress": round(rng.gauss(30, 5)), "kcal_active": int(rng.gauss(520, 130)),
            "intensity_mod": rng.randint(8, 40), "intensity_vig": rng.randint(0, 25), "load": round(rng.uniform(80, 420), 1)}


# ------------------------------------------------------------------------- WHOOP CSV

W_CYCLE_COLS = ["Cycle start time", "Cycle end time", "Cycle timezone", "Recovery score %", "Resting heart rate (bpm)",
                "Heart rate variability (ms)", "Skin temp (celsius)", "Blood oxygen %", "Day Strain", "Energy burned (cal)",
                "Max HR (bpm)", "Average HR (bpm)", "Sleep onset", "Wake onset", "Sleep performance %",
                "Respiratory rate (rpm)", "Asleep duration (min)", "In bed duration (min)", "Light sleep duration (min)",
                "Deep (SWS) duration (min)", "REM duration (min)", "Awake duration (min)", "Sleep need (min)",
                "Sleep debt (min)", "Sleep efficiency %", "Sleep consistency %"]
W_SLEEP_COLS = ["Cycle start time", "Cycle end time", "Cycle timezone", "Sleep onset", "Wake onset", "Sleep performance %",
                "Respiratory rate (rpm)", "Asleep duration (min)", "In bed duration (min)", "Light sleep duration (min)",
                "Deep (SWS) duration (min)", "REM duration (min)", "Awake duration (min)", "Sleep need (min)",
                "Sleep debt (min)", "Sleep efficiency %", "Sleep consistency %", "Nap"]
W_WORKOUT_COLS = ["Cycle start time", "Cycle end time", "Cycle timezone", "Workout start time", "Workout end time",
                  "Duration (min)", "Activity name", "Activity Strain", "Energy burned (cal)", "Max HR (bpm)",
                  "Average HR (bpm)", "HR Zone 1 %", "HR Zone 2 %", "HR Zone 3 %", "HR Zone 4 %", "HR Zone 5 %",
                  "GPS enabled", "Distance (meters)", "Altitude gain (meters)", "Altitude change (meters)"]
W_JOURNAL_COLS = ["Cycle start time", "Cycle end time", "Cycle timezone", "Question text", "Answered yes", "Notes"]
JOURNAL_QUESTIONS = ["Have any caffeine?", "Have any alcoholic drinks?", "Feeling sick or ill?", "Meditate? "]


def _local(utc_naive: datetime, off: int) -> datetime:
    return utc_naive + timedelta(minutes=off)


def write_whoop(out: Path, nights: dict[date, dict], days: dict[date, dict], end: date) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    wdates = [d for d in daterange(WHOOP_START, end) if d not in WHOOP_GAP]
    nxt_night = night_truth(end + timedelta(days=1))              # the in-progress cycle
    nxt_w, _ = device_values(nxt_night)
    cycles, sleeps, workouts, journal = [], [], [], []
    ts = lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")
    for i, d in enumerate(wdates):
        t = nights[d]
        w = t["_w"]
        off = t["off_min"]
        tz = f"UTC{_fmt_off(off)}"
        start = _local(w["onset"], off)
        if i + 1 < len(wdates):
            n2 = nights[wdates[i + 1]]
            end_dt = _local(n2["_w"]["onset"], off)
        else:
            end_dt = _local(nxt_w["onset"], off)
        dd = days[d]
        inbed = w["asleep"] + w["awake"]
        need = 480 + int(dd["strain"] * 3)
        perf = int(clip(round(100 * w["asleep"] / need), 20, 100))
        cyc = {
            "Cycle start time": ts(start), "Cycle end time": ts(end_dt), "Cycle timezone": tz,
            "Recovery score %": w["recovery"], "Resting heart rate (bpm)": w["rhr"],
            "Heart rate variability (ms)": int(round(w["hrv"])), "Skin temp (celsius)": f"{w['skin']:.2f}",
            "Blood oxygen %": f"{w['spo2']:.2f}", "Day Strain": f"{dd['strain']:.1f}",
            "Energy burned (cal)": int(2000 + dd["strain"] * 95), "Max HR (bpm)": 150 + int(dd["strain"] * 1.6),
            "Average HR (bpm)": 62 + int(dd["strain"]), "Sleep onset": ts(start), "Wake onset": ts(_local(w["wake"], off)),
            "Sleep performance %": perf, "Respiratory rate (rpm)": f"{w['resp']:.1f}", "Asleep duration (min)": w["asleep"],
            "In bed duration (min)": inbed, "Light sleep duration (min)": w["light"], "Deep (SWS) duration (min)": w["deep"],
            "REM duration (min)": w["rem"], "Awake duration (min)": w["awake"], "Sleep need (min)": need,
            "Sleep debt (min)": max(0, need - w["asleep"]), "Sleep efficiency %": int(round(100 * w["asleep"] / inbed)),
            "Sleep consistency %": w["consistency"]}
        cycles.append(cyc)
        sleeps.append({k: cyc[k] for k in W_SLEEP_COLS if k in cyc} | {"Nap": "false"})
        if d.day % 9 == 0:   # an occasional nap, which lives only in sleeps.csv
            nap_start = datetime.combine(d, time(14, 10))
            sleeps.append({"Cycle start time": ts(start), "Cycle end time": ts(end_dt), "Cycle timezone": tz,
                           "Sleep onset": ts(nap_start), "Wake onset": ts(nap_start + timedelta(minutes=25)),
                           "Sleep performance %": "", "Respiratory rate (rpm)": "14.2", "Asleep duration (min)": 22,
                           "In bed duration (min)": 25, "Light sleep duration (min)": 22, "Deep (SWS) duration (min)": 0,
                           "REM duration (min)": 0, "Awake duration (min)": 3, "Sleep need (min)": "", "Sleep debt (min)": "",
                           "Sleep efficiency %": 88, "Sleep consistency %": "", "Nap": "true"})
        for wo in dd["workouts"]:
            ws, we = wo["start_local"], wo["start_local"] + timedelta(minutes=wo["dur_min"])
            workouts.append({
                "Cycle start time": ts(start), "Cycle end time": ts(end_dt), "Cycle timezone": tz,
                "Workout start time": ts(ws), "Workout end time": ts(we), "Duration (min)": wo["dur_min"],
                "Activity name": wo["sport"], "Activity Strain": f"{wo['strain']:.1f}",
                "Energy burned (cal)": f"{wo['dur_min'] * 9.5:.1f}", "Max HR (bpm)": wo["max_hr"], "Average HR (bpm)": wo["avg_hr"],
                "HR Zone 1 %": 4, "HR Zone 2 %": 18, "HR Zone 3 %": 41, "HR Zone 4 %": 30, "HR Zone 5 %": 7,
                "GPS enabled": "true" if wo["distance_m"] else "false",
                "Distance (meters)": "" if wo["distance_m"] is None else wo["distance_m"],
                "Altitude gain (meters)": "" if wo["distance_m"] is None else "42.0",
                "Altitude change (meters)": "" if wo["distance_m"] is None else "-1.3"})
        jr = random.Random(f"{SEED}:journal:{d}")
        for q in JOURNAL_QUESTIONS:
            if jr.random() < 0.4:
                journal.append({"Cycle start time": ts(start), "Cycle end time": ts(end_dt), "Cycle timezone": tz,
                                "Question text": q, "Answered yes": "true" if jr.random() < 0.5 else "false", "Notes": ""})
    # the in-progress cycle: no end time, no strain, recovery already scored
    off = nxt_night["off_min"]
    inprog = {c: "" for c in W_CYCLE_COLS}
    on = _local(nxt_w["onset"], off)
    inprog.update({"Cycle start time": ts(on), "Cycle timezone": f"UTC{_fmt_off(off)}", "Sleep onset": ts(on),
                   "Wake onset": ts(_local(nxt_w["wake"], off)), "Recovery score %": nxt_w["recovery"],
                   "Resting heart rate (bpm)": nxt_w["rhr"], "Heart rate variability (ms)": int(round(nxt_w["hrv"])),
                   "Asleep duration (min)": nxt_w["asleep"], "In bed duration (min)": nxt_w["asleep"] + nxt_w["awake"],
                   "Light sleep duration (min)": nxt_w["light"], "Deep (SWS) duration (min)": nxt_w["deep"],
                   "REM duration (min)": nxt_w["rem"], "Awake duration (min)": nxt_w["awake"]})
    cycles.append(inprog)

    def dump(name: str, cols: list[str], rows: list[dict], newest_first_key="Cycle start time") -> None:
        rows = sorted(rows, key=lambda r: (r[newest_first_key], r.get("Workout start time", r.get("Sleep onset", ""))), reverse=True)
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            wr.writerows(rows)

    dump("physiological_cycles.csv", W_CYCLE_COLS, cycles)
    dump("sleeps.csv", W_SLEEP_COLS, sleeps)
    dump("workouts.csv", W_WORKOUT_COLS, workouts)
    dump("journal_entries.csv", W_JOURNAL_COLS, journal)
    return {"whoop_nights": len(wdates), "whoop_cycles_total": len(cycles), "whoop_workouts": len(workouts),
            "whoop_naps": sum(1 for s in sleeps if s["Nap"] == "true"), "whoop_journal": len(journal)}


# ------------------------------------------------------------------------- WHOOP API (September 2026)

def write_whoop_api(out: Path, nights: dict[date, dict], days: dict[date, dict], end: date) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    zms = lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
    dates = [d for d in daterange(max(date(2026, 9, 1), WHOOP_START), end) if d not in WHOOP_GAP]
    cycles, recov, sleeps, works = [], [], [], []
    for i, d in enumerate(dates):
        t, dd = nights[d], days[d]
        w = t["_w"]
        off = _fmt_off(t["off_min"])
        cid, sid = 93000 + i, str(uuid.UUID(int=random.Random(f"{SEED}:sid:{d}").getrandbits(128), version=4))
        end_ts = nights[dates[i + 1]]["_w"]["onset"] if i + 1 < len(dates) else None
        cyc = {"id": cid, "user_id": 10129, "created_at": zms(w["wake"]), "updated_at": zms(w["wake"] + timedelta(hours=3)),
               "start": zms(w["onset"]), "timezone_offset": off, "score_state": "SCORED",
               "score": {"strain": dd["strain"], "kilojoule": round((2000 + dd["strain"] * 95) * 4.184, 3),
                         "average_heart_rate": 62 + int(dd["strain"]), "max_heart_rate": 150 + int(dd["strain"] * 1.6)},
               "step_count": dd["steps"] if i % 5 else None}
        if end_ts is not None:
            cyc["end"] = zms(end_ts)
        cycles.append(cyc)
        recov.append({"cycle_id": cid, "sleep_id": sid, "user_id": 10129, "created_at": zms(w["wake"]),
                      "updated_at": zms(w["wake"]), "score_state": "SCORED",
                      "score": {"user_calibrating": False, "recovery_score": w["recovery"], "resting_heart_rate": w["rhr"],
                                "hrv_rmssd_milli": w["hrv"], "spo2_percentage": w["spo2"], "skin_temp_celsius": w["skin"]}})
        slp = {"id": sid, "cycle_id": cid, "user_id": 10129, "created_at": zms(w["wake"]), "updated_at": zms(w["wake"]),
               "start": zms(w["onset"]), "end": zms(w["wake"]), "timezone_offset": off, "nap": False}
        if d == end:     # the freshest night is still being scored: no `score` object
            slp["score_state"] = "PENDING_SCORE"
        else:
            slp["score_state"] = "SCORED"
            slp["score"] = {"stage_summary": {
                "total_in_bed_time_milli": (w["asleep"] + w["awake"]) * 60000, "total_awake_time_milli": w["awake"] * 60000,
                "total_no_data_time_milli": 0, "total_light_sleep_time_milli": w["light"] * 60000,
                "total_slow_wave_sleep_time_milli": w["deep"] * 60000, "total_rem_sleep_time_milli": w["rem"] * 60000,
                "sleep_cycle_count": 4, "disturbance_count": 9},
                "sleep_needed": {"baseline_milli": 27395716, "need_from_sleep_debt_milli": 352230,
                                 "need_from_recent_strain_milli": 208595, "need_from_recent_nap_milli": 0},
                "respiratory_rate": w["resp"] + 0.01, "sleep_performance_percentage": 90,
                "sleep_consistency_percentage": w["consistency"],
                "sleep_efficiency_percentage": round(100 * w["asleep"] / (w["asleep"] + w["awake"]) + 0.3, 8)}
        sleeps.append(slp)
        for wo in dd["workouts"]:
            ws = wo["start_local"] - timedelta(minutes=t["off_min"])
            works.append({"id": str(uuid.UUID(int=random.Random(f"{SEED}:wid:{ws}").getrandbits(128), version=4)),
                          "v1_id": 1043 + len(works), "user_id": 10129, "created_at": zms(ws), "updated_at": zms(ws),
                          "start": zms(ws), "end": zms(ws + timedelta(minutes=wo["dur_min"])), "timezone_offset": off,
                          "sport_name": wo["sport"].lower(), "sport_id": 0, "score_state": "SCORED",
                          "score": {"strain": wo["strain"], "average_heart_rate": wo["avg_hr"], "max_heart_rate": wo["max_hr"],
                                    "kilojoule": round(wo["dur_min"] * 9.5 * 4.184, 3), "percent_recorded": 100,
                                    "distance_meter": wo["distance_m"], "altitude_gain_meter": 46.6,
                                    "altitude_change_meter": -0.78,
                                    "zone_durations": {"zone_zero_milli": 0, "zone_one_milli": 600000, "zone_two_milli": 900000,
                                                       "zone_three_milli": 900000, "zone_four_milli": 600000, "zone_five_milli": 300000}}})
    # newest first, in two pages with a next_token, like the real collection endpoints
    for name, rows in (("cycles", cycles), ("recovery", recov), ("sleep", sleeps), ("workouts", works)):
        rows = rows[::-1]
        half = max(1, len(rows) // 2)
        (out / f"{name}_page1.json").write_text(json.dumps({"records": rows[:half], "next_token": "opaque-token-1"}, indent=1))
        (out / f"{name}_page2.json").write_text(json.dumps({"records": rows[half:], "next_token": None}, indent=1))
    return {"api_cycles": len(cycles), "api_workouts": len(works)}


# ------------------------------------------------------------------------- Garmin export

def _gmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + ".0"


def write_garmin(out: Path, nights: dict[date, dict], days: dict[date, dict], end: date) -> dict[str, int]:
    root = out / GARMIN_ROOT / "DI_CONNECT"
    wel, agg, fit, met = (root / n for n in ("DI-Connect-Wellness", "DI-Connect-Aggregator", "DI-Connect-Fitness", "DI-Connect-Metrics"))
    for p in (wel, agg, fit, met):
        p.mkdir(parents=True, exist_ok=True)
    all_days = list(daterange(GARMIN_START, end))
    sleep_rows, health_rows, uds_rows, load_rows = {}, {}, {}, {}
    n_sleep = 0
    for d in all_days:
        t, dd = nights[d], days[d]
        g = t["_g"]
        off = t["off_min"]
        if d not in GARMIN_GAP:
            n_sleep += 1
            sleep_rows[d] = {
                "sleepStartTimestampGMT": _gmt(g["onset"]), "sleepEndTimestampGMT": _gmt(g["wake"]),
                "sleepStartTimestampLocal": _gmt(g["onset"] + timedelta(minutes=off)),     # format: unconfirmed
                "sleepEndTimestampLocal": _gmt(g["wake"] + timedelta(minutes=off)),        # format: unconfirmed
                "calendarDate": d.isoformat(), "sleepWindowConfirmationType": "ENHANCED_CONFIRMED_FINAL",
                "deepSleepSeconds": g["deep"] * 60, "lightSleepSeconds": g["light"] * 60, "remSleepSeconds": g["rem"] * 60,
                "awakeSleepSeconds": g["awake"] * 60, "unmeasurableSeconds": 0, "retro": False,
                "sleepScores": {"overallScore": g["score"], "qualityScore": g["score"] - 2, "durationScore": g["score"] + 3,
                                "recoveryScore": g["score"] - 1, "deepScore": 80, "remScore": 78, "restfulnessScore": 83,
                                "feedback": "POSITIVE_LONG_AND_DEEP"},
                "averageRespiration": g["resp"], "lowestRespiration": g["resp"] - 3, "highestRespiration": g["resp"] + 4,
                "avgSleepStress": 21.4, "awakeCount": 2, "restlessMomentCount": 38,
                "spo2SleepSummary": {"averageSPO2": g["spo2"], "lowestSPO2": round(g["spo2"] - 6, 1)},
                "breathingDisruptionSeverity": "NONE"}
            health_rows[d] = {"calendarDate": d.isoformat(), "metrics": [
                {"type": "HRV", "value": g["hrv"], "baselineLowerLimit": 45.0, "baselineUpperLimit": 62.0,
                 "status": "BALANCED"},                                                     # format: unconfirmed
                {"type": "HR", "value": g["rhr"], "baselineLowerLimit": 48.0, "baselineUpperLimit": 58.0, "status": "BALANCED"},
                {"type": "SPO2", "value": g["spo2"], "baselineLowerLimit": 93.0, "baselineUpperLimit": 98.0, "status": "BALANCED"},
                {"type": "SKIN_TEMP_C", "value": g["skin"], "baselineLowerLimit": 33.0, "baselineUpperLimit": 35.0, "status": "BALANCED"},
                {"type": "RESPIRATION", "value": g["resp"], "baselineLowerLimit": 12.0, "baselineUpperLimit": 17.0, "status": "BALANCED"}]}
        steps = dd["steps"]
        uds_rows[d] = {
            "calendarDate": d.isoformat(), "totalSteps": steps, "totalDistanceMeters": int(steps * 0.78),
            "totalKilocalories": float(2060 + dd["kcal_active"]), "activeKilocalories": float(dd["kcal_active"]),
            "bmrKilocalories": 2060.0, "restingHeartRate": g["rhr"], "minHeartRate": g["rhr"] - 5, "maxHeartRate": 151,
            "moderateIntensityMinutes": dd["intensity_mod"], "vigorousIntensityMinutes": dd["intensity_vig"],
            "floorsAscendedInMeters": 30.5, "highlyActiveSeconds": 1200, "activeSeconds": 4100, "includesWellnessData": True,
            "wellnessStartTimeGmt": _gmt(datetime.combine(d, time(7, 0)) + timedelta(hours=0)),
            "allDayStress": {"aggregatorList": [{"type": "TOTAL", "averageStressLevel": int(g["stress"]), "maxStressLevel": 86,
                                                "highDuration": 900, "mediumDuration": 3600, "lowDuration": 14400,
                                                "restDuration": 28800, "totalDuration": 47700}]},
            "bodyBattery": {"chargedValue": 62, "drainedValue": 58, "bodyBatteryStatList": [      # list order is not reliable
                {"bodyBatteryStatType": "LOWEST", "statsValue": g["bb_low"]}, {"bodyBatteryStatType": "ENDOFDAY", "statsValue": 32},
                {"bodyBatteryStatType": "HIGHEST", "statsValue": g["bb_high"]}, {"bodyBatteryStatType": "STARTOFDAY", "statsValue": 70}]}}
        ms = int(datetime.combine(d, time(0, 0), tzinfo=UTC).timestamp() * 1000)
        load_rows[d] = {"calendarDate": ms, "dailyTrainingLoadAcute": dd["load"], "dailyTrainingLoadChronic": 210.0}  # format: unconfirmed
    # chunks of ~90 days with a 2 day overlap, so an importer must de-duplicate by calendarDate
    starts = list(range(0, len(all_days), 90))
    for i, s0 in enumerate(starts):
        chunk = all_days[s0: s0 + 92]
        a, b = chunk[0].isoformat(), chunk[-1].isoformat()
        (wel / f"{a}_{b}_{USER_ID}_sleepData.json").write_text(json.dumps([sleep_rows[d] for d in chunk if d in sleep_rows], indent=1))
        (wel / f"{a}_{b}_{USER_ID}_healthStatusData.json").write_text(json.dumps([health_rows[d] for d in chunk if d in health_rows], indent=1))
        (agg / f"UDSFile_{a}_{b}.json").write_text(json.dumps([uds_rows[d] for d in chunk], indent=1))   # filename pattern: unconfirmed
        (met / f"MetricsAcuteTrainingLoad_{a}_{b}_{USER_ID}.json").write_text(json.dumps([load_rows[d] for d in chunk], indent=1))
    acts = []
    for d in all_days:
        for wo in days[d]["workouts"]:
            off = nights[d]["off_min"]
            start_utc = wo["start_local"] - timedelta(minutes=off) + timedelta(seconds=random.Random(f"{SEED}:gj:{wo['start_local']}").randint(-45, 45))
            dist = wo["distance_m"]
            acts.append({"activityId": 12345678901 + len(acts), "name": f"{wo['sport']}", "activityType": wo["sport"].lower(),
                         "startTimeGmt": int(start_utc.replace(tzinfo=UTC).timestamp() * 1000),
                         "distance": None if dist is None else int(dist * 100),             # centimeters
                         "duration": wo["dur_min"] * 60000, "movingDuration": wo["dur_min"] * 60000 - 20000,
                         "elapsedDuration": wo["dur_min"] * 60000 + 15000, "avgHr": wo["avg_hr"] - 2, "maxHr": wo["max_hr"],
                         "avgSpeed": 3.0, "calories": int(wo["dur_min"] * 9.5), "steps": None if dist is None else int(dist / 0.9),
                         "aerobicTrainingEffect": wo["te"]})                                  # key: format unconfirmed
    (fit / f"{USER_ID}_0_summarizedActivities.json").write_text(json.dumps([{"summarizedActivitiesExport": acts}], indent=1))
    return {"garmin_nights": n_sleep, "garmin_days": len(all_days), "garmin_activities": len(acts)}


# ------------------------------------------------------------------------- Epic

def _opaque(rng: random.Random, prefix: str = "e") -> str:
    return prefix + base64.urlsafe_b64encode(rng.randbytes(15)).decode().rstrip("=")[:20]


def epic_bundles() -> dict[str, dict[str, Any]]:
    rng = random.Random(f"{SEED}:epic")
    F = EHR_FACTS
    base = "https://fhir.example-health.org/api/FHIR/R4"
    pid = _opaque(rng)
    subj = {"reference": f"Patient/{pid}"}
    cat = lambda code, system="http://terminology.hl7.org/CodeSystem/observation-category": [{"coding": [{"system": system, "code": code}]}]

    def bundle(rtype: str, resources: list[dict], extra_entries: list[dict] | None = None) -> dict:
        entries = [{"fullUrl": f"{base}/{r['resourceType']}/{r['id']}", "resource": r, "search": {"mode": "match"}} for r in resources]
        return {"resourceType": "Bundle", "type": "searchset", "total": len(resources),
                "link": [{"relation": "self", "url": f"{base}/{rtype}?patient={pid}"}], "entry": entries + (extra_entries or [])}

    p = F["patient"]
    patient = {"resourceType": "Patient", "id": pid, "active": True,
               "identifier": [{"use": "usual", "type": {"text": "MRN"}, "system": "urn:oid:1.2.840.114350.1.13.999.2.7.5.737384.14", "value": p["mrn"]},
                              {"use": "usual", "type": {"text": "EPI"}, "system": "urn:oid:1.2.840.114350.1.13.999.2.7.5.737384.63", "value": p["epi"]}],
               "name": [{"use": "official", "family": p["family"], "given": p["given"]}],
               "gender": p["gender"], "birthDate": p["birth"],
               "telecom": [{"system": "phone", "value": p["phone"], "use": "home"}, {"system": "email", "value": p["email"]}],
               "address": [{"use": "home", "line": [p["address"]["line"]], "city": p["address"]["city"],
                            "state": p["address"]["state"], "postalCode": p["address"]["postal"]}],
               "communication": [{"language": {"coding": [{"system": "urn:ietf:bcp:47", "code": "en"}]}, "preferred": True}],
               "extension": [{"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-birthsex", "valueCode": "F"}]}
    c = F["condition"]
    condition = {"resourceType": "Condition", "id": _opaque(rng),
                 "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
                 "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
                 "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item"}]}],
                 "code": {"coding": [{"system": "http://snomed.info/sct", "code": c["snomed"], "display": c["display"]},
                                     {"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": c["icd10"]}], "text": c["text"]},
                 "subject": subj, "onsetDateTime": c["onset"], "recordedDate": c["onset"]}
    meds = []
    for m in F["meds"]:
        meds.append({"resourceType": "MedicationRequest", "id": _opaque(rng), "status": m["status"], "intent": "order",
                     "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/medicationrequest-category", "code": "community"}]}],
                     "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": m["rxnorm"], "display": m["display"]}], "text": m["text"]},
                     "subject": subj, "authoredOn": m["authored"], "requester": {"display": "Dr. Sample, MD"},
                     "dosageInstruction": [{"text": m["dosage"], "route": {"text": "Oral"}}]})
    a = F["allergy"]
    allergy = {"resourceType": "AllergyIntolerance", "id": _opaque(rng),
               "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]},
               "category": ["medication"], "criticality": "low",
               "code": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": a["rxnorm"], "display": a["display"]}], "text": a["text"]},
               "patient": subj, "recordedDate": a["recorded"],
               "reaction": [{"manifestation": [{"text": a["reaction"]}], "severity": a["severity"]}]}
    labs = []
    for i, (loinc, disp, text, val, unit, eff, hi, rtext) in enumerate(F["labs"]):
        coding = [{"system": "http://loinc.org", "code": loinc, "display": disp}]
        if loinc == "4548-4":   # Epic-local code alongside LOINC, a documented quirk
            coding.append({"system": F["epic_local_oid"], "code": "EAP#1230"})
        o = {"resourceType": "Observation", "id": _opaque(rng), "status": "final", "category": cat("laboratory"),
             "code": {"coding": coding, "text": text}, "subject": subj, "effectiveDateTime": eff,
             "valueQuantity": {"value": val, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit}}
        if hi is not None:
            o["referenceRange"] = [{"high": {"value": hi, "unit": unit}, "text": rtext}]
            o["interpretation"] = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                                "code": "H" if val > hi else "N"}]}]
        labs.append(o)
    vitals = []
    for eff, sys_, dia in F["bp"]:
        vitals.append({"resourceType": "Observation", "id": _opaque(rng), "status": "final", "category": cat("vital-signs"),
                       "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9", "display": "Blood pressure panel with all children optional"}], "text": "Blood pressure"},
                       "subject": subj, "effectiveDateTime": eff,
                       "component": [{"code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic blood pressure"}]},
                                      "valueQuantity": {"value": sys_, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}},
                                     {"code": {"coding": [{"system": "http://loinc.org", "code": "8462-4", "display": "Diastolic blood pressure"}]},
                                      "valueQuantity": {"value": dia, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}}]})
    for eff, kg in F["weight"]:
        vitals.append({"resourceType": "Observation", "id": _opaque(rng), "status": "final", "category": cat("vital-signs"),
                       "code": {"coding": [{"system": "http://loinc.org", "code": "29463-7", "display": "Body weight"}]},
                       "subject": subj, "effectiveDateTime": eff,
                       "valueQuantity": {"value": kg, "unit": "kg", "system": "http://unitsofmeasure.org", "code": "kg"}})
    warn = {"fullUrl": "urn:uuid:" + str(uuid.UUID(int=rng.getrandbits(128), version=4)),
            "resource": {"resourceType": "OperationOutcome", "issue": [{"severity": "warning", "code": "informational",
                                                                        "details": {"text": "Some vital sign data was not returned (illustrative warning)"}}]},
            "search": {"mode": "outcome"}}
    return {"Patient.json": bundle("Patient", [patient]), "Condition.json": bundle("Condition", [condition]),
            "MedicationRequest.json": bundle("MedicationRequest", meds), "AllergyIntolerance.json": bundle("AllergyIntolerance", [allergy]),
            "Observation-laboratory.json": bundle("Observation", labs), "Observation-vital-signs.json": bundle("Observation", vitals, [warn])}


def _hl7(iso: str) -> str:
    return iso.replace("-", "").replace(":", "").replace("T", "").replace("Z", "") + ("+0000" if "T" in iso else "")


def epic_ccd() -> str:
    F = EHR_FACTS
    p = F["patient"]
    LOINC, SNOMED, RXN, ICD = "2.16.840.1.113883.6.1", "2.16.840.1.113883.6.96", "2.16.840.1.113883.6.88", "2.16.840.1.113883.6.90"
    def entry_lab(l):
        loinc, disp, text, val, unit, eff, hi, rtext = l
        rr = (f'<referenceRange><observationRange><text>{xml_escape(rtext)}</text><value xsi:type="IVL_PQ"><high value="{hi}" unit="{unit}"/></value>'
              f'</observationRange></referenceRange>') if hi is not None else ""
        return (f'<entry><organizer classCode="BATTERY"><component><observation classCode="OBS" moodCode="EVN">'
                f'<code code="{loinc}" codeSystem="{LOINC}" displayName="{disp}"/><text>{text}</text>'
                f'<effectiveTime value="{_hl7(eff)}"/><value xsi:type="PQ" value="{val}" unit="{unit}"/>{rr}'
                f'</observation></component></organizer></entry>')
    labs = "".join(entry_lab(l) for l in F["labs"])
    meds = "".join(
        f'<entry><substanceAdministration classCode="SBADM" moodCode="INT"><statusCode code="{"active" if m["status"] == "active" else "completed"}"/>'
        f'<effectiveTime xsi:type="IVL_TS"><low value="{_hl7(m["authored"][:10])}"/></effectiveTime>'
        f'<text>{m["dosage"]}</text><consumable><manufacturedProduct><manufacturedMaterial>'
        f'<code code="{m["rxnorm"]}" codeSystem="{RXN}" displayName="{m["display"]}"><originalText>{m["text"]}</originalText></code>'
        f'</manufacturedMaterial></manufacturedProduct></consumable></substanceAdministration></entry>' for m in F["meds"])
    c, a = F["condition"], F["allergy"]
    problems = (f'<entry><act classCode="ACT" moodCode="EVN"><statusCode code="active"/><entryRelationship typeCode="SUBJ">'
                f'<observation classCode="OBS" moodCode="EVN"><statusCode code="completed"/>'
                f'<effectiveTime><low value="{_hl7(c["onset"])}"/></effectiveTime>'
                f'<value xsi:type="CD" code="{c["snomed"]}" codeSystem="{SNOMED}" displayName="{c["display"]}">'
                f'<translation code="{c["icd10"]}" codeSystem="{ICD}"/></value><text>{c["text"]}</text></observation>'
                f'</entryRelationship></act></entry>')
    allergies = (f'<entry><act classCode="ACT" moodCode="EVN"><statusCode code="active"/><entryRelationship typeCode="SUBJ">'
                 f'<observation classCode="OBS" moodCode="EVN"><effectiveTime><low value="{_hl7(a["recorded"])}"/></effectiveTime>'
                 f'<participant typeCode="CSM"><participantRole classCode="MANU"><playingEntity classCode="MMAT">'
                 f'<code code="{a["rxnorm"]}" codeSystem="{RXN}" displayName="{a["display"]}"><originalText>{a["text"]}</originalText></code>'
                 f'</playingEntity></participantRole></participant>'
                 f'<entryRelationship typeCode="MFST"><observation classCode="OBS" moodCode="EVN">'
                 f'<value xsi:type="CD" displayName="{a["reaction"]}"/><text>{a["reaction"]}</text></observation></entryRelationship>'
                 f'</observation></entryRelationship></act></entry>')
    sec = lambda code, title, body: (f'<component><section><code code="{code}" codeSystem="{LOINC}"/><title>{title}</title>{body}</section></component>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<ClinicalDocument xmlns="urn:hl7-org:v3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
            f'<recordTarget><patientRole><id root="1.2.840.114350.1.13.999.2.7.5.737384.14" extension="{p["mrn"]}"/>'
            f'<addr><streetAddressLine>{p["address"]["line"]}</streetAddressLine><city>{p["address"]["city"]}</city>'
            f'<state>{p["address"]["state"]}</state><postalCode>{p["address"]["postal"]}</postalCode></addr>'
            f'<telecom value="tel:{p["phone"]}"/><telecom value="mailto:{p["email"]}"/>'
            f'<patient><name><given>{p["given"][0]}</given><given>{p["given"][1]}</given><family>{p["family"]}</family></name>'
            f'<administrativeGenderCode code="UN"/><birthTime value="{p["birth"].replace("-", "")}"/></patient></patientRole></recordTarget>\n'
            '<component><structuredBody>'
            + sec("48765-2", "Allergies", allergies) + sec("10160-0", "Medications", meds)
            + sec("11450-4", "Problems", problems) + sec("30954-2", "Results", labs)
            + '</structuredBody></component></ClinicalDocument>\n')


# ------------------------------------------------------------------------- memory seed

def memory_seed() -> list[dict[str, Any]]:
    mk = lambda **kw: {"status": "active", "supersedes": None, "superseded_by": None, "pinned": False,
                       "sensitivity": "normal", "evidence_count": 1, "extra": {}, "origin": "user", **kw}
    return [
        mk(id="r01", type="plan", content="Run the SF half in April 2027.", created_at="2025-11-02T10:00:00",
           updated_at="2025-11-02T10:00:00", source="user_stated", confidence=0.9, topic="plan:goal-race"),
        mk(id="r02", type="plan", content="Taking it easy until my BP is sorted, no racing this year.",
           created_at="2026-02-03T21:40:00", updated_at="2026-02-03T21:40:00", source="user_stated", confidence=0.8,
           topic="plan:goal-race-change", extra={"provisional_for": "r01", "volatile_because": "low recovery that day"}),
        mk(id="r03", type="open_loop",
           content=("Confirm: you changed 'Run the SF half in April 2027' to 'Taking it easy until my BP is sorted, "
                    "no racing this year' on 2026-02-03. Unanswered."),
           created_at="2026-02-03T21:41:00", updated_at="2026-02-03T21:41:00", source="inferred", confidence=0.8,
           topic="open:confirm:goal-race"),
        mk(id="r04", type="preference", content="Short answers.", created_at="2025-11-02T10:05:00",
           updated_at="2025-11-02T10:05:00", source="user_stated", confidence=0.9, topic="pref:style"),
        mk(id="r05", type="preference", content="No reflexive see-a-doctor advice.", created_at="2025-11-02T10:06:00",
           updated_at="2025-11-02T10:06:00", source="user_stated", confidence=0.9, topic="pref:no-reflexive-referral"),
    ]


# ------------------------------------------------------------------------- entry point

def build_truth(end: date) -> tuple[dict[date, dict], dict[date, dict]]:
    nights, days = {}, {}
    for d in daterange(GARMIN_START, end):
        t = night_truth(d)
        t["_w"], t["_g"] = device_values(t)
        nights[d], days[d] = t, day_truth(d)
    return nights, days


def generate(out: str | Path = "mock_exports", end: str | date | None = None) -> Truth:
    out = Path(out)
    end_d = date.fromisoformat(end) if isinstance(end, str) else (end or END)
    nights, days = build_truth(end_d + timedelta(days=1))   # one extra night feeds the in-progress WHOOP cycle
    truth = Truth()
    counts: dict[str, int] = {}
    counts.update(write_whoop(out / "whoop", {d: n for d, n in nights.items()}, days, end_d))
    counts.update(write_whoop_api(out / "whoop_api", nights, days, end_d))
    counts.update(write_garmin(out / "garmin", nights, days, end_d))
    epic = out / "epic"
    epic.mkdir(parents=True, exist_ok=True)
    for name, b in epic_bundles().items():
        (epic / name).write_text(json.dumps(b, indent=1))
    ccd_dir = epic / "ccda" / "MachineReadable_XDMFormat" / "IHE_XDM" / "Jordan1"
    ccd_dir.mkdir(parents=True, exist_ok=True)
    (ccd_dir / "DOC0001.XML").write_text(epic_ccd())
    (ccd_dir.parent / "METADATA.XML").write_text('<?xml version="1.0"?><XDSMetadata/>\n')
    (out / "memory_seed.json").write_text(json.dumps(memory_seed(), indent=2))
    for d in daterange(GARMIN_START, end_d):
        t = nights[d]
        truth.nights[d.isoformat()] = {"whoop_asleep": t["_w"]["asleep"] if d >= WHOOP_START and d not in WHOOP_GAP else None,
                                       "garmin_asleep": t["_g"]["asleep"] if d not in GARMIN_GAP else None,
                                       "whoop_rhr": t["_w"]["rhr"], "garmin_rhr": t["_g"]["rhr"], "recovery": t["recovery"],
                                       "whoop_hrv": t["_w"]["hrv"], "garmin_hrv": t["_g"]["hrv"], "off_min": t["off_min"],
                                       "zone": t["zone"], "onset_utc": t["onset_utc"].isoformat(), "wake_utc": t["wake_utc"].isoformat()}
    counts["garmin_gap_nights"] = len(GARMIN_GAP)
    truth.counts = counts
    return truth


if __name__ == "__main__":
    import sys
    generate(sys.argv[1] if len(sys.argv) > 1 else "mock_exports")
