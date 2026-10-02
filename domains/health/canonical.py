"""Canonical signal names, units and row shapes, plus the plain-JSON wearable store.

Wearable data is time series, not memory records. Every row keeps `source`, `source_file` and
`source_record_id` so a number in a trace can be traced to the file line or JSON object it came from.

Date alignment. A night belongs to the local calendar date on which the sleep ENDED, using the
offset recorded in the data (never the date it started, never the UTC date).
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SIGNALS = {   # signal -> (table, field, unit)
    "sleep_duration": ("nights", "asleep_min", "min"),
    "sleep_stages": ("nights", "deep_min", "min"),
    "resting_hr": ("nights", "resting_hr_bpm", "bpm"),
    "hrv": ("nights", "hrv_ms", "ms"),
    "recovery": ("daily", "recovery_pct", "%"),
    "steps": ("daily", "steps", "steps"),
    "activity": ("workouts", "sport", ""),
    "stress": ("daily", "stress_avg", "0-100"),
    "body_battery": ("daily", "body_battery_max", "0-100"),
    "spo2": ("nights", "spo2_pct", "%"),
}
HRV_METHOD = {"whoop": "rmssd_last_sws", "garmin": "overnight_avg"}
SOURCE_LABEL = {"whoop": "WHOOP", "garmin": "Garmin"}


@dataclass
class NightRow:
    date: str                       # local date the sleep ended, YYYY-MM-DD
    source: str
    sleep_start: str | None = None  # ISO 8601 with offset
    sleep_end: str | None = None
    asleep_min: int | None = None
    deep_min: int | None = None
    rem_min: int | None = None
    light_min: int | None = None
    awake_min: int | None = None
    efficiency_pct: float | None = None
    resting_hr_bpm: float | None = None
    hrv_ms: float | None = None
    hrv_method: str | None = None
    resp_rate: float | None = None
    skin_temp_c: float | None = None          # absolute, as the device reports it
    skin_temp_delta_c: float | None = None    # versus this source's own median, filled by the store
    spo2_pct: float | None = None
    tz_offset_min: int | None = None
    source_file: str = ""
    source_record_id: str = ""


@dataclass
class DailyRow:
    date: str
    source: str
    steps: int | None = None
    active_kcal: float | None = None
    strain: float | None = None               # WHOOP
    body_battery_max: int | None = None       # Garmin
    body_battery_min: int | None = None
    stress_avg: float | None = None           # Garmin
    recovery_pct: float | None = None         # WHOOP
    intensity_minutes: int | None = None
    training_load_acute: float | None = None  # Garmin metrics file
    source_file: str = ""
    source_record_id: str = ""


@dataclass
class Workout:
    start: str                       # ISO with offset
    end: str | None
    source: str
    sport: str
    avg_hr: float | None = None
    max_hr: float | None = None
    strain: float | None = None      # WHOOP activity strain
    training_effect: float | None = None
    distance_m: float | None = None
    date: str = ""                   # local date of the start
    source_file: str = ""
    source_record_id: str = ""


# ------------------------------------------------------------------ time helpers

_OFFSET = re.compile(r"^(?:UTC|GMT)?\s*([+-])(\d{1,2}):?(\d{2})$")


def parse_offset(text: str | None) -> int | None:
    """'UTC-08:00', '-08:00', '-0800' or 'Z' to minutes east of UTC; None when unparseable."""
    if text is None:
        return None
    t = text.strip()
    if t in ("Z", "UTC", "GMT"):
        return 0
    m = _OFFSET.match(t)
    if not m:
        return None
    mins = int(m.group(2)) * 60 + int(m.group(3))
    return -mins if m.group(1) == "-" else mins


def fmt_offset(mins: int) -> str:
    sign = "-" if mins < 0 else "+"
    return f"{sign}{abs(mins) // 60:02d}:{abs(mins) % 60:02d}"


def naive_local_to_iso(naive: str, offset_min: int) -> str:
    """'2026-03-12 07:02:40' (local wall clock) + offset -> '2026-03-12T07:02:40-07:00'."""
    return naive.strip().replace(" ", "T")[:19] + fmt_offset(offset_min)


def utc_to_local_iso(utc_naive: str, offset_min: int) -> str:
    """'2026-03-12T14:02:00.0' (naive UTC) + offset -> local ISO with offset. The date can change."""
    dt = datetime.fromisoformat(utc_naive.strip().replace("Z", "")[:19]) + timedelta(minutes=offset_min)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + fmt_offset(offset_min)


def local_date(iso_with_offset: str) -> str:
    """The calendar date in the local part of an ISO string (the part before the offset)."""
    return iso_with_offset[:10]


def utc_instant(iso_with_offset: str) -> datetime:
    return datetime.fromisoformat(iso_with_offset).astimezone(timezone.utc)


def ms_to_min(ms: float | None) -> int | None:
    return None if ms is None else int(round(ms / 60000))


def hm(minutes: float | None) -> str:
    """Minutes as '4h40'. Digits in this string are the digits a reply may quote."""
    if minutes is None:
        return "n/a"
    m = int(round(minutes))
    return f"{m // 60}h{m % 60:02d}"


# ------------------------------------------------------------------ store

class WearableStore:
    """Plain JSON under the data directory. Idempotent: a (source, date) night or day exists once."""

    FILE = "wearables.json"

    def __init__(self, nights=None, daily=None, workouts=None):
        self.nights: dict[tuple[str, str], NightRow] = {(r.source, r.date): r for r in (nights or [])}
        self.daily: dict[tuple[str, str], DailyRow] = {(r.source, r.date): r for r in (daily or [])}
        self.workouts: dict[tuple[str, str], Workout] = {(r.source, r.start): r for r in (workouts or [])}

    # io
    @classmethod
    def load(cls, data_dir: Path) -> "WearableStore":
        p = Path(data_dir) / cls.FILE
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        known = lambda dc: {f.name for f in fields(dc)}
        mk = lambda dc, d: dc(**{k: v for k, v in d.items() if k in known(dc)})
        return cls([mk(NightRow, d) for d in raw.get("nights", [])],
                   [mk(DailyRow, d) for d in raw.get("daily", [])],
                   [mk(Workout, d) for d in raw.get("workouts", [])])

    def save(self, data_dir: Path) -> None:
        self.recompute_skin_delta()
        raw = {"nights": [asdict(r) for _, r in sorted(self.nights.items(), key=lambda kv: (kv[0][1], kv[0][0]))],
               "daily": [asdict(r) for _, r in sorted(self.daily.items(), key=lambda kv: (kv[0][1], kv[0][0]))],
               "workouts": [asdict(r) for _, r in sorted(self.workouts.items(), key=lambda kv: (kv[0][1], kv[0][0]))]}
        p = Path(data_dir) / self.FILE
        p.write_text(json.dumps(raw, indent=1))
        p.chmod(0o600)

    # adds return True when the row was new
    def add_night(self, row: NightRow) -> bool:
        k = (row.source, row.date)
        if k in self.nights:
            return False
        self.nights[k] = row
        return True

    def add_daily(self, row: DailyRow) -> bool:
        k = (row.source, row.date)
        if k in self.daily:
            return False
        self.daily[k] = row
        return True

    def add_workout(self, row: Workout) -> bool:
        k = (row.source, row.start)
        if k in self.workouts:
            return False
        self.workouts[k] = row
        return True

    def recompute_skin_delta(self) -> None:
        for src in {s for s, _ in self.nights}:
            rows = [r for (s, _), r in self.nights.items() if s == src]
            vals = [r.skin_temp_c for r in rows if r.skin_temp_c is not None]
            med = statistics.median(vals) if vals else None
            for r in rows:
                r.skin_temp_delta_c = None if r.skin_temp_c is None or med is None else round(r.skin_temp_c - med, 2)

    # views
    def sources(self) -> list[str]:
        return sorted({s for s, _ in self.nights} | {s for s, _ in self.daily})

    def night_rows(self, source: str) -> list[NightRow]:
        return sorted((r for (s, _), r in self.nights.items() if s == source), key=lambda r: r.date)

    def daily_rows(self, source: str) -> list[DailyRow]:
        return sorted((r for (s, _), r in self.daily.items() if s == source), key=lambda r: r.date)

    def series(self) -> dict[str, list[dict[str, Any]]]:
        """Rows as dicts, oldest first: the shape a Scenario carries."""
        return {"nights": [asdict(r) for r in sorted(self.nights.values(), key=lambda r: (r.date, r.source))],
                "daily": [asdict(r) for r in sorted(self.daily.values(), key=lambda r: (r.date, r.source))],
                "workouts": [asdict(r) for r in sorted(self.workouts.values(), key=lambda r: (r.start, r.source))]}


# ------------------------------------------------------------------ import ledger

def ledger_path(data_dir: Path) -> Path:
    return Path(data_dir) / "imports.json"


def load_ledger(data_dir: Path) -> list[dict[str, Any]]:
    p = ledger_path(data_dir)
    return json.loads(p.read_text()) if p.exists() else []


def append_ledger(data_dir: Path, entry: dict[str, Any]) -> None:
    led = load_ledger(data_dir)
    led.append(entry)
    p = ledger_path(data_dir)
    p.write_text(json.dumps(led, indent=1))
    p.chmod(0o600)


def date_range(start: str, end: str) -> list[str]:
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    return [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]
