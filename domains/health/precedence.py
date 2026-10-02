"""Per-signal source precedence and disagreement between sources.

Precedence is the user's call, not learned: the first source in the list that has a value for that
night or day wins. Where two sources overlap, disagreement is computed and reported, never resolved by
averaging. HRV values are never compared across sources; only each source's direction against its own
baseline is.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .baselines import Baselines
from .canonical import SIGNALS, SOURCE_LABEL

DEFAULT_PRECEDENCE: dict[str, list[str]] = {
    "sleep_duration": ["whoop", "garmin"], "sleep_stages": ["whoop", "garmin"],
    "resting_hr": ["whoop", "garmin"], "hrv": ["whoop", "garmin"], "recovery": ["whoop"],
    "steps": ["garmin", "whoop"], "activity": ["garmin", "whoop"], "stress": ["garmin"],
    "body_battery": ["garmin"], "spo2": ["garmin", "whoop"],
}
STAGE_FIELDS = ("deep_min", "rem_min", "light_min", "awake_min")


def load_precedence(data_dir: Path) -> dict[str, list[str]]:
    """Read `precedence.json`, creating it with the defaults on first run."""
    p = Path(data_dir) / "precedence.json"
    if not p.exists():
        p.write_text(json.dumps(DEFAULT_PRECEDENCE, indent=2))
        p.chmod(0o600)
        return {k: list(v) for k, v in DEFAULT_PRECEDENCE.items()}
    user = json.loads(p.read_text())
    return {**{k: list(v) for k, v in DEFAULT_PRECEDENCE.items()}, **user}


@dataclass
class Pick:
    signal: str
    source: str
    value: Any
    row: dict[str, Any]
    preferred: str                  # the source the settings ranked first
    skipped: list[str]              # higher-ranked sources that had no value that day

    @property
    def label(self) -> str:
        return SOURCE_LABEL.get(self.source, self.source)


class WearView:
    """Read access to a scenario's wearable series with precedence applied."""

    def __init__(self, series: dict[str, list[dict[str, Any]]], precedence: dict[str, list[str]]):
        self.prec = precedence
        self.nights = {(r["source"], r["date"]): r for r in series.get("nights", [])}
        self.daily = {(r["source"], r["date"]): r for r in series.get("daily", [])}
        self.workouts: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for w in series.get("workouts", []):
            self.workouts.setdefault((w["source"], w["date"]), []).append(w)

    def pick(self, signal: str, day: str) -> Pick | None:
        table, field, _ = SIGNALS[signal]
        skipped: list[str] = []
        order = self.prec.get(signal, [])
        for src in order:
            if table == "workouts":
                ws = self.workouts.get((src, day))
                row, value = ({"workouts": ws}, ws) if ws else (None, None)
            else:
                row = (self.nights if table == "nights" else self.daily).get((src, day))
                value = None if row is None else row.get(field)
            if row is not None and value is not None:
                return Pick(signal, src, value, row, order[0], skipped)
            skipped.append(src)
        return None

    def night_dates(self) -> list[str]:
        return sorted({d for _, d in self.nights})

    def last_night(self) -> str | None:
        ds = self.night_dates()
        return ds[-1] if ds else None


def night_disagreement(view: WearView, base: Baselines, day: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    srcs = [s for s in view.prec.get("sleep_duration", []) if (s, day) in view.nights][:2]
    if len(srcs) < 2:
        return None
    a, b = (view.nights[(s, day)] for s in srcs)
    out: dict[str, Any] = {"date": day, "sources": srcs}
    if a.get("asleep_min") is not None and b.get("asleep_min") is not None:
        out["sleep_diff_min"] = a["asleep_min"] - b["asleep_min"]
        out["sleep_over"] = abs(out["sleep_diff_min"]) > cfg.get("sleep_disagreement_min", 45)
    if a.get("resting_hr_bpm") is not None and b.get("resting_hr_bpm") is not None:
        out["rhr_diff"] = round(a["resting_hr_bpm"] - b["resting_hr_bpm"], 1)
        out["rhr_over"] = abs(out["rhr_diff"]) > cfg.get("rhr_disagreement_bpm", 5)
    da, db = (base.direction(s, "hrv_ms", day) for s in srcs)
    out["hrv_direction"] = {srcs[0]: da, srcs[1]: db}
    out["hrv_mismatch"] = {da, db} == {"above", "below"}
    return out


def disagreement_summary(view: WearView, base: Baselines, as_of: str, cfg: dict[str, Any],
                         days: int = 14) -> dict[str, Any]:
    end = date.fromisoformat(as_of)
    rows = []
    for i in range(days):
        r = night_disagreement(view, base, (end - timedelta(days=i)).isoformat(), cfg)
        if r:
            rows.append(r)
    diffs = [r["sleep_diff_min"] for r in rows if "sleep_diff_min" in r]
    srcs = view.prec.get("sleep_duration", [])[:2]
    return {
        "days": days, "nights_compared": len(rows), "sources": srcs,
        "mean_sleep_diff_min": round(sum(diffs) / len(diffs), 1) if diffs else None,
        "sleep_over_nights": [r["date"] for r in rows if r.get("sleep_over")],
        "rhr_over_nights": [r["date"] for r in rows if r.get("rhr_over")],
        "hrv_mismatch_nights": [r["date"] for r in rows if r.get("hrv_mismatch")],
    }
