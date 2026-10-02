"""Baselines per signal and per source: a rolling 30-day median, an anchored median over a fixed early
window, divergence between them, and regime breaks that exclude days from both.

Never compares a value from one source to another source's baseline: each source is judged against
its own history. HRV in particular is not one signal (WHOOP RMSSD in the last slow-wave stretch,
Garmin overnight average), so only direction relative to the source's own baseline is comparable.
"""
from __future__ import annotations

import json
import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Any

FIELDS = {"sleep_duration": "asleep_min", "resting_hr": "resting_hr_bpm", "hrv": "hrv_ms"}
SLEEP_DROP = 0.10        # rolling below anchored by more than 10%
RHR_RISE = 5.0           # rolling above anchored by more than 5 bpm
DIVERGENCE_DAYS = 14
HRV_BAND = 0.05          # within +/-5% of its own median counts as "within"


# ------------------------------------------------------------------ regimes

def regimes_path(data_dir: Path) -> Path:
    return Path(data_dir) / "regimes.json"


def load_regimes(data_dir: Path) -> list[dict[str, Any]]:
    p = regimes_path(data_dir)
    return json.loads(p.read_text()) if p.exists() else []


def add_regime(data_dir: Path, start: str, end: str, kind: str, label: str, origin: str = "user") -> bool:
    """Mark a date range (inclusive) as its own regime: travel, illness, a clinical change. Idempotent."""
    date.fromisoformat(start), date.fromisoformat(end)
    regs = load_regimes(data_dir)
    if any(r["start"] == start and r["end"] == end and r["kind"] == kind for r in regs):
        return False
    regs.append({"start": start, "end": end, "kind": kind, "label": label, "origin": origin})
    p = regimes_path(data_dir)
    p.write_text(json.dumps(regs, indent=1))
    p.chmod(0o600)
    return True


def regime_days(regimes: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for r in regimes:
        d, end = date.fromisoformat(r["start"]), date.fromisoformat(r["end"])
        while d <= end:
            out.add(d.isoformat())
            d += timedelta(days=1)
    return out


# ------------------------------------------------------------------ baselines

class Baselines:
    def __init__(self, nights: list[dict[str, Any]], regimes: list[dict[str, Any]], cfg: dict[str, Any]):
        self.cfg = cfg
        self.excluded = regime_days(regimes)
        self.nights = nights
        self._cache: dict[tuple[str, str], dict[str, float]] = {}

    def points(self, source: str, field: str) -> dict[str, float]:
        """date -> value for one source and field, regime days removed."""
        key = (source, field)
        if key not in self._cache:
            self._cache[key] = {r["date"]: r[field] for r in self.nights
                                if r["source"] == source and r.get(field) is not None and r["date"] not in self.excluded}
        return self._cache[key]

    def first_date(self, source: str, field: str) -> str | None:
        """First date of data from this source, regime days included (the anchor window is calendar-based)."""
        dates = [r["date"] for r in self.nights if r["source"] == source and r.get(field) is not None]
        return min(dates) if dates else None

    def rolling(self, source: str, field: str, as_of: str, days: int | None = None, include_as_of: bool = True,
                min_n: int = 10) -> float | None:
        days = days or self.cfg.get("rolling_days", 30)
        end = date.fromisoformat(as_of)
        lo = (end - timedelta(days=days - 1 if include_as_of else days)).isoformat()
        hi = as_of if include_as_of else (end - timedelta(days=1)).isoformat()
        vals = [v for d, v in self.points(source, field).items() if lo <= d <= hi]
        return statistics.median(vals) if len(vals) >= min_n else None

    def anchor_window(self, source: str, field: str) -> tuple[str, str] | None:
        override = (self.cfg.get("anchor") or {}).get(source)
        if override:
            return override["start"], override["end"]
        first = self.first_date(source, field)
        if first is None:
            return None
        return first, (date.fromisoformat(first) + timedelta(days=self.cfg.get("anchor_days", 90) - 1)).isoformat()

    def anchored(self, source: str, field: str, min_n: int = 20) -> float | None:
        win = self.anchor_window(source, field)
        if win is None:
            return None
        vals = [v for d, v in self.points(source, field).items() if win[0] <= d <= win[1]]
        return statistics.median(vals) if len(vals) >= min_n else None

    def diverged(self, signal: str, rolling: float | None, anchored: float | None) -> bool:
        if rolling is None or anchored is None:
            return False
        if signal == "sleep_duration":
            return rolling < anchored * (1 - SLEEP_DROP)
        if signal == "resting_hr":
            return rolling > anchored + RHR_RISE
        return False

    def divergence(self, source: str, signal: str, as_of: str, lookback: int = 120) -> dict[str, Any]:
        """Is the rolling median past the threshold versus the anchored one, and for how many consecutive
        days up to `as_of`? Active at 14 or more."""
        field = FIELDS[signal]
        anchored = self.anchored(source, field)
        rolling_now = self.rolling(source, field, as_of)
        run, d = 0, date.fromisoformat(as_of)
        for _ in range(lookback):
            if self.diverged(signal, self.rolling(source, field, d.isoformat()), anchored):
                run += 1
                d -= timedelta(days=1)
            else:
                break
        delta = None if rolling_now is None or anchored is None else rolling_now - anchored
        pct = None if delta is None or not anchored else 100.0 * delta / anchored
        return {"source": source, "signal": signal, "rolling": rolling_now, "anchored": anchored, "delta": delta,
                "delta_pct": pct, "run_days": run, "active": run >= DIVERGENCE_DAYS}

    def direction(self, source: str, field: str, day: str) -> str | None:
        """Where one night sits against that source's own 30 days before it: above, below or within."""
        v = self.points(source, field).get(day)
        base = self.rolling(source, field, day, include_as_of=False)
        if v is None or base is None or base == 0:
            return None
        if v > base * (1 + HRV_BAND):
            return "above"
        if v < base * (1 - HRV_BAND):
            return "below"
        return "within"
