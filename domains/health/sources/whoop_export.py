"""WHOOP app data export (Data Export > Create Export): a zip or directory of CSVs.

Format (documented only by community parsers, see health_formats.md section 1.1; the whole importer is
built against it and needs checking against a real export): flat files physiological_cycles.csv,
sleeps.csv, workouts.csv, journal_entries.csv; rows newest first; timestamps are naive local time and
`Cycle timezone` ("UTC-08:00") is the offset for every timestamp in that row; the in-progress cycle has
an empty `Cycle end time` and is skipped. Headers are matched case-insensitively with punctuation
stripped, because casing varies between exports and localised exports exist.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path, PurePosixPath
from typing import Any

from ..canonical import (DailyRow, HRV_METHOD, NightRow, WearableStore, Workout, append_ledger, local_date,
                         naive_local_to_iso, parse_offset, date_range)
from .base import ImportResult, SourceInfo, Tree

INFO = SourceInfo("whoop", "wearable", ["WHOOP export zip", "directory of CSVs"],
                  "WHOOP app data export: cycles, sleeps, workouts, journal")


def norm(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", header.lower())


def read_rows(text: str) -> list[tuple[int, dict[str, str]]]:
    """(line number, {normalised header: value}) per data row."""
    reader = csv.reader(io.StringIO(text))
    try:
        header = [norm(h) for h in next(reader)]
    except StopIteration:
        return []
    out = []
    for i, row in enumerate(reader, start=2):
        if not any(c.strip() for c in row):
            continue
        out.append((i, {h: (row[j].strip() if j < len(row) else "") for j, h in enumerate(header)}))
    return out


def num(row: dict[str, str], key: str) -> float | None:
    v = row.get(key, "")
    try:
        return float(v) if v != "" else None
    except ValueError:
        return None


def whole(row: dict[str, str], key: str) -> int | None:
    v = num(row, key)
    return None if v is None else int(round(v))


class WhoopExportImporter:
    info = INFO

    def can_import(self, path: str | Path) -> bool:
        try:
            return any(PurePosixPath(n).name.lower() == "physiological_cycles.csv" for n in Tree(Path(path)).names())
        except FileNotFoundError:
            return False

    def import_path(self, path: str | Path, data_dir: Path, **kw: Any) -> ImportResult:
        tree = Tree(Path(path))
        files = {PurePosixPath(n).name.lower(): n for n in tree.names()}
        res = ImportResult("whoop")
        store = WearableStore.load(data_dir)
        if "physiological_cycles.csv" not in files and "sleeps.csv" not in files:
            res.anomalies.append("no physiological_cycles.csv or sleeps.csv found: nothing imported")
            return res

        nights: list[NightRow] = []
        daily: list[DailyRow] = []
        offsets: dict[str, int] = {}            # local date -> offset, for tz change reporting
        if "physiological_cycles.csv" in files:
            fn = files["physiological_cycles.csv"]
            res.files.append(fn)
            text = tree.read_text(fn)
            for line, r in read_rows(text):
                n, d = self._cycle(r, line, PurePosixPath(fn).name, res)
                if n is not None:
                    nights.append(n)
                    offsets[n.date] = n.tz_offset_min
                if d is not None:
                    daily.append(d)
        else:   # fall back to sleeps.csv for the main sleeps
            fn = files["sleeps.csv"]
            res.files.append(fn)
            for line, r in read_rows(tree.read_text(fn)):
                if r.get("nap", "").lower() == "true":
                    continue
                n, _ = self._cycle(r, line, PurePosixPath(fn).name, res)
                if n is not None:
                    nights.append(n)
                    offsets[n.date] = n.tz_offset_min
        if "sleeps.csv" in files:
            res.files.append(files["sleeps.csv"])
            naps = sum(1 for _, r in read_rows(tree.read_text(files["sleeps.csv"])) if r.get("nap", "").lower() == "true")
            res.add("naps", False, naps)

        for n in nights:
            res.add("nights", store.add_night(n))
        for d in daily:
            res.add("daily", store.add_daily(d))

        if "workouts.csv" in files:
            fn = files["workouts.csv"]
            res.files.append(fn)
            for line, r in read_rows(tree.read_text(fn)):
                w = self._workout(r, line, PurePosixPath(fn).name, res)
                if w is not None:
                    res.add("workouts", store.add_workout(w))
        if "journal_entries.csv" in files:
            res.files.append(files["journal_entries.csv"])
            res.add("journal", False, len(read_rows(tree.read_text(files["journal_entries.csv"]))))   # not used by v1

        self._anomalies(nights, offsets, res)
        store.save(data_dir)
        append_ledger(data_dir, res.ledger_entry(path))
        return res

    # ------------------------------------------------------------------ rows
    def _cycle(self, r: dict[str, str], line: int, fname: str, res: ImportResult):
        start, end = r.get("cyclestarttime", ""), r.get("cycleendtime", "")
        if not end and "physiological" in fname.lower():
            res.anomalies.append(f"{fname}:{line}: in-progress cycle (empty end time) starting {start} skipped")
            return None, None
        off = parse_offset(r.get("cycletimezone"))
        if off is None:
            res.anomalies.append(f"{fname}:{line}: unreadable Cycle timezone {r.get('cycletimezone')!r}; row skipped")
            return None, None
        onset, wake = r.get("sleeponset", ""), r.get("wakeonset", "")
        if not wake:
            res.anomalies.append(f"{fname}:{line}: no wake onset (unscored sleep); row skipped")
            return None, None
        wake_iso = naive_local_to_iso(wake, off)
        day = local_date(wake_iso)                                  # the local date the sleep ended
        skin = num(r, "skintempcelsius")
        if skin is not None and skin > 50:                          # a Fahrenheit surprise
            res.anomalies.append(f"{fname}:{line}: skin temp {skin} looks like Fahrenheit; converted to Celsius")
            skin = round((skin - 32) * 5 / 9, 2)
        asleep = whole(r, "asleepdurationmin")
        inbed = whole(r, "inbeddurationmin")
        night = NightRow(
            date=day, source="whoop", sleep_start=naive_local_to_iso(onset, off) if onset else None, sleep_end=wake_iso,
            asleep_min=asleep, deep_min=whole(r, "deepswsdurationmin"), rem_min=whole(r, "remdurationmin"),
            light_min=whole(r, "lightsleepdurationmin"), awake_min=whole(r, "awakedurationmin"),
            efficiency_pct=num(r, "sleepefficiency") if num(r, "sleepefficiency") is not None else
            (round(100.0 * asleep / inbed, 1) if asleep and inbed else None),
            resting_hr_bpm=num(r, "restingheartratebpm"), hrv_ms=num(r, "heartratevariabilityms"),
            hrv_method=HRV_METHOD["whoop"], resp_rate=num(r, "respiratoryraterpm"), skin_temp_c=skin,
            spo2_pct=num(r, "bloodoxygen"), tz_offset_min=off, source_file=fname, source_record_id=f"line {line}")
        daily = DailyRow(date=day, source="whoop", recovery_pct=num(r, "recoveryscore"), strain=num(r, "daystrain"),
                         source_file=fname, source_record_id=f"line {line}") if "physiological" in fname.lower() else None
        return night, daily

    def _workout(self, r: dict[str, str], line: int, fname: str, res: ImportResult) -> Workout | None:
        off = parse_offset(r.get("cycletimezone"))
        start, end = r.get("workoutstarttime", ""), r.get("workoutendtime", "")
        if off is None or not start:
            res.anomalies.append(f"{fname}:{line}: workout row without a start time or timezone skipped")
            return None
        s_iso = naive_local_to_iso(start, off)
        return Workout(start=s_iso, end=naive_local_to_iso(end, off) if end else None, source="whoop",
                       sport=r.get("activityname", "Activity") or "Activity", avg_hr=num(r, "averagehrbpm"),
                       max_hr=num(r, "maxhrbpm"), strain=num(r, "activitystrain"),
                       distance_m=num(r, "distancemeters"),            # blank on older exports
                       date=local_date(s_iso), source_file=fname, source_record_id=f"line {line}")

    def _anomalies(self, nights: list[NightRow], offsets: dict[str, int], res: ImportResult) -> None:
        if not nights:
            return
        dates = sorted({n.date for n in nights})
        have = set(dates)
        missing = [d for d in date_range(dates[0], dates[-1]) if d not in have]
        if missing:
            res.anomalies.append(f"{len(missing)} night(s) missing between {dates[0]} and {dates[-1]}: "
                                 f"{_runs(missing)} (strap off or not worn)")
        prev = None
        for d in dates:
            if prev is not None and offsets[d] != offsets[prev] and abs(offsets[d] - offsets[prev]) != 60:   # 60 min is DST
                res.anomalies.append(f"timezone change between {prev} (UTC{_fmt(offsets[prev])}) and {d} "
                                     f"(UTC{_fmt(offsets[d])}); each row uses its own Cycle timezone")
            prev = d


def _fmt(m: int) -> str:
    return f"{'-' if m < 0 else '+'}{abs(m) // 60:02d}:{abs(m) % 60:02d}"


def _runs(days: list[str]) -> str:
    """'2026-03-08..2026-03-13' style summary of consecutive dates."""
    from datetime import date, timedelta
    out, i = [], 0
    while i < len(days):
        j = i
        while j + 1 < len(days) and date.fromisoformat(days[j + 1]) - date.fromisoformat(days[j]) == timedelta(days=1):
            j += 1
        out.append(days[i] if i == j else f"{days[i]}..{days[j]}")
        i = j + 1
    return ", ".join(out[:6]) + (" ..." if len(out) > 6 else "")
