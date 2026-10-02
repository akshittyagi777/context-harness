"""Garmin Connect "Export Your Data": a zip or directory with DI_CONNECT/DI-Connect-* folders.

Built against open-source parsers of real exports (see health_formats.md section 2.1); Garmin
publishes no schema, so every parsing site marked `format: unconfirmed` needs checking against a real
export. Handled quirks: files chunked by date range (de-duplicated by calendarDate), sleep timestamps
that are naive UTC ending in ".0", activity start in epoch ms with distance in cm and duration in ms,
`calendarDate` as epoch ms in some metrics files, HRV in healthStatusData rather than sleepData.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

from ..canonical import (DailyRow, HRV_METHOD, NightRow, WearableStore, Workout, append_ledger, date_range, fmt_offset,
                         local_date, utc_to_local_iso)
from ..config import load_config
from .base import ImportResult, SourceInfo, Tree

INFO = SourceInfo("garmin", "wearable", ["Garmin Connect export zip", "directory containing DI_CONNECT"],
                  "Garmin Connect account export: sleep, health status (HRV), daily summaries, activities")


def cal_date(v: Any) -> str | None:
    """calendarDate as 'YYYY-MM-DD' or as epoch milliseconds (some metrics files). None if unreadable."""
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000, timezone.utc).date().isoformat()      # format: unconfirmed
    if isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}", v):
        return v[:10]
    return None


def _minutes(sec: Any) -> int | None:
    return None if sec is None else int(round(float(sec) / 60))


class GarminExportImporter:
    info = INFO

    def can_import(self, path: str | Path) -> bool:
        try:
            return any("di_connect" in n.lower() for n in Tree(Path(path)).names())
        except FileNotFoundError:
            return False

    def _files(self, tree: Tree, pattern: str) -> list[str]:
        rx = re.compile(pattern, re.I)
        return [n for n in tree.names() if "di_connect" in n.lower() or "di-connect" in n.lower()
                if rx.search(PurePosixPath(n).name)]

    def _json_rows(self, tree: Tree, name: str, res: ImportResult) -> list[dict[str, Any]]:
        try:
            data = json.loads(tree.read_text(name))
        except json.JSONDecodeError as exc:
            res.anomalies.append(f"{name}: invalid JSON ({exc.msg}); file skipped")
            return []
        return data if isinstance(data, list) else [data]

    def import_path(self, path: str | Path, data_dir: Path, **kw: Any) -> ImportResult:
        tree = Tree(Path(path))
        cfg = load_config(Path(data_dir))
        res = ImportResult("garmin")
        store = WearableStore.load(data_dir)
        home = ZoneInfo(cfg["home_tz"])
        nights: dict[str, NightRow] = {}
        offsets: dict[str, int] = {}
        assumed_tz = False

        # ---- sleep (one night per entry; chunks overlap, so de-duplicate by calendarDate)
        for fn in self._files(tree, r"sleepData\.json$"):
            res.files.append(fn)
            for i, e in enumerate(self._json_rows(tree, fn, res)):
                if not e.get("sleepEndTimestampGMT"):
                    continue
                end_gmt, start_gmt = e["sleepEndTimestampGMT"], e.get("sleepStartTimestampGMT")
                end_loc = e.get("sleepEndTimestampLocal")                                    # format: unconfirmed
                if end_loc:
                    off = int(round((datetime.fromisoformat(end_loc[:19]) - datetime.fromisoformat(end_gmt[:19])).total_seconds() / 60))
                else:
                    end_dt = datetime.fromisoformat(end_gmt[:19]).replace(tzinfo=timezone.utc)
                    off = int(end_dt.astimezone(home).utcoffset().total_seconds() // 60)
                    assumed_tz = True
                sleep_end = utc_to_local_iso(end_gmt, off)
                day = local_date(sleep_end)                              # the local date the sleep ended
                cd = cal_date(e.get("calendarDate"))
                if cd and cd != day:
                    res.anomalies.append(f"{PurePosixPath(fn).name}[{i}]: calendarDate {cd} differs from local sleep end {day}; "
                                         f"using {day}")
                if day in nights:
                    res.add("nights", False)                               # an overlapping chunk
                    continue
                if e.get("deepSleepSeconds") is None and e.get("lightSleepSeconds") is None:
                    res.anomalies.append(f"{PurePosixPath(fn).name}[{i}]: night {day} has no stage data (unconfirmed sleep); skipped")
                    continue
                deep, light, rem, awake = (_minutes(e.get(k)) for k in
                                           ("deepSleepSeconds", "lightSleepSeconds", "remSleepSeconds", "awakeSleepSeconds"))
                asleep = (deep or 0) + (light or 0) + (rem or 0)
                inbed = asleep + (awake or 0)
                nights[day] = NightRow(
                    date=day, source="garmin", sleep_start=utc_to_local_iso(start_gmt, off) if start_gmt else None,
                    sleep_end=sleep_end, asleep_min=asleep, deep_min=deep, rem_min=rem, light_min=light, awake_min=awake,
                    efficiency_pct=round(100.0 * asleep / inbed, 1) if inbed else None, hrv_method=HRV_METHOD["garmin"],
                    resp_rate=e.get("averageRespiration"), spo2_pct=(e.get("spo2SleepSummary") or {}).get("averageSPO2"),
                    tz_offset_min=off, source_file=PurePosixPath(fn).name, source_record_id=f"[{i}] {day}")
                offsets[day] = off
        if assumed_tz:
            res.anomalies.append(f"sleepData had no local timestamps; assumed {cfg['home_tz']} (format: unconfirmed)")

        # ---- health status: HRV, resting HR, skin temp live here, not in sleepData
        seen_health: set[str] = set()
        for fn in self._files(tree, r"healthStatusData\.json$"):
            res.files.append(fn)
            for e in self._json_rows(tree, fn, res):
                day = cal_date(e.get("calendarDate"))
                if day is None or day in seen_health:
                    continue
                seen_health.add(day)
                n = nights.get(day)
                if n is None:
                    continue
                for m in e.get("metrics", []):      # format: unconfirmed (type names, units)
                    t, v = m.get("type"), m.get("value")
                    if t == "HRV":
                        n.hrv_ms = v                  # overnight average RMSSD, ms (inferred)
                    elif t == "HR":
                        n.resting_hr_bpm = v
                    elif t == "SKIN_TEMP_C":
                        n.skin_temp_c = v
                    elif t == "SPO2" and n.spo2_pct is None:
                        n.spo2_pct = v
                    elif t == "RESPIRATION" and n.resp_rate is None:
                        n.resp_rate = v

        # ---- daily summaries (UDS)
        daily: dict[str, DailyRow] = {}
        uds_rhr: dict[str, float] = {}
        for fn in self._files(tree, r"^UDSFile_.*\.json$"):          # filename pattern: unconfirmed
            res.files.append(fn)
            for i, e in enumerate(self._json_rows(tree, fn, res)):
                day = cal_date(e.get("calendarDate"))
                if day is None:
                    continue
                if day in daily:
                    res.add("daily", False)
                    continue
                bb = {s.get("bodyBatteryStatType"): s.get("statsValue")                 # list order is not reliable
                      for s in (e.get("bodyBattery") or {}).get("bodyBatteryStatList", [])}
                stress = next((a.get("averageStressLevel") for a in (e.get("allDayStress") or {}).get("aggregatorList", [])
                               if a.get("type") == "TOTAL"), e.get("averageStressLevel"))   # flat key: unconfirmed
                mod, vig = e.get("moderateIntensityMinutes"), e.get("vigorousIntensityMinutes")
                daily[day] = DailyRow(
                    date=day, source="garmin", steps=e.get("totalSteps"),
                    active_kcal=e.get("activeKilocalories"), body_battery_max=bb.get("HIGHEST"),
                    body_battery_min=bb.get("LOWEST"), stress_avg=stress,
                    intensity_minutes=None if mod is None and vig is None else int((mod or 0) + 2 * (vig or 0)),  # Garmin counts vigorous double
                    source_file=PurePosixPath(fn).name, source_record_id=f"[{i}] {day}")
                if e.get("restingHeartRate") is not None:
                    uds_rhr[day] = e["restingHeartRate"]
        for day, n in nights.items():
            if n.resting_hr_bpm is None and day in uds_rhr:
                n.resting_hr_bpm = uds_rhr[day]

        # ---- acute training load: calendarDate here is epoch milliseconds
        for fn in self._files(tree, r"MetricsAcuteTrainingLoad_.*\.json$"):
            res.files.append(fn)
            for e in self._json_rows(tree, fn, res):
                day = cal_date(e.get("calendarDate"))
                if day in daily and daily[day].training_load_acute is None:
                    daily[day].training_load_acute = e.get("dailyTrainingLoadAcute")        # key name: unconfirmed

        # ---- activities: start in epoch ms, distance in cm, duration in ms
        workouts: list[Workout] = []
        for fn in self._files(tree, r"summarizedActivities\.json$"):
            res.files.append(fn)
            for wrapper in self._json_rows(tree, fn, res):
                for a in wrapper.get("summarizedActivitiesExport", [wrapper] if "activityId" in wrapper else []):
                    ms = a.get("startTimeGmt")
                    if ms is None:
                        continue
                    start_utc = datetime.fromtimestamp(ms / 1000, timezone.utc)
                    off = self._offset_for(start_utc, offsets, home)
                    s_iso = utc_to_local_iso(start_utc.strftime("%Y-%m-%dT%H:%M:%S"), off)
                    dur_ms = a.get("duration")
                    end_iso = None if dur_ms is None else utc_to_local_iso(
                        (start_utc + timedelta(milliseconds=dur_ms)).strftime("%Y-%m-%dT%H:%M:%S"), off)
                    dist_cm = a.get("distance")
                    dist_m = None if dist_cm is None else dist_cm / 100.0                       # centimeters
                    if dist_m is not None and dist_m > 1_000_000:
                        res.anomalies.append(f"activity {a.get('activityId')}: distance {dist_m / 1000:.0f} km looks like a unit surprise")
                    workouts.append(Workout(
                        start=s_iso, end=end_iso, source="garmin", sport=str(a.get("activityType", "activity")),
                        avg_hr=a.get("avgHr"), max_hr=a.get("maxHr"), training_effect=a.get("aerobicTrainingEffect"),  # key: unconfirmed
                        distance_m=dist_m, date=local_date(s_iso), source_file=PurePosixPath(fn).name,
                        source_record_id=str(a.get("activityId"))))

        for n in nights.values():
            res.add("nights", store.add_night(n))
        for d in daily.values():
            res.add("daily", store.add_daily(d))
        for w in workouts:
            res.add("workouts", store.add_workout(w))
        self._anomalies(nights, offsets, res)
        store.save(data_dir)
        append_ledger(data_dir, res.ledger_entry(path))
        return res

    @staticmethod
    def _offset_for(start_utc: datetime, offsets: dict[str, int], home: ZoneInfo) -> int:
        """Offset for an activity: that day's sleep offset when we have one, else the home timezone."""
        for back in (0, 1, -1):
            d = (start_utc.date() - timedelta(days=back)).isoformat()
            if d in offsets:
                return offsets[d]
        return int(start_utc.astimezone(home).utcoffset().total_seconds() // 60)

    @staticmethod
    def _anomalies(nights: dict[str, NightRow], offsets: dict[str, int], res: ImportResult) -> None:
        if not nights:
            return
        dates = sorted(nights)
        have = set(dates)
        missing = [d for d in date_range(dates[0], dates[-1]) if d not in have]
        if missing:
            res.anomalies.append(f"{len(missing)} night(s) missing between {dates[0]} and {dates[-1]}: {', '.join(missing[:8])}"
                                 f"{' ...' if len(missing) > 8 else ''} (watch charging or not worn)")
        prev = None
        for d in dates:
            if prev is not None and offsets[d] != offsets[prev] and abs(offsets[d] - offsets[prev]) != 60:
                res.anomalies.append(f"timezone change between {prev} (UTC{fmt_offset(offsets[prev])}) and {d} "
                                     f"(UTC{fmt_offset(offsets[d])}); nights are dated by the local sleep end")
            prev = d
