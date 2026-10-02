"""WHOOP developer API v2: syncer skeleton plus the mapper from API JSON to canonical rows.

Network and OAuth are NOT implemented. The mapper is real and tested on mock API JSON.

What a sync would need (developer.whoop.com, researched 2026-10-01):
  * an app in the WHOOP Developer Dashboard (client id and secret; secret server-side only), scopes
    read:recovery read:cycles read:workout read:sleep read:profile read:body_measurement offline;
  * OAuth 2.0 authorization code flow: https://api.prod.whoop.com/oauth/oauth2/auth and /token,
    `state` of 8+ characters; the refresh token rotates on every refresh, so persist the newest pair;
  * base https://api.prod.whoop.com/developer/v2/ : /cycle, /recovery, /activity/sleep, /activity/workout;
    page with `limit` (max 25) and `nextToken`, newest first; 100 requests/min, 10,000/day.

API versus CSV: UTC `Z` timestamps plus `timezone_offset` (the CSV is local plus offset); durations in
milliseconds (CSV minutes); energy in kilojoules; `score` exists only when `score_state == "SCORED"`;
HRV is a float `hrv_rmssd_milli`.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..canonical import (DailyRow, HRV_METHOD, NightRow, WearableStore, Workout, append_ledger, fmt_offset, local_date,
                         ms_to_min, parse_offset)
from .base import ImportResult, NotConfigured, SourceInfo

INFO = SourceInfo("whoop-api", "wearable", ["WHOOP developer API v2 JSON"], "WHOOP API sync (not implemented)")


def _local_iso(z: str, off_min: int) -> str:
    dt = datetime.fromisoformat(z.replace("Z", "+00:00")).replace(tzinfo=None) + timedelta(minutes=off_min)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + fmt_offset(off_min)


def map_records(cycles: list[dict], recoveries: list[dict], sleeps: list[dict], workouts: list[dict],
                result: ImportResult | None = None) -> tuple[list[NightRow], list[DailyRow], list[Workout]]:
    """Canonical rows from API records. Unscored records are skipped and reported."""
    res = result or ImportResult("whoop-api")
    recov_by_sleep = {r["sleep_id"]: r for r in recoveries}
    cycle_by_id = {c["id"]: c for c in cycles}
    nights: list[NightRow] = []
    daily: dict[str, DailyRow] = {}
    for s in sleeps:
        if s.get("nap"):
            res.add("naps", False)
            continue
        if s.get("score_state") != "SCORED" or "score" not in s:
            res.anomalies.append(f"sleep {s['id']}: score_state {s.get('score_state')}; skipped until scored")
            continue
        off = parse_offset(s.get("timezone_offset")) or 0
        end_iso = _local_iso(s["end"], off)
        day = local_date(end_iso)                                     # the local date the sleep ended
        st = s["score"]["stage_summary"]
        in_bed, awake = st["total_in_bed_time_milli"], st["total_awake_time_milli"]
        asleep = ms_to_min(in_bed - awake - st.get("total_no_data_time_milli", 0))
        rec = recov_by_sleep.get(s["id"])
        sc = rec["score"] if rec and rec.get("score_state") == "SCORED" and "score" in rec else {}
        nights.append(NightRow(
            date=day, source="whoop", sleep_start=_local_iso(s["start"], off), sleep_end=end_iso, asleep_min=asleep,
            deep_min=ms_to_min(st["total_slow_wave_sleep_time_milli"]), rem_min=ms_to_min(st["total_rem_sleep_time_milli"]),
            light_min=ms_to_min(st["total_light_sleep_time_milli"]), awake_min=ms_to_min(awake),
            efficiency_pct=round(s["score"].get("sleep_efficiency_percentage", 0.0), 1) or None,
            resting_hr_bpm=sc.get("resting_heart_rate"), hrv_ms=sc.get("hrv_rmssd_milli"), hrv_method=HRV_METHOD["whoop"],
            resp_rate=s["score"].get("respiratory_rate"), skin_temp_c=sc.get("skin_temp_celsius"),
            spo2_pct=sc.get("spo2_percentage"), tz_offset_min=off, source_file="whoop_api", source_record_id=str(s["id"])))
        c = cycle_by_id.get(s.get("cycle_id"))
        row = DailyRow(date=day, source="whoop", recovery_pct=sc.get("recovery_score"), source_file="whoop_api",
                       source_record_id=f"cycle {s.get('cycle_id')}")
        if c and c.get("score_state") == "SCORED" and "score" in c:
            row.strain = c["score"].get("strain")
            row.steps = c.get("step_count")           # null when the strap was not worn
        daily[day] = row
    wos: list[Workout] = []
    for w in workouts:
        if w.get("score_state") != "SCORED" or "score" not in w:
            res.anomalies.append(f"workout {w['id']}: score_state {w.get('score_state')}; skipped")
            continue
        off = parse_offset(w.get("timezone_offset")) or 0
        s_iso = _local_iso(w["start"], off)
        sc = w["score"]
        wos.append(Workout(start=s_iso, end=_local_iso(w["end"], off), source="whoop", sport=w.get("sport_name", "activity"),
                           avg_hr=sc.get("average_heart_rate"), max_hr=sc.get("max_heart_rate"), strain=sc.get("strain"),
                           distance_m=sc.get("distance_meter"), date=local_date(s_iso), source_file="whoop_api",
                           source_record_id=str(w["id"])))
    return nights, list(daily.values()), wos


class WhoopSyncer:
    info = INFO

    def requirements(self) -> list[str]:
        return ["a WHOOP developer app (developer-dashboard.whoop.com) with a client id and secret",
                "OAuth 2.0 authorization code flow with scopes read:recovery read:cycles read:workout read:sleep offline",
                "somewhere to persist the rotating refresh token"]

    def sync(self, since: str | None, data_dir: Path, credentials: dict[str, Any] | None = None) -> ImportResult:
        raise NotConfigured("WHOOP sync is not implemented. It would need: " + "; ".join(self.requirements()))

    def import_json(self, pages: dict[str, list[dict]], data_dir: Path) -> ImportResult:
        """Write API records (already fetched) into the store: the part of a sync that is implemented."""
        res = ImportResult("whoop-api")
        nights, daily, wos = map_records(pages.get("cycles", []), pages.get("recovery", []), pages.get("sleep", []),
                                         pages.get("workouts", []), res)
        store = WearableStore.load(data_dir)
        for n in nights:
            res.add("nights", store.add_night(n))
        for d in daily:
            res.add("daily", store.add_daily(d))
        for w in wos:
            res.add("workouts", store.add_workout(w))
        store.save(data_dir)
        append_ledger(data_dir, res.ledger_entry("whoop_api"))
        return res
