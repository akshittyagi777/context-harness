"""Garmin Health API: mapping kept for completeness, NOT a personal route.

The Garmin Connect Developer Program's Health API is for approved business and enterprise use, not
personal use (developer.garmin.com program FAQ). The personal route is the Connect account export
(`garmin_export.py`). The field names below come from open-source parsers of real payloads and the
official spec was not reachable: everything here is `format: unconfirmed`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..canonical import DailyRow, HRV_METHOD, NightRow, fmt_offset, local_date
from .base import ImportResult, NotConfigured, SourceInfo

INFO = SourceInfo("garmin-api", "wearable", ["Garmin Health API JSON"], "Garmin Health API (business use only)")


def _local_iso(epoch_s: int, off_s: int) -> str:
    from datetime import datetime, timedelta, timezone
    dt = datetime.fromtimestamp(epoch_s, timezone.utc).replace(tzinfo=None) + timedelta(seconds=off_s)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + fmt_offset(off_s // 60)


def map_sleep(s: dict[str, Any], hrv: dict[str, Any] | None = None) -> NightRow:
    """One `sleeps` summary (plus its `hrv` summary) to a NightRow. format: unconfirmed."""
    off = s.get("startTimeOffsetInSeconds", 0)
    start = _local_iso(s["startTimeInSeconds"], off)
    end = _local_iso(s["startTimeInSeconds"] + s["durationInSeconds"], off)
    deep, light, rem = (round(s.get(k, 0) / 60) for k in
                        ("deepSleepDurationInSeconds", "lightSleepDurationInSeconds", "remSleepInSeconds"))   # note: remSleepInSeconds
    awake = round(s.get("awakeDurationInSeconds", 0) / 60)
    asleep = deep + light + rem
    return NightRow(date=local_date(end), source="garmin", sleep_start=start, sleep_end=end, asleep_min=asleep,
                    deep_min=deep, rem_min=rem, light_min=light, awake_min=awake,
                    efficiency_pct=round(100.0 * asleep / (asleep + awake), 1) if asleep + awake else None,
                    hrv_ms=(hrv or {}).get("lastNightAvg"), hrv_method=HRV_METHOD["garmin"], resp_rate=s.get("respirationAvg"),
                    tz_offset_min=off // 60, source_file="garmin_health_api", source_record_id=str(s.get("summaryId", "")))


def map_daily(d: dict[str, Any]) -> DailyRow:
    """One `dailies` summary to a DailyRow. format: unconfirmed."""
    return DailyRow(date=d["calendarDate"], source="garmin", steps=d.get("steps"), active_kcal=d.get("activeKilocalories"),
                    body_battery_max=d.get("bodyBatteryHighestValue"), body_battery_min=d.get("bodyBatteryLowestValue"),
                    stress_avg=d.get("averageStressLevel"),
                    intensity_minutes=round((d.get("moderateIntensityDurationInSeconds", 0)
                                             + 2 * d.get("vigorousIntensityDurationInSeconds", 0)) / 60),
                    source_file="garmin_health_api", source_record_id=str(d.get("summaryId", "")))


class GarminSyncer:
    info = INFO

    def requirements(self) -> list[str]:
        return ["the Garmin Health API is for approved business use, not personal use (Connect Developer Program)",
                "for personal data use the Garmin Connect account export instead: `import garmin <export zip or folder>`"]

    def sync(self, since: str | None, data_dir: Path, credentials: dict[str, Any] | None = None) -> ImportResult:
        raise NotConfigured("Garmin sync is not available to individuals. " + " ".join(self.requirements()))
