"""Importers and syncers, parsed against the mock exports with exact row counts."""
from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from tests import support
from tests.support import PASSPHRASE, empty_data, fixture

import health_cli as cli
from domains.health.canonical import (WearableStore, hm, local_date, ms_to_min, naive_local_to_iso, parse_offset,
                              utc_instant, utc_to_local_iso)
from domains.health.sources import IMPORTERS, SYNCERS, NotConfigured
from domains.health.sources import garmin_api, whoop_api
from domains.health.sources.garmin_export import cal_date
from domains.health.sources.whoop_export import norm, read_rows
from domains.health.vault import Vault, VaultLocked


def import_into(name: str, path: Path, dd: Path | None = None):
    dd = dd or empty_data()
    res = IMPORTERS[name].import_path(path, dd)
    return res, WearableStore.load(dd), dd


class TimeHelperTests(unittest.TestCase):
    def test_parse_offset(self):
        self.assertEqual(parse_offset("UTC-08:00"), -480)
        self.assertEqual(parse_offset("-0700"), -420)
        self.assertEqual(parse_offset("Z"), 0)
        self.assertEqual(parse_offset("+05:30"), 330)
        self.assertIsNone(parse_offset("PST"))

    def test_naive_local_gets_the_row_offset(self):
        self.assertEqual(naive_local_to_iso("2026-03-12 07:02:40", -420), "2026-03-12T07:02:40-07:00")

    def test_utc_to_local_can_change_the_date(self):
        # 02:30 UTC on May 11 is 19:30 on May 10 in Pacific time: the date is the LOCAL one
        self.assertEqual(utc_to_local_iso("2026-05-11T02:30:00.0", -420), "2026-05-10T19:30:00-07:00")
        self.assertEqual(local_date(utc_to_local_iso("2026-05-11T02:30:00.0", -420)), "2026-05-10")

    def test_hm_and_ms(self):
        self.assertEqual(hm(280), "4h40")
        self.assertEqual(ms_to_min(16800000), 280)


class WhoopExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res, cls.store, cls.dd = import_into("whoop", fixture()["mock"] / "whoop")
        cls.counts = fixture()["truth"].counts

    def test_exact_row_counts(self):
        self.assertEqual(self.res.added["nights"], 359)
        self.assertEqual(self.res.added["daily"], 359)
        self.assertEqual(self.res.added["workouts"], self.counts["whoop_workouts"])
        self.assertEqual(len(self.store.night_rows("whoop")), 359)

    def test_naps_and_journal_are_not_nights(self):
        self.assertEqual(self.res.skipped["naps"], self.counts["whoop_naps"])
        self.assertEqual(self.res.skipped["journal"], self.counts["whoop_journal"])

    def test_in_progress_cycle_is_skipped_and_reported(self):
        self.assertTrue(any("in-progress cycle" in a for a in self.res.anomalies))
        self.assertNotIn("2026-10-01", {r.date for r in self.store.night_rows("whoop")})

    def test_strap_off_gap_reported(self):
        gap = [a for a in self.res.anomalies if "missing" in a]
        self.assertEqual(len(gap), 1)
        self.assertIn("6 night(s)", gap[0])
        self.assertIn("2026-03-08..2026-03-13", gap[0])
        dates = {r.date for r in self.store.night_rows("whoop")}
        self.assertTrue(all(f"2026-03-{d:02d}" not in dates for d in range(8, 14)))

    def test_timezone_changes_reported_but_dst_is_not(self):
        tz = [a for a in self.res.anomalies if "timezone change" in a]
        self.assertEqual(len(tz), 2)
        self.assertIn("UTC-04:00", tz[0])
        self.assertFalse(any("2025-11" in a or "2026-03-07" in a for a in tz))

    def test_row_values_and_units(self):
        row = next(r for r in self.store.night_rows("whoop") if r.date == "2026-09-29")
        self.assertEqual(row.asleep_min, 280)
        self.assertEqual(hm(row.asleep_min), "4h40")
        self.assertEqual(row.deep_min + row.rem_min + row.light_min, row.asleep_min)
        self.assertEqual(row.hrv_method, "rmssd_last_sws")
        self.assertTrue(row.source_file.endswith("physiological_cycles.csv"))
        self.assertRegex(row.source_record_id, r"^line \d+$")
        daily = self.store.daily["whoop", "2026-09-29"]
        self.assertEqual(daily.recovery_pct, 21)

    def test_red_eye_night_is_dated_by_local_sleep_end(self):
        row = next(r for r in self.store.night_rows("whoop") if r.date == "2026-05-10")
        self.assertEqual(row.sleep_start[:11], "2026-05-10T")                # 23:50 in Pacific is 02:50 on the Eastern clock
        self.assertIn(row.sleep_start[11:13], ("02", "03"))
        self.assertLess(abs((utc_instant(row.sleep_start) - datetime(2026, 5, 10, 6, 50, tzinfo=timezone.utc)).total_seconds()), 300)
        self.assertTrue(row.sleep_start.endswith("-04:00") and row.sleep_end.endswith("-04:00"))
        self.assertEqual(local_date(row.sleep_end), "2026-05-10")
        self.assertEqual(row.tz_offset_min, -240)

    def test_night_before_the_flight_is_still_pacific(self):
        row = next(r for r in self.store.night_rows("whoop") if r.date == "2026-05-09")
        self.assertTrue(row.sleep_end.endswith("-07:00"))

    def test_every_night_date_equals_the_local_date_of_its_end(self):
        for r in self.store.night_rows("whoop"):
            self.assertEqual(r.date, local_date(r.sleep_end))

    def test_idempotent_reimport_adds_nothing(self):
        res = IMPORTERS["whoop"].import_path(fixture()["mock"] / "whoop", self.dd)
        self.assertEqual(res.total_added, 0)
        self.assertEqual(res.skipped["nights"], 359)
        self.assertEqual(len(WearableStore.load(self.dd).night_rows("whoop")), 359)

    def test_newer_export_appends_only_new_dates(self):
        from domains.health.mock.generate import generate
        out = support._tmp() / "older"
        generate(out, end="2026-08-31")
        dd = empty_data()
        old = IMPORTERS["whoop"].import_path(out / "whoop", dd)
        new = IMPORTERS["whoop"].import_path(fixture()["mock"] / "whoop", dd)
        self.assertEqual(old.added["nights"] + new.added["nights"], 359)
        self.assertEqual(new.skipped["nights"], old.added["nights"])
        self.assertEqual(new.added["nights"], 30)

    def test_zip_import_equals_directory_import(self):
        buf = support._tmp() / "whoop.zip"
        with zipfile.ZipFile(buf, "w") as z:
            for p in (fixture()["mock"] / "whoop").iterdir():
                z.write(p, f"export/{p.name}")        # nested folder inside the zip
        self.assertTrue(IMPORTERS["whoop"].can_import(buf))
        res, store, _ = import_into("whoop", buf)
        self.assertEqual(res.added["nights"], 359)

    def test_a_zip_bomb_is_refused_before_it_is_decompressed(self):
        from domains.health.sources import ExportTooLarge
        from domains.health.sources.base import Tree
        buf = support._tmp() / "bomb.zip"
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("export/physiological_cycles.csv", b"a" * (20 * 1024 * 1024))
        with self.assertRaises(ExportTooLarge):
            Tree(buf).read_bytes("export/physiological_cycles.csv")
        with self.assertRaises(ExportTooLarge):
            IMPORTERS["whoop"].import_path(buf, empty_data())

    def _zip(self, members, declared=None):
        """A real zip of tiny members. `declared` overrides the uncompressed size written in the central directory
        (the local header and data stay real), which is what an archive-level cap reads."""
        from domains.health.sources.base import Tree
        buf = support._tmp() / "a.zip"
        with zipfile.ZipFile(buf, "w") as z:
            for name in members:
                z.writestr(name, b"x")
            for info, size in zip(z.filelist, declared or []):
                info.file_size = size
        return buf, Tree

    def test_an_archive_declaring_more_than_2_gib_in_total_is_refused_before_anything_is_read(self):
        from domains.health.sources import ExportTooLarge
        buf, Tree = self._zip([f"export/f{i}.csv" for i in range(5)], [500 * 1024 ** 2] * 5)      # each under the member cap
        with mock.patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("read before refusing")):
            with self.assertRaisesRegex(ExportTooLarge, "in total"):
                Tree(buf)
            with self.assertRaises(ExportTooLarge):
                IMPORTERS["whoop"].import_path(buf, empty_data())

    def test_the_total_cap_is_a_cap_not_a_count(self):
        from domains.health.sources import base
        buf, Tree = self._zip([f"export/f{i}.csv" for i in range(5)])           # five real one-byte members
        Tree(buf)
        with mock.patch.object(base, "MAX_ARCHIVE_BYTES", 4):
            with self.assertRaisesRegex(base.ExportTooLarge, "in total"):
                Tree(buf)
        with mock.patch.object(base, "MAX_ARCHIVE_BYTES", 5):
            Tree(buf)

    def test_an_archive_with_more_than_20000_members_is_refused(self):
        from domains.health.sources import ExportTooLarge
        from domains.health.sources.base import MAX_MEMBERS
        self.assertEqual(MAX_MEMBERS, 20_000)
        buf, Tree = self._zip([f"export/f{i}.csv" for i in range(MAX_MEMBERS + 1)])
        with self.assertRaisesRegex(ExportTooLarge, "members"):
            Tree(buf)
        buf, Tree = self._zip([f"export/f{i}.csv" for i in range(MAX_MEMBERS)])
        Tree(buf)

    def test_a_member_name_that_is_absolute_or_climbs_out_is_refused(self):
        from domains.health.sources import ExportRefused
        for name in ("../evil", "export/../../evil", "/etc/evil", "C:/evil", "..\\evil"):
            buf, Tree = self._zip(["export/ok.csv"])
            with zipfile.ZipFile(buf, "a") as z:
                z.writestr(zipfile.ZipInfo(name), b"x")
            with self.assertRaises(ExportRefused, msg=name):
                Tree(buf)
        buf, Tree = self._zip(["export/ok..v2.csv"])                  # '..' inside a name is not a path component
        Tree(buf)

    def test_the_cli_reports_a_refused_archive_in_one_line(self):
        buf, _ = self._zip(["export/ok.csv"])
        with zipfile.ZipFile(buf, "a") as z:
            z.writestr(zipfile.ZipInfo("../evil"), b"x")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cli.main(["--data-dir", str(empty_data()), "import", "whoop", str(buf)])
        self.assertEqual(rc, 2)
        self.assertRegex(err.getvalue().strip(), r"^refused: .*\.\./evil.*$")
        self.assertEqual(len(err.getvalue().strip().splitlines()), 1)

    def test_headers_are_case_insensitive_and_localised_variants_normalise(self):
        self.assertEqual(norm("Day Strain"), norm("Day strain"))
        self.assertEqual(norm("Zone FC 1 %"), norm("zone fc 1 %"))
        text = "CYCLE START TIME, cycle end time ,Cycle Timezone,Sleep onset,Wake onset,RECOVERY SCORE %,Day strain\n" \
               "2026-03-10 22:51:03,2026-03-11 23:05:12,UTC-07:00,2026-03-10 22:51:03,2026-03-11 06:44:19,58,12.4\n"
        d = support._tmp()
        (d / "physiological_cycles.csv").write_text(text)
        res, store, _ = import_into("whoop", d)
        self.assertEqual(res.added["nights"], 1)
        self.assertEqual(store.daily["whoop", "2026-03-11"].strain, 12.4)
        self.assertEqual(store.daily["whoop", "2026-03-11"].recovery_pct, 58)

    def test_fahrenheit_skin_temp_is_flagged_and_converted(self):
        d = support._tmp()
        (d / "physiological_cycles.csv").write_text(
            "Cycle start time,Cycle end time,Cycle timezone,Sleep onset,Wake onset,Skin temp (celsius)\n"
            "2026-03-10 22:51:03,2026-03-11 23:05:12,UTC-07:00,2026-03-10 22:51:03,2026-03-11 06:44:19,92.1\n")
        res, store, _ = import_into("whoop", d)
        self.assertTrue(any("Fahrenheit" in a for a in res.anomalies))
        self.assertAlmostEqual(store.night_rows("whoop")[0].skin_temp_c, 33.39, places=1)

    def test_not_a_whoop_export(self):
        self.assertFalse(IMPORTERS["whoop"].can_import(fixture()["mock"] / "garmin"))

    def test_skin_temp_delta_is_against_own_median(self):
        rows = [r for r in self.store.night_rows("whoop") if r.skin_temp_delta_c is not None]
        self.assertEqual(len(rows), 359)
        self.assertLess(abs(sorted(r.skin_temp_delta_c for r in rows)[len(rows) // 2]), 0.05)


class WhoopApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = fixture()["mock"] / "whoop_api"
        cls.pages = {n: [r for p in ("page1", "page2") for r in json.loads((d / f"{n}_{p}.json").read_text())["records"]]
                     for n in ("cycles", "recovery", "sleep", "workouts")}
        cls.nights, cls.daily, cls.workouts = whoop_api.map_records(cls.pages["cycles"], cls.pages["recovery"],
                                                                    cls.pages["sleep"], cls.pages["workouts"])
        _, cls.csv_store, _ = import_into("whoop", fixture()["mock"] / "whoop")

    def test_collections_are_paged_newest_first(self):
        first = json.loads((fixture()["mock"] / "whoop_api" / "sleep_page1.json").read_text())
        self.assertEqual(first["next_token"], "opaque-token-1")
        self.assertGreater(first["records"][0]["start"], first["records"][-1]["start"])

    def test_pending_score_is_skipped_not_guessed(self):
        self.assertEqual(len(self.nights), len(self.pages["sleep"]) - 1)
        res = whoop_api.ImportResult("whoop-api")
        whoop_api.map_records([], [], self.pages["sleep"], [], res)
        self.assertTrue(any("PENDING_SCORE" in a for a in res.anomalies))

    def test_units_converted_ms_to_minutes_and_hrv_kept_as_float(self):
        n = next(x for x in self.nights if x.date == "2026-09-15")
        c = self.csv_store.nights["whoop", "2026-09-15"]
        self.assertEqual(n.asleep_min, c.asleep_min)
        self.assertEqual((n.deep_min, n.rem_min, n.light_min, n.awake_min), (c.deep_min, c.rem_min, c.light_min, c.awake_min))
        self.assertEqual(round(n.hrv_ms), c.hrv_ms)
        self.assertIsInstance(n.hrv_ms, float)
        self.assertEqual(n.resting_hr_bpm, c.resting_hr_bpm)

    def test_utc_plus_offset_gives_the_same_local_dates_as_the_csv(self):
        for n in self.nights:
            c = self.csv_store.nights["whoop", n.date]
            self.assertEqual(n.sleep_end[10:13], c.sleep_end[10:13])          # same local hour
            self.assertEqual(local_date(n.sleep_end), local_date(c.sleep_end))
            self.assertEqual(n.tz_offset_min, c.tz_offset_min)

    def test_red_eye_style_date_from_utc_plus_offset(self):
        sleeps = [{"id": "s1", "cycle_id": 1, "start": "2026-05-11T01:50:00.000Z", "end": "2026-05-11T02:30:00.000Z",
                   "timezone_offset": "-07:00", "nap": False, "score_state": "SCORED",
                   "score": {"stage_summary": {"total_in_bed_time_milli": 3600000, "total_awake_time_milli": 600000,
                                               "total_no_data_time_milli": 0, "total_light_sleep_time_milli": 1800000,
                                               "total_slow_wave_sleep_time_milli": 600000, "total_rem_sleep_time_milli": 600000},
                             "sleep_efficiency_percentage": 83.3}}]
        nights, _, _ = whoop_api.map_records([], [], sleeps, [])
        self.assertEqual(nights[0].date, "2026-05-10")
        self.assertEqual(nights[0].sleep_end, "2026-05-10T19:30:00-07:00")

    def test_strain_steps_recovery_join_by_ids(self):
        _, daily, _ = self.nights, self.daily, self.workouts
        d = next(x for x in self.daily if x.date == "2026-09-15")
        c = self.csv_store.daily["whoop", "2026-09-15"]
        self.assertEqual(d.recovery_pct, c.recovery_pct)
        self.assertAlmostEqual(d.strain, c.strain, places=1)

    def test_workouts_map(self):
        self.assertEqual(len(self.workouts), len(self.pages["workouts"]))
        self.assertTrue(all(w.source == "whoop" and w.date for w in self.workouts))

    def test_syncer_is_a_skeleton_that_says_what_it_needs(self):
        with self.assertRaises(NotConfigured) as cm:
            SYNCERS["whoop"].sync(None, empty_data())
        self.assertIn("OAuth", str(cm.exception))
        self.assertTrue(SYNCERS["whoop"].requirements())

    def test_import_json_writes_idempotently(self):
        dd = empty_data()
        a = SYNCERS["whoop"].import_json(self.pages, dd)
        b = SYNCERS["whoop"].import_json(self.pages, dd)
        self.assertGreater(a.added["nights"], 20)
        self.assertEqual(b.total_added, 0)


class GarminExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res, cls.store, cls.dd = import_into("garmin", fixture()["mock"] / "garmin")
        cls.counts = fixture()["truth"].counts

    def test_exact_row_counts(self):
        self.assertEqual(self.res.added["nights"], 788)
        self.assertEqual(self.res.added["daily"], 791)
        self.assertEqual(self.res.added["workouts"], self.counts["garmin_activities"])

    def test_overlapping_chunks_deduplicate_by_calendar_date(self):
        self.assertEqual(self.res.skipped["nights"], 16)
        self.assertEqual(self.res.skipped["daily"], 16)

    def test_charging_nights_reported(self):
        gap = next(a for a in self.res.anomalies if "missing" in a)
        self.assertIn("3 night(s)", gap)
        for d in ("2025-06-14", "2025-12-02", "2026-07-21"):
            self.assertIn(d, gap)

    def test_naive_utc_converted_with_local_twin_offset(self):
        row = next(r for r in self.store.night_rows("garmin") if r.date == "2026-09-29")
        self.assertEqual(row.asleep_min, 245)
        self.assertTrue(row.sleep_end.endswith("-07:00"))
        self.assertEqual(row.hrv_method, "overnight_avg")
        self.assertEqual(local_date(row.sleep_end), "2026-09-29")

    def test_red_eye_and_timezone_change(self):
        row = next(r for r in self.store.night_rows("garmin") if r.date == "2026-05-10")
        self.assertEqual(row.tz_offset_min, -240)
        self.assertTrue(row.sleep_start.startswith("2026-05-10T02:") and row.sleep_start.endswith("-04:00"))
        tz = [a for a in self.res.anomalies if "timezone change" in a]
        self.assertEqual(len(tz), 2)
        # an instant is an instant: the same absolute time whichever offset renders it
        self.assertLessEqual(abs((utc_instant(row.sleep_start) - datetime(2026, 5, 10, 6, 50, tzinfo=timezone.utc)).total_seconds()), 200)

    def test_every_night_date_equals_the_local_date_of_its_end(self):
        for r in self.store.night_rows("garmin"):
            self.assertEqual(r.date, local_date(r.sleep_end))

    def test_hrv_comes_from_health_status_not_sleep_data(self):
        row = next(r for r in self.store.night_rows("garmin") if r.date == "2026-09-15")
        truth = fixture()["truth"].nights["2026-09-15"]
        self.assertEqual(row.hrv_ms, truth["garmin_hrv"])
        self.assertEqual(row.resting_hr_bpm, truth["garmin_rhr"])
        self.assertTrue(all(r.hrv_ms is not None for r in self.store.night_rows("garmin")))

    def test_body_battery_is_looked_up_by_type_not_position(self):
        d = self.store.daily["garmin", "2026-09-15"]
        self.assertGreater(d.body_battery_max, d.body_battery_min)
        self.assertIsNotNone(d.stress_avg)

    def test_epoch_ms_calendar_date_handled(self):
        self.assertEqual(cal_date(1773273600000), "2026-03-12")
        self.assertEqual(cal_date("2026-03-12"), "2026-03-12")
        self.assertEqual(cal_date("2026-03-12T00:00:00.0"), "2026-03-12")
        self.assertIsNone(cal_date(None))
        loaded = [d for d in self.store.daily.values() if d.training_load_acute is not None]
        self.assertEqual(len(loaded), 791)

    def test_activity_units_cm_ms_and_epoch(self):
        w = next(x for x in self.store.workouts.values() if x.sport == "running" and x.distance_m)
        self.assertTrue(1500 < w.distance_m < 20000)            # centimeters converted to meters
        self.assertTrue(w.end > w.start)
        self.assertEqual(w.start[:10], w.date)
        self.assertIsNotNone(w.training_effect)

    def test_garmin_and_whoop_workouts_are_separate_rows(self):
        _, both, _ = import_into("whoop", fixture()["mock"] / "whoop", self.dd)
        srcs = {w.source for w in both.workouts.values()}
        self.assertEqual(srcs, {"whoop", "garmin"})

    def test_idempotent_reimport(self):
        res = IMPORTERS["garmin"].import_path(fixture()["mock"] / "garmin", self.dd)
        self.assertEqual(res.total_added, 0)

    def test_zip_import(self):
        z = support._tmp() / "g.zip"
        root = fixture()["mock"] / "garmin"
        with zipfile.ZipFile(z, "w") as zf:
            for p in root.rglob("*.json"):
                zf.write(p, p.relative_to(root).as_posix())
        res, store, _ = import_into("garmin", z)
        self.assertEqual(res.added["nights"], 788)

    def test_invalid_json_file_is_reported_not_fatal(self):
        d = support._tmp() / "DI_CONNECT" / "DI-Connect-Wellness"
        d.mkdir(parents=True)
        (d / "2026-01-01_2026-01-02_1_sleepData.json").write_text("{not json")
        res, store, _ = import_into("garmin", d.parent.parent)
        self.assertTrue(any("invalid JSON" in a for a in res.anomalies))

    def test_garmin_health_api_mapping_is_marked_unconfirmed_and_unavailable(self):
        n = garmin_api.map_sleep({"summaryId": "x", "startTimeInSeconds": 1773294000, "startTimeOffsetInSeconds": -25200,
                                  "durationInSeconds": 28000, "deepSleepDurationInSeconds": 5400, "lightSleepDurationInSeconds": 15300,
                                  "remSleepInSeconds": 6300, "awakeDurationInSeconds": 1020}, {"lastNightAvg": 58})
        self.assertEqual((n.deep_min, n.light_min, n.rem_min, n.hrv_ms), (90, 255, 105, 58))
        self.assertTrue(n.sleep_end.endswith("-07:00"))
        with self.assertRaises(NotConfigured) as cm:
            SYNCERS["garmin"].sync(None, empty_data())
        self.assertIn("not available to individuals", str(cm.exception))
        self.assertIn("export", " ".join(SYNCERS["garmin"].requirements()))


class EpicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dd = empty_data()
        cls.res = IMPORTERS["epic"].import_path(fixture()["mock"] / "epic", cls.dd)
        cls.data = Vault(cls.dd).read("status")

    def test_exact_resource_counts(self):
        self.assertEqual(self.res.added, {"Patient": 1, "Condition": 1, "MedicationRequest": 2, "AllergyIntolerance": 1, "Observation": 18})
        d = self.data
        self.assertEqual((len(d["conditions"]), len(d["medications"]), len(d["allergies"]), len(d["labs"]), len(d["vitals"])),
                         (1, 2, 1, 13, 5))

    def test_operation_outcome_entry_is_reported(self):
        self.assertTrue(any("OperationOutcome" in a for a in self.res.anomalies))

    def test_epic_local_code_kept_beside_loinc(self):
        a1c = [l for l in self.data["labs"] if l["text"] == "Hemoglobin A1c"]
        self.assertEqual(len(a1c), 3)
        systems = {c["system"] for c in a1c[0]["codes"]}
        self.assertTrue(any(s.startswith("urn:oid:") for s in systems) and "http://loinc.org" in systems)

    def test_bp_components_and_units(self):
        bp = next(v for v in self.data["vitals"] if v["text"] == "Blood pressure")
        self.assertEqual([c["value"] for c in bp["components"]], [148, 94])
        self.assertEqual(bp["components"][0]["unit"], "mmHg")

    def test_local_ids_are_harness_ids_not_epic_ids(self):
        m = self.data["medications"][0]
        self.assertRegex(m["local_id"], r"^med\d+$")
        self.assertTrue(m["source_ids"] and m["source_ids"][0] != m["local_id"])

    def test_idempotent_reimport(self):
        res = IMPORTERS["epic"].import_path(fixture()["mock"] / "epic", self.dd)
        self.assertEqual(res.total_added, 0)
        self.assertEqual(len(Vault(self.dd).read("status")["labs"]), 13)

    def test_single_bundle_file_import(self):
        dd = empty_data()
        res = IMPORTERS["epic"].import_path(fixture()["mock"] / "epic" / "MedicationRequest.json", dd)
        self.assertEqual(res.added, {"MedicationRequest": 2})

    def test_ccda_carries_the_same_facts_for_its_four_sections(self):
        dd = empty_data()
        res = IMPORTERS["epic"].import_path(fixture()["mock"] / "epic" / "ccda", dd)
        got = Vault(dd).read("status")
        self.assertEqual((len(got["conditions"]), len(got["medications"]), len(got["allergies"]), len(got["labs"])), (1, 2, 1, 13))
        self.assertEqual(len(got["vitals"]), 0)                     # vital signs are outside the four sections
        key = lambda items, f: sorted(f(i) for i in items)
        self.assertEqual(key(got["labs"], lambda l: (l["codes"][0]["code"], l["value"], l["effective"][:16])),
                         key(self.data["labs"], lambda l: (l["codes"][0]["code"], l["value"], l["effective"][:16])))
        self.assertEqual(key(got["medications"], lambda m: (m["codes"][0]["code"], m["authored_on"][:10], m["status"])),
                         key(self.data["medications"], lambda m: (m["codes"][0]["code"], m["authored_on"][:10], m["status"])))
        self.assertEqual(got["patient"]["family"], self.data["patient"]["family"])
        self.assertEqual(got["patient"]["birth_date"], self.data["patient"]["birth_date"])

    def test_ccda_after_fhir_adds_nothing(self):
        res = IMPORTERS["epic"].import_path(fixture()["mock"] / "epic" / "ccda", self.dd)
        self.assertEqual(res.total_added, 0)

    def test_ccda_xdm_zip(self):
        z = support._tmp() / "xdm.zip"
        root = fixture()["mock"] / "epic" / "ccda"
        with zipfile.ZipFile(z, "w") as zf:
            for p in root.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(root).as_posix())
        res = IMPORTERS["epic"].import_path(z, empty_data())
        self.assertEqual(res.added["MedicationRequest"], 2)

    def test_ccda_tolerates_invalid_code_token(self):
        xml = (fixture()["mock"] / "epic" / "ccda" / "MachineReadable_XDMFormat" / "IHE_XDM" / "Jordan1" / "DOC0001.XML").read_text()
        xml = xml.replace('displayName="Penicillin G"', 'displayName="NO PREFERENC"')
        d = support._tmp()
        (d / "DOC0002.XML").write_text(xml)
        res = IMPORTERS["epic"].import_path(d, empty_data())
        self.assertEqual(res.added["AllergyIntolerance"], 1)

    def test_epic_import_refuses_without_a_passphrase(self):
        dd = empty_data()
        with mock.patch.dict(os.environ):
            os.environ.pop("HEALTH_HARNESS_PASSPHRASE", None)
            with self.assertRaises(VaultLocked) as cm:
                IMPORTERS["epic"].import_path(fixture()["mock"] / "epic", dd)
        self.assertIn("HEALTH_HARNESS_PASSPHRASE", str(cm.exception))
        self.assertFalse((dd / "ehr.fernet").exists())

    def test_epic_import_refuses_with_an_install_hint_when_cryptography_is_missing(self):
        import domains.health.vault as vault_mod
        dd = empty_data()
        with mock.patch.dict("sys.modules", {"cryptography": None, "cryptography.fernet": None}):
            with self.assertRaises(vault_mod.VaultUnavailable) as cm:
                IMPORTERS["epic"].import_path(fixture()["mock"] / "epic", dd)
        self.assertIn("pip install cryptography", str(cm.exception))
        self.assertFalse((dd / "ehr.fernet").exists())

    def test_epic_syncer_is_a_documented_skeleton(self):
        with self.assertRaises(NotConfigured):
            SYNCERS["epic"].sync(None, empty_data())
        text = " ".join(SYNCERS["epic"].requirements())
        self.assertIn("PKCE", text)
        self.assertIn("fhir.epic.com", text)
        self.assertIn("smart-configuration", text)

    def test_wearable_import_does_not_need_a_passphrase(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("HEALTH_HARNESS_PASSPHRASE", None)
            res = IMPORTERS["whoop"].import_path(fixture()["mock"] / "whoop", empty_data())
        self.assertEqual(res.added["nights"], 359)


if __name__ == "__main__":
    unittest.main()
