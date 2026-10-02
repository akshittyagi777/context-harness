"""Precedence, disagreement, baselines, regimes and the domain rules."""
from __future__ import annotations

import json
import unittest
from datetime import date, datetime, timedelta

from tests import support
from tests.support import data_copy, empty_data, fixture

from harness.memory import MemoryRecord, MemoryStore
from harness.resolve import resolve
from domains.health.baselines import Baselines, add_regime, load_regimes, regime_days
from domains.health.domain import build_domain
from domains.health.precedence import DEFAULT_PRECEDENCE, WearView, disagreement_summary, load_precedence, night_disagreement
from domains.health.rules import compute_values

CFG = {"anchor_days": 90, "anchor": {}, "rolling_days": 30, "sleep_disagreement_min": 45, "rhr_disagreement_bpm": 5}
START = date(2025, 1, 1)


def night(src, d, asleep=450, rhr=50.0, hrv=60.0):
    return {"date": d.isoformat() if isinstance(d, date) else d, "source": src, "asleep_min": asleep, "resting_hr_bpm": rhr,
            "hrv_ms": hrv, "deep_min": 80, "rem_min": 100, "light_min": 250, "awake_min": 30, "spo2_pct": 95.0}


def series_two_sources(n=60, whoop=lambda i: {}, garmin=lambda i: {}):
    rows = []
    for i in range(n):
        d = START + timedelta(days=i)
        rows.append(night("whoop", d, **whoop(i)))
        rows.append(night("garmin", d, **garmin(i)))
    return {"nights": rows, "daily": [], "workouts": []}


def at(series, day):
    return WearView(series, DEFAULT_PRECEDENCE), Baselines(series["nights"], [], CFG)


class PrecedenceTests(unittest.TestCase):
    def test_defaults_created_on_first_run_with_the_spec_values(self):
        dd = empty_data()
        got = load_precedence(dd)
        self.assertTrue((dd / "precedence.json").exists())
        self.assertEqual(got, DEFAULT_PRECEDENCE)
        self.assertEqual(got["sleep_duration"], ["whoop", "garmin"])
        self.assertEqual(got["steps"], ["garmin", "whoop"])
        self.assertEqual(got["recovery"], ["whoop"])
        self.assertEqual(got["stress"], ["garmin"])
        self.assertEqual(oct((dd / "precedence.json").stat().st_mode)[-3:], "600")

    def test_user_override_wins_and_missing_keys_fall_back(self):
        dd = empty_data()
        (dd / "precedence.json").write_text(json.dumps({"sleep_duration": ["garmin", "whoop"]}))
        got = load_precedence(dd)
        self.assertEqual(got["sleep_duration"], ["garmin", "whoop"])
        self.assertEqual(got["steps"], ["garmin", "whoop"])

    def test_first_source_with_a_row_wins(self):
        s = series_two_sources(3, whoop=lambda i: {"asleep": 480}, garmin=lambda i: {"asleep": 450})
        v = WearView(s, DEFAULT_PRECEDENCE)
        p = v.pick("sleep_duration", "2025-01-02")
        self.assertEqual((p.source, p.value, p.skipped), ("whoop", 480, []))
        v2 = WearView(s, {**DEFAULT_PRECEDENCE, "sleep_duration": ["garmin", "whoop"]})
        self.assertEqual(v2.pick("sleep_duration", "2025-01-02").source, "garmin")

    def test_missing_night_falls_through_and_says_so(self):
        s = series_two_sources(3)
        s["nights"] = [r for r in s["nights"] if not (r["source"] == "whoop" and r["date"] == "2025-01-02")]
        p = WearView(s, DEFAULT_PRECEDENCE).pick("sleep_duration", "2025-01-02")
        self.assertEqual((p.source, p.skipped, p.preferred), ("garmin", ["whoop"], "whoop"))

    def test_nothing_when_no_source_has_the_signal(self):
        v = WearView(series_two_sources(3), DEFAULT_PRECEDENCE)
        self.assertIsNone(v.pick("recovery", "2025-01-02"))       # whoop-only, and there are no daily rows
        self.assertIsNone(v.pick("sleep_duration", "2030-01-01"))

    def test_per_signal_precedence_differs(self):
        s = {"nights": [], "workouts": [], "daily": [
            {"date": "2025-01-01", "source": "whoop", "steps": 100, "recovery_pct": 50, "stress_avg": None},
            {"date": "2025-01-01", "source": "garmin", "steps": 9000, "recovery_pct": None, "stress_avg": 30}]}
        v = WearView(s, DEFAULT_PRECEDENCE)
        self.assertEqual(v.pick("steps", "2025-01-01").source, "garmin")
        self.assertEqual(v.pick("recovery", "2025-01-01").source, "whoop")
        self.assertEqual(v.pick("stress", "2025-01-01").value, 30)


class DisagreementTests(unittest.TestCase):
    def check(self, whoop, garmin, day="2025-02-20"):
        s = series_two_sources(60, whoop=lambda i: whoop if i == 50 else {}, garmin=lambda i: garmin if i == 50 else {})
        v, b = at(s, day)
        return night_disagreement(v, b, day, CFG)

    def test_sleep_threshold_is_45_minutes_exclusive(self):
        self.assertFalse(self.check({"asleep": 495}, {"asleep": 450})["sleep_over"])      # exactly 45
        d = self.check({"asleep": 496}, {"asleep": 450})
        self.assertTrue(d["sleep_over"])
        self.assertEqual(d["sleep_diff_min"], 46)
        self.assertTrue(self.check({"asleep": 400}, {"asleep": 450})["sleep_over"])         # either direction

    def test_resting_hr_threshold_is_5_bpm_exclusive(self):
        self.assertFalse(self.check({"rhr": 55}, {"rhr": 50})["rhr_over"])
        self.assertTrue(self.check({"rhr": 56}, {"rhr": 50})["rhr_over"])

    def test_hrv_is_compared_by_direction_never_by_value(self):
        # WHOOP 80 vs Garmin 40 is a huge gap in value but both sit above their own baseline
        d = self.check({"hrv": 80.0}, {"hrv": 40.0})
        self.assertNotIn("hrv_diff", d)
        self.assertFalse(any(k.startswith("hrv") and "diff" in k for k in d))
        self.assertEqual(d["hrv_direction"], {"whoop": "above", "garmin": "below"})
        self.assertTrue(d["hrv_mismatch"])           # garmin baseline is 60, so 40 is below: directions oppose

    def test_same_direction_is_not_a_mismatch_even_with_different_values(self):
        d = self.check({"hrv": 80.0}, {"hrv": 70.0})
        self.assertEqual(d["hrv_direction"], {"whoop": "above", "garmin": "above"})
        self.assertFalse(d["hrv_mismatch"])

    def test_summary_has_no_averaged_hrv_and_reports_over_threshold_nights(self):
        s = series_two_sources(60, whoop=lambda i: {"asleep": 520} if i in (50, 52) else {}, garmin=lambda i: {})
        v, b = at(s, "2025-02-20")
        summ = disagreement_summary(v, b, "2025-02-23", CFG)
        self.assertEqual(summ["nights_compared"], 14)
        self.assertEqual(sorted(summ["sleep_over_nights"]), ["2025-02-20", "2025-02-22"])
        self.assertEqual(summ["mean_sleep_diff_min"], round(2 * 70 / 14, 1))
        self.assertFalse(any("hrv" in k and "mean" in k for k in summ))
        self.assertEqual(summ["sources"], ["whoop", "garmin"])

    def test_no_comparison_when_only_one_source_has_the_night(self):
        s = series_two_sources(10)
        s["nights"] = [r for r in s["nights"] if not (r["source"] == "garmin" and r["date"] == "2025-01-05")]
        v, b = at(s, "2025-01-05")
        self.assertIsNone(night_disagreement(v, b, "2025-01-05", CFG))

    def test_mock_data_has_a_systematic_bias_and_about_eight_percent_over_threshold(self):
        s = WearView(_series(), DEFAULT_PRECEDENCE)
        both = [d for d in s.night_dates() if ("whoop", d) in s.nights and ("garmin", d) in s.nights]
        diffs = [s.nights["whoop", d]["asleep_min"] - s.nights["garmin", d]["asleep_min"] for d in both]
        self.assertEqual(len(both), 357)
        self.assertTrue(15 <= sum(diffs) / len(diffs) <= 30)
        share = sum(abs(x) > 45 for x in diffs) / len(diffs)
        self.assertTrue(0.04 <= share <= 0.13, share)


def _series():
    from domains.health.canonical import WearableStore
    return WearableStore.load(fixture()["data"]).series()


class BaselineTests(unittest.TestCase):
    def synth(self, values, src="whoop"):
        rows = [night(src, START + timedelta(days=i), asleep=v) for i, v in enumerate(values)]
        return Baselines(rows, [], CFG), (START + timedelta(days=len(values) - 1)).isoformat()

    def test_rolling_median_is_30_days_and_anchored_is_the_first_90(self):
        b, end = self.synth([450] * 90 + [390] * 60)
        self.assertEqual(b.rolling("whoop", "asleep_min", end), 390)
        self.assertEqual(b.anchored("whoop", "asleep_min"), 450)

    def test_rolling_follows_a_drift_that_anchored_catches(self):
        b, end = self.synth([450] * 90 + [450 - i * 0.5 for i in range(1, 121)])
        d = b.divergence("whoop", "sleep_duration", end)
        self.assertEqual(d["anchored"], 450)
        self.assertTrue(d["active"])
        self.assertLess(d["delta_pct"], -10)

    def test_divergence_needs_14_consecutive_days(self):
        b, end = self.synth([450] * 90 + [450] * 30 + [380] * 22)      # median drops only once 15 of 30 days are low
        d = b.divergence("whoop", "sleep_duration", end)
        self.assertEqual(d["run_days"], 7)       # the median drops below the threshold once 16 of 30 days are low
        self.assertFalse(d["active"])
        b2, end2 = self.synth([450] * 90 + [450] * 30 + [380] * 40)
        self.assertTrue(b2.divergence("whoop", "sleep_duration", end2)["active"])

    def test_exactly_ten_percent_is_not_a_divergence(self):
        b, end = self.synth([450] * 90 + [405] * 60)
        self.assertFalse(b.divergence("whoop", "sleep_duration", end)["active"])
        b, end = self.synth([450] * 90 + [404] * 60)
        self.assertTrue(b.divergence("whoop", "sleep_duration", end)["active"])

    def test_resting_hr_rule_is_more_than_5_bpm_above(self):
        rows = [night("whoop", START + timedelta(days=i), rhr=50.0 if i < 90 else 55.0) for i in range(150)]
        b = Baselines(rows, [], CFG)
        self.assertFalse(b.divergence("whoop", "resting_hr", (START + timedelta(days=149)).isoformat())["active"])
        rows = [night("whoop", START + timedelta(days=i), rhr=50.0 if i < 90 else 56.0) for i in range(150)]
        self.assertTrue(Baselines(rows, [], CFG).divergence("whoop", "resting_hr", (START + timedelta(days=149)).isoformat())["active"])

    def test_resting_hr_below_anchored_is_not_a_divergence(self):
        rows = [night("whoop", START + timedelta(days=i), rhr=54.0 if i < 90 else 47.0) for i in range(150)]
        d = Baselines(rows, [], CFG).divergence("whoop", "resting_hr", (START + timedelta(days=149)).isoformat())
        self.assertFalse(d["active"])
        self.assertEqual(d["delta"], -7.0)

    def test_each_source_has_its_own_anchor(self):
        rows = [night("whoop", START + timedelta(days=i), asleep=480) for i in range(120)] + \
               [night("garmin", START + timedelta(days=i), asleep=440) for i in range(120)]
        b = Baselines(rows, [], CFG)
        self.assertEqual(b.anchored("whoop", "asleep_min"), 480)
        self.assertEqual(b.anchored("garmin", "asleep_min"), 440)

    def test_anchor_window_is_user_configurable(self):
        rows = [night("whoop", START + timedelta(days=i), asleep=450 if i < 30 else 400) for i in range(150)]
        cfg = {**CFG, "anchor": {"whoop": {"start": "2025-02-01", "end": "2025-04-01"}}}
        self.assertEqual(Baselines(rows, [], cfg).anchored("whoop", "asleep_min"), 400)

    def test_regime_days_are_excluded_from_both_baselines(self):
        rows = [night("whoop", START + timedelta(days=i), asleep=450) for i in range(150)]
        for r in rows[100:130]:
            r["asleep_min"] = 100                                  # an illness-like stretch
        regimes = [{"start": (START + timedelta(days=100)).isoformat(), "end": (START + timedelta(days=129)).isoformat(),
                    "kind": "illness", "label": "flu"}]
        end = (START + timedelta(days=140)).isoformat()
        self.assertEqual(Baselines(rows, [], CFG).rolling("whoop", "asleep_min", end), 100)         # unmarked: the stretch counts
        marked = Baselines(rows, regimes, CFG)
        self.assertEqual(marked.rolling("whoop", "asleep_min", end), 450)
        early = [night("whoop", START + timedelta(days=i), asleep=100 if i < 20 else 450) for i in range(150)]
        reg2 = [{"start": START.isoformat(), "end": (START + timedelta(days=19)).isoformat(), "kind": "travel", "label": "x"}]
        self.assertEqual(Baselines(early, reg2, CFG).anchored("whoop", "asleep_min"), 450)
        self.assertIn(START.isoformat(), Baselines(early, [], CFG).points("whoop", "asleep_min"))
        self.assertNotIn(START.isoformat(), Baselines(early, reg2, CFG).points("whoop", "asleep_min"))

    def test_too_little_data_gives_no_baseline(self):
        b, end = self.synth([450] * 5)
        self.assertIsNone(b.rolling("whoop", "asleep_min", end))
        self.assertIsNone(b.anchored("whoop", "asleep_min"))

    def test_add_regime_is_idempotent_and_listed(self):
        dd = empty_data()
        self.assertTrue(add_regime(dd, "2026-05-10", "2026-05-17", "travel", "New York"))
        self.assertFalse(add_regime(dd, "2026-05-10", "2026-05-17", "travel", "New York"))
        self.assertEqual(len(load_regimes(dd)), 1)
        self.assertEqual(len(regime_days(load_regimes(dd))), 8)

    def test_mock_data_slow_decline_is_caught_by_anchored_not_by_rolling_alone(self):
        series = _series()
        b = Baselines(series["nights"], load_regimes(fixture()["data"]), CFG)
        d = b.divergence("whoop", "sleep_duration", "2026-09-29")
        self.assertTrue(d["active"], d)
        self.assertGreaterEqual(d["run_days"], 14)
        self.assertLess(d["delta_pct"], -10)
        # the rolling median at the start of the decline sits near the anchor: it followed the drift, it did not jump
        self.assertGreater(b.rolling("whoop", "asleep_min", "2026-08-01"), b.anchored("whoop", "asleep_min") * 0.9)


class RuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dd = fixture()["data"]
        cls.domain = build_domain(cls.dd)

    def sc(self, ts, text=None):
        from dataclasses import replace
        from harness.domain import Turn
        sc = self.domain.build_scenario(datetime.fromisoformat(ts))
        return replace(sc, incoming=Turn(sc.incoming.ts, "user", text, sc.incoming.session)) if text else sc

    def store(self):
        return MemoryStore.load(self.dd / "memory.json")

    def test_acute_night_values_keep_each_source(self):
        v = compute_values(self.sc("2026-09-29T08:10:00"))
        self.assertEqual(v["as_of"], "2026-09-29")
        self.assertEqual(v["last"]["sleep_duration"]["value"], 280)
        self.assertEqual(v["last"]["sleep_duration"]["label"], "WHOOP")
        self.assertEqual(v["sources"]["garmin"]["asleep_min"], 245)
        self.assertEqual(v["recovery"]["value"], 21)
        self.assertEqual(v["facts"], "Last night (2026-09-29): sleep 4h40 (WHOOP), resting HR 55 bpm (WHOOP), recovery 21% (WHOOP).")

    def test_hrv_is_reported_per_source_with_its_method(self):
        v = compute_values(self.sc("2026-09-29T08:10:00"))
        self.assertEqual(v["hrv"]["whoop"]["method"], "rmssd_last_sws")
        self.assertEqual(v["hrv"]["garmin"]["method"], "overnight_avg")
        self.assertEqual(v["hrv"]["whoop"]["direction"], "below")
        self.assertEqual(v["hrv"]["garmin"]["direction"], "below")
        self.assertFalse({k for k in v["disagreement_last"] if k.startswith("hrv") and k.endswith("diff")})

    def test_a_night_is_not_visible_before_the_sleep_ended(self):
        early = compute_values(self.sc("2026-09-29T03:00:00"))
        self.assertEqual(early["as_of"], "2026-09-28")
        late = compute_values(self.sc("2026-09-29T08:10:00"))
        self.assertEqual(late["as_of"], "2026-09-29")

    def test_todays_totals_are_not_visible_only_the_morning_recovery(self):
        sc = self.sc("2026-09-29T08:10:00")
        today = [r for r in sc.series["daily"] if r["date"] == "2026-09-29"]
        self.assertTrue(today and all(r["steps"] is None and r["strain"] is None for r in today))
        self.assertEqual([r["recovery_pct"] for r in today if r["source"] == "whoop"], [21])
        self.assertFalse(any(w["date"] == "2026-09-30" for w in sc.series["workouts"]))
        self.assertFalse(any(r["date"] == "2026-09-30" for r in sc.series["nights"]))

    def test_a_later_turn_sees_later_data(self):
        self.assertEqual(compute_values(self.sc("2026-09-30T08:30:00"))["as_of"], "2026-09-30")

    def test_strap_off_gap_falls_back_to_garmin_and_the_note_says_so(self):
        sc = self.sc("2026-03-10T09:00:00")
        v = compute_values(sc)
        self.assertEqual(v["as_of"], "2026-03-10")
        self.assertEqual(v["last"]["sleep_duration"]["label"], "Garmin")
        self.assertEqual(v["last"]["sleep_duration"]["skipped"], ["whoop"])
        res = resolve(self.domain, sc, self.store())
        self.assertTrue(any("WHOOP has no sleep duration for last night; using Garmin per your settings" in n.outcome for n in res.notes))
        self.assertNotIn("recovery", v)           # recovery is WHOOP-only

    def test_over_threshold_night_becomes_a_resolution_note(self):
        res = resolve(self.domain, self.sc("2026-09-10T08:30:00"), self.store())
        text = " ".join(n.outcome for n in res.notes)
        self.assertIn("Sources disagree on last night's sleep by 55 minutes; using WHOOP per your settings", text)

    def test_under_threshold_night_has_no_disagreement_note(self):
        res = resolve(self.domain, self.sc("2026-09-29T08:10:00"), self.store())      # 35 minutes
        self.assertFalse(any(n.topic.startswith("sleep-disagreement") for n in res.notes))

    def test_disagreement_is_never_resolved_by_averaging(self):
        v = compute_values(self.sc("2026-09-10T08:30:00"))
        self.assertEqual(v["last"]["sleep_duration"]["value"], v["sources"]["whoop"]["asleep_min"])
        self.assertNotEqual(v["last"]["sleep_duration"]["value"], (v["sources"]["whoop"]["asleep_min"] + v["sources"]["garmin"]["asleep_min"]) / 2)

    def test_clinical_change_regime_is_marked_automatically_and_noted(self):
        regs = load_regimes(self.dd)
        clin = [r for r in regs if r["kind"] == "clinical_change"]
        self.assertEqual((clin[0]["start"], clin[0]["end"], clin[0]["origin"]), ("2026-02-01", "2026-02-28", "derived_local"))
        res = resolve(self.domain, self.sc("2026-09-29T08:10:00"), self.store())
        self.assertTrue(any(n.topic == "regime:clinical" and "do not read the shift as a fitness change" in n.outcome for n in res.notes))
        v = compute_values(self.sc("2026-09-29T08:10:00"))
        self.assertEqual(v["clinical_regime"]["month"], "February 2026")
        self.assertIn("2026-02-14", v["regime_days_with_data"])

    def test_resting_hr_steps_down_about_six_bpm_after_the_change(self):
        series = _series()
        w = [r for r in series["nights"] if r["source"] == "whoop"]
        import statistics
        before = statistics.mean(r["resting_hr_bpm"] for r in w if "2025-10-01" <= r["date"] < "2026-02-01")
        after = statistics.mean(r["resting_hr_bpm"] for r in w if "2026-03-01" <= r["date"] < "2026-07-01")
        self.assertTrue(4.5 <= before - after <= 7.5, before - after)

    def test_travel_regime_is_noted_and_excluded(self):
        dd = data_copy()
        add_regime(dd, "2026-05-10", "2026-05-17", "travel", "New York")
        domain = build_domain(dd)
        sc = domain.build_scenario(datetime.fromisoformat("2026-05-20T08:10:00"))
        res = resolve(domain, sc, MemoryStore.load(dd / "memory.json"))
        self.assertTrue(any(n.topic.startswith("regime:travel") for n in res.notes))
        v = compute_values(sc)
        self.assertEqual(len([d for d in v["regime_days_with_data"] if d.startswith("2026-05")]), 8)

    def test_red_eye_night_and_the_night_after_are_distinct_local_dates(self):
        series = _series()
        dates = sorted(r["date"] for r in series["nights"] if r["source"] == "whoop" and "2026-05-08" <= r["date"] <= "2026-05-12")
        self.assertEqual(dates, ["2026-05-08", "2026-05-09", "2026-05-10", "2026-05-11", "2026-05-12"])

    def test_sensor_beats_self_report_names_the_source(self):
        res = resolve(self.domain, self.sc("2026-09-29T08:10:00", "I slept 8 hours, feeling fine"), self.store())
        note = next(r for r in res.resolutions if r.kind == "measured_vs_recorded")
        self.assertIn("WHOOP measured 4h40", note.outcome)
        self.assertIn("the measurement stands", note.outcome)
        self.assertTrue(note.about_incoming)

    def test_self_report_close_to_the_sensor_is_left_alone(self):
        res = resolve(self.domain, self.sc("2026-09-29T08:10:00", "I slept about 5 hours"), self.store())
        self.assertFalse(any(r.kind == "measured_vs_recorded" for r in res.resolutions))

    # ---- provisional changes on volatile days
    def test_seeded_statement_on_a_volatile_day_is_provisional(self):
        res = resolve(self.domain, self.sc("2026-09-29T08:10:00"), self.store())
        prov = next(r for r in res.resolutions if r.kind == "provisional")
        self.assertEqual(prov.loser_ids, ["r01"])
        self.assertIn("provisional", prov.outcome)
        self.assertIn("confirm on a later day", prov.outcome)
        self.assertFalse(any(op.op == "supersede" and op.target_id == "r01" for op in res.memory_ops))
        self.assertEqual([p.id for p in res.plans_active], ["r02"])         # wins for the turn

    def test_a_new_change_on_a_volatile_day_writes_an_open_loop_and_does_not_supersede(self):
        store = MemoryStore([r for r in self.store().records if r.id in ("r01", "r04", "r05")])
        sc = self.sc("2026-09-29T20:00:00", "I'm done with racing. Feeling wrecked.")
        res = resolve(self.domain, sc, store)
        adds = [op for op in res.memory_ops if op.op == "add"]
        self.assertEqual({op.record.type for op in adds}, {"plan", "open_loop"})
        loop = next(op.record for op in adds if op.record.type == "open_loop")
        self.assertTrue(loop.content.startswith("Confirm: you changed 'Run the SF half in April 2027' to "))
        self.assertIn("2026-09-29", loop.content)
        plan = next(op.record for op in adds if op.record.type == "plan")
        self.assertEqual(plan.extra["provisional_for"], "r01")
        self.assertFalse(any(op.op == "supersede" and op.target_id == "r01" for op in res.memory_ops))

    def test_each_volatility_reason_triggers_it(self):
        base = [r for r in self.store().records if r.id in ("r01", "r04", "r05")]
        for ts, text, why in (("2026-09-29T20:00:00", "forget the half, I'm done racing", "recovery 21% today"),
                              ("2026-09-25T20:00:00", "forget the half, feeling exhausted", "crash language"),):
            res = resolve(self.domain, self.sc(ts, text), MemoryStore([MemoryRecord(**vars(r)) for r in base]))
            prov = next(r for r in res.resolutions if r.kind == "provisional")
            self.assertIn(why, prov.evidence, (ts, text))

    def test_same_day_contradiction_counts(self):
        from dataclasses import replace
        from harness.domain import Turn
        sc = self.sc("2026-09-25T20:00:00", "forget the half, I'm wrecked, no more racing")
        sc = replace(sc, history=sc.history + [Turn("2026-09-25T09:00:00", "user", "feeling great and strong today", 9)])
        store = MemoryStore([r for r in self.store().records if r.id in ("r01", "r04", "r05")])
        res = resolve(self.domain, sc, store)
        prov = next(r for r in res.resolutions if r.kind == "provisional")
        self.assertIn("same-day contradiction", prov.evidence)

    def test_a_change_on_an_ordinary_day_supersedes_immediately(self):
        store = MemoryStore([r for r in self.store().records if r.id in ("r01", "r04", "r05")])
        sc = self.sc("2026-09-24T09:00:00", "no more racing for me, I'm switching to trail running")
        v = compute_values(sc)
        self.assertGreaterEqual(v["recovery"]["value"], 33)
        res = resolve(self.domain, sc, store)
        self.assertTrue(any(op.op == "supersede" and op.target_id == "r01" for op in res.memory_ops))
        self.assertFalse(any(r.kind == "provisional" for r in res.resolutions))

    def test_confirmation_on_a_later_day_supersedes(self):
        store = self.store()
        res = resolve(self.domain, self.sc("2026-10-01T09:00:00", "Yes, confirmed, still the plan: no racing."), store)
        ops = [(op.op, op.target_id, op.by) for op in res.memory_ops]
        self.assertIn(("supersede", "r01", "r02"), ops)
        self.assertIn(("supersede", "r03", None), ops)
        applied = store.apply(res.memory_ops, datetime.fromisoformat("2026-10-01T09:00:00"))
        self.assertEqual(store.get("r01").status, "superseded")
        self.assertEqual(store.get("r01").superseded_by, "r02")
        self.assertEqual(store.get("r03").status, "superseded")

    def test_confirmation_language_is_needed(self):
        store = self.store()
        res = resolve(self.domain, self.sc("2026-10-01T09:00:00", "how did I sleep?"), store)
        self.assertFalse(any(op.op == "supersede" and op.target_id == "r01" for op in res.memory_ops))
        self.assertTrue(any(r.kind == "provisional" for r in res.resolutions))

    def test_medication_question_overrides_the_no_reflexive_referral_preference_with_a_reason(self):
        res = resolve(self.domain, self.sc("2026-09-29T08:10:00", "can I take melatonin tonight?"), self.store())
        note = next(r for r in res.notes if r.topic == "preference-override:medication")
        self.assertIn("overridden for medication questions", note.outcome)
        self.assertIn("prescriber", note.outcome)
        self.assertIn("by design", note.rule)

    def test_no_override_note_for_other_questions(self):
        res = resolve(self.domain, self.sc("2026-09-29T08:10:00", "how did I sleep?"), self.store())
        self.assertFalse(any(r.topic == "preference-override:medication" for r in res.notes))


if __name__ == "__main__":
    unittest.main()
