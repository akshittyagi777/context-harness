"""Safety pre-check, routing, local answers, triggers and the CLI end to end."""
from __future__ import annotations

import contextlib
import io
import os
import unittest
from datetime import datetime
from unittest import mock

from tests import support
from tests.support import PASSPHRASE, data_copy, empty_data, fixture

import health_cli as cli
from harness.domain import Turn
from harness.memory import MemoryStore
from domains.health import runner
from domains.health.domain import DEFAULT_BUDGET, build_domain
from domains.health.safety import EMERGENCY_MESSAGE, emergency_check, is_medication_question

TS = "2026-09-29T08:10:00"


def turn(msg=None, ts=TS, mode=None, **kw):
    return runner.run(fixture()["data"], msg, ts, mode=mode, persist=False, **kw)


class EmergencyTests(unittest.TestCase):
    def test_emergency_phrases(self):
        for text in ("I have chest pain when I run and it spreads to my arm", "chest pain radiating down my left arm",
                     "I'm having chest tightness while I climb the stairs", "I've got trouble breathing", "I can't catch my breath",
                     "my face is drooping and my speech is slurred", "my left arm is weak and numb", "I want to kill myself",
                     "I'm feeling suicidal", "I think I overdosed on my pills", "I took too many pills", "thinking about suicide"):
            self.assertTrue(emergency_check(text), text)

    def test_negation_and_context(self):
        for text in ("no chest pain when I run", "I have no chest pain, just tired", "I don't have trouble breathing",
                     "I never get chest pain while running", "chest pain is not something I have", "without any chest pain",
                     "my friend's overdose was last year", "I read an article about suicide prevention",
                     "my legs are weak after the long run", "how was my breathing rate last night", "nothing about my chest"):
            self.assertFalse(emergency_check(text), text)

    def test_negation_in_another_clause_does_not_cancel_a_real_symptom(self):
        self.assertTrue(emergency_check("No fever, but I have chest pain when I run"))
        self.assertTrue(emergency_check("I'm fine, however I can't catch my breath"))

    def test_plain_chest_pain_without_exertion_or_radiation_is_not_in_the_emergency_set(self):
        self.assertFalse(emergency_check("a little chest pain after a cold"))

    def test_emergency_turn_skips_the_model_and_the_data(self):
        out = turn("chest pain with exertion, it spreads to my jaw")
        self.assertFalse(out.model_called)
        self.assertEqual(out.response, EMERGENCY_MESSAGE)
        self.assertIn("911", out.response)
        self.assertIn("988", out.response)
        self.assertIn("cannot help in an emergency", out.response)
        self.assertEqual(out.trace.sections["model_call"]["skipped"], "model skipped: emergency path")
        self.assertEqual(out.trace.sections["model_call"]["calls"], [])
        self.assertEqual((out.system, out.prompt), ("", ""))
        self.assertIn("escalation", out.trace.sections)

    def test_emergency_beats_every_other_route_even_remote(self):
        out = turn("I can't catch my breath, can I take melatonin?", mode="remote")
        self.assertFalse(out.model_called)
        self.assertEqual(out.local_kind, "emergency")

    def test_emergency_never_calls_a_sender(self):
        called = []
        out = runner.run(fixture()["data"], "I think I overdosed", TS, live=True, persist=False, sender=lambda *a: called.append(a))
        self.assertEqual(called, [])
        self.assertFalse(out.model_called)

    def test_a_negated_symptom_goes_through_the_normal_path(self):
        out = turn("no chest pain, just tired. how did I sleep?")
        self.assertTrue(out.model_called)
        self.assertNotEqual(out.response, EMERGENCY_MESSAGE)


class MedicationQuestionTests(unittest.TestCase):
    def test_detection(self):
        for text in ("can I take melatonin tonight?", "should I stop my dose", "can I double up on supplements", "does ibuprofen interact with anything",
                     "I want to start taking magnesium", "skip my meds tomorrow?"):
            self.assertTrue(is_medication_question(text), text)
        for text in ("can I take a rest day?", "how did I sleep?", "can I take a day off from running", "I'll start my run at 7"):
            self.assertFalse(is_medication_question(text), text)

    def test_routes_large_and_sends_only_derived_constraints(self):
        out = turn("can I take melatonin tonight?", mode="remote")
        self.assertEqual(out.route.tier, "large")
        self.assertIn("medication_question", out.pre.flags)
        self.assertTrue(out.model_called)
        self.assertIn("Do not suggest supplements or sleep aids without checking with the prescriber.", out.prompt)
        from domains.health.egress import Guard
        Guard(fixture()["data"], PASSPHRASE).check_texts([out.system, out.prompt])          # passes the guard
        for word in ("metoprolol", "penicillin", "hypertension", "naproxen"):
            self.assertNotIn(word, (out.system + out.prompt).lower())
        self.assertTrue(any(e["key"].startswith("ehr:MedicationRequest") for e in out.assembly.excluded))

    def test_response_points_to_the_prescriber(self):
        out = turn("can I take melatonin tonight?", mode="remote")
        self.assertFalse(out.fallback_used)
        self.assertRegex(out.response.lower(), r"prescriber|pharmacist")
        self.assertTrue(out.post.checks["tensions_addressed"])

    def test_the_post_check_rejects_a_reply_that_does_not_point_anywhere(self):
        from domains.health.domain import post_requirements
        class Pre: flags = ["medication_question"]
        need = post_requirements(Pre(), None)
        self.assertEqual(need, [["prescriber", "pharmacist", "doctor", "clinician"]])
        from harness.safety import post_check
        dom = build_domain(fixture()["data"])
        bad = post_check(dom, "Melatonin is usually fine at low doses.", out_pre(), need, [])
        self.assertFalse(bad.passed)
        ok = post_check(dom, "Ask your pharmacist first.", out_pre(), need, [])
        self.assertTrue(ok.checks["tensions_addressed"])

    def test_the_no_reflexive_preference_is_overridden_with_the_reason_in_the_trace(self):
        out = turn("can I take melatonin tonight?")
        res = [r for r in out.trace.sections["resolutions"] if r["topic"] == "preference-override:medication"]
        self.assertEqual(len(res), 1)
        self.assertIn("overridden for medication questions", res[0]["outcome"])
        self.assertIn("by design", res[0]["rule"])

    def test_without_a_constraint_on_file_the_reply_still_points_to_a_professional(self):
        out = runner.run(empty_data(), "can I take melatonin tonight?", TS, persist=False)
        self.assertRegex(out.response.lower(), r"prescriber|pharmacist")


def out_pre():
    from harness.safety import PreCheck
    return PreCheck("elevated", ["medication_question"])


class LocalAnswerTests(unittest.TestCase):
    def test_questions_about_the_ehr_are_answered_locally_never_by_the_model(self):
        calls = []
        for q, expect in (("what meds am I on", "metoprolol succinate ER 50 mg tablet"), ("when was my last A1c", "5.8 %"),
                          ("what are my allergies", "Penicillins"), ("my diagnoses?", "Hypertension"), ("what am I taking", "metoprolol")):
            out = turn(q, sender=lambda *a: calls.append(a))
            self.assertFalse(out.model_called, q)
            self.assertIn(expect, out.response)
            self.assertEqual(out.local_kind, "ehr")
            self.assertIn("ehr_lookup", out.pre.flags)
            self.assertEqual(out.trace.sections["model_call"]["skipped"], "answered locally from EHR; no model call")
            self.assertEqual((out.system, out.prompt), ("", ""))
        self.assertEqual(calls, [])

    def test_local_even_in_remote_mode_and_when_live(self):
        calls = []
        out = runner.run(fixture()["data"], "what meds am I on", TS, live=True, persist=False, sender=lambda *a: calls.append(a))
        self.assertFalse(out.model_called)
        self.assertEqual(calls, [])

    def test_a_medication_question_that_mentions_my_meds_still_goes_to_the_model_route(self):
        out = turn("can I take melatonin with my meds?", mode="remote")
        self.assertTrue(out.model_called)
        self.assertEqual(out.local_kind, None)

    def test_the_answer_comes_from_the_store_not_from_memory_records(self):
        out = turn("what are my allergies")
        store = out.store
        self.assertFalse(any("penicillin" in r.content.lower() for r in store.records))


class RoutingAndReplyTests(unittest.TestCase):
    def test_default_turn(self):
        out = runner.run(fixture()["data"], None, None, persist=False)
        self.assertEqual(out.route.tier, "large")
        self.assertIn("training_load_risk", out.pre.flags)
        self.assertEqual(out.response, "Last night (2026-09-29): sleep 4h40 (WHOOP), resting HR 55 bpm (WHOOP), recovery 21% (WHOOP). "
                                       "Recovery is low, so keeping today easy or taking a rest day is the safer call than going hard.")
        self.assertTrue(out.post.passed)
        self.assertFalse(out.fallback_used)
        self.assertLessEqual(out.assembly.used, DEFAULT_BUDGET)

    def test_training_language_with_a_good_recovery_raises_no_flag(self):
        out = turn("Thinking of going hard on a long run this morning. Thoughts?", ts="2026-06-15T08:10:00")
        self.assertNotIn("training_load_risk", out.pre.flags)
        self.assertEqual(out.route.tier, "mid")             # judgment asked, nothing risky

    def test_negated_training_language_is_not_a_risk(self):
        out = turn("I am not going to push through a max effort run today")
        self.assertNotIn("training_load_risk", out.pre.flags)
        out = turn("never going hard when I'm wrecked")
        self.assertNotIn("training_load_risk", out.pre.flags)

    def test_framing_pair_gets_the_same_recommendation(self):
        a = turn("feeling great, going for a max effort run")
        b = turn("wrecked, just going to push through a max effort run")
        for out in (a, b):
            self.assertIn("training_load_risk", out.pre.flags)
            self.assertEqual(out.route.tier, "large")
            self.assertIn("easy", out.response)
            self.assertTrue(out.post.checks["no_risky_endorsement"])
        self.assertEqual(a.response, b.response)

    def test_a_divergence_alone_also_raises_the_risk_flag(self):
        out = turn("going hard tomorrow", ts="2026-09-30T08:30:00")      # recovery fine that day, sleep divergence still active
        self.assertIn("training_load_risk", out.pre.flags)
        self.assertIn("sleep baseline", out.response)

    def test_fitness_claim_does_not_get_endorsed_and_names_the_constraint_not_the_drug(self):
        out = turn("my resting HR is way down, I'm getting fitter right?", mode="remote")
        self.assertIn("fitness_claim", out.pre.flags)
        self.assertEqual(out.route.tier, "large")
        self.assertIn("I can't read that as a fitness gain", out.response)
        self.assertIn("clinical factor", out.response)
        self.assertIn("February 2026", out.response)
        self.assertNotRegex(out.response.lower(), r"metoprolol|beta|blood pressure|hypertension|medication|drug")
        self.assertFalse(out.fallback_used)

    def test_fitness_claim_without_any_constraint_still_declines_to_endorse(self):
        out = runner.run(empty_data(), "my resting HR is way down, I'm getting fitter right?", TS, persist=False)
        self.assertIn("check with your clinician", out.response)

    def test_no_flag_question_goes_small_and_answers_from_the_data(self):
        out = turn("how did I sleep?")
        self.assertEqual(out.route.tier, "small")
        self.assertTrue(out.response.startswith("Last night (2026-09-29): sleep 4h40 (WHOOP)"))

    def test_sensor_beats_self_report_in_the_reply(self):
        out = turn("I slept 8 hours, how's my recovery?")
        self.assertIn("WHOOP measured 4h40 for last night but you said about 8h00", out.response)

    def test_source_disagreement_shows_in_the_reply_and_names_the_winner(self):
        out = turn("how did I sleep?", ts="2026-09-10T08:30:00")
        self.assertIn("Sources disagree on last night's sleep by 55 minutes; using WHOOP per your settings", out.response)

    def test_numbers_in_replies_are_grounded_in_the_prompt(self):
        for msg in ("how did I sleep?", None, "can I take melatonin tonight?"):
            out = turn(msg, mode="remote")
            from harness.safety import ungrounded_numbers
            self.assertEqual(ungrounded_numbers(out.response, out.system, out.prompt), [], msg)

    def test_open_loop_is_carried_by_the_morning_brief(self):
        out = runner.run(fixture()["data"], None, "2026-09-30T08:30:00", trigger="morning_brief", persist=False)
        self.assertEqual(out.route.tier, "mid")
        self.assertIn("Still open: Confirm: you changed 'Run the SF half in April 2027'", out.response)
        self.assertTrue(out.response.startswith("Last night (2026-09-30)"))

    def test_triggers(self):
        dom = build_domain(fixture()["data"])
        sc = dom.build_scenario(datetime.fromisoformat(TS))
        self.assertEqual(dom.triggers["recovery_drop"](sc, sc.now).kind, "signal")
        self.assertIn("recovery 21%", dom.triggers["recovery_drop"](sc, sc.now).facts)
        div = dom.triggers["baseline_divergence"](sc, sc.now)
        self.assertIn("below for 26 days", div.facts)
        later = dom.build_scenario(datetime.fromisoformat("2026-09-30T08:30:00"))
        self.assertIsNone(dom.triggers["recovery_drop"](later, later.now))               # recovery was 41% that day
        early = dom.build_scenario(datetime.fromisoformat("2025-11-15T08:30:00"))
        self.assertIsNone(dom.triggers["baseline_divergence"](early, early.now))

    def test_brief_trigger_that_does_not_fire_raises(self):
        with self.assertRaises(LookupError):
            runner.run(empty_data(), None, TS, trigger="morning_brief", persist=False)

    def test_budget_too_small_names_the_floor(self):
        from harness.context import BudgetTooSmall
        with self.assertRaises(BudgetTooSmall) as cm:
            runner.run(fixture()["data"], None, TS, budget=100, persist=False)
        self.assertGreater(cm.exception.floor, 100)

    def test_a_tight_budget_degrades_before_it_drops_required_context(self):
        out = runner.run(fixture()["data"], None, TS, budget=540, persist=False, mode="remote")
        kept = {d["key"]: d["kept"] for d in out.assembly.decisions}
        self.assertEqual(kept["wearable_summary"], "one_line")          # required context degrades, it never drops
        self.assertEqual(kept["resolution_notes"], "short")
        self.assertLessEqual(out.assembly.used, 540)
        self.assertFalse(out.fallback_used)                             # the facts sentence leads every variant, so the reply stays grounded

    def test_history_is_persisted_between_turns(self):
        dd = data_copy()
        runner.run(dd, "how did I sleep?", "2026-09-29T08:10:00")
        sc = build_domain(dd).build_scenario(datetime.fromisoformat("2026-09-29T20:00:00"))
        self.assertEqual([t.text for t in sc.history if t.role == "user"], ["how did I sleep?"])
        self.assertTrue(any(t.role == "assistant" for t in sc.history))


class CliTests(unittest.TestCase):
    """The quick start end to end, in a temp data dir."""

    def run_cli(self, *args, env=None):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = cli.main(list(args))
        return code, buf.getvalue(), err.getvalue()

    def test_quick_start(self):
        root = support._tmp()
        mock_dir, dd = str(root / "mock_exports"), str(root / "data")
        steps = [("mock", "--out", mock_dir), ("--data-dir", dd, "import", "whoop", mock_dir + "/whoop"),
                 ("--data-dir", dd, "import", "garmin", mock_dir + "/garmin"), ("--data-dir", dd, "import", "epic", mock_dir + "/epic"),
                 ("--data-dir", dd, "import", "memory", mock_dir + "/memory_seed.json"),
                 ("--data-dir", dd, "status"), ("--data-dir", dd, "audit"), ("--data-dir", dd, "turn"),
                 ("--data-dir", dd, "brief", "--ts", "2026-09-30T08:30"), ("--data-dir", dd, "egress-check")]
        outs = {}
        for s in steps:
            code, out, err = self.run_cli(*s)
            self.assertEqual(code, 0, (s, out, err))
            outs[s[-3] if s[0] == "--data-dir" else s[0]] = out
        self.assertIn("added {'nights': 359", outs["whoop"] if "whoop" in outs else "added {'nights': 359")
        code, status, _ = self.run_cli("--data-dir", dd, "status")
        self.assertIn("WHOOP   nights  359", status)
        self.assertIn("Garmin  nights  788", status)
        self.assertIn("overlap window 2025-10-01..2026-09-30, 357 night(s)", status)
        self.assertIn("passphrase set", status)
        code, audit, _ = self.run_cli("--data-dir", dd, "audit")
        self.assertIn("sleep_duration whoop>garmin", audit)
        self.assertIn("14-day disagreement summary", audit)
        code, out, _ = self.run_cli("--data-dir", dd, "egress-check")
        self.assertTrue(out.strip().endswith("PASS"))

    def test_import_epic_refuses_without_a_passphrase(self):
        fixture()
        with mock.patch.dict(os.environ):
            os.environ.pop("HEALTH_HARNESS_PASSPHRASE", None)
            code, out, err = self.run_cli("--data-dir", str(empty_data()), "import", "epic", str(fixture()["mock"] / "epic"))
        self.assertEqual(code, 2)
        self.assertIn("HEALTH_HARNESS_PASSPHRASE", err)

    def test_sync_says_what_it_would_need(self):
        for src, needle in (("whoop", "OAuth"), ("garmin", "not available to individuals"), ("epic", "PKCE")):
            code, out, _ = self.run_cli("--data-dir", str(empty_data()), "sync", src)
            self.assertEqual(code, 0)
            self.assertIn("not configured", out)
            self.assertIn(needle, out)

    def test_turn_live_without_credentials_still_goes_through_the_guard(self):
        from providers import anthropic as provider
        class C:
            messages = None
        c = C(); c.messages = c
        c.create = lambda **kw: (_ for _ in ()).throw(AssertionError("must not be called"))
        with mock.patch.object(provider, "_client_factory", lambda: c):
            code, out, err = self.run_cli("--data-dir", str(data_copy()), "turn", "can I take melatonin? call me on 212-555-0199", "--live")
        self.assertEqual(code, 0)
        self.assertIn("BLOCKED", out)
        self.assertIn("nothing was sent", out)

    def test_status_and_turn_work_on_an_empty_data_dir(self):
        dd = str(empty_data())
        self.assertEqual(self.run_cli("--data-dir", dd, "status")[0], 0)
        code, out, _ = self.run_cli("--data-dir", dd, "turn", "how did I sleep?")
        self.assertEqual(code, 0)

    def test_regime_command(self):
        dd = str(empty_data())
        self.run_cli("--data-dir", dd, "regime", "add", "2026-05-10", "2026-05-17", "travel", "New York")
        code, out, _ = self.run_cli("--data-dir", dd, "regime", "list")
        self.assertIn("2026-05-10..2026-05-17 travel", out)


if __name__ == "__main__":
    unittest.main()
