"""Stub-mode tests: stdlib unittest, no network. Run with `python3 -m unittest discover tests`."""
import sys
import types
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import harness.turn as turn_mod
from domains.lab import DOMAIN
from harness.context import Candidate, _order, assemble, assemble_with_note, build_candidates, render, required_floor
from harness.domain import Turn
from harness.memory import MemoryStore
from harness.model import ModelResult, call_model
from harness.resolve import resolve
from harness.router import TIERS, Route, escalate
from harness.safety import pre_check
from harness.tokens import count_tokens
from harness.tools import RECALL_TOOL, make_executor
from harness.turn import run_turn
from providers.anthropic import build_request, send

BUDGET = 1050


def fresh(now=None, message=None):
    sc = DOMAIN.build_scenario(now)
    if message:
        sc = replace(sc, incoming=Turn(sc.incoming.ts, "user", message, sc.incoming.session))
    return sc, MemoryStore.load(DOMAIN.seed_path)


def prompt_inputs(sc, store):
    pinned = [r for r in store.active() if r.pinned]
    pre = pre_check(DOMAIN, sc.incoming.text, pinned)
    res = resolve(DOMAIN, sc, store)
    ranked = store.read(sc.incoming.text, sc.now, DOMAIN.query_synonyms)
    return pre, res, build_candidates(DOMAIN, sc, ranked, res, pre)


class MemoryAndResolutionTests(unittest.TestCase):
    def test_scenario_filters_series_by_date(self):
        sc, _ = fresh()
        self.assertEqual(max(r["date"] for r in sc.series["readouts"]), "2026-09-28")
        later, _ = fresh(datetime.fromisoformat("2026-09-30T08:30:00"))
        self.assertEqual(max(r["date"] for r in later.series["shaker"]), "2026-09-29")

    def test_stale_plan_superseded_and_tension_surfaced(self):
        sc, store = fresh()
        res = resolve(DOMAIN, sc, store)
        plan = next(r for r in res.resolutions if r.topic == "plan:expression")
        self.assertEqual(plan.winner_id, "r07")
        self.assertEqual(plan.loser_ids, ["r01"])
        self.assertEqual(len(res.tensions), 1)
        t = res.tensions[0]
        for word in ("37", "induction"):                  # the user's own words
            self.assertIn(word, t.hits)
            self.assertIn(word, t.text)
        self.assertIn("2026-09-26", t.text)
        self.assertEqual((t.old.id, t.new.id), ("r01", "r07"))
        self.assertNotIn("r01", {p.id for p in res.plans_active})

    def test_no_tension_when_only_one_word_overlaps(self):
        sc, store = fresh(message="Setting up the 18°C induction tonight, anything I should watch?")
        self.assertEqual(resolve(DOMAIN, sc, store).tensions, [])

    def test_preference_reversal_latest_wins(self):
        sc, store = fresh()
        ids = {p.id for p in resolve(DOMAIN, sc, store).prefs_active}
        self.assertIn("r06", ids)
        self.assertNotIn("r02", ids)

    def test_same_day_conflict_excluded_from_trend(self):
        sc, store = fresh()
        res = resolve(DOMAIN, sc, store)
        self.assertIn("2026-09-26", res.excluded)
        self.assertEqual(res.values["yield_excluded"], ["2026-09-26"])
        self.assertEqual(res.values["yield_n"], 3)          # five preps, one hidden by date, one excluded

    def test_measured_beats_recorded(self):
        sc, store = fresh()
        res = resolve(DOMAIN, sc, store)
        note = next(r for r in res.resolutions if r.topic == "concentration:2026-09-24")
        self.assertIsNone(note.winner_id)
        self.assertIn("0.6 mg/mL", note.outcome)
        self.assertIn("kept as what was recorded", note.outcome)

    def test_read_path_breakdown_and_pinned_floor(self):
        sc, store = fresh()
        ranked = store.read(sc.incoming.text, sc.now, DOMAIN.query_synonyms)
        by_id = {m["id"]: m for m in ranked}
        for key in ("relevance", "recency", "type_priority", "score"):
            self.assertIn(key, by_id["r12"])
        self.assertGreaterEqual(by_id["r04"]["score"], 0.75 * (0.6 + 0.4 * 1.0) - 1e-6)
        self.assertGreater(by_id["r12"]["relevance"], 0.9)   # old fact, found by relevance to a 37 C plan

    def test_write_path_reinforces_dedupes_and_never_deletes(self):
        sc, store = fresh()
        before = len(store.records)
        run_turn(DOMAIN, sc, store, BUDGET)
        self.assertEqual(store.get("r01").status, "superseded")
        self.assertEqual(store.get("r01").superseded_by, "r07")
        self.assertEqual(store.get("r02").status, "superseded")
        self.assertGreater(store.get("r05").confidence, 0.6)
        self.assertGreater(len(store.records), before)       # records are added, none removed
        self.assertTrue(any(r.type == "open_loop" for r in store.active()))
        self.assertEqual(store.conflicts(), {})


class AssemblyTests(unittest.TestCase):
    def test_budget_is_hard_at_four_sizes(self):
        for b in (600, 900, 1500, 4000):
            sc, store = fresh()
            out = run_turn(DOMAIN, sc, store, b)
            self.assertLessEqual(out.assembly.used, b, f"budget {b} exceeded")
            self.assertLessEqual(count_tokens(out.system) + count_tokens(out.prompt), b,
                                 f"rendered prompt over budget {b}")

    def test_required_candidates_degrade_but_never_drop(self):
        sc, store = fresh()
        _, _, cands = prompt_inputs(sc, store)
        floor = required_floor(cands)
        tight = assemble(cands, floor + 3)
        kept = {d["key"]: d["kept"] for d in tight.decisions}
        for c in cands:
            if c.required:
                self.assertNotEqual(kept[c.key], "DROPPED", c.key)
        self.assertEqual(kept["readout_summary"], "one_line")
        self.assertEqual(kept["plans_prefs"], "compact")
        with self.assertRaises(ValueError):
            assemble(cands, floor - 1)

    def test_degrades_before_dropping_and_records_costs(self):
        sc, store = fresh()
        out = run_turn(DOMAIN, sc, store, BUDGET)
        kept = {d["key"]: d for d in out.assembly.decisions}
        self.assertEqual(kept["raw_table"]["kept"], "5_rows")              # one degrade
        self.assertEqual(kept["memory:r10"]["kept"], "DROPPED")            # one drop
        self.assertEqual(set(kept["memory:r10"]["variant_costs"]), {"full"})
        self.assertTrue(all(d["why"] for d in out.assembly.decisions))

    def test_score_per_token_ordering_within_a_tier(self):
        words = lambda n: " ".join(["word"] * n)
        big = Candidate("memory:big", "memory", 3, [("full", words(60))], score=0.60)
        cheap = Candidate("memory:cheap", "memory", 3, [("full", words(6))], score=0.40)
        mid = Candidate("memory:mid", "memory", 3, [("full", words(20))], score=0.45)
        req = Candidate("sys", "system", 0, [("v", "x")], required=True)
        self.assertEqual([c.key for c in _order([big, cheap, mid, req])],
                         ["sys", "memory:cheap", "memory:mid", "memory:big"])
        # room for the cheap one only: the valuable-but-large item must not block it
        room = count_tokens("x") + count_tokens(words(6)) + 40
        asm = assemble([req, big, cheap], room)
        self.assertEqual([p.candidate.key for p in asm.placed], ["sys", "memory:cheap"])

    def test_sensitive_gate_closed_and_open(self):
        sc, store = fresh(message="Best way to label 200 tubes quickly?")
        closed = run_turn(DOMAIN, sc, store, BUDGET)
        self.assertIn("prefer precast gels and TCEP", closed.prompt)
        self.assertNotIn("Confidential occupational-health", closed.prompt)
        self.assertEqual(closed.assembly.gates, [{"id": "r04", "gate": "closed", "shipped": "constraint"}])
        sc, store = fresh(message="Should I pour my own acrylamide gels to save money?")
        opened = run_turn(DOMAIN, sc, store, BUDGET)
        self.assertIn("Confidential occupational-health", opened.prompt)
        self.assertEqual(opened.assembly.gates, [{"id": "r04", "gate": "open", "shipped": "full"}])
        self.assertIn("gate", opened.trace.to_markdown())

    def test_stub_never_acknowledges_a_closed_gate(self):
        forbid = MemoryStore.load(DOMAIN.seed_path).get("r04").extra["forbid"]
        for message in (None, "Gel looks rough, I'll just push through it tonight."):
            sc, store = fresh(message=message)
            out = run_turn(DOMAIN, sc, store, BUDGET)
            self.assertEqual(out.assembly.gates[0]["gate"], "closed")
            self.assertTrue(out.pre.risk_flags)
            low = out.response.lower()
            self.assertNotIn("standing constraint", low)
            self.assertNotIn("accommodation", low)
            self.assertFalse([t for t in forbid if t.lower() in low])
            self.assertFalse(any("r04" in r for r in out.pre.risk_reasons))
        sc, store = fresh(message="Should I pour my own acrylamide gels to save money?")
        opened = run_turn(DOMAIN, sc, store, BUDGET)
        self.assertEqual(opened.assembly.gates[0]["gate"], "open")
        self.assertTrue(opened.response.strip())
        self.assertFalse([t for t in forbid if t.lower() in opened.response.lower()])

    def test_missing_context_note_when_tier3_memory_drops(self):
        sc, store = fresh()
        _, _, cands = prompt_inputs(sc, store)
        asm = assemble_with_note(cands, required_floor(cands) + 60)
        dropped = [d["key"] for d in asm.decisions if d["kept"] == "DROPPED" and d["key"].startswith("memory:")
                   and d["tier"] <= 3]
        self.assertTrue(dropped)
        self.assertIn("relevant record(s) were left out for budget; call recall if you need them.", asm.note)
        for key in dropped:
            self.assertIn(key.split(":")[1], asm.note)
        self.assertIn(asm.note, render(asm)[1])
        self.assertLessEqual(asm.used, required_floor(cands) + 60)
        sc, store = fresh()
        roomy = run_turn(DOMAIN, sc, store, 4000)
        self.assertEqual(roomy.assembly.note, "")


class RecallToolTests(unittest.TestCase):
    def test_schema(self):
        self.assertEqual(RECALL_TOOL["name"], "recall")
        self.assertIs(RECALL_TOOL["strict"], True)
        self.assertEqual(RECALL_TOOL["input_schema"]["required"], [])
        self.assertIs(RECALL_TOOL["input_schema"]["additionalProperties"], False)
        self.assertEqual(set(RECALL_TOOL["input_schema"]["properties"]), {"query", "ids"})

    def _tight_turn(self):
        sc, store = fresh()
        _, _, cands = prompt_inputs(sc, store)
        asm = assemble_with_note(cands, required_floor(cands) + 60)
        return sc, store, asm

    def test_returns_dropped_records_and_logs_calls(self):
        sc, store, asm = self._tight_turn()
        dropped = {d["key"].split(":")[1] for d in asm.decisions if d["kept"] == "DROPPED"
                   and d["key"].startswith("memory:") and d["tier"] <= 3}
        self.assertTrue(dropped)
        log = []
        run = make_executor(store, DOMAIN, sc.now, [], asm.placed_ids() | {"r01", "r02"}, log)
        text = run("recall", {"query": "shaker setpoint hot room"})
        self.assertIn("[r12] (fact, 2026-08-18)", text)
        self.assertEqual(log[0]["query"], "shaker setpoint hot room")
        self.assertIn("r12", log[0]["ids_returned"])
        self.assertLessEqual(len(log[0]["ids_returned"]), 3)
        self.assertNotIn("r01", log[0]["ids_returned"])
        self.assertIn("[r11]", run("recall", {"ids": ["r11"]}))

    def test_never_returns_sensitive_content_when_gate_closed(self):
        sc, store, _ = self._tight_turn()
        closed = make_executor(store, DOMAIN, sc.now, [], set(), [])("recall", {"ids": ["r04"]})
        self.assertIn("prefer precast gels and TCEP", closed)
        self.assertNotIn("Confidential", closed)
        by_query = make_executor(store, DOMAIN, sc.now, [], set(), [])("recall", {"query": "reproductive toxicants handling"})
        self.assertNotIn("Confidential", by_query)
        opened = make_executor(store, DOMAIN, sc.now, ["reagent_question"], set(), [])("recall", {"ids": ["r04"]})
        self.assertIn("Confidential", opened)


class SafetyAndRoutingTests(unittest.TestCase):
    def test_emergency_path_skips_the_model(self):
        def boom(*a, **k):
            raise AssertionError("model must not be called on the emergency path")

        orig, turn_mod.call_model = turn_mod.call_model, boom
        try:
            sc, store = fresh(message="I splashed BME on my arm.")
            out = run_turn(DOMAIN, sc, store, BUDGET)
        finally:
            turn_mod.call_model = orig
        self.assertEqual(out.pre.level, "emergency")
        self.assertFalse(out.model_called)
        self.assertEqual(out.response, DOMAIN.emergency_message)
        self.assertEqual(out.trace.sections["model_call"]["skipped"], "model skipped: emergency path")
        esc = out.trace.sections["escalation"]
        self.assertEqual(esc["contact"], "lab safety officer")
        self.assertIn("prefer precast gels and TCEP", " ".join(esc["pinned_safe_text"]))
        self.assertNotIn("Confidential", str(esc))
        self.assertIn("model skipped: emergency path", out.trace.to_markdown())

    def test_failing_post_check_falls_back_to_the_template(self):
        calls = []

        def fake(system, prompt, route, live=False, stub=None, executor=None, **_):
            calls.append(route.tier)
            return ModelResult("Go for it, that should be fine.", "stub", 1, route.model)

        orig, turn_mod.call_model = turn_mod.call_model, fake
        try:
            sc, store = fresh()
            out = run_turn(DOMAIN, sc, store, BUDGET)
        finally:
            turn_mod.call_model = orig
        self.assertEqual(len(calls), 2)                        # one retry, escalated
        self.assertTrue(out.fallback_used)
        self.assertIn("can't advise on this one right now", out.response)
        self.assertEqual(out.response.count("?"), 1)
        self.assertNotIn("go for it", out.response.lower())
        self.assertTrue(out.trace.sections["fallback"]["used"])
        self.assertIn("## Fallback", out.trace.to_markdown())

    def test_refusal_goes_straight_to_the_fallback(self):
        calls = []

        def fake(system, prompt, route, live=False, stub=None, executor=None, **_):
            calls.append(1)
            return ModelResult("", "live", 1, route.model, stop="refusal")

        orig, turn_mod.call_model = turn_mod.call_model, fake
        try:
            sc, store = fresh()
            out = run_turn(DOMAIN, sc, store, BUDGET)
        finally:
            turn_mod.call_model = orig
        self.assertEqual(len(calls), 1)
        self.assertTrue(out.fallback_used)
        self.assertEqual(out.trace.sections["fallback"]["reason"], "model refused the turn")

    def test_post_check_catches_forbidden_terms(self):
        from harness.safety import post_check
        pre = pre_check(DOMAIN, "hello", [])
        bad = post_check(DOMAIN, "Try DMF as the solvent.", pre, [], ["dmf", "acrylamide"])
        self.assertFalse(bad.checks["no_forbidden_recommendation"])
        ok = post_check(DOMAIN, "Try a precast gel.", pre, [], ["dmf", "acrylamide"])
        self.assertTrue(ok.passed)

    def test_forbidden_check_is_per_sentence_and_allows_avoidance(self):
        from harness.safety import post_check
        pre = pre_check(DOMAIN, "hello", [])
        forbid = ["acrylamide", "dmf"]
        cases = [
            ("I'd avoid acrylamide; precast gels instead.", True),
            ("I\u2019d avoid acrylamide; precast gels instead.", True),
            ("Pour your own acrylamide gels, it's cheaper.", False),
            ("Skip DMF here. Pour your own acrylamide gels, it's cheaper.", False),   # cue is in another sentence
            ("Pour your own acrylamide gels, it's cheaper. Precast gels are the alternative.", False),
            ("Rather than acrylamide, buy a precast gel.", True),
            ("Use 0.2 mM of it. Nothing about forbidden things.", True),
        ]
        for text, ok in cases:
            with self.subTest(text=text):
                got = post_check(DOMAIN, text, pre, [], forbid).checks["no_forbidden_recommendation"]
                self.assertEqual(got, ok)

    def test_routing_tiers_and_escalation(self):
        sc, store = fresh()
        self.assertEqual(run_turn(DOMAIN, sc, store, BUDGET).route.tier, "large")
        sc, store = fresh(message="Best way to label 200 tubes quickly?")
        self.assertEqual(run_turn(DOMAIN, sc, store, BUDGET).route.tier, "small")
        r = Route("small", TIERS["small"]["model"], None)
        self.assertEqual(escalate(r, "x").tier, "mid")
        top = escalate(escalate(escalate(r, "x"), "y"), "z")
        self.assertEqual((top.tier, top.effort), ("large", "high"))


class SafetyRegexTests(unittest.TestCase):
    def test_emergency_patterns_need_a_body_or_person(self):
        pinned = [r for r in MemoryStore.load(DOMAIN.seed_path).active() if r.pinned]
        cases = [
            ("Band looks burnt", False),
            ("samples were exposed to room temp", False),
            ("burned through the DTT stock", False),
            ("the gel burned", False),
            ("I splashed BME on my arm", True),
            ("got some in my eye", True),
            ("I think I inhaled some fumes", True),
            ("burned my hand on the autoclave", True),
            ("some acid spilled on my hands", True),
            ("I was exposed to it", True),
            ("burned through my DTT", False),
            ("I burnt my wrist on the hot plate", True),
            ("got some BME on my skin", True),
            ("avoid skin contact", False),
        ]
        for text, emergency in cases:
            with self.subTest(text=text):
                self.assertEqual(pre_check(DOMAIN, text, pinned).level == "emergency", emergency)

    def test_negation_silences_push_through_and_temperature(self):
        pinned = [r for r in MemoryStore.load(DOMAIN.seed_path).active() if r.pinned]
        cases = [
            ("I will NOT push through at 37C", False),
            ("won't go to 37\u00b0C", False),
            ("I never go above 37\u00b0C", False),
            ("I'm no longer going to 37C", False),
            ("going to push through at 37C", True),
            ("going to do 37\u00b0C tonight", True),
            ("not sure, but I'll just run it anyway", False),
        ]
        for text, risk in cases:
            with self.subTest(text=text):
                self.assertEqual("protocol_risk" in pre_check(DOMAIN, text, pinned).flags, risk)


class GatedEpisodeTests(unittest.TestCase):
    FIRST = "Should I pour my own acrylamide gels to save money?"
    SECOND = "Best way to label 200 tubes quickly?"

    def _two_turns(self):
        sc, store = fresh(message=self.FIRST)
        run_turn(DOMAIN, sc, store, BUDGET)
        sc2, _ = fresh(datetime.fromisoformat("2026-09-29T09:00:00"), self.SECOND)
        return store, run_turn(DOMAIN, sc2, store, BUDGET)

    def test_episode_of_a_gated_turn_is_stored_sensitive(self):
        store, _ = self._two_turns()
        ep = next(r for r in store.records if r.type == "episode" and "acrylamide" in r.content)
        self.assertEqual(ep.sensitivity, "high")
        self.assertEqual(ep.extra["reveal_on"], ["reagent_question"])
        self.assertEqual(ep.extra["constraint"], "Earlier turn on a gated topic; details withheld.")
        plain = [r for r in store.records if r.type == "episode" and r.id != ep.id and r.created_at > "2026-09-29T09"]
        self.assertTrue(all(r.sensitivity == "normal" for r in plain))   # the ungated turn stays ordinary

    def test_closed_gate_never_replays_the_gated_episode(self):
        store, out = self._two_turns()
        outside_pinned = "\n".join(l for l in out.prompt.splitlines() if "prefer precast gels and TCEP" not in l)
        self.assertNotIn("acrylamide", outside_pinned.lower())
        self.assertNotIn("save money", out.prompt)
        self.assertIn("Earlier turn on a gated topic; details withheld.", out.prompt)
        gates = {g["id"]: g for g in out.assembly.gates}
        self.assertEqual(gates["r04"]["gate"], "closed")
        ep_id = next(r.id for r in store.records if r.type == "episode" and "acrylamide" in r.content)
        self.assertEqual(gates[ep_id], {"id": ep_id, "gate": "closed", "shipped": "constraint"})
        self.assertNotIn("acrylamide", out.response.lower())

    def test_closed_gated_episode_is_optional_and_drops_first(self):
        sc, store = fresh(message=self.FIRST)
        run_turn(DOMAIN, sc, store, BUDGET)
        sc2, _ = fresh(datetime.fromisoformat("2026-09-29T09:00:00"), self.SECOND)
        pre, res, cands = prompt_inputs(sc2, store)
        cand = next(c for c in cands if c.meta.get("gate") == "closed" and c.key.startswith("memory:"))
        self.assertFalse(cand.required)
        self.assertEqual((cand.tier, [v for v, _ in cand.variants]), (5, ["constraint"]))
        asm = assemble(cands, required_floor(cands))
        self.assertEqual({d["key"] for d in asm.decisions if d["kept"] == "DROPPED"} >= {cand.key}, True)

    def test_open_gate_ranks_the_episode_like_any_memory(self):
        store, _ = self._two_turns()
        sc3, _ = fresh(datetime.fromisoformat("2026-09-29T10:00:00"), "Is acrylamide really that bad for gels?")
        out = run_turn(DOMAIN, sc3, store, 4000)
        self.assertIn("save money", out.prompt)
        self.assertTrue(any(g["gate"] == "open" for g in out.assembly.gates if g["id"] != "r04"))

    def test_recall_covers_non_pinned_sensitive_records(self):
        store, _ = self._two_turns()
        ep = next(r for r in store.records if r.type == "episode" and "acrylamide" in r.content)
        sc, _ = fresh()
        closed = make_executor(store, DOMAIN, sc.now, [], set(), [])
        by_id = closed("recall", {"ids": [ep.id]})
        by_query = closed("recall", {"query": "pour my own acrylamide gels save money"})
        for text in (by_id, by_query):
            self.assertNotIn("acrylamide gels", text)
            self.assertNotIn("save money", text)
        self.assertIn("details withheld", by_id)
        opened = make_executor(store, DOMAIN, sc.now, ["reagent_question"], set(), [])("recall", {"ids": [ep.id]})
        self.assertIn("save money", opened)

    def test_cli_repro_two_runs_through_memory_file(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            r1, r2 = Path(tmp, "r1"), Path(tmp, "r2")
            run = lambda *a: subprocess.run([sys.executable, str(ROOT / "run.py"), "--stub", *a],
                                            capture_output=True, text=True, check=True)
            run("--out", str(r1), "--message", self.FIRST)
            run("--memory", str(r1 / "memory_after.json"), "--ts", "2026-09-29T09:00",
                "--message", self.SECOND, "--out", str(r2))
            prompt = (r2 / "prompt.txt").read_text()
        rest = "\n".join(l for l in prompt.splitlines() if "prefer precast gels and TCEP" not in l)
        self.assertNotIn("acrylamide", rest.lower())


class GroundingTests(unittest.TestCase):
    def test_numbers_are_extracted_with_units_ranges_and_decimals(self):
        from harness.safety import _numbers
        self.assertEqual(_numbers("37\u00b0C, 0.2 mM, residues 140-160, day 09"), {"37", "0.2", "140", "160", "9"})
        self.assertEqual(_numbers("no digits here"), set())

    def test_default_stub_reply_is_grounded(self):
        sc, store = fresh()
        out = run_turn(DOMAIN, sc, store, BUDGET)
        self.assertTrue(out.post.checks["numbers_grounded"])
        self.assertTrue(out.post.passed)
        self.assertFalse(out.fallback_used)

    def test_invented_numbers_fail_and_trigger_escalate_then_fallback(self):
        from harness.safety import _numbers
        calls = []

        def fake(system, prompt, route, live=False, stub=None, executor=None, **_):
            calls.append(route.tier)
            return ModelResult("Use 5 mM DTT at 42\u00b0C.", "stub", 1, route.model)

        orig, turn_mod.call_model = turn_mod.call_model, fake
        try:
            sc, store = fresh()
            out = run_turn(DOMAIN, sc, store, BUDGET)
        finally:
            turn_mod.call_model = orig
        known = _numbers(out.system + "\n" + out.prompt)
        self.assertFalse({"5", "42"} & known)                  # the premise: neither is in the prompt
        self.assertFalse(out.post.checks["numbers_grounded"])
        self.assertIn("numbers_grounded", out.post.notes)
        self.assertEqual(len(calls), 2)
        self.assertTrue(out.fallback_used)
        self.assertIn("can't advise on this one right now", out.response)


class RecallGroundingTests(unittest.TestCase):
    def test_numbers_from_a_recall_result_count_as_grounded(self):
        from harness.safety import _numbers
        sc, store = fresh()
        secret = "73"
        store.records.append(type(store.records[0])(
            id="r90", type="fact", content=f"Freezer B holds {secret} aliquots of the stock.",
            created_at="2026-08-01T09:00:00", updated_at="2026-08-01T09:00:00", source="user_stated",
            confidence=0.5, topic="fact:freezer"))
        asked = []

        def fake(system, prompt, route, live=False, stub=None, executor=None, **_):
            asked.append(executor("recall", {"ids": ["r90"]}))
            self.assertNotIn(secret, _numbers(system + "\n" + prompt))   # only the tool result has it
            return ModelResult(f"Freezer B has {secret} aliquots.", "live", 1, route.model)

        orig, turn_mod.call_model = turn_mod.call_model, fake
        try:
            out = run_turn(DOMAIN, sc, store, BUDGET)
        finally:
            turn_mod.call_model = orig
        self.assertIn(secret, asked[0])
        self.assertTrue(out.post.checks["numbers_grounded"])
        self.assertNotIn("numbers_grounded", out.post.notes)      # (other checks may still fail this reply)

        sc, store = fresh()                                       # same reply, no recall: ungrounded
        fake2 = lambda system, prompt, route, live=False, stub=None, executor=None, **_: ModelResult(
            f"Freezer B has {secret} aliquots.", "live", 1, route.model)
        turn_mod.call_model = fake2
        try:
            out2 = run_turn(DOMAIN, sc, store, BUDGET)
        finally:
            turn_mod.call_model = orig
        self.assertFalse(out2.post.checks["numbers_grounded"])


class StubReplyTests(unittest.TestCase):
    def test_risk_sentence_keeps_the_pinned_limit(self):
        sc, store = fresh()
        out = run_turn(DOMAIN, sc, store, BUDGET)
        self.assertIn("On file: construct v1 has an aggregation-prone loop (residues 140-160); "
                      "it aggregates when induced above 25\u00b0C. Staying under that is the safer call.", out.response)


class BudgetErrorTests(unittest.TestCase):
    def _run(self, *argv):
        import subprocess
        return subprocess.run([sys.executable, *argv], capture_output=True, text=True, cwd=ROOT)

    def test_library_still_raises_value_error(self):
        sc, store = fresh()
        with self.assertRaises(ValueError):
            run_turn(DOMAIN, sc, store, 10)

    def test_cli_prints_one_line_and_exits_2(self):
        import re
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run("run.py", "--stub", "--budget", "50", "--out", tmp)
        self.assertEqual(r.returncode, 2)
        self.assertNotIn("Traceback", r.stderr)
        lines = r.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1)
        m = re.fullmatch(r"budget 50 is below the required context floor \((\d+) tokens\); "
                         r"raise --budget to at least (\d+)", lines[0])
        self.assertIsNotNone(m)
        floor = int(m.group(1))
        self.assertEqual(m.group(1), m.group(2))
        sc, store = fresh()
        run_turn(DOMAIN, sc, store, floor)                      # the floor it names is enough
        sc, store = fresh()
        with self.assertRaises(ValueError):
            run_turn(DOMAIN, sc, store, floor - 1)

    def test_evals_cli_exits_2_on_a_tiny_case_budget(self):
        import evals.lab_cases as cases
        from evals.run import main
        orig = cases.CASES[0].budget
        cases.CASES[0].budget = 50
        try:
            import io
            from contextlib import redirect_stderr, redirect_stdout
            err = io.StringIO()
            sys.argv = ["evals.run", "--domain", "lab", "--only", cases.CASES[0].name]
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
        finally:
            cases.CASES[0].budget = orig
        self.assertRegex(err.getvalue(), rf"^case {cases.CASES[0].name} needs a budget of at least \d+ tokens "
                                         r"\(it was given 50\); raise the case's budget in evals/lab_cases.py\n$")


class ProactiveTests(unittest.TestCase):
    def test_second_turn_closes_the_open_loop(self):
        sc, store = fresh()
        run_turn(DOMAIN, sc, store, BUDGET)
        loop = next(r for r in store.active() if r.type == "open_loop")
        self.assertIn("Unanswered", loop.content)
        sc2, _ = fresh(datetime.fromisoformat("2026-09-29T09:00:00"), "18°C it is, forget the 37 plan.")
        out = run_turn(DOMAIN, sc2, store, BUDGET)
        self.assertEqual(store.get(loop.id).status, "superseded")
        self.assertEqual(store.conflicts().get("plan:expression"), None)
        self.assertTrue(any(w["op"] == "supersede" and w["type"] == "open_loop" for w in out.memory_ops))

    def test_scheduled_trigger_with_open_loop_routes_at_least_mid(self):
        sc, store = fresh()
        run_turn(DOMAIN, sc, store, BUDGET)
        sc2 = DOMAIN.build_scenario(datetime.fromisoformat("2026-09-30T08:30:00"))
        trig = DOMAIN.triggers["morning_brief"](sc2, sc2.now)
        out = run_turn(DOMAIN, sc2, store, BUDGET, trigger=trig)
        self.assertIn(out.route.tier, ("mid", "large"))
        self.assertTrue(any("open loop" in r for r in out.route.reasons))
        self.assertEqual(out.trace.sections["mode"]["kind"], "proactive:scheduled")
        self.assertIn("Still open", out.response)
        self.assertLessEqual(out.response.count("?"), 1)
        self.assertTrue(any(w["op"] == "add" and "assistant initiated" in w["content"] for w in out.memory_ops))

    def test_signal_trigger_fires_on_the_excursion_and_not_the_night_before(self):
        sc = DOMAIN.build_scenario(datetime.fromisoformat("2026-09-29T07:00:00"))
        trig = DOMAIN.triggers["temperature_excursion"](sc, sc.now)
        self.assertIsNotNone(trig)
        self.assertIn("21.6", trig.text)
        before = DOMAIN.build_scenario(datetime.fromisoformat("2026-09-28T07:00:00"))
        self.assertIsNone(DOMAIN.triggers["temperature_excursion"](before, before.now))
        _, store = fresh()
        out = run_turn(DOMAIN, sc, store, BUDGET, trigger=trig)
        self.assertIn("protocol_risk", out.pre.flags)
        self.assertIn(out.route.tier, ("mid", "large"))
        self.assertEqual(out.resolved.tensions, [])             # a trigger is not a user message


class LiveRequestTests(unittest.TestCase):
    def test_request_shapes_per_tier(self):
        reqs = {t: build_request("SYS", "hello", Route(t, TIERS[t]["model"], TIERS[t]["effort"]))
                for t in TIERS}
        for kw in reqs.values():
            self.assertEqual(kw["system"][0]["cache_control"], {"type": "ephemeral"})
            self.assertEqual(kw["tools"], [RECALL_TOOL])
            self.assertEqual(kw["tool_choice"], {"type": "auto"})
            self.assertEqual(kw["max_tokens"], 4096)
        small, mid, large = reqs["small"], reqs["mid"], reqs["large"]
        self.assertNotIn("thinking", small)
        self.assertNotIn("output_config", small)
        self.assertNotIn("extra_headers", small)
        self.assertEqual(mid["thinking"], {"type": "adaptive"})
        self.assertEqual(mid["output_config"], {"effort": "low"})
        self.assertNotIn("extra_headers", mid)
        self.assertEqual(large["output_config"], {"effort": "medium"})
        self.assertEqual(large["extra_headers"], {"anthropic-beta": "server-side-fallback-2026-07-01"})
        self.assertEqual(large["extra_body"]["fallbacks"], "default")

    def _fake_anthropic(self, replies):
        mod = types.ModuleType("anthropic")

        class APIStatusError(Exception):
            pass

        class APIConnectionError(Exception):
            pass

        seen = []

        class Messages:
            def create(self, **kw):
                seen.append({"messages": list(kw["messages"]), "kw": kw})
                out = replies.pop(0)
                if isinstance(out, Exception):
                    raise out
                return out

        class Client:
            messages = Messages()

        mod.APIStatusError, mod.APIConnectionError, mod.Anthropic = APIStatusError, APIConnectionError, Client
        mod.seen = seen
        return mod

    @staticmethod
    def _resp(stop, blocks, **usage):
        u = types.SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=3,
                                  cache_creation_input_tokens=None)
        return types.SimpleNamespace(stop_reason=stop, content=blocks, usage=u)

    def test_tool_loop_refusal_and_error_markers(self):
        T = lambda t: types.SimpleNamespace(type="text", text=t)
        U = lambda i: types.SimpleNamespace(type="tool_use", id=i, name="recall", input={"ids": ["r11"]})
        route = Route("large", TIERS["large"]["model"], "medium")
        calls = []
        execute = lambda name, inp: calls.append((name, inp)) or "[r11] found"

        fake = self._fake_anthropic([self._resp("tool_use", [T("thinking"), U("a"), U("b")]),
                                     self._resp("end_turn", [T("Final "), T("answer.")])])
        sys.modules["anthropic"] = fake
        try:
            out = call_model("S", "P", route, live=True, executor=execute, sender=send)
            self.assertEqual(out.text, "Final answer.")
            self.assertEqual(out.usage, {"input_tokens": 20, "output_tokens": 10,
                                         "cache_read_input_tokens": 6, "cache_creation_input_tokens": 0})
            self.assertEqual(len(calls), 2)
            second = fake.seen[1]["messages"]
            self.assertEqual(second[1]["role"], "assistant")
            self.assertEqual(len(second[1]["content"]), 3)               # full content list, not just text
            self.assertEqual(second[2]["role"], "user")
            self.assertEqual([b["tool_use_id"] for b in second[2]["content"]], ["a", "b"])  # one message
            fake2 = self._fake_anthropic([self._resp("refusal", [])])
            sys.modules["anthropic"] = fake2
            self.assertEqual(call_model("S", "P", route, live=True, executor=execute, sender=send).stop, "refusal")
            replies = []
            fake3 = self._fake_anthropic(replies)
            replies.append(fake3.APIConnectionError("down"))
            sys.modules["anthropic"] = fake3
            err = call_model("S", "P", route, live=True, executor=execute, sender=send)
            self.assertEqual(err.stop, "error")
            fake4 = self._fake_anthropic([self._resp("tool_use", [U("a")])] * 3)
            sys.modules["anthropic"] = fake4
            self.assertEqual(call_model("S", "P", route, live=True, executor=execute, sender=send).stop, "error")
            self.assertEqual(len(fake4.seen), 3)                         # at most three rounds
        finally:
            sys.modules.pop("anthropic", None)

    def test_failing_tool_reports_is_error(self):
        route = Route("small", TIERS["small"]["model"], None)
        U = types.SimpleNamespace(type="tool_use", id="a", name="recall", input={})
        fake = self._fake_anthropic([self._resp("tool_use", [U]),
                                     self._resp("end_turn", [types.SimpleNamespace(type="text", text="ok")])])
        sys.modules["anthropic"] = fake
        try:
            def broken(name, inp):
                raise RuntimeError("nope")
            call_model("S", "P", route, live=True, executor=broken, sender=send)
            block = fake.seen[1]["messages"][2]["content"][0]
            self.assertTrue(block["is_error"])
        finally:
            sys.modules.pop("anthropic", None)


class EvalSuiteTests(unittest.TestCase):
    def test_every_eval_case_passes_in_stub_mode(self):
        from evals.run import evaluate, run_lab_case as run_case
        from evals.lab_cases import CASES
        for case in CASES:
            failed = [c["check"] for c in evaluate(case, run_case(case)) if not c["passed"]]
            self.assertEqual(failed, [], case.name)


if __name__ == "__main__":
    unittest.main()
