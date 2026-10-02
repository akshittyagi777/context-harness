"""The core's generic extensions: origin, mode, sender, boundary hooks, local responders."""
from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from tests import support  # noqa: F401  (path and env setup)

from harness.context import Candidate, BudgetTooSmall, assemble, assemble_with_note, build_candidates, render
from harness.domain import Domain, LocalReply, Scenario, Turn
from harness.memory import MemoryRecord, MemoryStore
from harness.model import ModelResult, SendBlocked, SenderError
from harness.safety import pre_check
from harness.turn import Boundary, run_turn

NOW = datetime.fromisoformat("2026-09-29T08:10:00")


def rec(rid, content, origin="user", pinned=False, type_="fact", topic=None):
    return MemoryRecord(id=rid, type=type_, content=content, created_at=NOW.isoformat(), updated_at=NOW.isoformat(),
                        source="user_stated", confidence=0.9, topic=topic or f"t:{rid}", pinned=pinned, origin=origin)


def toy_domain(local=None, emergency_check=None, stub=None, **kw) -> Domain:
    return Domain(
        name="toy", system_prompt="You are a toy.", seed_path=Path("none"), rules=[], candidates=lambda sc, res: [],
        build_scenario=lambda now=None: toy_scenario(), pre_patterns={"lookup": [r"\bmy secret\b"]},
        emergency_patterns=[r"\bfire\b"], emergency_message="Leave now.", risk_flag_rules=[], endorsement_patterns=[],
        triggers={}, query_synonyms={}, escalation_contact="nobody", local_responders=local or {},
        emergency_check=emergency_check, stub_reply=stub, **kw)


def toy_scenario(text="hello there") -> Scenario:
    return Scenario("u", {}, Turn(NOW.isoformat(), "user", text, 1), [], {}, [], NOW)


class OriginTests(unittest.TestCase):
    def test_default_origin_is_user(self):
        self.assertEqual(Candidate("k", "data", 1, [("v", "x")]).origin, "user")
        self.assertEqual(rec("r1", "x").origin, "user")

    def test_memory_origin_roundtrips_through_json(self):
        import tempfile
        s = MemoryStore([rec("r1", "a", origin="derived_local")])
        with tempfile.TemporaryDirectory() as d:
            s.save(Path(d) / "m.json")
            self.assertEqual(MemoryStore.load(Path(d) / "m.json").records[0].origin, "derived_local")


class AssembleModeTests(unittest.TestCase):
    def cands(self):
        return [Candidate("system", "system", 0, [("v", "sys")], required=True),
                Candidate("incoming", "incoming", 0, [("v", "hi")], required=True),
                Candidate("ehr:Condition/c1", "data", 0, [("full", "A very identifying line about a condition")], required=True, origin="ehr"),
                Candidate("ehr:Observation/l1", "data", 3, [("full", "optional ehr line")], origin="ehr", score=0.9),
                Candidate("wear", "data", 3, [("full", "sleep 7h")], origin="wearable", score=0.5)]

    def test_local_mode_may_render_every_origin(self):
        asm = assemble(self.cands(), 500, "local")
        keys = {p.candidate.key for p in asm.placed}
        self.assertIn("ehr:Condition/c1", keys)
        self.assertIn("ehr:Observation/l1", keys)
        self.assertEqual(asm.excluded, [])

    def test_remote_mode_drops_ehr_even_when_required(self):
        asm = assemble(self.cands(), 500, "remote")
        keys = {p.candidate.key for p in asm.placed}
        self.assertEqual(keys, {"system", "incoming", "wear"})
        self.assertEqual({e["key"] for e in asm.excluded}, {"ehr:Condition/c1", "ehr:Observation/l1"})
        self.assertTrue(all(e["why"] == "excluded: remote mode, origin ehr" for e in asm.excluded))
        text = "\n".join(render(asm))
        self.assertNotIn("identifying", text)
        self.assertNotIn("optional ehr", text)

    def test_excluded_record_names_key_and_tokens_only(self):
        asm = assemble(self.cands(), 500, "remote")
        for e in asm.excluded:
            self.assertEqual(set(e), {"key", "tokens", "why"})

    def test_ehr_not_degraded_into_prompt_under_tight_budget(self):
        cands = self.cands()
        cands[2] = Candidate("ehr:Condition/c1", "data", 0, [("full", "long identifying text " * 5), ("short", "identifying")],
                             required=True, origin="ehr")
        asm = assemble(cands, 60, "remote")
        self.assertNotIn("identifying", "\n".join(render(asm)))

    def test_floor_ignores_excluded_required_ehr(self):
        cands = self.cands()
        with self.assertRaises(BudgetTooSmall):
            assemble(cands, 12, "local")
        assemble(cands, 25, "remote")          # fits only because the required ehr candidate is not counted

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            assemble(self.cands(), 500, "cloud")

    def test_missing_note_path_accepts_mode(self):
        asm = assemble_with_note(self.cands(), 500, "remote")
        self.assertEqual(asm.mode, "remote")

    def test_record_origin_reaches_candidate(self):
        store = MemoryStore([rec("r1", "ehr fact", origin="ehr", pinned=True), rec("r2", "wear fact", origin="wearable")])
        dom = toy_domain()
        sc = toy_scenario()
        from harness.resolve import resolve
        res = resolve(dom, sc, store)
        pre = pre_check(dom, sc.incoming.text, [], sc)
        cands = build_candidates(dom, sc, store.read("fact", NOW), res, pre)
        by = {c.key: c.origin for c in cands}
        self.assertEqual(by["pinned:r1"], "ehr")
        self.assertEqual(by["memory:r2"], "wearable")


class RemoteTurnTests(unittest.TestCase):
    def test_remote_turn_never_reads_ehr_records(self):
        store = MemoryStore([rec("r1", "SECRETCONDITION noted", origin="ehr", pinned=True),
                             rec("r2", "SECRETDRUG started", origin="ehr"), rec("r3", "likes tea", origin="user")])
        out = run_turn(toy_domain(), toy_scenario("tea SECRETDRUG?"), store, 400, boundary=Boundary("remote"))
        blob = out.system + out.prompt + out.trace.to_json()
        self.assertNotIn("SECRETCONDITION", blob)
        self.assertNotIn("SECRETDRUG started", blob)
        self.assertIn("memory:r2", out.trace.sections["remote_exclusions"]["memory_records"])
        self.assertEqual(out.mode, "remote")

    def test_local_turn_may_use_ehr_records(self):
        store = MemoryStore([rec("r1", "SECRETCONDITION noted", origin="ehr", pinned=True)])
        out = run_turn(toy_domain(), toy_scenario(), store, 400)
        self.assertIn("SECRETCONDITION", out.prompt)

    def test_recall_tool_excludes_ehr_in_remote_mode(self):
        from harness.tools import make_executor
        store = MemoryStore([rec("r1", "needle ehr", origin="ehr"), rec("r2", "needle user")])
        ex = make_executor(store, toy_domain(), NOW, [], set(), [], ("ehr",))
        self.assertNotIn("needle ehr", ex("recall", {"query": "needle"}))
        self.assertNotIn("needle ehr", ex("recall", {"ids": ["r1"]}))
        self.assertIn("needle user", ex("recall", {"query": "needle"}))

    def test_writes_still_go_to_the_real_store(self):
        store = MemoryStore([rec("r1", "x", origin="ehr")])
        run_turn(toy_domain(), toy_scenario(), store, 400, boundary=Boundary("remote"))
        self.assertTrue(any(r.type == "episode" for r in store.records))
        self.assertTrue(any(r.id == "r1" for r in store.records))

    def test_scrub_applies_to_remote_prompt_only(self):
        sc = toy_scenario("meet on 2026-02-03")
        scrub = lambda t: (t.replace("2026-02-03", "Feb 2026"), t.count("2026-02-03"))
        remote = run_turn(toy_domain(), sc, MemoryStore([]), 400, boundary=Boundary("remote", scrub=scrub))
        local = run_turn(toy_domain(), sc, MemoryStore([]), 400, boundary=Boundary("local", scrub=scrub))
        self.assertIn("Feb 2026", remote.prompt)
        self.assertNotIn("2026-02-03", remote.prompt)
        self.assertIn("2026-02-03", local.prompt)
        self.assertEqual(remote.trace.sections["context_assembly"]["scrubbed_dates"], 1)


class SenderTests(unittest.TestCase):
    """A sender is `(system, prompt, route, executor) -> ModelResult`; the core knows no wire format."""

    @staticmethod
    def ok(system, prompt, route, executor):
        return ModelResult("a fine reply", "live", 1, route.model)

    def test_live_requires_remote_mode(self):
        with self.assertRaises(ValueError):
            run_turn(toy_domain(), toy_scenario(), MemoryStore([]), 400, live=True, boundary=Boundary("local", sender=self.ok))

    def test_live_requires_a_sender(self):
        with self.assertRaises(ValueError):
            run_turn(toy_domain(), toy_scenario(), MemoryStore([]), 400, live=True, boundary=Boundary("remote"))

    def test_live_goes_through_the_injected_sender(self):
        seen = []

        def sender(system, prompt, route, executor):
            seen.append((system, prompt, route, executor))
            return self.ok(system, prompt, route, executor)
        out = run_turn(toy_domain(), toy_scenario(), MemoryStore([]), 400, live=True, boundary=Boundary("remote", sender=sender))
        self.assertEqual(len(seen), 1)
        self.assertEqual(out.response, "a fine reply")
        self.assertIn("hello there", seen[0][1])
        self.assertTrue(callable(seen[0][3]))

    def test_send_blocked_becomes_a_templated_reply(self):
        def sender(system, prompt, route, executor):
            raise SendBlocked("egress blocked: categories date, medication", ["date", "medication"])
        out = run_turn(toy_domain(), toy_scenario(), MemoryStore([]), 400, live=True, boundary=Boundary("remote", sender=sender))
        self.assertTrue(out.fallback_used)
        self.assertIn("egress blocked: categories date, medication", out.trace.sections["fallback"]["reason"])
        self.assertIn("hello there", out.response)

    def test_sender_error_is_an_api_error_fallback(self):
        def sender(system, prompt, route, executor):
            raise SenderError("APIConnectionError: down")
        out = run_turn(toy_domain(), toy_scenario(), MemoryStore([]), 400, live=True, boundary=Boundary("remote", sender=sender))
        self.assertIn("API error", out.trace.sections["fallback"]["reason"])

    def test_preflight_blocks_stub_calls_too(self):
        def pre(system, prompt):
            raise SendBlocked("egress blocked: categories name", ["name"])
        out = run_turn(toy_domain(), toy_scenario(), MemoryStore([]), 400, boundary=Boundary("remote", preflight=pre))
        self.assertTrue(out.fallback_used)
        self.assertFalse(out.model_called)
        self.assertEqual(out.trace.sections["send_blocked"]["reason"], "egress blocked: categories name")


class LocalResponderTests(unittest.TestCase):
    def test_flag_routes_to_local_responder_without_model(self):
        calls = []
        dom = toy_domain(local={"lookup": lambda text, pre, sc: LocalReply("from local code", "answered locally; no model call", "ehr")},
                         stub=lambda inp: calls.append(1) or "model said")
        out = run_turn(dom, toy_scenario("what is my secret"), MemoryStore([]), 400)
        self.assertEqual(out.response, "from local code")
        self.assertFalse(out.model_called)
        self.assertEqual(calls, [])
        self.assertEqual(out.local_kind, "ehr")
        self.assertEqual(out.trace.sections["model_call"]["skipped"], "answered locally; no model call")
        self.assertEqual(out.system, "")

    def test_a_star_responder_is_offered_every_reactive_turn_before_flag_responders(self):
        seen = []
        dom = toy_domain(local={"*": lambda text, pre, sc: (seen.append(text), LocalReply("star", "answered locally; no model call", "ehr"))[1],
                                "lookup": lambda text, pre, sc: LocalReply("flag", "n", "ehr")},
                         stub=lambda inp: "model said")
        out = run_turn(dom, toy_scenario("what is my secret"), MemoryStore([]), 400)
        self.assertEqual((out.response, seen), ("star", ["what is my secret"]))
        out = run_turn(toy_domain(local={"*": lambda text, pre, sc: None}, stub=lambda inp: "model said"),
                       toy_scenario("hello there"), MemoryStore([]), 400)
        self.assertEqual(out.response, "model said")

    def test_responder_may_decline(self):
        dom = toy_domain(local={"lookup": lambda text, pre, sc: None}, stub=lambda inp: "model said")
        out = run_turn(dom, toy_scenario("what is my secret"), MemoryStore([]), 400)
        self.assertEqual(out.response, "model said")
        self.assertTrue(out.model_called)

    def test_emergency_pattern_still_short_circuits(self):
        out = run_turn(toy_domain(), toy_scenario("there is a fire"), MemoryStore([]), 400)
        self.assertEqual(out.response, "Leave now.")
        self.assertFalse(out.model_called)

    def test_emergency_check_hook(self):
        dom = toy_domain(emergency_check=lambda text: "smoke" in text)
        out = run_turn(dom, toy_scenario("I smell smoke"), MemoryStore([]), 400)
        self.assertEqual(out.response, "Leave now.")
        self.assertEqual(out.local_kind, "emergency")

    def test_domain_stub_replaces_core_stub(self):
        out = run_turn(toy_domain(stub=lambda inp: "domain stub"), toy_scenario(), MemoryStore([]), 400)
        self.assertEqual(out.response, "domain stub")

    def test_post_requirements_hook(self):
        dom = toy_domain(stub=lambda inp: "no pointer here", post_requirements=lambda pre, res: [["prescriber"]])
        out = run_turn(dom, toy_scenario(), MemoryStore([]), 400)
        self.assertTrue(out.fallback_used)
        self.assertIn("tensions_addressed", out.trace.sections["fallback"]["reason"])


if __name__ == "__main__":
    unittest.main()
