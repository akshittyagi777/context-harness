"""The egress guard, the model boundary, and the EHR-stays-local guarantee as a whole."""
from __future__ import annotations

import ast
import json
import os
import re
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from tests import support
from tests.support import ROOT, PASSPHRASE, data_copy, empty_data, fixture

from harness.model import SenderError
from domains.health import model_boundary, runner
from providers import anthropic as provider
from harness.router import TIERS, Route
from domains.health.egress import EgressBlocked, Guard, build_denylist, redact, scan, scrub_dates
from domains.health.vault import Vault


def guard(dd=None) -> Guard:
    return Guard(dd or fixture()["data"], PASSPHRASE)


# every kind of EHR token in the mock record, embedded in an otherwise ordinary prompt line
TOKENS = {
    "name": ["Jordan", "Rivera", "jordan rivera", "A. Rivera"],
    "identifier": ["Z0048213", "E3318420"],
    "dob": ["1990-04-12", "April 12, 1990", "04/12/1990", "12 Apr 1990"],
    "address": ["1450 Valencia Street", "94110", "San Francisco"],
    "phone": ["(415) 555-0142", "415-555-0142", "4155550142"],
    "email": ["jordan.rivera@example.com"],
    "medication": ["metoprolol", "Metoprolol Succinate", "Toprol-XL", "Lopressor", "naproxen", "Aleve", "penicillin", "Penicillins"],
    "condition": ["Hypertension", "essential hypertension"],
    "code": ["I10", "59621000", "4548-4", "13457-7", "866924", "EAP#1230", "RxNorm 7980", "SNOMED 59621000"],
    "lab_value": ["5.8 %", "5.8%", "118 mg/dL", "104mg/dl", "0.9 mg/dL", "140 mmol/L", "148/94", "128/82 mmHg"],
    "date": ["2026-02-03", "2026-02-10", "2026-06-09", "2025-03-18", "Feb 3, 2026", "February 10, 2026", "02/03/2026", "3 Feb 2026"],
}


class DenylistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = guard()
        cls.dl = cls.g.denylist()

    def test_every_ehr_token_is_blocked_under_its_own_category(self):
        for cat, tokens in TOKENS.items():
            if cat == "lab_value":
                tokens = [t for t in tokens if t not in ("128/82 mmHg",)]       # 128/82 is not in the record (126/80 is)
            for tok in tokens:
                text = f"Context: last night went fine. {tok} noted. Respond briefly."
                cats = scan(text, self.dl)
                self.assertIn(cat if cat != "identifier" else "identifier", cats if cat != "address" else cats + ["address"], f"{tok!r} -> {cats}")
                with self.assertRaises(EgressBlocked) as cm:
                    self.g.check_texts(["ok system", text])
                self.assertNotIn(tok, str(cm.exception))                                # categories, never values

    def test_a_value_not_in_the_record_is_not_blocked_by_accident(self):
        self.assertEqual(scan("Reading 128/82 mmHg and 5.9 %", self.dl), [])

    def test_epic_opaque_ids_are_blocked(self):
        data = Vault(fixture()["data"], PASSPHRASE).read("egress guard")
        sid = data["medications"][0]["source_ids"][0]
        self.assertIn("identifier", scan(f"see {sid}", self.dl))

    def test_blood_pressure_values_and_short_numeric_codes_do_not_hit_ordinary_numbers(self):
        # steps and minutes are everywhere in a wearable prompt: 7980 is a penicillin code but only with a code-system word
        self.assertEqual(scan("Garmin steps 7980, 140 bpm, 94 percent, 148 minutes", self.dl), [])
        self.assertEqual(scan("Sleep 4h40, recovery 21%, HRV 36 ms, resting HR 55 bpm, 5.7 hours", self.dl), [])

    def test_wearable_dates_pass_but_ehr_event_dates_do_not(self):
        self.assertEqual(scan("Last night (2026-09-29) and 2026-09-10", self.dl), [])
        self.assertEqual(scan("on 2026-02-10", self.dl), ["date"])
        self.assertEqual(scan("ISO date-time 2026-06-09T16:20:00Z", self.dl), ["date"])

    def test_month_precision_is_allowed(self):
        self.assertEqual(scan("A clinical factor lowers resting heart rate from February 2026", self.dl), [])
        self.assertEqual(scan("Feb 2026", self.dl), [])

    def test_safe_harbor_patterns_work_without_any_ehr_store(self):
        none = None
        self.assertIn("phone", scan("call 212-555-0199 now", none))
        self.assertIn("email", scan("write to someone@example.org", none))
        self.assertIn("ssn_like", scan("id 123-45-6789", none))
        self.assertIn("mrn_like", scan("MRN: 00482137", none))
        self.assertIn("mrn_like", scan("patient Q1234567", none))
        self.assertIn("street_address", scan("lives at 221 Baker Street", none))
        self.assertEqual(scan("Sleep 4h40 on 2026-09-29, HRV 36 ms, steps 12345", none), [])

    def test_guard_with_no_store_only_applies_the_patterns(self):
        g = Guard(empty_data(), PASSPHRASE)
        g.check_texts(["metoprolol is just a word here"])
        with self.assertRaises(EgressBlocked):
            g.check_texts(["call 415-555-0142"])

    def test_fail_closed_when_a_store_exists_but_cannot_be_read(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("HEALTH_HARNESS_PASSPHRASE", None)
            g = Guard(fixture()["data"])
            with self.assertRaises(EgressBlocked) as cm:
                g.check_texts(["a perfectly innocent prompt"])
            self.assertEqual(cm.exception.categories, ("denylist_unavailable",))
        with self.assertRaises(EgressBlocked):
            Guard(fixture()["data"], "wrong passphrase").check_texts(["innocent"])

    def test_scrub_coarsens_ehr_dates_to_a_month(self):
        text, n = scrub_dates("stated 2026-02-03 and again Feb 3, 2026, but 2026-09-29 stays", self.dl)
        self.assertEqual(n, 2)
        self.assertEqual(text, "stated Feb 2026 and again Feb 2026, but 2026-09-29 stays")
        self.assertEqual(scan(text, self.dl), [])

    def test_scrub_does_not_touch_date_of_birth(self):
        text, n = scrub_dates("born 1990-04-12", self.dl)
        self.assertEqual(n, 0)
        self.assertEqual(scan(text, self.dl), ["dob"])           # still blocked

    def test_redact_removes_tokens_for_storage(self):
        out, n = redact("can I take melatonin with metoprolol? seen 2026-02-03, call 415-555-0142", self.dl)
        self.assertEqual(scan(out, self.dl), [])
        self.assertNotIn("metoprolol", out)
        self.assertIn("[redacted: medication]", out)
        self.assertGreaterEqual(n, 3)

    def test_request_scan_covers_system_user_tool_results_and_sdk_objects(self):
        g = self.g
        class Block:                                  # stands in for an SDK content block
            def model_dump(self): return {"type": "text", "text": "metoprolol"}
        for req in ({"system": [{"type": "text", "text": "Rivera"}], "messages": []},
                    {"system": "ok", "messages": [{"role": "user", "content": "Rivera"}]},
                    {"system": "ok", "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "4548-4"}]}]},
                    {"system": "ok", "messages": [{"role": "assistant", "content": [Block()]}]}):
            with self.assertRaises(EgressBlocked):
                g.check_request({"model": "m", "max_tokens": 1, **req})
        g.check_request({"model": "Rivera-model-name-is-ignored", "max_tokens": 5, "system": "ok", "messages": [{"role": "user", "content": "hi"}]})


class DefaultPromptTests(unittest.TestCase):
    def test_egress_check_passes_the_default_remote_prompt(self):
        r = runner.egress_check(fixture()["data"])
        self.assertTrue(r["passed"], r["categories"])
        self.assertTrue(r["ehr_store"])
        self.assertGreaterEqual(len(r["excluded"]), 13)
        self.assertEqual(r["route"], "large")

    def test_the_default_remote_prompt_contains_the_derived_constraints_and_no_ehr_content(self):
        r = runner.egress_check(fixture()["data"])
        p = r["prompt"]
        self.assertIn("A clinical factor lowers resting heart rate and may blunt heart-rate response from February 2026", p)
        self.assertIn("Do not suggest specific medications.", p)
        for word in ("metoprolol", "penicillin", "hypertension", "A1c", "Rivera", "naproxen"):
            self.assertNotIn(word.lower(), (p + r["system"]).lower())

    def test_dates_that_coincide_with_ehr_events_are_coarsened_not_sent(self):
        r = runner.egress_check(fixture()["data"])
        self.assertGreaterEqual(r["scrubbed_dates"], 3)               # the seeded memory is dated 2026-02-03
        self.assertNotIn("2026-02-03", r["prompt"])
        self.assertIn("Feb 2026", r["prompt"])

    def test_it_fails_when_an_unscrubbable_ehr_token_is_injected(self):
        # a bare lab value (no lab term beside it) and identifiers that are not the patient's are not scrubbed: they block
        for msg, cat in (("Hard run today? the result was 5.8 % last time", "lab_value"),
                         ("Hard run today? call me 212-555-0199", "phone"),
                         ("Hard run today? id 123-45-6789", "ssn_like")):
            r = runner.egress_check(fixture()["data"], message=msg)
            self.assertFalse(r["passed"], msg)
            self.assertIn(cat, r["categories"], (msg, r["categories"]))

    def test_identifiers_and_codes_are_scrubbed_not_blocked(self):
        for msg, want in (("Going hard today, Dr. Rivera said", "Dr. [an identifier] said"),
                          ("Hard run today? call me 415-555-0142", "call me [an identifier]"),
                          ("Hard run today? my code is 59621000", "my code is [a code]"),
                          ("Hard run today? born 1990-04-12", "born [an identifier]"),
                          ("Hard run today? I live at 1450 Valencia Street", "I live at [an identifier]")):
            r = runner.egress_check(fixture()["data"], message=msg)
            self.assertTrue(r["passed"], (msg, r["categories"]))
            self.assertIn(want, r["prompt"])

    def test_a_typed_medication_or_condition_is_scrubbed_not_blocked(self):
        for msg, placeholder in (("Thinking of going hard on a run, I take metoprolol", "[a medication on file]"),
                                 ("Thinking of going hard. Hypertension runs in my family", "[a condition on file]")):
            r = runner.egress_check(fixture()["data"], message=msg)
            self.assertTrue(r["passed"], (msg, r["categories"]))
            self.assertIn(placeholder, r["prompt"])
            self.assertEqual(scan(r["system"] + r["prompt"], guard().denylist()), [])

    def test_a_typed_ehr_date_is_coarsened_before_the_guard_and_still_blocked_if_it_gets_past(self):
        r = runner.egress_check(fixture()["data"], message="Hard run today? the clinic visit was on 2026-06-09")
        self.assertTrue(r["passed"])
        self.assertNotIn("2026-06-09", r["prompt"])
        self.assertIn("Jun 2026", r["prompt"])
        with self.assertRaises(EgressBlocked):
            guard().check_texts(["the clinic visit was on 2026-06-09"])         # the guard itself never lets a full date through

    def test_it_fails_when_an_ehr_candidate_leaks_into_assembly(self):
        # simulate the failure mode the guard exists for: a bug that lets an ehr-origin line through
        import harness.context as ctx
        real = ctx.remove_ehr_for_remote
        try:
            ctx.remove_ehr_for_remote = lambda cands, mode: (cands, [])
            leaked = runner.egress_check(fixture()["data"])
            with mock.patch.object(Guard, "scrub_detail", lambda self, t: (t, 0, {})):      # and the scrub misses it too
                r = runner.egress_check(fixture()["data"])
        finally:
            ctx.remove_ehr_for_remote = real
        self.assertTrue(leaked["passed"])                     # the scrub alone repairs the leaked text...
        self.assertNotIn("metoprolol", leaked["prompt"].lower())
        self.assertFalse(r["passed"])                         # ...and the guard still stops it if the scrub does not
        self.assertTrue({"medication", "lab_value"} & set(r["categories"]), r["categories"])

    def test_cli_egress_check_prints_pass(self):
        import io, contextlib, health_cli as cli
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"HEALTH_HARNESS_DATA": str(fixture()["data"])}), contextlib.redirect_stdout(buf):
            code = cli.main(["egress-check"])
        self.assertEqual(code, 0)
        self.assertEqual(buf.getvalue().strip().splitlines()[-1], "PASS")

    def test_cli_egress_check_prints_blocked_categories_and_exits_nonzero(self):
        import io, contextlib, health_cli as cli
        dd = data_copy()
        mem = MemoryStore_with_ehr_note(dd)
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"HEALTH_HARNESS_DATA": str(dd)}), contextlib.redirect_stdout(buf):
            code = cli.main(["egress-check"])
        self.assertEqual(code, 1)
        self.assertIn("BLOCKED: categories", buf.getvalue())
        self.assertIn("phone", buf.getvalue())
        self.assertNotIn("metoprolol", buf.getvalue().lower())
        self.assertNotIn("212", buf.getvalue())


def MemoryStore_with_ehr_note(dd):
    """A user-origin memory with a drug name and a phone number: the drug is scrubbed, the phone number blocks."""
    from harness.memory import MemoryRecord, MemoryStore
    s = MemoryStore.load(dd / "memory.json")
    s.records.append(MemoryRecord(id="r90", type="constraint", content="Never take metoprolol on race days, call 212-555-0199", created_at="2026-09-01T10:00:00",
                                  updated_at="2026-09-01T10:00:00", source="user_stated", confidence=0.9, pinned=True, topic="c:x"))
    s.save(dd / "memory.json")
    return s


class BoundaryTests(unittest.TestCase):
    ROUTE = Route("small", TIERS["small"]["model"], None)

    class Resp:
        stop_reason = "end_turn"
        usage = type("U", (), {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})()
        content = [type("B", (), {"type": "text", "text": "response"})()]

    class FakeClient:
        def __init__(self, log, fail=None):
            self.log, self.fail = log, fail
            self.messages = self

        def create(self, **kw):
            self.log.append(kw)
            if self.fail:
                raise self.fail
            return BoundaryTests.Resp()

    def send(self, text):
        return model_boundary.send_remote("s", text, self.ROUTE, lambda name, inp: "", guard())

    def test_blocked_request_never_creates_a_client_or_sends(self):
        made, sent = [], []
        factory = lambda: (made.append(1), self.FakeClient(sent))[1]
        with mock.patch.object(provider, "_client_factory", factory):
            out = self.send("I take metoprolol")
        self.assertEqual(out.stop, "blocked")
        self.assertIn("egress blocked: categories", out.error)
        self.assertNotIn("metoprolol", out.error)
        self.assertEqual((made, sent), ([], []))

    def test_clean_request_is_sent_exactly_once(self):
        sent = []
        with mock.patch.object(provider, "_client_factory", lambda: self.FakeClient(sent)):
            out = self.send("sleep was short")
        self.assertEqual((out.stop, out.text), ("ok", "response"))
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["messages"][0]["content"], "sleep was short")

    def test_sdk_errors_become_error_markers(self):
        class APIConnectionError(Exception):
            __module__ = "anthropic._exceptions"
        with mock.patch.object(provider, "_client_factory", lambda: self.FakeClient([], APIConnectionError("down"))):
            out = self.send("hi")
        self.assertEqual(out.stop, "error")
        self.assertIn("APIConnectionError: down", out.error)

    def test_other_errors_are_not_swallowed(self):
        with mock.patch.object(provider, "_client_factory", lambda: self.FakeClient([], KeyError("bug"))):
            with self.assertRaises(KeyError):
                self.send("hi")

    def test_the_guard_checks_every_round_of_the_tool_loop(self):
        sent = []

        class Tool:
            type, id, name, input = "tool_use", "t1", "recall", {}

        class ToolResp(self.Resp):
            stop_reason = "tool_use"
            content = [Tool()]
        replies = [ToolResp(), self.Resp()]
        client = self.FakeClient(sent)
        client.create = lambda **kw: (sent.append(kw), replies.pop(0))[1]
        with mock.patch.object(provider, "_client_factory", lambda: client):
            out = model_boundary.send_remote("s", "hello", self.ROUTE, lambda name, inp: "your metoprolol dose", guard())
        self.assertEqual((out.stop, len(sent)), ("blocked", 1))      # the second request carries the tool result, and is stopped
        self.assertNotIn("metoprolol", out.error)

    def test_live_turn_routes_through_the_guard_and_sends_only_remote_assembly(self):
        sent = []

        class Resp:
            stop_reason = "end_turn"
            usage = type("U", (), {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})()
            content = [type("B", (), {"type": "text", "text": "Recovery is low, so an easy day is the safer call than going hard."})()]
        client = self.FakeClient(sent)
        client.create = lambda **kw: (sent.append(kw), Resp())[1]
        with mock.patch.object(provider, "_client_factory", lambda: client):
            out = runner.run(fixture()["data"], live=True, persist=False)
        self.assertEqual(out.mode, "remote")
        self.assertEqual(len(sent), 1)
        wire = json.dumps(sent[0], default=str).lower()
        for word in ("metoprolol", "rivera", "hypertension", "penicillin", "4548-4", "2026-02-03"):
            self.assertNotIn(word, wire)
        self.assertIn("a clinical factor lowers resting heart rate", wire)
        self.assertEqual(out.response.startswith("Recovery is low"), True)

    def test_live_turn_with_an_ehr_token_in_the_message_makes_no_call(self):
        made = []
        with mock.patch.object(provider, "_client_factory", lambda: (made.append(1), self.FakeClient([]))[1]):
            out = runner.run(fixture()["data"], "going hard today, and call me on 212-555-0199", live=True, persist=False)
        self.assertEqual(made, [])
        self.assertTrue(out.fallback_used)
        self.assertIn("egress blocked: categories", out.trace.sections["fallback"]["reason"])
        self.assertIn("phone", out.trace.sections["fallback"]["reason"])
        self.assertNotIn("212", out.trace.sections["fallback"]["reason"])

    def test_live_turn_with_a_locked_store_makes_no_call(self):
        made = []
        with mock.patch.dict(os.environ):
            os.environ.pop("HEALTH_HARNESS_PASSPHRASE", None)
            with mock.patch.object(provider, "_client_factory", lambda: (made.append(1), self.FakeClient([]))[1]):
                out = runner.run(fixture()["data"], live=True, persist=False)
        self.assertEqual(made, [])
        self.assertIn("denylist_unavailable", out.trace.sections["fallback"]["reason"])

    def test_local_mode_cannot_go_live(self):
        with self.assertRaises(ValueError):
            runner.run(fixture()["data"], live=True, mode="local", persist=False)


class ModeTests(unittest.TestCase):
    def test_remote_assembly_excludes_ehr_candidates_and_the_trace_says_so(self):
        out = runner.run(fixture()["data"], mode="remote", persist=False)
        ex = out.trace.sections["context_assembly"]["excluded"]
        self.assertTrue(ex and all(e["why"] == "excluded: remote mode, origin ehr" for e in ex))
        self.assertTrue(all(re.match(r"^ehr:[A-Za-z]+/[a-z]+\d+$", e["key"]) and isinstance(e["tokens"], int) for e in ex))
        placed = {p.candidate.key for p in out.assembly.placed}
        self.assertFalse(any(k.startswith("ehr:") for k in placed))
        self.assertIn("excluded: remote mode, origin ehr", out.trace.to_markdown())

    def test_local_assembly_may_render_ehr_candidates(self):
        out = runner.run(fixture()["data"], mode="local", persist=False)
        placed = {p.candidate.key for p in out.assembly.placed}
        self.assertTrue(any(k.startswith("ehr:") for k in placed))
        self.assertEqual(out.trace.sections["context_assembly"]["excluded"], [])
        self.assertIn("metoprolol", out.prompt.lower())
        self.assertIn("A clinical factor lowers", out.prompt)

    def test_stub_defaults_to_local_and_live_to_remote(self):
        from domains.health.config import egress_mode
        self.assertEqual((egress_mode(False), egress_mode(True)), ("local", "remote"))

    def test_the_stub_reply_never_depends_on_ehr_content(self):
        local = runner.run(fixture()["data"], mode="local", persist=False)
        remote = runner.run(fixture()["data"], mode="remote", persist=False)
        self.assertEqual(local.response, remote.response)


class AtRestTests(unittest.TestCase):
    """Traces, history, memory and the access log never hold EHR values, whatever the user typed."""

    @classmethod
    def setUpClass(cls):
        cls.dd = data_copy()
        for msg, kw in (("Thinking of going hard on a long run this morning. Thoughts?", {}),
                        ("what meds am I on", {}), ("when was my last A1c", {}), ("what are my allergies", {}),
                        ("can I take melatonin tonight?", {"mode": "remote"}),
                        ("can I take melatonin with metoprolol? I saw it on 2026-02-10", {"mode": "remote"}),
                        ("my resting HR is way down, I'm getting fitter right?", {"mode": "remote"}),
                        ("chest pain when I run, spreading to my arm", {}),
                        ("my A1c is 5.8 % and Rivera is my name, 415-555-0142", {"mode": "remote"})):
            runner.run(cls.dd, msg, "2026-09-29T08:10:00", persist=True, **kw)
        runner.run(cls.dd, None, "2026-09-30T08:30:00", trigger="morning_brief", persist=True)
        cls.dl = Guard(cls.dd, PASSPHRASE).denylist()

    def files(self):
        return sorted((self.dd / "traces").glob("*")) + [self.dd / "access.log", self.dd / "history.json"]

    def test_traces_access_log_and_history_contain_no_denylisted_token(self):
        n = 0
        for f in self.files():
            self.assertEqual(scan(f.read_text(), self.dl), [], f.name)
            n += 1
        self.assertGreaterEqual(n, 20)

    def test_memory_written_by_turns_holds_no_ehr_token_either(self):
        from harness.memory import MemoryStore
        for r in MemoryStore.load(self.dd / "memory.json").records:
            if r.origin == "user" and r.type == "episode":
                self.assertEqual(scan(r.content, self.dl), [], r.content)
        self.assertTrue(any("[redacted: medication]" in r.content for r in MemoryStore.load(self.dd / "memory.json").records))

    def test_ehr_candidates_appear_in_traces_only_as_type_slash_local_id_with_tokens(self):
        traces = [json.loads(p.read_text()) for p in (self.dd / "traces").glob("*.json")]
        t = next(t for t in traces if t["mode"]["assembly_mode"] == "local" and t.get("context_assembly")
                 and any(d["key"].startswith("ehr:") for d in t["context_assembly"]["decisions"]))
        rows = [d for d in t["context_assembly"]["decisions"] if d["key"].startswith("ehr:")]
        self.assertTrue(rows)
        for d in rows:
            self.assertRegex(d["key"], r"^ehr:[A-Za-z]+/[a-z]+\d+$")
            self.assertIsInstance(d["tokens"], int)
            self.assertTrue(set(d) <= {"tier", "key", "variant_costs", "required", "kept", "tokens", "why"})

    def test_local_answers_are_traced_without_the_answer(self):
        traces = [json.loads(p.read_text()) for p in (self.dd / "traces").glob("*.json")]
        local = [t for t in traces if t.get("local_response", {}).get("kind") == "ehr"]
        self.assertEqual(len(local), 4)         # meds, A1c, allergies, and the message stating an A1c value (a lab result question)
        for t in local:
            self.assertEqual(t["model_call"]["skipped"], "answered locally from EHR; no model call")
            self.assertEqual(t["routing"]["tier"], "local")
        history = (self.dd / "history.json").read_text()
        self.assertIn("(answered locally from your records)", history)
        self.assertNotIn("5.8", history)

    def test_access_log_has_purposes_types_and_never_values(self):
        entries = Vault(self.dd, PASSPHRASE).read_access_log()
        self.assertTrue(entries)
        for e in entries:
            self.assertEqual(set(e), {"ts", "purpose", "resource_types"})
            self.assertTrue(set(e["resource_types"]) <= {"Patient", "Condition", "MedicationRequest", "AllergyIntolerance", "Observation"})
        purposes = {e["purpose"] for e in entries}
        self.assertTrue({"import", "derive", "local answer", "local assembly", "egress guard"} <= purposes, purposes)

    def test_a_blocked_turn_stores_a_templated_reply_without_the_drug(self):
        h = json.loads((self.dd / "history.json").read_text())
        blocked = [t for t in h if "melatonin with" in t["text"]]
        self.assertTrue(blocked)
        self.assertTrue(all("metoprolol" not in t["text"].lower() for t in h))


if __name__ == "__main__":
    unittest.main()


class BoundaryErrorTests(unittest.TestCase):
    def test_missing_sdk_and_missing_credentials_are_error_markers_not_crashes(self):
        def no_sdk():
            raise ImportError("No module named anthropic")
        class AnthropicError(Exception):
            __module__ = "anthropic"
        def no_key():
            raise AnthropicError("Could not resolve authentication method")
        for factory, needle in ((no_sdk, "pip install anthropic"), (no_key, "authentication")):
            with mock.patch.object(provider, "_client_factory", factory):
                out = model_boundary.send_remote("s", "hi", BoundaryTests.ROUTE, lambda n, i: "", guard())
            self.assertEqual(out.stop, "error")
            self.assertIn(needle, out.error)

    def test_a_live_turn_with_no_credentials_falls_back_instead_of_crashing(self):
        def no_key():
            class AnthropicError(Exception):
                __module__ = "anthropic"
            raise AnthropicError("no api key")
        with mock.patch.object(provider, "_client_factory", no_key):
            out = runner.run(fixture()["data"], live=True, persist=False)
        self.assertTrue(out.fallback_used)
        self.assertIn("API error", out.trace.sections["fallback"]["reason"])


TS_SCRUB = "2026-09-29T08:10:00"


def _remote(msg, dd=None, **kw):
    return runner.run(dd or fixture()["data"], msg, TS_SCRUB, mode="remote", persist=False, **kw)


def _plan_store(dd, content):
    from harness.memory import MemoryRecord, MemoryStore
    s = MemoryStore.load(dd / "memory.json")
    s.records.append(MemoryRecord(id="r91", type="plan", content=content, created_at="2026-09-20T10:00:00",
                                  updated_at="2026-09-20T10:00:00", source="user_stated", confidence=0.9, topic="plan:week"))
    s.save(dd / "memory.json")
    return s


class SynonymMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dl = guard().denylist()

    def terms(self):
        return set(self.dl.scrub_terms)

    def test_the_required_synonyms_are_active_for_facts_in_the_record(self):
        want = {"high blood pressure", "blood pressure", "bp", "htn",
                "toprol", "toprol-xl", "lopressor", "beta blocker", "beta-blocker",
                "aleve", "naprosyn",
                "a1c", "hemoglobin a1c", "glycated hemoglobin", "hba1c",
                "ldl", "ldl cholesterol", "bad cholesterol"}
        self.assertEqual(want - self.terms(), set())
        self.assertEqual(self.dl.scrub_terms["bp"], "condition")
        self.assertEqual(self.dl.scrub_terms["aleve"], "medication")
        self.assertEqual(self.dl.scrub_terms["a1c"], "lab")
        self.assertEqual(self.dl.scrub_terms["penicillins"], "allergy")

    def test_amoxicillin_is_not_a_synonym_of_the_penicillin_allergy(self):
        self.assertNotIn("amoxicillin", self.terms())
        self.assertEqual(scrub_terms_of("amoxicillin and penicillin", self.dl), "amoxicillin and [an allergy on file]")

    def test_only_terms_for_facts_in_the_store_are_active(self):
        ehr = Vault(fixture()["data"], PASSPHRASE).read("egress guard")
        ehr = {**ehr, "conditions": [], "labs": [l for l in ehr["labs"] if "ldl" not in l["text"].lower()],
               "medications": [m for m in ehr["medications"] if "naproxen" not in m["display"].lower()]}
        t = set(build_denylist(ehr).scrub_terms)
        self.assertTrue({"bp", "htn", "high blood pressure", "ldl", "bad cholesterol", "aleve", "naprosyn"}.isdisjoint(t))
        self.assertTrue({"toprol", "a1c"} <= t)
        self.assertEqual(scrub_terms_of("my BP and Aleve", build_denylist(ehr)), "my BP and Aleve")

    def test_no_store_means_nothing_to_scrub(self):
        self.assertEqual(scrub_terms_of("metoprolol for my BP", None), "metoprolol for my BP")


def scrub_terms_of(text, dl):
    from domains.health.egress import scrub_terms
    return scrub_terms(text, dl)[0]


class ScrubThenGuardTests(unittest.TestCase):
    def test_a_typed_medication_name_is_scrubbed_routed_large_and_flagged(self):
        out = _remote("is my metoprolol why I feel tired?")
        sent = out.system + "\n" + out.prompt
        self.assertTrue(out.model_called)
        self.assertFalse(out.fallback_used)
        self.assertNotIn("send_blocked", out.trace.sections)
        self.assertIn("is my [a medication on file] why I feel tired?", out.prompt)
        self.assertEqual(scan(sent, guard().denylist()), [])
        self.assertNotIn("metoprolol", sent.lower())
        self.assertEqual(out.route.tier, "large")
        self.assertIn("medication_question", out.pre.flags)
        self.assertRegex(out.response.lower(), r"prescriber|pharmacist")
        self.assertTrue(out.post.passed, out.post.checks)

    def test_the_derived_constraints_reach_the_model_on_a_medication_question(self):
        out = _remote("is my metoprolol why I feel tired?")
        self.assertIn("Do not suggest specific medications.", out.prompt)

    def test_the_stub_reply_for_a_medication_question_names_no_drug_and_points_to_the_prescriber(self):
        out = _remote("is my metoprolol why I feel tired?")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", out.response) if s]
        self.assertEqual(len(sentences), 2, sentences)
        self.assertRegex(sentences[1], r"prescriber or pharmacist")
        self.assertRegex(sentences[1], r"next visit")
        self.assertEqual(scan(out.response, guard().denylist()), [])
        for word in ("metoprolol", "toprol", "naproxen", "penicillin", "beta"):
            self.assertNotIn(word, out.response.lower())

    def test_brand_names_are_scrubbed_too(self):
        out = _remote("I switched from Toprol to Aleve last month, thoughts?")
        self.assertIn("I switched from [a medication on file] to [a medication on file]", out.prompt)
        self.assertNotRegex(out.system + out.prompt, r"(?i)toprol|aleve")
        self.assertEqual(out.trace.sections["egress"]["by_category"].get("medication"), 2)

    def test_a_stored_plan_with_a_synonym_is_scrubbed(self):
        dd = data_copy()
        _plan_store(dd, "Taking it easy until my BP is sorted")
        out = _remote("how should I train this week?", dd)
        self.assertRegex(out.prompt, r"Taking it easy until my \[a condition on file\] is sorted")
        self.assertNotRegex(out.system + out.prompt, r"\bBP\b")
        self.assertEqual(scan(out.system + out.prompt, Guard(dd, PASSPHRASE).denylist()), [])
        self.assertTrue(out.model_called)

    def test_user_history_and_memory_origins_are_all_scrubbed(self):
        dd = data_copy()
        _plan_store(dd, "Keep Lopressor and the high blood pressure log going")
        out = _remote("what is my plan?", dd)
        self.assertIn("Keep [a medication on file] and the [a condition on file] log going", out.prompt)

    def test_the_trace_records_counts_and_the_guard_result_never_the_terms(self):
        out = _remote("is my metoprolol why I feel tired? my A1c worries me too")
        lines = [f"{e['stage']}: {e['msg']}" for e in out.trace.events if e["stage"] == "egress"]
        self.assertRegex(lines[0], r"^egress: scrubbed \d+ term\(s\) by category \{.*'medication': 1.*\}$")
        self.assertIn("'lab': 1", lines[0])
        self.assertEqual(lines[-1], "egress: guard passed")
        self.assertEqual(out.trace.sections["egress"]["guard"], "passed")
        blob = out.trace.to_markdown().split("## Event log")[1]
        self.assertNotRegex(blob.lower(), r"metoprolol|a1c")

    def test_the_guard_still_blocks_what_the_scrub_leaves_behind(self):
        noop = lambda self, text: (text, 0, {})
        with mock.patch.object(Guard, "scrub_detail", noop), mock.patch.object(Guard, "scrub", lambda self, t: (t, 0)):
            out = _remote("is my metoprolol why I feel tired?")
            self.assertTrue(out.fallback_used)
            self.assertFalse(out.model_called)
            self.assertRegex(out.trace.sections["send_blocked"]["reason"], r"^egress blocked: categories .*medication")
            self.assertEqual(out.trace.sections["egress"]["guard"], "blocked")
            r = runner.egress_check(fixture()["data"], message="is my metoprolol why I feel tired?")
            self.assertFalse(r["passed"])
            self.assertIn("medication", r["categories"])
            dd = data_copy()
            _plan_store(dd, "Taking it easy until my BP is sorted")
            self.assertFalse(runner.egress_check(dd, message="how should I train this week?")["passed"])

    def test_a_live_turn_sends_the_scrubbed_prompt_and_blocks_when_the_scrub_is_off(self):
        sent = []

        class Resp:
            stop_reason = "end_turn"
            usage = type("U", (), {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})()
            content = [type("B", (), {"type": "text", "text": "A fair question; your prescriber or pharmacist is the right person to ask."})()]
        client = BoundaryTests.FakeClient(sent)
        client.create = lambda **kw: (sent.append(kw), Resp())[1]
        with mock.patch.object(provider, "_client_factory", lambda: client):
            out = runner.run(fixture()["data"], "is my metoprolol why I feel tired?", TS_SCRUB, live=True, persist=False)
            self.assertFalse(out.fallback_used)
            self.assertEqual(len(sent), 1)
            wire = json.dumps(sent[0], default=str)
            self.assertIn("[a medication on file]", wire)
            self.assertNotIn("metoprolol", wire.lower())
            self.assertEqual(sent[0]["model"], "claude-opus-5")
            sent.clear()
            noop = lambda self, text: (text, 0, {})
            with mock.patch.object(Guard, "scrub_detail", noop):
                out = runner.run(fixture()["data"], "is my metoprolol why I feel tired?", TS_SCRUB, live=True, persist=False)
            self.assertEqual(sent, [])
            self.assertTrue(out.fallback_used)

    def test_local_mode_keeps_the_original_text(self):
        dd = data_copy()
        _plan_store(dd, "Taking it easy until my BP is sorted")
        out = runner.run(dd, "is my metoprolol why I feel tired?", TS_SCRUB, mode="local", persist=False)
        self.assertIn("is my metoprolol why I feel tired?", out.prompt)
        self.assertIn("my BP is sorted", out.prompt)
        self.assertNotIn("on file]", out.prompt)
        self.assertNotIn("egress", out.trace.sections)
        self.assertNotIn("medication_question", out.pre.flags)       # local flagging is unchanged

    def test_remote_without_the_scrubbed_term_behaves_as_before(self):
        a = _remote("how did I sleep?")
        self.assertNotIn("medication", a.trace.sections["egress"]["by_category"])    # only the seeded plan's BP is scrubbed
        self.assertNotIn("medication_question", a.pre.flags)
        self.assertEqual(a.route.tier, "small")

    def test_a_local_lookup_still_answers_locally_in_remote_mode(self):
        out = _remote("when was my last A1c")
        self.assertFalse(out.model_called)
        self.assertEqual(out.local_kind, "ehr")


class LabContextAndValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dl = guard().denylist()

    def s(self, text):
        from domains.health.egress import scrub_terms
        return scrub_terms(text, self.dl)[0]

    def test_specific_clinical_terms_are_scrubbed_everywhere(self):
        for text in ("A1c is a thing I read about", "creatinine again", "what is HbA1c anyway", "glycated hemoglobin", "LDL"):
            self.assertIn("[a lab result on file]", self.s(text), text)

    def test_ordinary_word_analytes_pass_outside_lab_context(self):
        for text in ("I had a salty, high-sodium dinner", "glucose gels on my long run", "calcium chews and potassium-rich bananas",
                     "added a pinch of sodium to the bottle"):
            self.assertEqual(self.s(text), text)
            self.assertEqual(scan(text, self.dl), [], text)

    def test_ordinary_word_analytes_are_scrubbed_and_guarded_in_lab_context(self):
        self.assertEqual(self.s("my sodium was 133"), "my [a lab result on file] was [a value]")
        self.assertEqual(self.s("glucose level came back high"), "[a lab result on file] level came back high")
        self.assertEqual(self.s("glucose gels on my long run, thoughts?"), "glucose gels on my long run, thoughts?")
        self.assertEqual(self.s("my potassium result worried me"), "my [a lab result on file] result worried me")
        self.assertEqual(self.s("calcium 9.4 mg/dL"), "[a lab result on file] [a value]")
        for text in ("glucose level came back high", "my sodium was 133", "glucose panel"):
            self.assertEqual(scan(text, self.dl), ["lab_name"], text)

    def test_a_value_after_a_lab_term_is_scrubbed_with_or_without_a_unit(self):
        self.assertEqual(self.s("LDL cholesterol 104 mg/dL, fine"), "[a lab result on file] [a value], fine")
        self.assertEqual(self.s("my A1c was 5.8%"), "my [a lab result on file] was [a value]")
        self.assertEqual(self.s("my A1c was 5.8, ok"), "my [a lab result on file] was [a value], ok")
        self.assertEqual(self.s("A1c 5.8 % and creatinine 0.9 mg/dL"), "[a lab result on file] [a value] and [a lab result on file] [a value]")

    def test_a_number_far_from_the_lab_term_is_left_alone(self):
        self.assertEqual(self.s("my A1c came back after 3 weeks of waiting"), "my [a lab result on file] came back after 3 weeks of waiting")

    def test_a_lab_term_in_a_message_that_is_not_a_results_question_goes_to_the_model_scrubbed(self):
        out = _remote("How can I lower my A1c from 5.8?")
        self.assertIn("How can I lower my [a lab result on file] from [a value]?", out.prompt)
        self.assertNotIn("5.8", out.system + out.prompt)
        self.assertTrue(out.model_called)
        self.assertFalse(out.fallback_used)
        self.assertEqual(scan(out.system + out.prompt, guard().denylist()), [])
        self.assertIn("lab_question", out.pre.flags)
        self.assertIn(out.route.tier, ("mid", "large"))
        self.assertRegex(out.response.lower(), r"clinician|doctor")
        self.assertEqual(out.trace.sections["egress"]["by_category"].get("value"), 1)
        self.assertTrue(out.post.passed, out.post.checks)
        out = _remote("should I eat before my A1c test tomorrow?")
        self.assertIn("should I eat before my [a lab result on file] test tomorrow?", out.prompt)
        self.assertTrue(out.model_called)

    def test_the_post_check_requires_a_clinician_for_a_lab_question(self):
        from domains.health.domain import post_requirements
        from harness.safety import PreCheck
        self.assertIn(["clinician", "doctor"], post_requirements(PreCheck("elevated", ["lab_question"]), None))

    def test_a_name_is_scrubbed(self):
        out = _remote("Hi, it's Jordan")
        self.assertIn("Hi, it's [an identifier]", out.prompt)
        self.assertFalse(out.fallback_used)
        self.assertEqual(scan(out.system + out.prompt, guard().denylist()), [])

    def test_a_residual_hit_still_blocks_when_the_scrub_is_off(self):
        with mock.patch.object(Guard, "scrub_detail", lambda self, t: (t, 0, {})):
            for msg, cat in (("Hi, it's Jordan", "name"), ("my result was 5.8 % last time", "lab_value"),
                             ("a glucose panel tomorrow", "lab_name")):
                out = _remote(msg)
                self.assertTrue(out.fallback_used, msg)
                self.assertFalse(out.model_called, msg)
                self.assertIn(cat, out.trace.sections["send_blocked"]["reason"], msg)

    def test_lookups_stay_local(self):
        for msg in ("when was my last A1c", "what was my A1c in 2025"):
            out = _remote(msg)
            self.assertEqual(out.local_kind, "ehr", msg)

    def test_the_rendered_timestamp_and_other_lines_do_not_make_a_lab_context(self):
        # the incoming line is rendered as "[2026-09-29T08:10] <text>": that timestamp is not a number near the analyte
        for msg in ("glucose gels on my long run, thoughts?", "I had a salty, high-sodium dinner, will that hurt my sleep?"):
            out = _remote(msg)
            self.assertIn(f"] {msg}", out.prompt, msg)
            self.assertNotIn("[a lab result on file]", out.prompt.split("## Incoming message")[1], msg)
            self.assertEqual(scan(out.system + out.prompt, guard().denylist()), [], msg)
            self.assertTrue(out.model_called, msg)
        from domains.health.egress import in_lab_context
        text = "[2026-09-29T08:10] glucose gels on my long run"
        self.assertFalse(in_lab_context(text, 19, 26))
        self.assertTrue(in_lab_context("[2026-09-29T08:10] glucose 96 mg/dL", 19, 26))
        self.assertFalse(in_lab_context("Last night: 4h40\nglucose gels", 17, 24))        # another line's numbers do not count

    def test_a_number_near_the_analyte_is_scrubbed_even_when_it_is_a_gel_dose(self):
        # ambiguous by design: "25 g" is next to "glucose", so it is treated as lab context and scrubbed (fail safe)
        out = _remote("gel with 25 g glucose before the long run")
        self.assertIn("gel with 25 g [a lab result on file] before the long run", out.prompt)

    def test_redaction_at_rest_covers_ordinary_words_in_lab_context(self):
        from domains.health.egress import redact
        text, n = redact("my sodium was 133. Later a long ride with sodium chews", self.dl)
        self.assertNotIn("sodium was", text)
        self.assertIn("with sodium chews", text)


class RecallToolResultTests(unittest.TestCase):
    def test_recall_results_are_scrubbed_before_they_go_back_to_the_model(self):
        from harness.memory import MemoryRecord, MemoryStore
        dd = data_copy()
        s = MemoryStore.load(dd / "memory.json")
        s.records.append(MemoryRecord(id="r92", type="fact", content="Started metoprolol in the winter; my A1c was 5.8 at that visit.",
                                      created_at="2026-03-01T10:00:00", updated_at="2026-03-01T10:00:00",
                                      source="user_stated", confidence=0.9, topic="fact:visit"))
        s.save(dd / "memory.json")
        sent = []

        def block(kind, **kw):
            b = type("B", (), kw)()
            b.type = kind
            b.model_dump = lambda: {"type": kind, **kw}
            return b

        class Resp:
            def __init__(self, stop, content):
                self.stop_reason, self.content = stop, content
                self.usage = type("U", (), {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})()

        def create(**kw):
            sent.append(json.loads(json.dumps(kw, default=lambda o: o.model_dump())))
            if len(sent) == 1:
                return Resp("tool_use", [block("tool_use", id="t1", name="recall", input={"ids": ["r92"]})])
            return Resp("end_turn", [block("text", text="Please ask your prescriber or pharmacist; your doctor can interpret it.")])

        client = BoundaryTests.FakeClient([])
        client.create = create
        with mock.patch.object(provider, "_client_factory", lambda: client):
            out = runner.run(dd, "what did I note about my last visit?", TS_SCRUB, live=True, persist=False)
        self.assertEqual(len(sent), 2, out.trace.sections.get("send_blocked"))
        result = sent[1]["messages"][-1]["content"][0]
        self.assertEqual(result["type"], "tool_result")
        self.assertIn("[r92]", result["content"])
        self.assertIn("Started [a medication on file] in the winter; my [a lab result on file] was [a value] at that visit.", result["content"])
        wire = json.dumps(sent[1]).lower()
        for word in ("metoprolol", "a1c", "5.8"):
            self.assertNotIn(word, wire)
        self.assertEqual(out.trace.sections["model_call"]["calls"][0]["tool_calls"][0]["ids_returned"], ["r92"])


class LabResultQuestionTests(unittest.TestCase):
    NOTE = "This is flagged for your clinician; I can note it for your next visit."

    def ask(self, msg, mode):
        calls = []
        out = runner.run(fixture()["data"], msg, TS_SCRUB, mode=mode, persist=False, sender=lambda *a: calls.append(a))
        self.assertEqual(calls, [])
        self.assertFalse(out.model_called, msg)
        self.assertEqual(out.local_kind, "ehr")
        self.assertEqual((out.system, out.prompt), ("", ""))
        self.assertEqual(out.trace.sections["model_call"]["skipped"], "answered locally from EHR; no model call")
        return out.response

    def test_questions_about_own_results_are_answered_locally_in_both_modes(self):
        for mode in ("local", "remote"):
            r = self.ask("My A1c was 5.8, should I worry?", mode)
            self.assertIn("Hemoglobin A1c on file: 5.8 % (Jun 2026), 5.6 % (Nov 2025), 5.4 % (Mar 2025).", r)
            self.assertIn("The 5.8 you mention matches the Jun 2026 result.", r)
            self.assertIn("Reference range <5.7 %; the latest result is outside it.", r)
            self.assertIn("A flagged trend is on file: results have been going up.", r)
            self.assertIn(self.NOTE, r)
            r = self.ask("is my A1c bad?", mode)
            self.assertIn("Hemoglobin A1c on file: 5.8 % (Jun 2026)", r)
            self.assertNotIn("you mention", r)
            self.assertIn(self.NOTE, r)
            r = self.ask("my sodium was 133 last week", mode)
            self.assertIn("Sodium on file: 140 mmol/L (Jun 2026).", r)
            self.assertIn("I don't see a result of 133 on file, so I can't confirm that number.", r)
            self.assertIn(self.NOTE, r)

    def test_no_interpretation_beyond_the_records_own_range(self):
        r = self.ask("is my A1c bad?", "remote")
        for word in ("diabet", "prediabet", "risk", "should", "concerning", "healthy"):
            self.assertNotIn(word, r.lower())

    def test_a_typed_value_that_matches_nothing_says_so_and_lists_what_is_on_file(self):
        r = self.ask("My A1c was 5.9, should I worry?", "remote")
        self.assertIn("I don't see a result of 5.9 on file", r)
        self.assertIn("5.8 % (Jun 2026), 5.6 % (Nov 2025), 5.4 % (Mar 2025)", r)

    def test_the_answer_names_only_the_test_asked_about(self):
        r = self.ask("is my LDL bad?", "local")
        self.assertIn("LDL cholesterol on file: 104 mg/dL (Jun 2026), 118 mg/dL (Nov 2025).", r)
        self.assertNotIn("A1c", r)

    def test_advice_and_non_result_mentions_still_go_to_the_model(self):
        for msg in ("How can I lower my A1c from 5.8?", "should I eat before my A1c test tomorrow?", "glucose gels on my long run, thoughts?"):
            out = runner.run(fixture()["data"], msg, TS_SCRUB, mode="remote", persist=False)
            self.assertTrue(out.model_called, msg)

    def test_the_stored_history_and_trace_hold_no_answer_or_value(self):
        dd = data_copy()
        runner.run(dd, "My A1c was 5.8, should I worry?", TS_SCRUB, mode="remote", persist=True)
        h = (dd / "history.json").read_text()
        self.assertIn("(answered locally from your records)", h)
        self.assertNotIn("5.8", h)
        dl = Guard(dd, PASSPHRASE).denylist()
        for f in (dd / "traces").glob("*"):
            self.assertEqual(scan(f.read_text(), dl), [], f.name)


class ReviewRegressionTests(unittest.TestCase):
    """Fixes from the 2026-10-01 safety review."""

    def test_memory_is_redacted_before_it_is_clipped(self):
        from domains.health.runner import load_memory
        dd = data_copy()
        msg = "a " * 67 + "metoprolol tonight, slept badly"     # the drug name straddles the 140-character clip
        store = load_memory(dd)
        runner.run(dd, msg, "2026-09-29T08:10:00", persist=False, store=store)
        written = [r.content for r in store.records if r.origin == "user" and r.type == "episode"
                   and "a a a" in r.content]
        self.assertTrue(written)
        for c in written:
            self.assertNotIn("metopr", c.lower())
            self.assertIn("[redac", c)                       # the placeholder itself may be clipped

    def test_a_partial_ehr_date_does_not_break_the_denylist_or_a_turn(self):
        dd = data_copy()

        def partial(data):
            data["conditions"][0]["onset"] = "2019"
            data["conditions"][0]["recorded"] = "2019-03"
        Vault(dd, PASSPHRASE).update("import", partial)
        dl = Guard(dd, PASSPHRASE).denylist()
        self.assertIsNotNone(dl)
        runner.run(dd, "slept badly, should I rest?", "2026-09-29T08:10:00", persist=True)
        runner.run(dd, "slept badly, should I rest?", "2026-09-29T08:20:00", mode="remote", persist=True)
