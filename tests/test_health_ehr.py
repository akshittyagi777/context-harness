"""Local EHR handling: the vault, derived constraints, trends and local answers."""
from __future__ import annotations

import json
import os
import re
import unittest
from datetime import datetime
from unittest import mock

from tests import support
from tests.support import PASSPHRASE, data_copy, empty_data, fixture

from harness.memory import MemoryStore
from domains.health.baselines import load_regimes
from domains.health.egress import build_denylist, scan
from domains.health.ehr import answer
from domains.health.ehr.classes import BRANDS, classify, names_for
from domains.health.ehr.derive import (TEMPLATES, apply_to_memory, derive_constraints, is_allowlisted, month_label, run_derive)
from domains.health.ehr.model import EHR, add_resource, natural_key
from domains.health.ehr.trends import flagged_trends
from domains.health.vault import Vault, VaultError, VaultLocked, empty_store


def ehr_from(**kw):
    store = empty_store()
    for k, v in kw.items():
        store[k] = v
    return EHR(store)


def med(display, status="active", authored="2026-02-10T17:40:00Z", codes=None):
    return {"local_id": "m", "display": display, "text": display, "codes": codes or [], "status": status, "authored_on": authored}


def lab(code, value, eff, hi=None, lo=None):
    return {"local_id": "l", "display": "Test " + code, "codes": [{"system": "http://loinc.org", "code": code}], "value": value,
            "unit": "x", "effective": eff, "ref_high": hi, "ref_low": lo}


def real_ehr():
    return EHR(Vault(fixture()["data"]).read("derive"))


class VaultTests(unittest.TestCase):
    def test_round_trip(self):
        dd = empty_data()
        v = Vault(dd, PASSPHRASE)
        data = empty_store()
        data["conditions"].append({"display": "Roundtrip condition"})
        v.write("import", data)
        self.assertEqual(Vault(dd, PASSPHRASE).read("derive", ["conditions"])["conditions"][0]["display"], "Roundtrip condition")

    def test_file_is_encrypted_at_rest(self):
        raw = (fixture()["data"] / "ehr.fernet").read_bytes()
        for token in (b"metoprolol", b"Rivera", b"Hypertension", b"4548-4", b"Z0048213"):
            self.assertNotIn(token, raw)
        self.assertTrue(raw.startswith(b"gAAAA"))          # a Fernet token

    def test_refuses_without_a_passphrase_and_says_why(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("HEALTH_HARNESS_PASSPHRASE", None)
            with self.assertRaises(VaultLocked) as cm:
                Vault(fixture()["data"]).read("derive")
            self.assertIn("HEALTH_HARNESS_PASSPHRASE", str(cm.exception))
            with self.assertRaises(VaultLocked):
                Vault(empty_data()).write("import", empty_store())

    def test_wrong_passphrase_is_refused(self):
        with self.assertRaises(VaultLocked):
            Vault(fixture()["data"], "wrong").read("derive")
        self.assertFalse(Vault(fixture()["data"], "wrong").unlocked())

    def test_key_is_scrypt_with_a_random_salt_next_to_the_data(self):
        a, b = empty_data(), empty_data()
        for d in (a, b):
            Vault(d, PASSPHRASE).write("import", empty_store())
        self.assertEqual(len((a / "ehr.salt").read_bytes()), 16)
        self.assertNotEqual((a / "ehr.salt").read_bytes(), (b / "ehr.salt").read_bytes())
        self.assertNotEqual((a / "ehr.fernet").read_bytes()[:40], (b / "ehr.fernet").read_bytes()[:40])
        import inspect, domains.health.vault as vm
        self.assertIn("hashlib.scrypt", inspect.getsource(vm))

    def test_files_are_owner_only(self):
        dd = fixture()["data"]
        for name in ("ehr.fernet", "ehr.salt", "access.log"):
            self.assertEqual(oct((dd / name).stat().st_mode)[-3:], "600", name)

    def test_data_dir_is_created_0700(self):
        from domains.health.config import resolve_data_dir
        d = support._tmp() / "fresh" / "nested"
        resolve_data_dir(d)
        self.assertEqual(oct(d.stat().st_mode)[-3:], "700")

    def test_missing_cryptography_gives_an_install_hint(self):
        with mock.patch.dict("sys.modules", {"cryptography": None, "cryptography.fernet": None}):
            with self.assertRaises(VaultError) as cm:
                Vault(empty_data(), PASSPHRASE).write("import", empty_store())
        self.assertIn("pip install cryptography", str(cm.exception))

    def test_every_read_is_logged_with_types_and_no_values(self):
        dd = data_copy()
        v = Vault(dd, PASSPHRASE)
        before = len(v.read_access_log())
        v.read("local answer", ["allergies"])
        entry = v.read_access_log()[-1]
        self.assertEqual(len(v.read_access_log()), before + 1)
        self.assertEqual(set(entry), {"ts", "purpose", "resource_types"})
        self.assertEqual((entry["purpose"], entry["resource_types"]), ("local answer", ["AllergyIntolerance"]))

    def test_access_log_is_append_only_json_lines(self):
        dd = data_copy()
        v = Vault(dd, PASSPHRASE)
        first = v.log_path.read_text()
        v.read("derive", ["labs"])
        self.assertTrue(v.log_path.read_text().startswith(first))
        for line in v.log_path.read_text().splitlines():
            json.loads(line)

    def test_reading_returns_only_what_was_asked_for(self):
        got = Vault(fixture()["data"]).read("local answer", ["allergies"])
        self.assertEqual(set(got) - {"version"}, {"allergies"})

    def test_a_missing_store_reads_empty_and_logs_nothing(self):
        dd = empty_data()
        v = Vault(dd, PASSPHRASE)
        self.assertEqual(v.read("derive")["conditions"], [])
        self.assertEqual(v.read_access_log(), [])

    def test_purposes_in_the_real_log(self):
        purposes = {e["purpose"] for e in Vault(fixture()["data"]).read_access_log()}
        self.assertTrue({"import", "derive"} <= purposes, purposes)


class MergeTests(unittest.TestCase):
    def test_same_fact_from_two_sources_is_added_once(self):
        s = empty_store()
        item = {"display": "x", "codes": [{"system": "http://loinc.org", "code": "1-1"}], "effective": "2026-01-01T10:00:00Z"}
        self.assertTrue(add_resource(s, "labs", item, "a"))
        self.assertFalse(add_resource(s, "labs", item, "b"))
        self.assertEqual(s["labs"][0]["source_ids"], ["a", "b"])
        self.assertEqual(s["labs"][0]["local_id"], "lab1")

    def test_natural_keys(self):
        self.assertEqual(natural_key("allergies", {"codes": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "7980"}]}),
                         ("rxnorm:7980",))


class ClassTests(unittest.TestCase):
    def test_name_and_atc_classification(self):
        self.assertEqual(classify({"display": "metoprolol succinate 50 MG", "text": "", "codes": []}), "beta_blocker")
        self.assertEqual(classify({"display": "Tab X", "codes": [{"system": "http://www.whocc.no/atc", "code": "C07AB02"}]}), "beta_blocker")
        self.assertEqual(classify({"display": "sertraline"}), "ssri_snri")
        self.assertIsNone(classify({"display": "zorbazine 5 mg"}))

    def test_names_include_generic_and_brand(self):
        names = {n.lower() for n in names_for({"display": "metoprolol succinate 50 MG 24 HR Extended Release Oral Tablet", "text": "metoprolol succinate ER 50 mg tablet"})}
        self.assertTrue({"metoprolol", "toprol-xl", "lopressor"} <= names)
        self.assertIn("pen-vee k", {n.lower() for n in names_for({"display": "Penicillin G", "text": "Penicillins"})})
        self.assertIn("Aleve", BRANDS["naproxen"])


class DeriveTests(unittest.TestCase):
    def test_the_mock_ehr_derives_exactly_four_constraints(self):
        d = derive_constraints(real_ehr())
        self.assertEqual({x.template_id for x in d}, {"rhr_clinical_factor", "supp_sleep_check", "lab_trend_flag", "no_specific_meds"})

    def test_every_constraint_is_an_allowlisted_template_sentence(self):
        for d in derive_constraints(real_ehr()):
            self.assertTrue(is_allowlisted(d.text), d.text)
            self.assertIn(d.template_id, TEMPLATES)

    def test_allowlist_rejects_anything_else(self):
        for bad in ("Avoid metoprolol.", "A clinical factor lowers resting heart rate and may blunt heart-rate response from 2026-02-10; do not "
                    "read a lower resting heart rate since then as a fitness gain.", "Do not suggest specific medications like penicillin.",
                    "A clinical factor lowers resting heart rate and may blunt heart-rate response from Febuary 2026; do not read a lower "
                    "resting heart rate since then as a fitness gain."):
            self.assertFalse(is_allowlisted(bad), bad)
        self.assertTrue(is_allowlisted(TEMPLATES["supp_sleep_check"]))

    def test_no_constraint_contains_a_denylisted_token(self):
        dl = build_denylist(Vault(fixture()["data"]).read("egress guard"))
        texts = [d.text for d in derive_constraints(real_ehr())]
        self.assertEqual(len(texts), 4)
        for t in texts:
            self.assertEqual(scan(t, dl), [], t)
        for t in TEMPLATES.values():
            self.assertEqual(scan(t.format(month="February 2026") if "{month}" in t else t, dl), [], t)

    def test_the_only_date_in_a_constraint_is_a_month(self):
        d = next(x for x in derive_constraints(real_ehr()) if x.template_id == "rhr_clinical_factor")
        self.assertIn("from February 2026;", d.text)
        self.assertFalse(re.search(r"\d{4}-\d{2}-\d{2}|\b\d{1,2}/\d{1,2}/\d{2,4}\b", d.text))
        self.assertEqual(month_label("2026-02-10T17:40:00Z"), "February 2026")

    def test_regime_marks_the_start_month(self):
        d = next(x for x in derive_constraints(real_ehr()) if x.template_id == "rhr_clinical_factor")
        self.assertEqual((d.regime["start"], d.regime["end"], d.regime["kind"]), ("2026-02-01", "2026-02-28", "clinical_change"))
        self.assertTrue(any(r["kind"] == "clinical_change" for r in load_regimes(fixture()["data"])))

    def test_unknown_medication_maps_to_the_conservative_default(self):
        d = derive_constraints(ehr_from(medications=[med("zorbazine 5 MG Oral Tablet")]))
        self.assertEqual([x.template_id for x in d], ["supp_sleep_check"])

    def test_beta_blocker_adds_the_clinical_factor_template(self):
        d = derive_constraints(ehr_from(medications=[med("atenolol 25 MG Oral Tablet")]))
        self.assertEqual({x.template_id for x in d}, {"rhr_clinical_factor", "supp_sleep_check"})

    def test_known_non_interacting_class_and_inactive_meds_derive_nothing(self):
        self.assertEqual(derive_constraints(ehr_from(medications=[med("naproxen 220 MG")])), [])
        self.assertEqual(derive_constraints(ehr_from(medications=[med("metoprolol 50", status="stopped")])), [])

    def test_only_drug_allergies_derive_a_constraint(self):
        food = {"local_id": "a", "display": "Peanut", "category": "food", "status": "active"}
        drug = {"local_id": "a", "display": "Sulfa", "category": "medication", "status": "active"}
        self.assertEqual(derive_constraints(ehr_from(allergies=[food])), [])
        self.assertEqual([x.template_id for x in derive_constraints(ehr_from(allergies=[drug]))], ["no_specific_meds"])

    def test_lab_trend_needs_three_monotonic_results_that_cross_the_range(self):
        L = lambda vals, **kw: ehr_from(labs=[lab("A-1", v, f"2025-0{i + 1}-01T10:00:00Z", **kw) for i, v in enumerate(vals)])
        self.assertEqual(len(flagged_trends(L([5.4, 5.6, 5.8], hi=5.6))), 1)
        self.assertEqual(flagged_trends(L([5.4, 5.8], hi=5.6)), [])                  # only two results
        self.assertEqual(flagged_trends(L([5.4, 5.8, 5.6], hi=5.6)), [])             # not monotonic
        self.assertEqual(flagged_trends(L([5.0, 5.2, 5.4], hi=5.6)), [])             # never crosses
        self.assertEqual(flagged_trends(L([5.8, 5.9, 6.0], hi=5.6)), [])             # was never inside the range
        self.assertEqual(len(flagged_trends(L([9.0, 8.0, 6.0], lo=7.0))), 1)         # a falling trend below the low bound
        self.assertEqual(flagged_trends(L([5.4, 5.6, 5.8])), [])                      # no reference range, no flag
        names = [t["code"] for t in flagged_trends(real_ehr())]
        self.assertEqual(names, ["loinc:4548-4"])                                     # the LDL has two results

    def test_derived_records_are_pinned_derived_local_constraints(self):
        store = MemoryStore.load(fixture()["data"] / "memory.json")
        derived = [r for r in store.active() if r.origin == "derived_local"]
        self.assertEqual(len(derived), 4)
        for r in derived:
            self.assertTrue(r.pinned and r.type == "constraint" and r.id.startswith("dl-"))
            self.assertTrue(is_allowlisted(r.content))

    def test_apply_is_idempotent_and_supersedes_stale_constraints(self):
        store = MemoryStore([])
        now = datetime.fromisoformat("2026-10-01T10:00:00")
        d = derive_constraints(ehr_from(medications=[med("atenolol 25 MG")]))
        self.assertEqual(len(apply_to_memory(store, d, now)), 2)
        self.assertEqual(apply_to_memory(store, d, now), [])                           # nothing changes the second time
        applied = apply_to_memory(store, [], now)                                      # the medication was stopped
        self.assertEqual({a["op"] for a in applied}, {"supersede"})
        self.assertEqual(store.active(), [])

    def test_derive_reads_are_logged_under_the_derive_purpose(self):
        dd = data_copy()
        run_derive(dd, Vault(dd, PASSPHRASE), datetime.fromisoformat("2026-10-01T10:00:00"))
        last = Vault(dd, PASSPHRASE).read_access_log()[-1]
        self.assertEqual(last["purpose"], "derive")
        self.assertNotIn("Patient", last["resource_types"])

    def test_derive_twice_writes_nothing_new(self):
        dd = data_copy()
        before = MemoryStore.load(dd / "memory.json")
        out = run_derive(dd, Vault(dd, PASSPHRASE), datetime.fromisoformat("2026-10-01T10:00:00"))
        after = MemoryStore.load(dd / "memory.json")
        self.assertEqual(out["applied"], [])
        self.assertEqual(len(before.records), len(after.records))
        self.assertEqual(len(load_regimes(dd)), 1)


class AnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ehr = real_ehr()

    def test_medications(self):
        a = answer.answer_question("what meds am I on", self.ehr)
        self.assertIn("metoprolol succinate ER 50 mg tablet", a)
        self.assertIn("started 2026-02-10", a)
        self.assertIn("Not active: naproxen 220 mg tablet (stopped, 2025-08-19)", a)
        self.assertTrue(a.endswith("no model was used.)"))

    def test_allergies(self):
        a = answer.answer_question("what are my allergies?", self.ehr)
        self.assertIn("Penicillins (Hives, moderate)", a)
        self.assertIn("recorded 2012-05-01", a)

    def test_last_a1c_gives_value_date_range_and_history(self):
        a = answer.answer_question("when was my last A1c", self.ehr)
        self.assertIn("Hemoglobin A1c: 5.8 % on 2026-06-09 (reference <5.7 %)", a)
        self.assertIn("earlier 5.6 % on 2025-11-12; 5.4 % on 2025-03-18", a)
        self.assertNotIn("LDL", a)                                                      # only the test that was asked about

    def test_labs_and_conditions(self):
        a = answer.answer_question("show my labs", self.ehr)
        self.assertIn("LDL cholesterol: 104 mg/dL on 2026-06-09", a)
        self.assertIn("Sodium: 140 mmol/L", a)
        c = answer.answer_question("what are my conditions", self.ehr)
        self.assertIn("Hypertension (active, since 2026-02-03)", c)

    def test_unrelated_questions_are_not_answered(self):
        self.assertIsNone(answer.answer_question("how did I sleep", self.ehr))

    def test_responder_stays_out_of_medication_questions(self):
        from domains.health.safety import PRE_PATTERNS
        resp = answer.make_responder(fixture()["data"])
        class P: flags = ["medication_question", "ehr_lookup"]
        self.assertIsNone(resp("can I take melatonin with my meds", P(), None))

    def test_locked_and_empty_stores_answer_locally_too(self):
        class P: flags = ["ehr_lookup"]
        r = answer.make_responder(empty_data())("what meds am I on", P(), None)
        self.assertIn("No EHR data has been imported yet", r.text)
        with mock.patch.dict(os.environ):
            os.environ.pop("HEALTH_HARNESS_PASSPHRASE", None)
            r = answer.make_responder(fixture()["data"])("what meds am I on", P(), None)
        self.assertIn("locked", r.text)
        self.assertEqual(r.note, "answered locally from EHR; no model call")

    def test_ehr_candidates_carry_local_ids_only(self):
        cands = answer.ehr_candidates(fixture()["data"])
        self.assertTrue(cands and all(c.origin == "ehr" for c in cands))
        for c in cands:
            self.assertRegex(c.key, r"^ehr:[A-Za-z]+/(med|cond|alg|lab)\d+$")


if __name__ == "__main__":
    unittest.main()
