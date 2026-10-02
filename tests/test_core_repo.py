"""Repository rules: layering, the single SDK import, the health domain's only road to a model, no mock
EHR values outside the generator and the tests, banned words, docs and config as specified."""
from __future__ import annotations

import ast
import codecs
import re
import subprocess
import unittest
from pathlib import Path

from tests.support import ROOT

from domains.health.mock.generate import EHR_FACTS as F

SKIP_DIRS = {".git", "__pycache__", "runs", "mock_exports", ".venv", "tests", "data"}
MOCK_DIR = Path("domains/health/mock")           # the one place mock EHR values may live (besides tests/)
GENERIC_DRUG_NAMES_OK_IN = {Path("domains/health/ehr/classes.py"), Path("domains/health/ehr/synonyms.py")}   # the local class and synonym maps name drugs generically
PROVIDER_FILE = "providers/anthropic.py"
BOUNDARY_FILE = "domains/health/model_boundary.py"


def repo_files(skip=SKIP_DIRS):
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if p.is_file() and not (set(rel.parts) & skip) and p.suffix != ".pyc":
            yield rel, p


def py_files():
    """Every python file in the repo, tests included; only the virtualenv and caches are skipped."""
    for rel, p in repo_files(SKIP_DIRS - {"tests"}):
        if p.suffix == ".py":
            yield rel, p


def imported_modules(tree: ast.AST, package: str = "") -> list[str]:
    """Absolute module names imported anywhere in the tree (relative imports resolved against `package`)."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[:len(package.split(".")) - node.level + 1]
                out.append(".".join(base + ([node.module] if node.module else [])))
            else:
                out.append(node.module or "")
    return out


def patient_values() -> list[str]:
    p = F["patient"]
    vals = [p["family"], p["given"][0], p["mrn"], p["epi"], p["phone"], p["email"], p["address"]["line"], p["address"]["postal"],
            p["birth"], F["condition"]["display"], F["allergy"]["display"], F["allergy"]["rxnorm"].join(["rxnorm:", ""]), F["epic_local_oid"]]
    vals += [F["condition"]["snomed"], F["condition"]["icd10"] + '"', F["condition"]["onset"]]
    for m in F["meds"]:
        vals += [m["rxnorm"], m["display"], m["text"], m["authored"][:10]]
    for l in F["labs"]:
        vals += [l[0], l[1], l[5][:10], f"{l[3]:g} {l[4]}"]
    for eff, s, d in F["bp"]:
        vals += [eff[:10], f"{s}/{d}"]
    return [v for v in vals if len(v) >= 4]


class LayeringTests(unittest.TestCase):
    """Rule 1: the core never imports from domains/ or providers/ (or from anything that sits on top of it)."""

    def test_the_core_never_imports_domains_or_providers(self):
        for p in sorted((ROOT / "harness").glob("*.py")):
            mods = imported_modules(ast.parse(p.read_text()), f"harness.{p.stem}")
            bad = [m for m in mods if m.split(".")[0] in ("domains", "providers", "evals", "tests", "run", "health_cli")]
            self.assertEqual(bad, [], p.name)

    def test_the_core_stays_domain_free(self):
        for p in (ROOT / "harness").glob("*.py"):
            text = p.read_text()
            for word in ("WHOOP", "Garmin", "Epic", "HbA1c"):
                self.assertNotIn(word, text, p.name)

    def test_the_two_domains_do_not_import_each_other(self):
        for dom, other in (("lab", "health"), ("health", "lab")):
            for p in (ROOT / "domains" / dom).rglob("*.py"):
                mods = imported_modules(ast.parse(p.read_text()), ".".join(p.relative_to(ROOT).with_suffix("").parts))
                self.assertEqual([m for m in mods if m.startswith(f"domains.{other}")], [], str(p))

    def test_there_is_one_core_and_no_vendored_copy(self):
        self.assertFalse((ROOT / "harness" / "VENDORED_FROM").exists())
        self.assertNotIn("VENDORED_FROM", (ROOT / "harness" / "__init__.py").read_text())
        self.assertFalse((ROOT / "health").exists() or (ROOT / "mock").exists() or (ROOT / "cli.py").exists())


class SdkImportTests(unittest.TestCase):
    def test_only_the_provider_module_imports_anthropic(self):          # rule 2, whole repo, tests included
        offenders = []
        for rel, p in py_files():
            tree = ast.parse(p.read_text())
            if any(m.split(".")[0] == "anthropic" for m in imported_modules(tree)):
                offenders.append(rel.as_posix())
            for node in ast.walk(tree):             # importlib tricks count too
                if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in ("import_module", "__import__") \
                        and node.args and isinstance(node.args[0], ast.Constant) and "anthropic" in str(node.args[0].value):
                    offenders.append(rel.as_posix())
        self.assertEqual(sorted(set(offenders)), [PROVIDER_FILE])

    def test_the_provider_does_no_egress_logic(self):
        text = (ROOT / PROVIDER_FILE).read_text()
        tree = ast.parse(text)
        mods = imported_modules(tree)
        self.assertEqual([m for m in mods if m.startswith("domains")], [])
        for word in ("scrub", "denylist", "EgressBlocked", "Guard"):
            self.assertNotIn(word, text)


class HealthRoadToTheModelTests(unittest.TestCase):
    """Rule 3: the health domain reaches the provider only through send_remote, which guards first."""

    def test_only_the_boundary_module_in_the_health_domain_touches_providers(self):
        users = []
        for p in (ROOT / "domains" / "health").rglob("*.py"):
            rel = p.relative_to(ROOT)
            mods = imported_modules(ast.parse(p.read_text()), ".".join(rel.with_suffix("").parts))
            if any(m.split(".")[0] == "providers" for m in mods):
                users.append(rel.as_posix())
        self.assertEqual(users, [BOUNDARY_FILE])

    def test_nothing_in_the_health_domain_reaches_the_provider_by_name_or_string(self):
        for p in (ROOT / "domains" / "health").rglob("*.py"):
            if p.relative_to(ROOT).as_posix() == BOUNDARY_FILE:
                continue
            text = p.read_text()
            self.assertNotIn("providers", text, p.name)

    def test_send_remote_hands_the_provider_the_guard_as_its_check(self):
        from unittest import mock
        from domains.health import model_boundary
        from domains.health.egress import EgressBlocked
        from harness.router import Route
        seen = {}

        class Block:
            def check_request(self, request):
                raise EgressBlocked(["medication"])

        def fake_send(system, prompt, route, executor, check=None):
            seen["check"] = check
            return "sentinel"
        with mock.patch.object(model_boundary.provider, "send", fake_send):
            out = model_boundary.send_remote("s", "p", Route("small", "m", None), lambda n, i: "", Block())
        self.assertEqual(out, "sentinel")
        with self.assertRaises(EgressBlocked):                      # not vacuous: the check it passed is the guard's
            seen["check"]({"messages": []})

    def test_a_live_health_turn_is_always_built_with_scrub_guard_and_send_remote(self):
        from domains.health import runner
        from tests.support import fixture
        b = runner.make_boundary(fixture()["data"], "remote", True)
        self.assertEqual(b.mode, "remote")
        for hook in (b.sender, b.preflight, b.scrub, b.scrub_detail):
            self.assertIsNotNone(hook)
        self.assertEqual(b.sender.__module__, "domains.health.model_boundary")
        self.assertIsNone(runner.make_boundary(fixture()["data"], "local", True).sender)


class HygieneTests(unittest.TestCase):
    def test_no_mock_ehr_value_outside_the_generator_and_tests(self):          # rule 6
        needles = patient_values()
        bare = [F["meds"][0]["text"].split()[0], F["meds"][1]["text"].split()[0], F["allergy"]["text"].lower()[:-1]]    # generic drug names
        found = []
        for rel, p in repo_files():
            if MOCK_DIR in rel.parents:
                continue
            try:
                text = p.read_text()
            except UnicodeDecodeError:
                continue
            low = text.lower()
            for n in needles:
                if n.lower() in low:
                    found.append((str(rel), n))
            if rel not in GENERIC_DRUG_NAMES_OK_IN:
                for n in bare:
                    if re.search(rf"\b{n}", low):
                        found.append((str(rel), n))
        self.assertEqual(found, [])

    def test_gitignore_is_the_union_of_both_domains(self):
        ignore = (ROOT / ".gitignore").read_text().split()
        for pat in ("runs/", "mock_exports/", "data/", "*.fernet", "*.ndjson", "exports/", "__pycache__/", ".venv/", ".DS_Store"):
            self.assertIn(pat, ignore)

    def test_nothing_that_looks_like_real_data_is_tracked(self):
        try:
            out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        except Exception:
            self.skipTest("no git")
        bad = [f for f in out if f.startswith(("mock_exports/", "data/", "runs/")) or f.endswith((".fernet", ".ndjson", ".salt"))]
        self.assertEqual(bad, [])

    def test_readme_is_short_and_ends_as_specified(self):
        lines = (ROOT / "README.md").read_text().rstrip("\n").splitlines()
        self.assertLess(len(lines), 200)
        self.assertEqual(lines[-1], "Built with Claude Code, which wrote most first drafts.")

    def test_package_metadata_and_dependencies(self):
        py = (ROOT / "pyproject.toml").read_text()
        self.assertIn('name = "context-harness"', py)
        self.assertIn('version = "0.2.0"', py)
        self.assertIn("dependencies = []", py)                      # the core is stdlib-only
        self.assertRegex(py, r'live = \["anthropic>=1\.0"\]')
        self.assertRegex(py, r'health = \["cryptography>=42"\]')
        req = (ROOT / "requirements.txt").read_text()
        self.assertIn("cryptography", req)
        self.assertIn("anthropic", req)
        self.assertRegex(req, r"(?m)^#.*health")
        self.assertRegex(req, r"(?m)^#.*(live|either domain)")


# Rule 7. Stored rot13-encoded so this file does not trip its own check. Common drug names are
# deliberately not banned: the health domain's class map needs them.
BANNED = [codecs.decode(w, "rot13") for w in (
    "oybbz", "znln", "yvyyl", "fpbgg", "frnggyr", "plc1n2", "gnxr-ubzr", "nffvtazrag", "fhozvffvba",
    "gur oevrs", "gur qrpx", "cneg 1", "cneg 2", "cneg 3",
    "fhggre", "zl urnygu bayvar")]


class BannedWordTests(unittest.TestCase):
    def tracked_and_unignored(self) -> list[Path]:
        try:
            out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.splitlines()
        except Exception:
            return [rel for rel, _ in repo_files(SKIP_DIRS - {"tests"})]
        return [Path(f) for f in out if (ROOT / f).is_file()]

    def test_no_banned_word_in_any_file(self):
        hits = []
        for rel in self.tracked_and_unignored():
            if rel.name == "test_core_repo.py":
                continue
            try:
                low = (ROOT / rel).read_text().lower()
            except UnicodeDecodeError:
                continue
            hits += [(str(rel), w) for w in BANNED if w in low]
            hits += [(str(rel), "name") for w in BANNED if w in str(rel).lower()]
        self.assertEqual(hits, [])

    def test_no_banned_word_in_this_files_own_source_in_plain_form(self):
        low = Path(__file__).read_text().lower()
        self.assertEqual([w for w in BANNED if w in low], [])

    def test_no_banned_word_in_any_commit_message(self):
        try:
            log = subprocess.run(["git", "log", "--all", "--format=%B"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.lower()
        except Exception:
            self.skipTest("no git")
        self.assertEqual([w for w in BANNED if w in log], [])


if __name__ == "__main__":
    unittest.main()
