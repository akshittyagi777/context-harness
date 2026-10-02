"""Local deterministic answers to questions about the user's own EHR. The model is never involved: the
text comes from code reading the encrypted store, and the turn records "answered locally from EHR; no
model call". Also builds the origin-`ehr` context candidates used in local assembly only."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from harness.context import Candidate
from harness.domain import LocalReply, Scenario

from ..config import ENV_PASSPHRASE
from ..egress import in_lab_context
from ..vault import Vault, VaultError
from .model import EHR, LabResult, date10
from .synonyms import split_lab_terms
from .trends import flagged_trends

NOTE = "answered locally from EHR; no model call"
SUFFIX = " (From your local record; no model was used.)"
_STOP = {"the", "and", "blood", "serum", "plasma", "total", "level", "test", "last", "latest", "recent", "what", "was", "when",
         "are", "for", "with", "lab", "labs", "results", "result", "my", "in", "of", "is", "me", "tell", "show", "value", "values"}


# What makes a message a question about the user's own lab result, as opposed to a lookup or a request for advice.
RESULT_CUES = re.compile(r"(?i)\b(?:worr\w*|bad|abnormal|normal|ok|okay|fine|good|concern\w*|serious|dangerous|came back|come back|"
                         r"trend\w*|rising|dropp\w*|worse|better|improv\w*|elevated|high|low)\b")
PREP_CUES = re.compile(r"(?i)\b(?:fast\w*|prepar\w*|schedul\w*|book\w*|appointment|order\w*|how often|how do i|how can i|how to|"
                       r"diet|eat\w*|drink\w*|supplement\w*|lower my|raise my|reduce my|improve my)\b")
LAB_NOTE = "This is flagged for your clinician; I can note it for your next visit."
_UNIT_RE = r"(?:%|mg/dL|mmol/L|mEq/L|g/dL|ng/mL|IU/L|U/L)"
VALUE_AFTER_TERM = re.compile(r"^(?:[\s,:;=()\-]+[^\s\d\[\]]+){0,2}?[\s,:;=()\-]+(?<![\w.])(\d+(?:\.\d+)?)(?!\w|\.\d)")


def _kinds(low: str) -> list[str]:
    kinds = []
    if re.search(r"allerg", low):
        kinds.append("allergies")
    if re.search(r"\b(?:meds|medications?|prescriptions?|taking|am i on)\b|what meds", low):
        kinds.append("medications")
    if re.search(r"diagnos|\bconditions\b|problem list", low):
        kinds.append("conditions")
    if re.search(r"\blabs?\b|lab results|blood work|a1c|hba1c|ldl|cholesterol", low):
        kinds.append("labs")
    return kinds


def _tests(labs: list[LabResult]) -> dict[str, list[LabResult]]:
    out: dict[str, list[LabResult]] = {}
    for lab in labs:
        out.setdefault(lab.primary_code() or lab.display, []).append(lab)
    for v in out.values():
        v.sort(key=lambda l: l.effective, reverse=True)
    return out


def _val(l: LabResult) -> str:
    return f"{l.value:g} {l.unit}".strip() if l.value is not None else "no value"


def answer_question(text: str, ehr: EHR) -> str | None:
    low = text.lower()
    kinds = _kinds(low)
    if not kinds:
        return None
    parts: list[str] = []
    if "medications" in kinds:
        act, rest = ehr.active_medications(), [m for m in ehr.medications if m.status != "active"]
        parts.append("Active medications: " + ("; ".join(f"{m.text or m.display}" + (f" ({m.dosage})" if m.dosage else "")
                                                       + (f", started {date10(m.authored_on)}" if m.authored_on else "") for m in act)
                                               or "none on file") + ".")
        if rest:
            parts.append("Not active: " + "; ".join(f"{m.text or m.display} ({m.status}, {date10(m.authored_on)})" for m in rest) + ".")
    if "allergies" in kinds:
        parts.append("Allergies: " + ("; ".join(
            f"{a.text or a.display}" + (f" ({', '.join(r['manifestation'] + (', ' + r['severity'] if r['severity'] else '') for r in a.reactions)})"
                                        if a.reactions else "") + (f", recorded {date10(a.recorded)}" if a.recorded else "")
            for a in ehr.allergies) or "none on file") + ".")
    if "conditions" in kinds:
        parts.append("Conditions: " + ("; ".join(f"{c.text or c.display} ({c.status}" + (f", since {date10(c.onset or c.recorded)}" if c.onset or c.recorded else "") + ")"
                                                for c in ehr.conditions) or "none on file") + ".")
    if "labs" in kinds:
        tests = _tests(ehr.labs)
        qtok = {t for t in re.findall(r"[a-z0-9]+", low) if len(t) >= 3 and t not in _STOP}
        asked = {k: v for k, v in tests.items()
                 if qtok & {t for t in re.findall(r"[a-z0-9]+", f"{v[0].display} {v[0].text}".lower()) if len(t) >= 3 and t not in _STOP}}
        shown = asked or tests
        lines = []
        for k, results in shown.items():
            l = results[0]
            rng = f" (reference {l.ref_text})" if l.ref_text else ""
            prev = "; ".join(f"{_val(p)} on {date10(p.effective)}" for p in results[1:4])
            lines.append(f"{l.text or l.display}: {_val(l)} on {date10(l.effective)}{rng}" + (f"; earlier {prev}" if prev else ""))
        parts.append("Labs: " + ("; ".join(lines) or "none on file") + ".")
    return " ".join(parts) + SUFFIX


def make_responder(data_dir: Path) -> Callable[[str, Any, Scenario], LocalReply | None]:
    def respond(text: str, pre: Any, sc: Scenario) -> LocalReply | None:
        if "medication_question" in pre.flags:        # a question about acting on a medication goes to the model route
            return None
        if PREP_CUES.search(text) and _kinds(text.lower()) == ["labs"]:   # advice about a lab ("how can I lower my A1c") is not a lookup
            return None
        kinds = _kinds(text.lower())
        if not kinds:
            return None
        vault = Vault(Path(data_dir))
        if not vault.exists():
            return LocalReply("No EHR data has been imported yet. Import it with `python3 health_cli.py import epic <path>`.",
                              NOTE, "ehr")
        try:
            data = vault.read("local answer", kinds)
        except VaultError:
            return LocalReply(f"The EHR store is locked. Set {ENV_PASSPHRASE} and ask again; nothing was sent anywhere.", NOTE, "ehr")
        ans = answer_question(text, EHR(data))
        return LocalReply(ans or "I could not find that in your local record.", NOTE, "ehr")
    return respond


def ehr_candidates(data_dir: Path) -> list[Candidate]:
    """Origin-`ehr` candidates, rendered only in local assembly. Keys carry the harness-local id only."""
    vault = Vault(Path(data_dir))
    if not vault.exists():
        return []
    try:
        ehr = EHR(vault.read("local assembly", ["medications", "conditions", "allergies", "labs"]))
    except VaultError:
        return []        # locked: local assembly simply has no EHR candidates
    out: list[Candidate] = []

    def add(rtype: str, lid: str, text: str) -> None:
        out.append(Candidate(f"ehr:{rtype}/{lid}", "data", 3, [("full", text)], score=0.4, origin="ehr",
                             meta={"ids": []}, why="local EHR record, local assembly only"))

    for m in ehr.active_medications():
        add("MedicationRequest", m.local_id, f"- Active medication: {m.text or m.display}. {m.dosage}".strip())
    for c in ehr.conditions:
        add("Condition", c.local_id, f"- Condition: {c.text or c.display}.")
    for a in ehr.allergies:
        add("AllergyIntolerance", a.local_id, f"- Allergy: {a.text or a.display}.")
    for results in _tests(ehr.labs).values():
        l = results[0]
        add("Observation", l.local_id, f"- Latest {l.text or l.display}: {_val(l)} on {date10(l.effective)}.")
    return out


# ------------------------------------------------------------------ lab result questions

def _month_year(iso: str) -> str:
    y, m = date10(iso).split("-")[:2]
    return f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(m) - 1]} {y}"


def _lab_groups(ehr: EHR) -> dict[str, list[LabResult]]:
    """Results by test, newest first, for numeric laboratory results."""
    out: dict[str, list[LabResult]] = {}
    for lab in ehr.labs:
        if lab.value is not None:
            out.setdefault(lab.primary_code() or lab.display, []).append(lab)
    for v in out.values():
        v.sort(key=lambda l: l.effective, reverse=True)
    return out


def _mentioned(text: str, results: list[LabResult]) -> re.Match[str] | None:
    """The first mention of this test in `text`: a specific term anywhere, an ordinary-word analyte in lab context."""
    specific: set[str] = set()
    ordinary: set[str] = set()
    for l in results:
        sp, od = split_lab_terms({"display": l.display, "text": l.text})
        specific |= sp
        ordinary |= od
    best = None
    for terms, ctx in ((specific, False), (ordinary, True)):
        if not terms:
            continue
        rx = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")(?![A-Za-z0-9])", re.I)
        for m in rx.finditer(text):
            if (not ctx or in_lab_context(text, m.start(), m.end())) and (best is None or m.start() < best.start()):
                best = m
                break
    return best


def _range_text(l: LabResult) -> str:
    if l.ref_text:
        return l.ref_text
    parts = ([f">= {l.ref_low:g}"] if l.ref_low is not None else []) + ([f"<= {l.ref_high:g}"] if l.ref_high is not None else [])
    return " and ".join(parts)


def lab_answer(text: str, ehr: EHR) -> str | None:
    """Answer a question about the user's own lab result from the store, or None when `text` is not one.
    Lists what is on file (month and year, unit, the reference range the record carries, a flagged trend direction
    when trends.py finds one), says whether a typed value matches a result on file, and adds no interpretation
    beyond in or out of the record's own range."""
    groups = [(code, rs, m) for code, rs in _lab_groups(ehr).items() if (m := _mentioned(text, rs))]
    if not groups or PREP_CUES.search(text):
        return None
    typed = {code: VALUE_AFTER_TERM.search(text[m.end():]) for code, _, m in groups}
    if not any(typed.values()) and not RESULT_CUES.search(text):
        return None
    flagged = {f["code"]: f for f in flagged_trends(ehr)}
    parts: list[str] = []
    for code, rs, _ in groups:
        latest = rs[0]
        name = latest.text or latest.display
        listing = ", ".join(f"{_val(r)} ({_month_year(r.effective)})" for r in rs)
        sentence = f"{name} on file: {listing}."
        t = typed[code]
        if t:
            v = float(t.group(1))
            hit = next((r for r in rs if abs(r.value - v) < 1e-9), None)
            sentence += (f" The {t.group(1)} you mention matches the {_month_year(hit.effective)} result." if hit else
                         f" I don't see a result of {t.group(1)} on file, so I can't confirm that number.")
        rng = _range_text(latest)
        if rng:
            out = ((latest.ref_high is not None and latest.value > latest.ref_high) or (latest.ref_low is not None and latest.value < latest.ref_low))
            sentence += f" Reference range {rng}; the latest result is {'outside' if out else 'within'} it."
        if code in flagged:
            sentence += f" A flagged trend is on file: results have been going {flagged[code]['direction']}."
        parts.append(sentence)
    return " ".join(parts) + " " + LAB_NOTE + SUFFIX


def make_lab_responder(data_dir: Path) -> Callable[[str, Any, Scenario], LocalReply | None]:
    """Offered every reactive turn. Answers questions about the user's own lab results locally, in both modes."""
    cache: dict[str, Any] = {}

    def respond(text: str, pre: Any, sc: Scenario) -> LocalReply | None:
        if "medication_question" in pre.flags:
            return None
        if "ehr" not in cache:
            vault = Vault(Path(data_dir))
            try:
                cache["ehr"] = EHR(vault.read("local answer", ["labs"])) if vault.exists() else None
            except VaultError:
                cache["ehr"] = None           # locked: not this responder's business; the usual paths report it
        ehr = cache["ehr"]
        ans = lab_answer(text, ehr) if ehr is not None else None
        return LocalReply(ans, NOTE, "ehr") if ans else None
    return respond
