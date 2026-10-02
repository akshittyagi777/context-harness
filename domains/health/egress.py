"""The egress guard: the last check before anything is sent to a model API.

It scans the fully rendered request and refuses it if it contains EHR content. The denylist is built
from the local EHR store: patient name parts, identifiers (MRN, source ids), date of birth, address,
phone, email, every medication and allergen name (generic and brand), condition names, every code value,
every lab or vital value with its unit, and every date that appears in EHR resources. It also applies
the HIPAA Safe Harbor identifiers that can be matched textually: phone, email, SSN-like, MRN-like,
street-address-like. A hit raises EgressBlocked naming the CATEGORIES (never the values).

Fail closed: if an EHR store exists but cannot be read (no passphrase, wrong passphrase), nothing is
sent. If no EHR store exists, only the textual Safe Harbor patterns apply.

Redact, then guard. Before the guard scan, `scrub` replaces every active medication, allergy, condition and
lab-name term (own names plus the synonyms in `ehr/synonyms.py`), patient name parts, identifiers, contact
details, date of birth, and codes with a category placeholder such as "[a medication on file]", in every
origin. A lab value that follows a scrubbed lab term within three words becomes "[a value]". Ordinary-word
analytes (sodium, glucose, ...) are scrubbed, and guarded, only in lab context. The guard then runs over what
is left and still blocks on any hit, so a term the scrub misses fails closed. A bare lab value with no lab
term next to it is not scrubbed: it blocks.

Limits, stated plainly: the denylist matches the exact strings in the user's own EHR plus the local
class map's generic and brand names. It cannot recognise a clinical fact written in other words, which
is why EHR content reaches a model only through the closed template table in `ehr/derive.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from harness.model import SendBlocked

from .ehr.classes import names_for
from .ehr.synonyms import all_terms, split_lab_terms
from .ehr.derive import MONTHS
from .vault import Vault, VaultError

MONTH_SHORT = [m[:3] for m in MONTHS]
BOUND_L, BOUND_R = r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])"

# Textual Safe Harbor patterns that need no vault.
GENERIC: list[tuple[str, re.Pattern[str]]] = [
    ("phone", re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")),
    ("ssn_like", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    ("mrn_like", re.compile(r"(?i)\b(?:mrn|medical record(?: number| no\.?)?|patient id)\W{0,3}[A-Za-z]?\d{4,}|\b[A-Z]{1,2}\d{7,9}\b")),
    ("street_address", re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Dr|Drive|Ln|Lane|Way|Ct|Court|Pl|Place)\b")),
]


# Category placeholders for scrubbed terms, keyed by scrub category.
PLACEHOLDERS: dict[str, str] = {
    "medication": "[a medication on file]", "condition": "[a condition on file]", "lab": "[a lab result on file]",
    "allergy": "[an allergy on file]", "identifier": "[an identifier]", "code": "[a code]", "value": "[a value]",
}
LAB_PLACEHOLDER = PLACEHOLDERS["lab"]

# An ordinary-word analyte (sodium) is a lab reference only near a number, a unit, or a lab-context word.
_CTX_WORDS = re.compile(r"(?i)(?<![A-Za-z])(?:levels?|labs?|tests?|results?|panel|blood ?work|reading)(?![A-Za-z])")
_CTX_NUM_UNIT = re.compile(r"(?i)\d|%|(?<![A-Za-z])(?:mg/dl|mmol/l|meq/l|g/dl)(?![A-Za-z])")
_VALUE_AFTER = re.compile(
    r"(?P<lab>" + re.escape(LAB_PLACEHOLDER) + r")(?P<gap>(?:[\s,:;=()\-]+[^\s\d\[\]]+){0,2}?[\s,:;=()\-]+)"
    r"(?P<val>(?<![\w.])\d+(?:\.\d+)?(?:\s*(?:%|mg/dL|mmol/L|mEq/L|g/dL|ng/mL|IU/L|U/L))?)(?![\w])", re.I)


# Timestamps and dates are not lab numbers: the rendered prompt prefixes the message with "[2026-09-29T08:10]".
_DATETIME_TOKEN = re.compile(r"^\W*(?:\d{4}-\d\d-\d\d(?:T[\d:.]*)?|\d{1,2}:\d\d(?::\d\d)?|\d{1,2}/\d{1,2}/\d{2,4})\W*$")


def _words(chunk: str) -> list[str]:
    return [w for w in chunk.split() if not _DATETIME_TOKEN.match(w)]


_REDACTED_VALUE_AFTER = re.compile(_VALUE_AFTER.pattern.replace(re.escape(LAB_PLACEHOLDER), re.escape("[redacted: lab_name]")), re.I)


def in_lab_context(text: str, start: int, end: int) -> bool:
    """Within three words of the match, on the same line: a number, a unit, or a lab-context word.
    Date and time tokens do not count as numbers and are skipped, not counted as words."""
    before = _words(text[:start].rsplit("\n", 1)[-1])[-3:]
    after = _words(text[end:].split("\n", 1)[0])[:3]
    around = " ".join(before + after)
    return bool(_CTX_NUM_UNIT.search(around) or _CTX_WORDS.search(around))


class EgressBlocked(SendBlocked):
    def __init__(self, categories: Iterable[str]):
        cats = sorted(set(categories))
        super().__init__("egress blocked: categories " + ", ".join(cats), cats)


_FULL_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _full_date(value: str) -> str | None:
    """The day of an ISO date, or None for a partial one. FHIR allows "2019" or "2019-03"; those carry
    no day to match, and guarding a bare year would block every mention of it."""
    return value[:10] if _FULL_DATE.match(value) else None


def _date_regexes(iso: str) -> list[re.Pattern[str]]:
    y, m, d = (int(x) for x in iso[:10].split("-"))
    mon_s, mon_l = MONTH_SHORT[m - 1], MONTHS[m - 1]
    dd = f"0?{d}"
    mm = f"0?{m}"
    forms = [rf"{y}[-/]{m:02d}[-/]{d:02d}", rf"{mm}/{dd}/{y}",
             rf"(?:{mon_s}|{mon_l})\.?\s+{dd}(?:st|nd|rd|th)?,?\s+{y}", rf"{dd}\s+(?:{mon_s}|{mon_l})\.?,?\s+{y}"]
    return [re.compile(r"(?i)(?<![\d-])" + f + r"(?!\d)") for f in forms]


def month_label(iso: str) -> str:
    y, m, _ = iso[:10].split("-")
    return f"{MONTH_SHORT[int(m) - 1]} {y}"


@dataclass
class Denylist:
    phrases: dict[str, set[str]] = field(default_factory=dict)       # category -> literal phrases (word-bounded, case-insensitive)
    patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list)
    event_dates: set[str] = field(default_factory=set)               # ISO dates of EHR events (blocked; scrubbed to month first)
    date_patterns: list[tuple[str, re.Pattern[str], str]] = field(default_factory=list)   # (category, regex, month label)
    local_ids: set[str] = field(default_factory=set)
    scrub_terms: dict[str, str] = field(default_factory=dict)        # lowercase term -> scrub category (placeholder key)
    context_terms: set[str] = field(default_factory=set)             # ordinary-word terms: scrubbed and guarded only in lab context
    scrub_patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list)   # (scrub category, regex): dob, short codes
    _scrub_rx: re.Pattern[str] | None = field(default=None, repr=False)
    _ctx_rx: re.Pattern[str] | None = field(default=None, repr=False)

    def add(self, category: str, phrase: str | None, scrub: str | None = None, context: bool = False) -> None:
        """A guarded phrase. With `scrub`, the same phrase is also replaced by that category's placeholder.
        A `context` phrase is an ordinary word: it is scrubbed and guarded only in lab context."""
        if phrase and len(phrase.strip()) >= 2:
            low = " ".join(phrase.lower().split())
            if context:
                self.context_terms.add(low)
                self._ctx_rx = None
            else:
                self.phrases.setdefault(category, set()).add(low)
            if scrub:
                self.scrub_terms.setdefault(low, scrub)
                self._scrub_rx = None


def _digits_code(code: str) -> bool:
    return code.isdigit()


def build_denylist(ehr: dict[str, Any]) -> Denylist:
    dl = Denylist()
    dates: dict[str, str] = {}        # iso -> category
    p = ehr.get("patient") or {}
    for g in p.get("given", []):
        dl.add("name", g.strip("."), "identifier")
    dl.add("name", p.get("family"), "identifier")
    if p.get("family") and p.get("given"):
        dl.add("name", f"{p['given'][0]} {p['family']}", "identifier")
    for i in p.get("identifiers", []):
        dl.add("identifier", i.get("value"), "identifier")
    for sid in p.get("source_ids", []):
        dl.add("identifier", sid, "identifier")
    if p.get("birth_date") and _full_date(p["birth_date"]):
        dates[_full_date(p["birth_date"])] = "dob"
    for a in p.get("addresses", []):
        for line in a.get("line", []):
            dl.add("address", line, "identifier")
        dl.add("address", a.get("city"), "identifier")
        dl.add("address", a.get("postal"), "identifier")
    for ph in p.get("phones", []):
        digits = re.sub(r"\D", "", ph)[-10:]
        if len(digits) == 10:
            a, b, c = digits[:3], digits[3:6], digits[6:]
            for v in (f"({a}) {b}-{c}", f"{a}-{b}-{c}", f"{a}.{b}.{c}", digits, f"{a} {b} {c}", f"+1 {a} {b} {c}"):
                dl.add("phone", v, "identifier")
        dl.add("phone", ph, "identifier")
    for em in p.get("emails", []):
        dl.add("email", em, "identifier")

    def codes(item: dict[str, Any]) -> None:
        for c in item.get("codes", []):
            code = c.get("code", "")
            if not code:
                continue
            if _digits_code(code) and len(code) < 5:      # a short number is only a code next to a code-system word
                rx = re.compile(rf"(?i)\b(?:rxnorm|rxcui|snomed(?: ct)?|sct|loinc|icd-?10(?:-cm)?|code)\W{{0,4}}{re.escape(code)}(?!\d)")
                dl.patterns.append(("code", rx))
                dl.scrub_patterns.append(("code", rx))
            else:
                dl.add("code", code, "code")
        for sid in item.get("source_ids", []):
            dl.add("identifier", sid, "identifier")

    for kind in ("conditions", "medications", "allergies", "labs", "vitals"):
        for item in ehr.get(kind, []):
            codes(item)
            for key in ("onset", "recorded", "authored_on", "effective"):
                if item.get(key) and _full_date(item[key]):
                    dates[_full_date(item[key])] = "date"
            if kind == "conditions":
                dl.add("condition", item.get("display"), "condition")
                dl.add("condition", item.get("text"), "condition")
            elif kind in ("medications", "allergies"):
                for name in names_for(item):           # an allergen is guarded as a medication, scrubbed as an allergy
                    dl.add("medication", name, "medication" if kind == "medications" else "allergy")
            if kind == "labs":                           # specific terms always; ordinary words only in lab context
                specific, ordinary = split_lab_terms(item)
                for t in specific:
                    dl.add("lab_name", t, "lab")
                for t in ordinary:
                    dl.add("lab_name", t, "lab", context=True)
            else:
                for syn in all_terms(kind, [item]):     # lay terms, abbreviations and brands, only for facts on file
                    if kind == "conditions":
                        dl.add("condition", syn, "condition")
                    else:
                        dl.add("medication", syn, "medication" if kind == "medications" else "allergy")
            for comp in item.get("components", []):
                codes(comp)
            # every value with its unit
            vals = []
            if item.get("value") is not None:
                vals.append((item["value"], item.get("unit", "")))
            vals += [(c["value"], c.get("unit", "")) for c in item.get("components", []) if c.get("value") is not None]
            for v, unit in vals:
                forms = {str(v)} | ({str(int(v))} if float(v).is_integer() else set())
                for f in forms:
                    if unit:
                        sep = r"\s*" if unit in ("%",) else r"\s*"
                        dl.patterns.append(("lab_value", re.compile(rf"(?i)(?<![\d.]){re.escape(f)}{sep}{re.escape(unit)}(?![A-Za-z0-9/])")))
            comps = [c.get("value") for c in item.get("components", [])]
            if len(comps) == 2 and None not in comps:           # blood pressure written as 120/80
                dl.patterns.append(("lab_value", re.compile(rf"(?<![\d.]){int(comps[0])}\s*/\s*{int(comps[1])}(?![\d.])")))
    for iso, cat in dates.items():
        for rx in _date_regexes(iso):
            if cat == "dob":
                dl.patterns.append(("dob", rx))
                dl.scrub_patterns.append(("identifier", rx))
            else:
                dl.date_patterns.append(("date", rx, month_label(iso)))
        if cat != "dob":
            dl.event_dates.add(iso)
    return dl


def _alt(phrases: set[str]) -> re.Pattern[str]:
    return re.compile(BOUND_L + "(?:" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + ")" + BOUND_R, re.I)


def _ctx_regex(dl: Denylist) -> re.Pattern[str] | None:
    if not dl.context_terms:
        return None
    if dl._ctx_rx is None:
        dl._ctx_rx = _alt(dl.context_terms)
    return dl._ctx_rx


def _context_hits(text: str, dl: Denylist) -> list[re.Match[str]]:
    rx = _ctx_regex(dl)
    return [m for m in rx.finditer(text) if in_lab_context(text, m.start(), m.end())] if rx else []


def scan(text: str, dl: Denylist | None) -> list[str]:
    """Categories (never values) found in `text`."""
    cats = {c for c, rx in GENERIC if rx.search(text)}
    if dl is not None:
        for cat, phrases in dl.phrases.items():
            if phrases and _alt(phrases).search(text):
                cats.add(cat)
        cats |= {c for c, rx in dl.patterns if rx.search(text)}
        cats |= {c for c, rx, _ in dl.date_patterns if rx.search(text)}
        if _context_hits(text, dl):
            cats.add("lab_name")
    return sorted(cats)


def scrub_dates(text: str, dl: Denylist | None) -> tuple[str, int]:
    """Coarsen every EHR event date in `text` to month precision. Returns (text, number of changes)."""
    if dl is None:
        return text, 0
    n = 0
    for _, rx, label in dl.date_patterns:
        text, k = rx.subn(label, text)
        n += k
    return text, n


def _scrub_regex(dl: Denylist) -> re.Pattern[str] | None:
    if not dl.scrub_terms:
        return None
    if dl._scrub_rx is None:
        alts = sorted(dl.scrub_terms, key=lambda t: (-len(t), t))                 # longest match first
        dl._scrub_rx = re.compile(BOUND_L + "(?:" + "|".join(r"\s+".join(re.escape(w) for w in a.split()) for a in alts) + ")" + BOUND_R, re.I)
    return dl._scrub_rx


def scrub_terms(text: str, dl: Denylist | None) -> tuple[str, dict[str, int]]:
    """Replace every active scrub term (names and synonyms of medications, allergies, conditions, lab results and
    EHR identifiers) with its category placeholder. Returns (text, counts by category); never the terms."""
    counts: dict[str, int] = {}
    if dl is None:
        return text, counts
    rx = _scrub_regex(dl)

    def sub(m: re.Match[str]) -> str:
        term = " ".join(m.group(0).lower().split())
        if term in dl.context_terms and not in_lab_context(m.string, m.start(), m.end()):
            return m.group(0)                                  # an ordinary word outside lab context
        cat = dl.scrub_terms[term]
        counts[cat] = counts.get(cat, 0) + 1
        return PLACEHOLDERS[cat]

    def sub_pattern(cat: str):
        def f(m: re.Match[str]) -> str:
            counts[cat] = counts.get(cat, 0) + 1
            return PLACEHOLDERS[cat]
        return f

    if rx is not None:
        text = rx.sub(sub, text)
    for cat, prx in dl.scrub_patterns:                         # date of birth, short numeric codes
        text = prx.sub(sub_pattern(cat), text)

    def value(m: re.Match[str]) -> str:                        # a value stated right after a lab term
        counts["value"] = counts.get("value", 0) + 1
        return m.group("lab") + m.group("gap") + PLACEHOLDERS["value"]

    return _VALUE_AFTER.sub(value, text), counts


def redact(text: str, dl: Denylist | None) -> tuple[str, int]:
    """Replace every denylisted token with `[redacted: category]`. Used for what is stored on disk
    (traces, memory written this turn), never for what is sent: sending is guarded, not repaired."""
    n = 0
    if dl is None:
        return text, 0
    for cat, phrases in dl.phrases.items():
        if phrases:
            text, k = _alt(phrases).subn(f"[redacted: {cat}]", text)
            n += k
    hits = _context_hits(text, dl)
    for m in reversed(hits):                      # ordinary-word analytes, only in lab context
        text, n = text[:m.start()] + "[redacted: lab_name]" + text[m.end():], n + 1
    for cat, rx in dl.patterns:
        text, k = rx.subn(f"[redacted: {cat}]", text)
        n += k
    for _, rx, _label in dl.date_patterns:
        text, k = rx.subn("[redacted: date]", text)
        n += k
    text, k = _REDACTED_VALUE_AFTER.subn(lambda m: m.group("lab") + m.group("gap") + "[redacted: lab_value]", text)   # a value typed after a lab term
    return text, n + k


def _flatten(obj: Any) -> Iterable[str]:
    if obj is None or isinstance(obj, (int, float, bool)):
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flatten(v)
    elif hasattr(obj, "model_dump"):
        yield from _flatten(obj.model_dump())
    else:
        yield str(obj)


class Guard:
    """Holds the denylist for one data directory. Build it once per process; it reads the vault once."""

    def __init__(self, data_dir: Path, passphrase: str | None = None, vault: Vault | None = None):
        self.vault = vault or Vault(Path(data_dir), passphrase)
        self._dl: Denylist | None = None
        self._built = False

    def denylist(self) -> Denylist | None:
        """None when there is no EHR store at all. Raises EgressBlocked when a store exists but cannot be read."""
        if not self._built:
            if not self.vault.exists():
                self._dl = None
            else:
                try:
                    self._dl = build_denylist(self.vault.read("egress guard"))
                except VaultError as exc:
                    raise EgressBlocked(["denylist_unavailable"]) from exc
            self._built = True
        return self._dl

    def check_texts(self, texts: Iterable[str]) -> None:
        dl = self.denylist()
        cats: set[str] = set()
        for t in texts:
            cats |= set(scan(t, dl))
        if cats:
            raise EgressBlocked(cats)

    def check_request(self, request: dict[str, Any]) -> None:
        """Every string in a Messages API request: system, user turns, tool results, tool inputs."""
        self.check_texts(_flatten({k: v for k, v in request.items() if k not in ("model", "max_tokens")}))

    def preflight(self, system: str, prompt: str) -> None:
        self.check_texts([system, prompt])

    def scrub_detail(self, text: str) -> tuple[str, int, dict[str, int]]:
        """(text, dates coarsened, term counts by category). Terms are replaced first, then dates are coarsened."""
        try:
            dl = self.denylist()
            text, counts = scrub_terms(text, dl)
            text, n_dates = scrub_dates(text, dl)
            return text, n_dates, counts
        except EgressBlocked:      # locked store: leave the text alone, the pre-send check will refuse it
            return text, 0, {}

    def scrub(self, text: str) -> tuple[str, int]:
        text, n_dates, counts = self.scrub_detail(text)
        return text, n_dates + sum(counts.values())

    def redact(self, text: str) -> str:
        try:
            return redact(text, self.denylist())[0]
        except EgressBlocked:      # store exists but is locked: store nothing we cannot vouch for
            return "[redacted: EHR store locked]"
