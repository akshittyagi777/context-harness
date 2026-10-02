"""Local EHR dataclasses and the merge rules the importers share.

The vault holds plain dicts (JSON); these classes give typed access for derive, trends and answers.
Every resource gets a harness-local id (`med1`, `cond1`, ...) that is safe to show in traces. The
source system's own opaque ids are kept in `source_ids` and are on the egress denylist.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
SYSTEM_PRIORITY = [LOINC, RXNORM, SNOMED, ICD10]
LOCAL_PREFIX = {"conditions": "cond", "medications": "med", "allergies": "alg", "labs": "lab", "vitals": "vit"}


def date10(s: str | None) -> str:
    return (s or "")[:10]


@dataclass
class Code:
    system: str
    code: str
    display: str = ""


def _mk(cls, d: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Patient:
    local_id: str = "pat1"
    source_ids: list[str] = field(default_factory=list)
    given: list[str] = field(default_factory=list)
    family: str = ""
    birth_date: str = ""
    gender: str = ""
    identifiers: list[dict[str, str]] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    addresses: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Condition:
    local_id: str
    display: str
    text: str = ""
    codes: list[dict[str, str]] = field(default_factory=list)
    onset: str = ""
    recorded: str = ""
    status: str = "active"
    category: str = ""
    source_ids: list[str] = field(default_factory=list)


@dataclass
class Medication:
    local_id: str
    display: str
    text: str = ""
    codes: list[dict[str, str]] = field(default_factory=list)
    status: str = "active"
    authored_on: str = ""
    dosage: str = ""
    requester: str = ""
    source_ids: list[str] = field(default_factory=list)


@dataclass
class Allergy:
    local_id: str
    display: str
    text: str = ""
    codes: list[dict[str, str]] = field(default_factory=list)
    category: str = ""
    criticality: str = ""
    reactions: list[dict[str, str]] = field(default_factory=list)
    recorded: str = ""
    status: str = "active"
    source_ids: list[str] = field(default_factory=list)


@dataclass
class LabResult:
    local_id: str
    display: str
    text: str = ""
    codes: list[dict[str, str]] = field(default_factory=list)
    value: float | None = None
    unit: str = ""
    components: list[dict[str, Any]] = field(default_factory=list)   # e.g. BP systolic/diastolic
    effective: str = ""
    ref_high: float | None = None
    ref_low: float | None = None
    ref_text: str = ""
    interpretation: str = ""
    category: str = "laboratory"
    source_ids: list[str] = field(default_factory=list)

    def primary_code(self) -> str:
        return primary_code(self.codes)


SYSTEM_NAME = {LOINC: "loinc", RXNORM: "rxnorm", SNOMED: "snomed", ICD10: "icd10"}


def primary_code(codes: list[dict[str, str]]) -> str:
    for system in SYSTEM_PRIORITY:
        for c in codes:
            if c.get("system") == system and c.get("code"):
                return f"{SYSTEM_NAME[system]}:{c['code']}"
    return f"{codes[0].get('system', '')}:{codes[0]['code']}" if codes else ""


def natural_key(kind: str, item: dict[str, Any]) -> tuple:
    code = primary_code(item.get("codes", [])) or (item.get("display") or "").lower()
    if kind == "conditions":
        return (code, date10(item.get("onset") or item.get("recorded")))
    if kind == "medications":
        return (code, date10(item.get("authored_on")))
    if kind == "allergies":
        return (code,)
    return (code, (item.get("effective") or "")[:16])   # labs and vitals


def add_resource(store: dict[str, Any], kind: str, item: dict[str, Any], source_id: str = "") -> bool:
    """Add one resource to the vault dict unless the same fact is already there (same code and date,
    whichever file or format it came from). Returns True when new."""
    bucket: list[dict[str, Any]] = store[kind]
    key = natural_key(kind, item)
    for existing in bucket:
        if natural_key(kind, existing) == key:
            if source_id and source_id not in existing.setdefault("source_ids", []):
                existing["source_ids"].append(source_id)
            return False
    item = dict(item)
    item["local_id"] = f"{LOCAL_PREFIX[kind]}{len(bucket) + 1}"
    item["source_ids"] = [source_id] if source_id else []
    bucket.append(item)
    return True


class EHR:
    """Typed view over a vault dict."""

    def __init__(self, data: dict[str, Any]):
        self.patient = _mk(Patient, data["patient"]) if data.get("patient") else None
        self.conditions = [_mk(Condition, d) for d in data.get("conditions", [])]
        self.medications = [_mk(Medication, d) for d in data.get("medications", [])]
        self.allergies = [_mk(Allergy, d) for d in data.get("allergies", [])]
        self.labs = [_mk(LabResult, d) for d in data.get("labs", [])]
        self.vitals = [_mk(LabResult, d) for d in data.get("vitals", [])]

    def active_medications(self) -> list[Medication]:
        return [m for m in self.medications if m.status == "active"]
