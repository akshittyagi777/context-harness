"""Epic FHIR R4 patient-access responses: searchset Bundles (one per resource search), as one or more
files or a directory of them. EHR content goes into the encrypted vault and nowhere else.

Epic quirks handled: ids are opaque and site specific; Observation searches return `category`
laboratory or vital-signs bundles separately; Epic-local codes can sit beside LOINC; OperationOutcome
entries can appear inside a bundle; `meta.lastUpdated` is usually absent. Built against
health_formats.md section 3.2 and Epic's published shapes; Epic extras were not seen in a real response
(format: unconfirmed).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..canonical import append_ledger
from ..ehr.model import add_resource
from ..vault import Vault
from .base import ImportResult, SourceInfo

INFO = SourceInfo("epic", "ehr", ["FHIR R4 searchset Bundle JSON (one or more files, or a directory)"],
                  "Epic FHIR R4 patient-access bundles into the encrypted vault")


def codings(cc: dict[str, Any] | None) -> list[dict[str, str]]:
    return [{"system": c.get("system", ""), "code": c.get("code", ""), "display": c.get("display", "")}
            for c in (cc or {}).get("coding", []) if c.get("code")]


def first_display(cc: dict[str, Any] | None) -> str:
    cc = cc or {}
    return (cc.get("coding") or [{}])[0].get("display") or cc.get("text", "")


def status_code(cc: dict[str, Any] | None) -> str:
    return ((cc or {}).get("coding") or [{}])[0].get("code", "")


def category_codes(r: dict[str, Any]) -> list[str]:
    out = []
    for c in r.get("category", []):
        if isinstance(c, str):
            out.append(c)
        else:
            out += [x.get("code", "") for x in c.get("coding", [])]
    return out


def _quantity(q: dict[str, Any] | None) -> tuple[float | None, str]:
    q = q or {}
    return q.get("value"), q.get("unit", "")


class EpicFhirImporter:
    info = INFO

    def _bundle_files(self, path: Path) -> list[Path]:
        return sorted(p for p in path.glob("*.json")) if path.is_dir() else [path]

    def can_import(self, path: str | Path) -> bool:
        p = Path(path)
        if p.is_dir():
            return any(p.glob("*.json"))
        return p.suffix.lower() == ".json"

    def import_path(self, path: str | Path, data_dir: Path, *, passphrase: str | None = None, **kw: Any) -> ImportResult:
        vault = Vault(Path(data_dir), passphrase)
        vault.require_passphrase()                      # refuses (with the reason) before anything is parsed
        res = ImportResult("epic")
        bundles: list[tuple[str, dict[str, Any]]] = []
        for f in self._bundle_files(Path(path)):
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                res.anomalies.append(f"{f.name}: invalid JSON ({exc.msg}); file skipped")
                continue
            if doc.get("resourceType") != "Bundle":
                res.anomalies.append(f"{f.name}: not a FHIR Bundle (resourceType {doc.get('resourceType')}); skipped")
                continue
            res.files.append(str(f))
            bundles.append((f.name, doc))

        def apply(store: dict[str, Any]) -> None:
            for fname, doc in bundles:
                digest = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
                if not any(s.get("sha256") == digest for s in store["sources"]):
                    store["sources"].append({"file": fname, "sha256": digest})
                for entry in doc.get("entry", []):
                    self._resource(store, entry.get("resource", {}), res)

        vault.update("import", apply)
        append_ledger(Path(data_dir), res.ledger_entry(path))
        return res

    # ------------------------------------------------------------------ resources
    def _resource(self, store: dict[str, Any], r: dict[str, Any], res: ImportResult) -> None:
        t, rid = r.get("resourceType"), r.get("id", "")
        if t == "Patient":
            self._patient(store, r, res)
        elif t == "Condition":
            cats = category_codes(r)
            item = {"display": first_display(r.get("code")), "text": (r.get("code") or {}).get("text", ""),
                    "codes": codings(r.get("code")), "onset": r.get("onsetDateTime", ""), "recorded": r.get("recordedDate", ""),
                    "status": status_code(r.get("clinicalStatus")) or "active", "category": cats[0] if cats else ""}
            res.add("Condition", add_resource(store, "conditions", item, rid))
        elif t == "MedicationRequest":
            cc = r.get("medicationCodeableConcept")
            dosage = (r.get("dosageInstruction") or [{}])[0].get("text", "")
            item = {"display": first_display(cc), "text": (cc or {}).get("text", ""), "codes": codings(cc),
                    "status": r.get("status", ""), "authored_on": r.get("authoredOn", ""), "dosage": dosage,
                    "requester": (r.get("requester") or {}).get("display", "")}
            res.add("MedicationRequest", add_resource(store, "medications", item, rid))
        elif t == "AllergyIntolerance":
            cats = category_codes(r)
            reactions = [{"manifestation": m.get("text") or first_display(m), "severity": x.get("severity", "")}
                         for x in r.get("reaction", []) for m in x.get("manifestation", [])]
            item = {"display": first_display(r.get("code")), "text": (r.get("code") or {}).get("text", ""),
                    "codes": codings(r.get("code")), "category": cats[0] if cats else "", "criticality": r.get("criticality", ""),
                    "reactions": reactions, "recorded": r.get("recordedDate", ""),
                    "status": status_code(r.get("clinicalStatus")) or "active"}
            res.add("AllergyIntolerance", add_resource(store, "allergies", item, rid))
        elif t == "Observation":
            self._observation(store, r, res)
        elif t == "OperationOutcome":
            msg = ((r.get("issue") or [{}])[0].get("details") or {}).get("text", "no detail")
            res.anomalies.append(f"OperationOutcome entry in a bundle ({msg}); ignored")
        else:
            res.add(f"{t}(not imported)", False)

    def _patient(self, store: dict[str, Any], r: dict[str, Any], res: ImportResult) -> None:
        name = (r.get("name") or [{}])[0]
        tele = r.get("telecom", [])
        p = {"local_id": "pat1", "source_ids": [r.get("id", "")], "given": name.get("given", []), "family": name.get("family", ""),
             "birth_date": r.get("birthDate", ""), "gender": r.get("gender", ""),
             "identifiers": [{"system": i.get("system", ""), "value": i.get("value", ""),
                              "type": (i.get("type") or {}).get("text", "")} for i in r.get("identifier", [])],
             "phones": [t["value"] for t in tele if t.get("system") == "phone"],
             "emails": [t["value"] for t in tele if t.get("system") == "email"],
             "addresses": [{"line": a.get("line", []), "city": a.get("city", ""), "state": a.get("state", ""),
                            "postal": a.get("postalCode", "")} for a in r.get("address", [])]}
        res.add("Patient", store.get("patient") is None)
        if store.get("patient") is None:
            store["patient"] = p

    def _observation(self, store: dict[str, Any], r: dict[str, Any], res: ImportResult) -> None:
        cats = category_codes(r)
        kind = "labs" if "laboratory" in cats else "vitals" if "vital-signs" in cats else None
        if kind is None:
            res.add("Observation(other category)", False)
            return
        val, unit = _quantity(r.get("valueQuantity"))
        comps = []
        for c in r.get("component", []):
            cv, cu = _quantity(c.get("valueQuantity"))
            comps.append({"display": first_display(c.get("code")), "codes": codings(c.get("code")), "value": cv, "unit": cu})
        rr = (r.get("referenceRange") or [{}])[0]
        interp = status_code((r.get("interpretation") or [{}])[0]) if r.get("interpretation") else ""
        item = {"display": first_display(r.get("code")), "text": (r.get("code") or {}).get("text", ""),
                "codes": codings(r.get("code")), "value": val, "unit": unit, "components": comps,
                "effective": r.get("effectiveDateTime", ""), "ref_high": (rr.get("high") or {}).get("value"),
                "ref_low": (rr.get("low") or {}).get("value"), "ref_text": rr.get("text", ""), "interpretation": interp,
                "category": cats[0]}
        res.add("Observation", add_resource(store, kind, item, r.get("id", "")))
