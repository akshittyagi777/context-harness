"""C-CDA (HL7 CCD) documents from an Epic organisation that offers the XDM download
(MachineReadable_XDMFormat/IHE_XDM/<Name>1/DOC000n.XML): a zip, a directory, or one XML file.

Secondary route: Some portals give only a password-protected PDF instead, and FHIR is the
primary route (`epic_fhir.py`). Only four sections are read: allergies (48765-2), medications (10160-0),
problems (11450-4) and results (30954-2). Timestamps are HL7 TS (YYYYMMDD[HHMM[SS]][+ZZZZ]) and are
normalised to UTC. Parsing is lenient: Epic exports have been reported with invalid code tokens, which a
plain attribute read tolerates. Built against the published C-CDA conventions only (format: unconfirmed).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from ..canonical import append_ledger
from ..ehr.model import ICD10, LOINC, RXNORM, SNOMED, add_resource
from ..vault import Vault
from .base import ImportResult, SourceInfo, Tree

INFO = SourceInfo("epic-ccda", "ehr", ["C-CDA CCD XML (XDM zip, directory or file)"], "Epic C-CDA documents into the encrypted vault")
H = "{urn:hl7-org:v3}"
XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"
OID = {"2.16.840.1.113883.6.1": LOINC, "2.16.840.1.113883.6.96": SNOMED, "2.16.840.1.113883.6.88": RXNORM,
       "2.16.840.1.113883.6.90": ICD10}
SECTIONS = {"48765-2": "allergies", "10160-0": "medications", "11450-4": "problems", "30954-2": "results"}


def hl7_ts(v: str | None) -> str:
    """HL7 TS to ISO: a date stays a date; a time with an offset becomes UTC ('...Z')."""
    if not v:
        return ""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})(?:(\d{2})(\d{2})(\d{2})?)?\s*([+-]\d{4})?$", v.strip())
    if not m:
        return ""
    y, mo, d, hh, mm, ss, off = m.groups()
    if hh is None:
        return f"{y}-{mo}-{d}"
    dt = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss or 0))
    if off:
        sign = 1 if off[0] == "+" else -1
        dt -= sign * timedelta(hours=int(off[1:3]), minutes=int(off[3:5]))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _code(el: ET.Element | None) -> list[dict[str, str]]:
    if el is None:
        return []
    out = []
    for e in [el] + el.findall(f"{H}translation"):
        if e.get("code"):
            out.append({"system": OID.get(e.get("codeSystem", ""), e.get("codeSystem", "")), "code": e.get("code", ""),
                        "display": e.get("displayName", "")})
    return out


def _attr(el: ET.Element | None, name: str) -> str:
    return el.get(name, "") if el is not None else ""


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


class EpicCcdaImporter:
    info = INFO

    def can_import(self, path: str | Path) -> bool:
        p = Path(path)
        if p.is_file() and p.suffix.lower() == ".xml":
            return True
        try:
            return any(PurePosixPath(n).name.upper().startswith("DOC") and n.lower().endswith(".xml") for n in Tree(p).names())
        except FileNotFoundError:
            return False

    def _docs(self, path: Path) -> list[tuple[str, str]]:
        if path.is_file() and path.suffix.lower() == ".xml":
            return [(path.name, path.read_text(encoding="utf-8"))]
        tree = Tree(path)
        return [(n, tree.read_text(n)) for n in tree.names()
                if PurePosixPath(n).name.upper().startswith("DOC") and n.lower().endswith(".xml")]

    def import_path(self, path: str | Path, data_dir: Path, *, passphrase: str | None = None, **kw: Any) -> ImportResult:
        vault = Vault(Path(data_dir), passphrase)
        vault.require_passphrase()
        res = ImportResult("epic")
        docs = self._docs(Path(path))
        if not docs:
            res.anomalies.append("no DOC*.XML documents found")
        parsed = []
        for name, text in docs:
            try:
                parsed.append((name, ET.fromstring(text)))
                res.files.append(name)
            except ET.ParseError as exc:
                res.anomalies.append(f"{name}: XML parse error ({exc}); skipped")

        def apply(store: dict[str, Any]) -> None:
            for name, root in parsed:
                self._patient(store, root, res)
                for sec in root.iter(f"{H}section"):
                    code = _attr(sec.find(f"{H}code"), "code")
                    if code in SECTIONS:
                        getattr(self, "_" + SECTIONS[code])(store, sec, name, res)

        vault.update("import", apply)
        append_ledger(Path(data_dir), res.ledger_entry(path))
        return res

    # ------------------------------------------------------------------ sections
    def _patient(self, store: dict[str, Any], root: ET.Element, res: ImportResult) -> None:
        pr = root.find(f"{H}recordTarget/{H}patientRole")
        if pr is None or store.get("patient") is not None:
            return
        pat = pr.find(f"{H}patient")
        name = pat.find(f"{H}name") if pat is not None else None
        addr = pr.find(f"{H}addr")
        tel = [t.get("value", "") for t in pr.findall(f"{H}telecom")]
        bt = pat.find(f"{H}birthTime") if pat is not None else None
        store["patient"] = {
            "local_id": "pat1", "source_ids": [], "given": [_text(g) for g in name.findall(f"{H}given")] if name is not None else [],
            "family": _text(name.find(f"{H}family")) if name is not None else "", "birth_date": hl7_ts(bt.get("value") if bt is not None else ""),
            "gender": "", "identifiers": [{"system": i.get("root", ""), "value": i.get("extension", ""), "type": "MRN"}
                                          for i in pr.findall(f"{H}id")],
            "phones": [t[4:] for t in tel if t.startswith("tel:")], "emails": [t[7:] for t in tel if t.startswith("mailto:")],
            "addresses": [{"line": [_text(addr.find(f"{H}streetAddressLine"))], "city": _text(addr.find(f"{H}city")),
                           "state": _text(addr.find(f"{H}state")), "postal": _text(addr.find(f"{H}postalCode"))}] if addr is not None else []}
        res.add("Patient", True)

    def _allergies(self, store, sec, doc, res) -> None:
        for e in sec.findall(f"{H}entry"):
            code = e.find(f".//{H}playingEntity/{H}code")
            if code is None:
                continue
            codes = _code(code)
            low = e.find(f".//{H}effectiveTime/{H}low")
            react = e.find(f".//{H}entryRelationship/{H}observation/{H}value")
            item = {"display": code.get("displayName", ""), "text": _text(code.find(f"{H}originalText")), "codes": codes,
                    "category": "medication" if any(c["system"] == RXNORM for c in codes) else "",
                    "reactions": [{"manifestation": react.get("displayName", ""), "severity": ""}] if react is not None else [],
                    "recorded": hl7_ts(low.get("value") if low is not None else ""), "status": "active", "criticality": ""}
            res.add("AllergyIntolerance", add_resource(store, "allergies", item, doc))

    def _medications(self, store, sec, doc, res) -> None:
        for e in sec.findall(f"{H}entry"):
            sa = e.find(f"{H}substanceAdministration")
            code = e.find(f".//{H}manufacturedMaterial/{H}code")
            if sa is None or code is None:
                continue
            low = sa.find(f"{H}effectiveTime/{H}low")
            status = _attr(sa.find(f"{H}statusCode"), "code")
            item = {"display": code.get("displayName", ""), "text": _text(code.find(f"{H}originalText")), "codes": _code(code),
                    "status": "active" if status == "active" else "stopped" if status == "completed" else status,
                    "authored_on": hl7_ts(low.get("value") if low is not None else ""), "dosage": _text(sa.find(f"{H}text")),
                    "requester": ""}
            res.add("MedicationRequest", add_resource(store, "medications", item, doc))

    def _problems(self, store, sec, doc, res) -> None:
        for e in sec.findall(f"{H}entry"):
            for obs in e.iter(f"{H}observation"):
                val = obs.find(f"{H}value")
                if val is None or val.get(XSI_TYPE) != "CD" or not val.get("code"):
                    continue
                low = obs.find(f"{H}effectiveTime/{H}low")
                item = {"display": val.get("displayName", ""), "text": _text(obs.find(f"{H}text")), "codes": _code(val),
                        "onset": hl7_ts(low.get("value") if low is not None else ""), "recorded": "", "status": "active",
                        "category": "problem-list-item"}
                res.add("Condition", add_resource(store, "conditions", item, doc))

    def _results(self, store, sec, doc, res) -> None:
        for obs in sec.iter(f"{H}observation"):
            code, val = obs.find(f"{H}code"), obs.find(f"{H}value")
            if code is None or val is None or val.get(XSI_TYPE) != "PQ":
                continue
            eff = obs.find(f"{H}effectiveTime")
            rr = obs.find(f"{H}referenceRange/{H}observationRange")
            hi = rr.find(f".//{H}high") if rr is not None else None
            try:
                v = float(val.get("value", ""))
            except ValueError:
                continue
            item = {"display": code.get("displayName", ""), "text": _text(obs.find(f"{H}text")), "codes": _code(code),
                    "value": v, "unit": val.get("unit", ""), "components": [], "effective": hl7_ts(eff.get("value") if eff is not None else ""),
                    "ref_high": float(hi.get("value")) if hi is not None and hi.get("value") else None, "ref_low": None,
                    "ref_text": _text(rr.find(f"{H}text")) if rr is not None else "", "interpretation": "", "category": "laboratory"}
            res.add("Observation", add_resource(store, "labs", item, doc))
