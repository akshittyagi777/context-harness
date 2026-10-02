"""A small local code-class map: medication name or code to a coarse class.

Used only on the machine, to choose derived-constraint templates and to build the egress denylist
(generic and brand names). It is not a clinical reference: the classes cover the mock medications
plus a few common neighbours, and anything unknown falls to the conservative default.
"""
from __future__ import annotations

import re
from typing import Any

# generic name -> class
CLASS_BY_NAME: dict[str, str] = {
    "metoprolol": "beta_blocker", "atenolol": "beta_blocker", "propranolol": "beta_blocker",
    "carvedilol": "beta_blocker", "bisoprolol": "beta_blocker", "nebivolol": "beta_blocker",
    "atorvastatin": "statin", "rosuvastatin": "statin", "simvastatin": "statin", "pravastatin": "statin",
    "sertraline": "ssri_snri", "fluoxetine": "ssri_snri", "escitalopram": "ssri_snri",
    "citalopram": "ssri_snri", "venlafaxine": "ssri_snri", "duloxetine": "ssri_snri",
    "lisinopril": "ace_arb", "losartan": "ace_arb", "valsartan": "ace_arb", "enalapril": "ace_arb",
    "metformin": "antidiabetic", "glipizide": "antidiabetic",
    "naproxen": "nsaid", "ibuprofen": "nsaid", "meloxicam": "nsaid",
    "levothyroxine": "thyroid",
}
BRANDS: dict[str, list[str]] = {
    "metoprolol": ["Toprol-XL", "Toprol", "Lopressor"], "atenolol": ["Tenormin"],
    "propranolol": ["Inderal"], "carvedilol": ["Coreg"], "atorvastatin": ["Lipitor"],
    "rosuvastatin": ["Crestor"], "sertraline": ["Zoloft"], "lisinopril": ["Zestril", "Prinivil"],
    "metformin": ["Glucophage"], "naproxen": ["Aleve", "Naprosyn"], "ibuprofen": ["Advil", "Motrin"],
    "levothyroxine": ["Synthroid"], "penicillin": ["Pen-Vee K", "Veetids"],
}
# ATC prefix -> class (WHO ATC codes appear in some coding systems)
ATC_PREFIX = {"C07": "beta_blocker", "C10AA": "statin", "N06AB": "ssri_snri", "C09": "ace_arb",
              "A10": "antidiabetic", "M01A": "nsaid", "H03A": "thyroid"}
ATC_SYSTEM = "http://www.whocc.no/atc"
# Classes with a known interaction with sleep aids or common supplements. An unknown medication is
# treated the same way: the conservative default.
INTERACTS_WITH_SUPPLEMENTS = {"beta_blocker", "ssri_snri", "statin", "ace_arb", "antidiabetic", "thyroid"}
KNOWN_NON_INTERACTING = {"nsaid"}


def generic_names(text: str) -> list[str]:
    low = text.lower()
    return [g for g in CLASS_BY_NAME if re.search(rf"(?<![a-z]){g}", low)]


def classify(med: dict[str, Any]) -> str | None:
    """Class for a medication dict (display/text/codes), or None when unknown."""
    for g in generic_names(f"{med.get('display', '')} {med.get('text', '')}"):
        return CLASS_BY_NAME[g]
    for c in med.get("codes", []):
        if c.get("system") == ATC_SYSTEM:
            for prefix, cls in ATC_PREFIX.items():
                if c.get("code", "").upper().startswith(prefix):
                    return cls
    return None


def names_for(item: dict[str, Any]) -> set[str]:
    """Every name a medication or allergen can be called: its own strings, generic names, brands."""
    out = {s.strip() for s in (item.get("display", ""), item.get("text", "")) if s and s.strip()}
    for g in generic_names(f"{item.get('display', '')} {item.get('text', '')}"):
        out.add(g)
        out.update(BRANDS.get(g, []))
    low = f"{item.get('display', '')} {item.get('text', '')}".lower()
    for g, brands in BRANDS.items():
        if g in low:
            out.add(g)
            out.update(brands)
    return out
