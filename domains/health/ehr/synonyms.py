"""Local synonym map: lay terms, abbreviations and brand names for the EHR facts the denylist knows.

A denylist of the EHR's own display names cannot catch "taking it easy until my BP is sorted": the
condition is named by a synonym. This module lists, for each condition class, drug and lab class, the other
ways people refer to it. `terms_for` returns only the terms for facts that are actually present in the local
EHR store, so a store without hypertension never makes "BP" a denylist term.

Local, small, and not a clinical reference. A synonym missing here is a gap the denylist cannot see, which is
why EHR content still reaches a model only as derived templates. Deliberately absent: terms that name a
different thing (amoxicillin is a related penicillin-class drug, not another name for the allergen on file).
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from .classes import BRANDS, CLASS_BY_NAME, generic_names

# --- conditions: class -> (substrings of a condition's own text that select it, extra terms)
CONDITION_CLASSES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "hypertension": (("hypertension", "high blood pressure", "htn"),
                     ("high blood pressure", "blood pressure", "BP", "HTN", "hypertensive")),
    "hyperlipidemia": (("hyperlipidemia", "dyslipidemia", "high cholesterol", "hypercholesterolemia"),
                       ("high cholesterol", "hyperlipidemia", "dyslipidemia", "hypercholesterolemia")),
    "type2_diabetes": (("type 2 diabetes", "diabetes mellitus", "type ii diabetes", "t2dm"),
                       ("diabetes", "diabetic", "type 2 diabetes", "T2D", "T2DM", "sugar diabetes")),
    "prediabetes": (("prediabetes", "pre-diabetes"), ("prediabetes", "pre-diabetes", "borderline diabetes")),
    "asthma": (("asthma",), ("asthma", "asthmatic")),
    "hypothyroidism": (("hypothyroid",), ("hypothyroidism", "underactive thyroid", "low thyroid")),
    "depression": (("depress",), ("depression", "depressed")),
    "anxiety": (("anxiety",), ("anxiety", "anxious", "GAD")),
}

# --- drugs: class-level lay terms apply to every drug in the class; drug-level terms to one drug
CLASS_TERMS: dict[str, tuple[str, ...]] = {
    "beta_blocker": ("beta blocker", "beta-blocker", "beta blockers", "beta-blockers"),
    "statin": ("statin", "statins"),
    "ssri_snri": ("SSRI", "SSRIs", "SNRI", "SNRIs", "antidepressant", "antidepressants"),
    "ace_arb": ("ACE inhibitor", "ACE inhibitors", "ARB", "ARBs"),
    "antidiabetic": ("diabetes medication", "diabetes meds"),
    "nsaid": ("NSAID", "NSAIDs"),
    "thyroid": ("thyroid medication", "thyroid pill", "thyroid meds"),
}
DRUG_TERMS: dict[str, tuple[str, ...]] = {      # brands beyond classes.BRANDS, and other names
    "metoprolol": ("Toprol", "Toprol-XL", "Lopressor"),
    "naproxen": ("Aleve", "Naprosyn"),
    "penicillin": ("penicillins", "Pen-Vee K", "Veetids", "penicillin V"),   # NOT amoxicillin
    "bisoprolol": ("Zebeta",), "nebivolol": ("Bystolic",),
    "simvastatin": ("Zocor",), "pravastatin": ("Pravachol",),
    "fluoxetine": ("Prozac",), "escitalopram": ("Lexapro",), "citalopram": ("Celexa",),
    "venlafaxine": ("Effexor",), "duloxetine": ("Cymbalta",),
    "losartan": ("Cozaar",), "valsartan": ("Diovan",), "enalapril": ("Vasotec",),
    "glipizide": ("Glucotrol",), "meloxicam": ("Mobic",),
    "levothyroxine": ("Levoxyl", "Synthroid"),
}

# --- labs: class -> (substrings of a lab's own text that select it, extra terms). Codes and values are
# matched by the denylist's own patterns, not here.
LAB_CLASSES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "hba1c": (("a1c", "glycated", "glycosylated", "glycohemoglobin"),
              ("HbA1c", "A1c", "A1C", "hemoglobin A1c", "glycated hemoglobin")),
    "ldl": (("ldl",), ("LDL", "LDL cholesterol", "bad cholesterol")),
    "hdl": (("hdl",), ("HDL", "HDL cholesterol", "good cholesterol")),
    "triglycerides": (("triglycerid",), ("triglycerides", "triglyceride")),
    "glucose": (("glucose",), ("blood sugar", "blood glucose")),
    "egfr": (("egfr", "glomerular"), ("eGFR", "kidney function")),
    "tsh": (("tsh", "thyrotropin"), ("TSH", "thyroid stimulating hormone")),
}
# Analytes that are also ordinary English words or everyday phrases. They are lab references only in lab context
# ("my sodium was 133"), not in "a high-sodium dinner" or "glucose gels on a long run".
ORDINARY_ANALYTES: frozenset[str] = frozenset({
    "sodium", "potassium", "calcium", "glucose", "chloride", "protein", "iron", "magnesium", "carbon dioxide",
    "albumin", "phosphorus", "bicarbonate", "zinc", "vitamin d", "blood sugar", "blood glucose", "kidney function",
})


def _item_text(item: dict[str, Any]) -> str:
    return f"{item.get('display', '')} {item.get('text', '')}".lower()


def _has(low: str, key: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(key)}", low) is not None


def drug_terms(item: dict[str, Any]) -> set[str]:
    """Synonyms for a medication or allergen item: brands, lay names, its class's names."""
    low = _item_text(item)
    out: set[str] = set()
    for g in set(generic_names(low)) | {g for g in DRUG_TERMS if _has(low, g)} | {g for g in BRANDS if _has(low, g)}:
        out.update(BRANDS.get(g, ()))
        out.update(DRUG_TERMS.get(g, ()))
        if g in CLASS_BY_NAME:
            out.update(CLASS_TERMS.get(CLASS_BY_NAME[g], ()))
    return out


def condition_terms(item: dict[str, Any]) -> set[str]:
    low = _item_text(item)
    out: set[str] = set()
    for keys, terms in CONDITION_CLASSES.values():
        if any(k in low for k in keys):
            out.update(terms)
    return out


def split_lab_terms(item: dict[str, Any]) -> tuple[set[str], set[str]]:
    """(specific clinical terms, ordinary-word analytes) for a lab: its own names plus its class's synonyms.
    Specific terms are scrubbed everywhere; ordinary-word ones only in lab context. Values are handled elsewhere."""
    names = _lab_names(item)
    ordinary = {n for n in names if n.lower() in ORDINARY_ANALYTES}
    return names - ordinary, ordinary


def lab_terms(item: dict[str, Any]) -> set[str]:
    """The specific (not ordinary-word) names and synonyms for a lab."""
    return split_lab_terms(item)[0]


def _lab_names(item: dict[str, Any]) -> set[str]:
    low = _item_text(item)
    out = {s.strip() for s in (item.get("display", ""), item.get("text", "")) if s and s.strip()}
    for keys, terms in LAB_CLASSES.values():
        if any(k in low for k in keys):
            out.update(terms)
    return out


def all_terms(kind: str, items: Iterable[dict[str, Any]]) -> set[str]:
    """Names and synonyms for the items of one EHR kind (conditions, medications, allergies, labs); {} for other kinds."""
    fn = {"conditions": condition_terms, "medications": drug_terms, "allergies": drug_terms, "labs": lab_terms}.get(kind)
    if fn is None:
        return set()
    out: set[str] = set()
    for it in items:
        out |= fn(it)
    return out
