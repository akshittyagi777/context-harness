"""Local lab trend detection: three or more numeric results for one test, strictly monotonic, with the
latest result outside the reference range while an earlier one was inside it."""
from __future__ import annotations

from typing import Any

from .model import EHR, LabResult


def _out_of_range(v: float, lab: LabResult) -> bool:
    return (lab.ref_high is not None and v > lab.ref_high) or (lab.ref_low is not None and v < lab.ref_low)


def flagged_trends(ehr: EHR) -> list[dict[str, Any]]:
    by_test: dict[str, list[LabResult]] = {}
    for lab in ehr.labs:
        if lab.value is not None and lab.primary_code():
            by_test.setdefault(lab.primary_code(), []).append(lab)
    flagged = []
    for code, labs in by_test.items():
        labs.sort(key=lambda l: l.effective)
        if len(labs) < 3:
            continue
        vals = [l.value for l in labs]
        up = all(b > a for a, b in zip(vals, vals[1:]))
        down = all(b < a for a, b in zip(vals, vals[1:]))
        latest = labs[-1]
        has_range = latest.ref_high is not None or latest.ref_low is not None
        crossed = has_range and _out_of_range(vals[-1], latest) and any(not _out_of_range(v, latest) for v in vals[:-1])
        if (up or down) and crossed:
            flagged.append({"code": code, "display": latest.display, "n": len(labs),
                            "direction": "up" if up else "down"})
    return flagged
