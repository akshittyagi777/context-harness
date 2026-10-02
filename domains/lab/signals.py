"""Context candidates the lab adds: a computed summary and the raw rows behind it."""
from __future__ import annotations

from harness.context import Candidate
from harness.domain import Scenario
from harness.resolve import Resolved


def _summary(v: dict) -> tuple[str, str]:
    dev = v["shaker_deviations"]
    full = (f"Concentration averaged {v['conc_first_avg']} mg/mL over {v['conc_first_from']} to "
            f"{v['conc_first_to']} and {v['conc_last_avg']} mg/mL over {v['conc_last_from']} to "
            f"{v['conc_last_to']}. Latest A260/280 is {v['a260_latest']} on {v['a260_date']} "
            f"(clean limit {v['a260_clean_max']}). "
            + (f"Shaker was off setpoint by more than 2°C on {', '.join(dev)}. " if dev
               else "Shaker stayed within 2°C of setpoint. "))
    one = (f"Conc {v['conc_first_avg']} -> {v['conc_last_avg']} mg/mL; A260/280 {v['a260_latest']} "
           f"(clean max {v['a260_clean_max']}); {len(dev)} shaker excursion(s)")
    if "yield_first" in v:
        skipped = f", leaving out {', '.join(v['yield_excluded'])}" if v["yield_excluded"] else ""
        full += (f"Yield went from {v['yield_first']} to {v['yield_last']} mg/L across {v['yield_n']} preps "
                 f"({v['yield_from']} to {v['yield_to']}{skipped}); typical is {v['typical_yield']} mg/L.")
        one += f"; yield {v['yield_first']} -> {v['yield_last']} mg/L."
    else:
        one += "."
    return full.strip(), one


def candidates(sc: Scenario, res: Resolved) -> list[Candidate]:
    v = res.values
    if not v:
        return []
    full, one = _summary(v)
    out = [Candidate("readout_summary", "data", 2, [("full", full), ("one_line", one)], required=True,
                     why="computed after resolution; the only variant that carries the resolver's decisions")]
    sh = {r["date"]: r for r in sc.series["shaker"]}
    rows = [f"{r['date'][5:]}  {r['conc_mg_ml']:.2f}  {r['a260_280']:.2f}  "
            f"{sh[r['date']]['setpoint_c']}  {sh[r['date']]['actual_c']}"
            for r in sc.series["readouts"] if r["date"] in sh]
    head = "date  mg/mL  A260/280  set_C  actual_C\n"
    out.append(Candidate("raw_table", "data", 4,
                         [("10_rows", "Raw rows:\n" + head + "\n".join(rows[-10:])),
                          ("5_rows", "Raw rows:\n" + head + "\n".join(rows[-5:]))],
                         score=0.2, why="raw rows are evidence, not analysis; digits tokenize expensively"))
    return out
