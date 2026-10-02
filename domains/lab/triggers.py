"""Lab triggers: a scheduled morning brief and a signal on shaker temperature."""
from __future__ import annotations

from datetime import datetime

from harness.domain import Scenario, Trigger


def _shaker(sc: Scenario, now: datetime) -> list[dict]:
    """Only nights that have happened: a row dated today has not."""
    return [r for r in sc.series.get("shaker", []) if r["date"] < now.date().isoformat()]


def _readouts(sc: Scenario, now: datetime) -> list[dict]:
    return [r for r in sc.series.get("readouts", []) if r["date"] < now.date().isoformat()]


def morning_brief(sc: Scenario, now: datetime) -> Trigger | None:
    sh, ro = _shaker(sc, now), _readouts(sc, now)
    if not sh or not ro:
        return None
    s, r = sh[-1], ro[-1]
    facts = (f"Shaker last night ({s['date']}): setpoint {s['setpoint_c']}°C, actual {s['actual_c']}°C. "
             f"Latest readout ({r['date']}): {r['conc_mg_ml']} mg/mL, A260/280 {r['a260_280']}.")
    return Trigger(
        kind="scheduled", name="morning_brief", ts=now.isoformat(),
        text=(f"[scheduled trigger: morning brief, set for 08:30] {facts} Compose a short brief. "
              f"Address any open loop. Ask at most one question."),
        query="morning shaker readout today plan induction reagent",
        facts=facts, why="a standing 08:30 brief was requested; the assistant is answering a standing request")


def temperature_excursion(sc: Scenario, now: datetime) -> Trigger | None:
    sh = _shaker(sc, now)
    if not sh or abs(sh[-1]["actual_c"] - sh[-1]["setpoint_c"]) <= 2:
        return None
    s = sh[-1]
    prior = [r["actual_c"] for r in sh[-4:-1]]
    facts = (f"Shaker night of {s['date']}: setpoint {s['setpoint_c']}°C, actual {s['actual_c']}°C; "
             f"the prior three nights read {prior}.")
    return Trigger(
        kind="signal", name="temperature_excursion", ts=now.isoformat(),
        text=(f"[signal trigger: {facts}] Decide whether this is worth interrupting the user about and, if so, "
              f"say what it changes about the induction plan. If not, say one line."),
        query="shaker temperature induction plan aggregation", facts=facts,
        flags=["protocol_risk"],
        why=f"last night's shaker deviated {abs(s['actual_c'] - s['setpoint_c']):.1f}°C from setpoint, over the 2°C threshold")


TRIGGERS = {"morning_brief": morning_brief, "temperature_excursion": temperature_excursion}
