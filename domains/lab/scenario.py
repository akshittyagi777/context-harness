"""Sample scenario: Sam, a postdoc expressing and purifying a protein for crystallography.

What makes it hard:
  * Plan drift.          Sep 19: express construct v1 at 37 C. Sep 26: switch to 18 C
                         overnight. Today's message goes back to 37 C. Memory holds both plans.
  * Preference reversal. "Don't remind me about buffer prep" (Sep 19) versus "keep me honest
                         on making fresh DTT" (Sep 26).
  * Same-day conflict.   Sep 26 morning "SEC run looks clean", evening "huge void-volume peak".
  * Measured vs recorded. The notebook says about 2 mg/mL on Sep 24; the instrument says 0.6.
  * A sensitive pinned record whose name is gated but whose constraint always ships.
Timezone: America/Los_Angeles. All timestamps are naive local ISO-8601.
"""
from __future__ import annotations

from datetime import datetime

from harness.domain import Scenario, Turn

NOW = datetime.fromisoformat("2026-09-29T08:10:00")

DEFAULT_MESSAGE = ("Gel looks rough again. Thinking I'll just push through a 37°C induction on the "
                   "2 L prep tomorrow to get enough protein for the trays. Thoughts?")

PROFILE = {
    "name": "Sam", "role": "postdoc", "timezone": "America/Los_Angeles",
    "project": "express and purify a protein for crystallography",
    "typical_yield_mg_per_l": 4.0, "a260_280_clean_max": 0.7,
}

HISTORY = [
    Turn("2026-09-19T09:05:00", "user",
         "Plan is to express construct v1 in BL21 at 37°C, 1 mM IPTG, 4 h. "
         "Please don't remind me about buffer prep, I've got it.", 1),
    Turn("2026-09-22T17:40:00", "user", "First prep done. Yield looked fine, gel had a strong band.", 2),
    Turn("2026-09-24T18:15:00", "user", "A280 looked great, about 2 mg/mL.", 3),
    Turn("2026-09-26T09:00:00", "user", "SEC run looks clean.", 4),
    Turn("2026-09-26T21:30:00", "user",
         "Actually the SEC trace had a huge void-volume peak, it's aggregated. Switching to 18°C "
         "overnight induction with 0.2 mM IPTG. Also keep me honest on making fresh DTT, "
         "I keep using old stock.", 4),
    Turn("2026-09-28T20:00:00", "user", "Started the 18°C culture. Used the DTT from Thursday though, oops.", 5),
]

SESSION_SUMMARIES = [
    {"session": 1, "date": "2026-09-19", "text": "Planned construct v1 expression in BL21; asked not to be reminded about buffer prep."},
    {"session": 2, "date": "2026-09-22", "text": "First prep completed; reported a strong gel band."},
    {"session": 3, "date": "2026-09-24", "text": "Reported a good A280 reading from the notebook."},
]

# One row per day, Sep 20 to Sep 29. The Sep 29 rows exist so a turn on Sep 30 sees them.
_CONC = [1.10, 1.12, 1.08, 1.05, 0.60, 0.52, 0.47, 0.44, 0.41, 0.40]
_RATIO = [0.60, 0.60, 0.62, 0.65, 0.75, 0.90, 1.05, 1.18, 1.27, 1.30]
_ACTUAL = [37.2, 36.9, 37.4, 37.9, 37.1, 36.8, 37.3, 18.3, 21.6, 18.2]


def _series() -> dict[str, list[dict]]:
    days = [f"2026-09-{d:02d}" for d in range(20, 30)]
    readouts = [{"date": d, "conc_mg_ml": c, "a260_280": r} for d, c, r in zip(days, _CONC, _RATIO)]
    shaker = [{"date": d, "setpoint_c": 37 if d <= "2026-09-26" else 18, "actual_c": a}
              for d, a in zip(days, _ACTUAL)]
    yields = [
        {"date": "2026-09-22", "mg_per_l": 3.8},
        {"date": "2026-09-24", "mg_per_l": 2.6},
        {"date": "2026-09-26", "mg_per_l": 1.9, "note": "same-day conflict: clean in the morning, aggregated in the evening"},
        {"date": "2026-09-27", "mg_per_l": 1.2},
        {"date": "2026-09-29", "mg_per_l": 0.9},
    ]
    return {"readouts": readouts, "shaker": shaker, "yield": yields}


def build_scenario(now: datetime | None = None) -> Scenario:
    """Everything dated strictly before `now.date()` is visible; a turn on a later day sees later data."""
    now = now or NOW
    today = now.date().isoformat()
    series = {k: [dict(r) for r in rows if r["date"] < today] for k, rows in _series().items()}
    history = [t for t in HISTORY if t.ts < now.isoformat()]
    summaries = [s for s in SESSION_SUMMARIES if s["date"] <= today]
    incoming = Turn(now.isoformat(), "user", DEFAULT_MESSAGE, 6)
    return Scenario("sam", dict(PROFILE), incoming, history, series, summaries, now)
