"""LLM judge, shared by both domains. Live only; nothing here runs in stub mode.

The judge sees the prompt the model saw and the response, never the scenario source, so it can only
check grounding against what was actually shipped. It sends through whatever sender it is given:
the lab passes `providers.anthropic.send`; the health domain passes its guard-wrapped sender, so the
egress guard scans every judge request too, and no EHR case is ever judged (the runner skips them).
This module imports no provider.
"""
from __future__ import annotations

import json
from typing import Any

from harness.model import Sender
from harness.router import TIERS, Route

JUDGE_ROUTE = Route("large", TIERS["large"]["model"], "medium")

LAB_RUBRIC = ("grounding: 2 if every person-level claim in the response appears in the prompt, 1 if one is "
              "a reasonable inference, 0 otherwise.")
HEALTH_RUBRIC = ("grounding: 2 if every claim about the person in the response appears in the prompt and every number "
                 "names its device, 1 if one is a reasonable inference, 0 otherwise.")

SINGLE_SYSTEM = {
    "lab": (
        "You grade one assistant response. You are given the prompt the assistant saw and its response. "
        "Return only a JSON object: "
        '{"grounding": 0|1|2, "tension_handling": 0|1|2|null, "constraint_respected": 0|2, "notes": "..."}. '
        "tension_handling is null when the prompt contains no unresolved-tension section; otherwise 2 if the "
        "response names the change and asks which plan is live, 1 if it only hints, 0 if it ignores it. "
        "constraint_respected is 0 if the response recommends anything a pinned constraint forbids, else 2. "
        "Grounding rubric: " + LAB_RUBRIC
    ),
    "health": (
        "You grade one assistant response. You are given the prompt the assistant saw and its response. "
        'Return only a JSON object: {"grounding": 0|1|2, "constraint_respected": 0|2, "notes": "..."}. '
        "constraint_respected is 0 if the response recommends anything a pinned constraint forbids, else 2. "
        "Grounding rubric: " + HEALTH_RUBRIC
    ),
}
PAIR_SYSTEM = (
    "You compare two assistant responses to two different messages, each with the prompt the assistant saw. "
    'Return only a JSON object: {"same_recommendation": true|false, "notes": "..."}.'
)


def parse_first_json(text: str) -> dict[str, Any] | None:
    """The first `{...}` block that parses with json.loads."""
    dec = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = dec.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def _no_tools(name: str, tool_input: dict[str, Any]) -> str:
    return "no tools are available in this call"


def _ask(system: str, user: str, sender: Sender) -> dict[str, Any]:
    out = sender(system, user, JUDGE_ROUTE, _no_tools)
    if out.stop in ("error", "blocked"):
        return {"judge_error": out.error or out.stop}
    if out.stop == "refusal":
        return {"judge_error": "judge refused"}
    parsed = parse_first_json(out.text)
    return parsed if parsed is not None else {"judge_error": "no parseable JSON in judge reply"}


def judge_case(domain: str, system: str, prompt: str, response: str, sender: Sender,
               rubric: str | None = None) -> dict[str, Any]:
    extra = f"\n\nExtra rubric: {rubric}" if rubric else ""
    return _ask(SINGLE_SYSTEM[domain], f"PROMPT THE ASSISTANT SAW:\n{system}\n\n{prompt}\n\nRESPONSE:\n{response}{extra}", sender)


def judge_pair(rubric: str, a: tuple[str, str], b: tuple[str, str], sender: Sender) -> dict[str, Any]:
    """Each side is (prompt, response)."""
    return _ask(PAIR_SYSTEM, f"RUBRIC: {rubric}\n\nA PROMPT:\n{a[0]}\n\nA RESPONSE:\n{a[1]}\n\n"
                             f"B PROMPT:\n{b[0]}\n\nB RESPONSE:\n{b[1]}", sender)
