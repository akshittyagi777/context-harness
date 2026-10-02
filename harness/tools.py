"""The `recall` tool: lets the model fetch records that were left out of the context.

Sensitive records are gated here too: only their safe constraint text comes back unless
a flag listed in `extra.reveal_on` is up for this turn.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from .domain import Domain
from .memory import MemoryRecord, MemoryStore

RECALL_TOOL: dict[str, Any] = {
    "name": "recall",
    "description": ("Fetch stored records about the user that were not included in the context. "
                    "Use when you need a fact you were told exists but cannot see."),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Words describing what to look for."},
            "ids": {"type": "array", "items": {"type": "string"}, "description": "Record ids to fetch."},
        },
        "required": [],
        "additionalProperties": False,
    },
}


def _line(rec: MemoryRecord, flags: list[str]) -> str:
    text = rec.content
    if rec.sensitivity == "high" and not (set(rec.extra.get("reveal_on", [])) & set(flags)):
        text = rec.extra.get("constraint", "[withheld]")
    tag = "" if rec.status == "active" else f" [{rec.status}]"
    return f"[{rec.id}] ({rec.type}, {rec.day}){tag} {text}"


def make_executor(store: MemoryStore, domain: Domain, now: datetime, flags: list[str],
                  exclude: set[str], log: list[dict[str, Any]],
                  exclude_origins: tuple[str, ...] = ()) -> Callable[[str, dict[str, Any]], str]:
    """Return `execute(name, tool_input) -> text`. `exclude` holds ids already in the
    prompt (or superseded this turn); every call is appended to `log`."""

    def execute(name: str, tool_input: dict[str, Any]) -> str:
        if name != "recall":
            raise ValueError(f"unknown tool {name!r}")
        query = str(tool_input.get("query") or "")
        ids = [str(i) for i in (tool_input.get("ids") or [])]
        if ids:
            recs = [r for r in (store.get(i) for i in ids) if r is not None and r.origin not in exclude_origins]
        else:
            ranked = store.read(query, now, domain.query_synonyms)
            recs = [m["record"] for m in ranked
                    if m["id"] not in exclude and m["record"].origin not in exclude_origins][:3]
        log.append({"query": query, "ids_requested": ids, "ids_returned": [r.id for r in recs]})
        return "\n".join(_line(r, flags) for r in recs) or "No matching records."

    return execute
