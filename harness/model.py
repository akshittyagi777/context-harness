"""The model call: a live path through an injected sender, and a generic stub.

The core knows no vendor and no wire format. A live call is `sender(system, prompt, route, executor)
-> ModelResult`: the sender owns the request shape, the tool loop, the SDK, the credentials and any
outbound guard (see `providers/anthropic.py`), and calls `executor` to run the model's recall tool.
With no sender, a live call is refused; stub mode needs none.
The stub is deliberately plain: it is built from the turn's structured inputs and never
mentions a record that is not in the prompt, so what it says is traceable to the
harness's decisions rather than to model skill.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable

from .domain import Resolution, Tension, Trigger
from .memory import MemoryRecord
from .router import Route


class SenderError(Exception):
    """The sender could not complete the request (network or API status error)."""


class SendBlocked(Exception):
    """A guard stopped the call before anything left the machine. `categories` names why, never values."""

    def __init__(self, message: str, categories: tuple[str, ...] | list[str] = ()):
        super().__init__(message)
        self.categories = tuple(categories)


Executor = Callable[[str, dict[str, Any]], str]          # (tool name, tool input) -> tool result text
Sender = Callable[[str, str, Route, Executor], "ModelResult"]   # (system, prompt, route, executor)
Preflight = Callable[[str, str], None]    # (system, prompt); raises SendBlocked


@dataclass
class ModelResult:
    text: str
    mode: str                        # live | stub | skipped
    latency_ms: int
    model: str
    usage: dict[str, int] | None = None
    stop: str = "ok"                 # ok | refusal | error
    error: str | None = None
    rounds: int = 1


@dataclass
class ShippedPinned:
    id: str
    text: str
    sensitive: bool
    revealed: bool = False           # the gate was open and the full content shipped
    constraint: str = ""             # the record's safe constraint text, when it has one
    forbid: list[str] = field(default_factory=list)
    safe_note: str = ""              # reply text the record's owner vetted for use once the gate is open


@dataclass
class StubInputs:
    """What the stub is allowed to see, beyond the prompt text itself."""
    tensions: list[Tension] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    risk_reasons: list[str] = field(default_factory=list)
    pinned: list[ShippedPinned] = field(default_factory=list)
    notes: list[Resolution] = field(default_factory=list)       # only notes shipped at full detail; the stub cites those about the incoming message
    open_loops: list[MemoryRecord] = field(default_factory=list)  # only loops shipped in the prompt
    trigger: Trigger | None = None
    incoming: str = ""               # the user's message, for quoting their own words back
    values: dict[str, Any] = field(default_factory=dict)   # the resolver's computed values (domain stubs read these)


# ------------------------------------------------------------------------ stub

def _first_lower(text: str) -> str:
    """Lowercase the first letter unless it starts an acronym or identifier (second char is upper)."""
    if len(text) > 1 and text[1].isupper():
        return text
    return text[:1].lower() + text[1:]


def _short_plan(content: str) -> str:
    text = re.sub(r"^\s*revised plan:\s*", "", content, flags=re.I)
    m = re.search(r",|\.(?=\s|$)", text)
    return _first_lower((text[:m.start()] if m else text).strip())


def _user_words(hits: list[str], message: str) -> str:
    """The user's hit words as they wrote them: one phrase if they sit side by side, else joined."""
    spans = []
    for w in hits:
        m = re.search(rf"(?<!\w){re.escape(w)}(?:\s*°\s*[A-Za-z](?!\w))?", message, re.I)
        if m is None:
            return " and ".join(hits)
        spans.append(m.span())
    spans.sort()
    if all(not message[a[1]:b[0]].strip() for a, b in zip(spans, spans[1:])):
        return message[spans[0][0]:spans[-1][1]]
    return " and ".join(message[a:b] for a, b in spans)


def _mentionable(text: str, forbid: list[str]) -> bool:
    low = text.lower()
    return bool(text.strip()) and not any(t.lower() in low for t in forbid)


def _stub(inp: StubInputs) -> str:
    parts: list[str] = []
    if inp.trigger is not None and inp.trigger.facts:
        parts.append(inp.trigger.facts)

    for t in inp.tensions:
        if t.old is not None and t.new is not None:
            d = datetime.fromisoformat(t.new.day)
            words = _user_words(t.hits, inp.incoming) if t.hits else "the old plan"
            parts.append(f"Before that: on {d:%b} {d.day} you switched to {_short_plan(t.new.content)}; "
                         f"today's message goes back to {words}. Which plan is live?")
        else:
            parts.append("Before that: a plan changed and today's message points at the old one. "
                         "Which plan is live?")

    if inp.flags:
        def named(p: ShippedPinned) -> bool:
            return any(_names(r, p.id) for r in inp.risk_reasons)
        plain = [p for p in inp.pinned if not p.sensitive and (named(p) or inp.trigger is not None)]
        gated = [p for p in inp.pinned if p.sensitive and p.revealed]   # a closed gate is never acknowledged
        sentences = [f"On file: {_first_lower(p.text.strip().removesuffix('.'))}. "
                     f"Staying under that is the safer call." for p in plain]
        for p in gated:   # never quote a sensitive record; use its safe constraint text only if it is clean
            if _mentionable(p.safe_note, p.forbid):
                sentences.append(f"I'd steer away from that here. {p.safe_note}")
            elif _mentionable(p.constraint, p.forbid):
                sentences.append(f"That runs into a standing constraint on file: {p.constraint}")
            else:
                sentences.append("That runs into a standing constraint on file, so I'd steer toward an "
                                 "alternative that avoids it.")
        if not sentences:
            sentences.append("This is flagged as a risk, so the conservative option is to pause and verify "
                             "before continuing.")
        parts.append(" ".join(sentences))

    for n in inp.notes:
        if n.kind == "measured_vs_recorded" and n.about_incoming:
            parts.append(f"On the data: {n.outcome}.")

    if inp.trigger is not None and inp.trigger.kind == "scheduled" and inp.open_loops:
        loop = inp.open_loops[0]
        parts.append(f"Still open: {loop.content.replace(' Unanswered.', '')} Which is it?")
    elif inp.trigger is not None and not inp.tensions and not inp.flags:
        parts.append("Nothing is open from last time.")

    return " ".join(parts) or "Noted. Anything you want me to keep an eye on?"


def _names(reason: str, rid: str) -> bool:
    return re.search(rf"\b{re.escape(rid)}\b", reason) is not None


# ----------------------------------------------------------------------- entry

def call_model(system: str, prompt: str, route: Route, live: bool = False,
               stub: StubInputs | None = None,
               executor: Executor | None = None,
               sender: Sender | None = None, preflight: Preflight | None = None,
               stub_fn: Callable[[StubInputs], str] | None = None) -> ModelResult:
    t0 = time.perf_counter()
    if preflight is not None:      # runs for stub and live alike, before anything is "sent"
        try:
            preflight(system, prompt)
        except SendBlocked as exc:
            return ModelResult("", "blocked", int((time.perf_counter() - t0) * 1000), route.model,
                               stop="blocked", error=str(exc))
    if live:
        if executor is None:
            raise ValueError("live calls need a tool executor")
        if sender is None:
            raise ValueError("live calls need a sender (the boundary that owns the SDK and the guard)")
        try:
            return sender(system, prompt, route, executor)
        except SendBlocked as exc:
            return ModelResult("", "live", int((time.perf_counter() - t0) * 1000), route.model,
                               stop="blocked", error=str(exc))
        except SenderError as exc:
            return ModelResult("", "live", int((time.perf_counter() - t0) * 1000), route.model,
                               stop="error", error=f"{type(exc).__name__}: {exc}")
    text = (stub_fn or _stub)(stub or StubInputs())
    return ModelResult(text, "stub", int((time.perf_counter() - t0) * 1000), route.model)
