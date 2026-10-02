"""Run one turn end to end against a data directory: build the domain and scenario, wire the boundary
(mode, guard, sender), call the core, persist memory, history and the trace.

Mode: `--live` means a model API is involved, so assembly is `remote` (ehr-origin content never
assembled) and the call goes through the egress guard. `--stub` is `local`. A stubbed remote dry run
(`mode="remote"`, live False) renders exactly what would be sent and runs the same guard.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.domain import Turn
from harness.memory import MemoryStore
from harness.turn import Boundary, TurnResult, run_turn

from .config import egress_mode, resolve_data_dir
from .domain import DEFAULT_BUDGET, DEFAULT_MESSAGE, DEFAULT_TS, append_history, build_domain
from .egress import EgressBlocked, Guard
from .model_boundary import make_sender


def memory_path(data_dir: Path) -> Path:
    return Path(data_dir) / "memory.json"


def load_memory(data_dir: Path) -> MemoryStore:
    p = memory_path(data_dir)
    return MemoryStore.load(p) if p.exists() else MemoryStore([])


def save_memory(data_dir: Path, store: MemoryStore) -> None:
    store.save(memory_path(data_dir))
    memory_path(data_dir).chmod(0o600)


def make_boundary(data_dir: Path, mode: str, live: bool, guard: Guard | None = None, sender: Any = None) -> Boundary:
    guard = guard or Guard(data_dir)
    if mode == "local":
        return Boundary("local", redact=guard.redact)
    return Boundary("remote", sender=(sender or make_sender(guard)) if live else None,
                    preflight=guard.preflight, scrub=guard.scrub, scrub_detail=guard.scrub_detail, redact=guard.redact)


def run(data_dir: str | Path | None = None, message: str | None = None, ts: str | None = None, live: bool = False,
        mode: str | None = None, budget: int = DEFAULT_BUDGET, trigger: str | None = None, persist: bool = True,
        store: MemoryStore | None = None, guard: Guard | None = None, sender: Any = None) -> TurnResult:
    dd = resolve_data_dir(data_dir)
    mode = mode or egress_mode(live)
    domain = build_domain(dd)
    now = datetime.fromisoformat(ts or DEFAULT_TS)
    sc = domain.build_scenario(now)
    text = message if message is not None else DEFAULT_MESSAGE
    sc = replace(sc, incoming=Turn(sc.incoming.ts, "user", text, sc.incoming.session))
    store = store if store is not None else load_memory(dd)
    trig = None
    if trigger:
        trig = domain.triggers[trigger](sc, now)
        if trig is None:
            raise LookupError(f"trigger '{trigger}' did not fire on this data; nothing to do")
    out = run_turn(domain, sc, store, budget, live=live, trigger=trig, boundary=make_boundary(dd, mode, live, guard, sender))
    if persist:
        g = guard or Guard(dd)
        stamp = sc.now.isoformat()
        for rec in store.records:       # what was written this turn never keeps an EHR token the user typed
            if rec.created_at == stamp and rec.origin == "user":
                rec.content = g.redact(rec.content)
        save_memory(dd, store)
        (dd / "traces").mkdir(exist_ok=True, mode=0o700)
        tid = out.trace.turn_id
        (dd / "traces" / f"{tid}.json").write_text(g.redact(out.trace.to_json()))
        (dd / "traces" / f"{tid}.md").write_text(g.redact(out.trace.to_markdown()))
        said = "(answered locally from your records)" if out.local_kind == "ehr" else g.redact(out.response)
        if trig is None:
            append_history(dd, [Turn(sc.incoming.ts, "user", g.redact(text), sc.incoming.session),
                                Turn(sc.incoming.ts, "assistant", said, sc.incoming.session)])
    return out


def egress_check(data_dir: str | Path | None = None, message: str | None = None, ts: str | None = None,
                 budget: int = DEFAULT_BUDGET, guard: Guard | None = None) -> dict[str, Any]:
    """Render the remote prompt for a turn and run the guard over it. Nothing is sent."""
    dd = resolve_data_dir(data_dir)
    guard = guard or Guard(dd)
    out = run(dd, message, ts, live=False, mode="remote", budget=budget, persist=False, guard=guard,
              store=load_memory(dd))
    result: dict[str, Any] = {"passed": True, "categories": [], "prompt_tokens": out.trace.sections["model_call"].get("prompt_tokens"),
                              "excluded": [e["key"] for e in (out.assembly.excluded if out.assembly else [])],
                              "scrubbed_dates": (out.assembly and out.trace.sections["context_assembly"]["scrubbed_dates"]) or 0,
                              "scrubbed_terms": dict((out.trace.sections.get("egress") or {}).get("by_category", {})),
                              "route": out.route.tier, "system": out.system, "prompt": out.prompt,
                              "ehr_store": guard.vault.exists()}
    try:
        guard.check_texts([out.system, out.prompt])
    except EgressBlocked as exc:
        result["passed"], result["categories"] = False, list(exc.categories)
    return result
