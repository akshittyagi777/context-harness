#!/usr/bin/env python3
"""Run one turn of the harness against the sample scenario.

  python3 run.py --stub                          # default turn, stubbed model
  python3 run.py --stub --budget 600             # watch the assembler degrade and drop
  python3 run.py --live                          # real model call (needs credentials)
  python3 run.py --trigger morning_brief --memory runs/memory_after.json --ts 2026-09-30T08:30
  python3 run.py --stub --out examples           # regenerate the committed example
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from domains.lab import DOMAIN  # noqa: E402
from harness.clip import clip  # noqa: E402
from harness.context import BudgetTooSmall  # noqa: E402
from harness.domain import Turn  # noqa: E402
from harness.memory import MemoryStore  # noqa: E402
from harness.turn import Boundary, run_turn  # noqa: E402
from providers import anthropic as provider  # noqa: E402

DEFAULT_BUDGET = 1050   # tight enough that the default turn degrades some context and drops some


def budget_message(exc: BudgetTooSmall) -> str:
    return (f"budget {exc.budget} is below the required context floor ({exc.floor} tokens); "
            f"raise --budget to at least {exc.floor}")


def pick_mode(live: bool, stub: bool) -> tuple[bool, str]:
    if live:
        return True, "forced by --live"
    if stub:
        return False, "forced by --stub"
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if os.environ.get(var):
            return True, f"{var} is set"
    return False, "no ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN set"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="hard token budget for the prompt")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--live", action="store_true", help="force the live model call")
    g.add_argument("--stub", action="store_true", help="force the stub")
    ap.add_argument("--memory", default=str(DOMAIN.seed_path), help="memory file to start from")
    ap.add_argument("--message", default=None, help="override the incoming message")
    ap.add_argument("--ts", default=None, help="override the turn time (ISO)")
    ap.add_argument("--out", default=str(ROOT / "runs"), help="directory for the output files")
    ap.add_argument("--trigger", default=None, choices=sorted(DOMAIN.triggers), help="run a proactive turn")
    args = ap.parse_args()

    live, why = pick_mode(args.live, args.stub)
    print(f"mode: {'live' if live else 'stub'} ({why})")
    if live:
        if not provider.available():
            sys.exit("live mode needs the anthropic package: pip install -r requirements.txt")

    now = datetime.fromisoformat(args.ts) if args.ts else None
    sc = DOMAIN.build_scenario(now)
    if args.message:
        sc = replace(sc, incoming=Turn(sc.incoming.ts, "user", args.message, sc.incoming.session))
    store = MemoryStore.load(args.memory)
    trigger = None
    if args.trigger:
        trigger = DOMAIN.triggers[args.trigger](sc, sc.now)
        if trigger is None:
            sys.exit(f"trigger '{args.trigger}' did not fire on this data; nothing to do")

    try:
        boundary = Boundary("remote", sender=provider.send) if live else None    # the lab has no EHR origin: remote only enables the live call
        out = run_turn(DOMAIN, sc, store, args.budget, live=live, trigger=trigger, boundary=boundary)
    except BudgetTooSmall as exc:
        print(budget_message(exc), file=sys.stderr)
        sys.exit(2)

    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "trace.json").write_text(out.trace.to_json())
    (od / "trace.md").write_text(out.trace.to_markdown())
    sent = (f"=== SYSTEM ===\n{out.system}\n\n=== USER ===\n{out.prompt}\n" if out.model_called
            else "(nothing was sent: emergency path)\n")
    (od / "prompt.txt").write_text(sent)
    (od / "response.txt").write_text(out.response + "\n")
    store.save(od / "memory_after.json")

    rt = out.route
    print(f"\n=== RESPONSE ({out.trace.sections['mode']['kind']}) ===\n\n{out.response}")
    print(f"\n=== ROUTING ===\ntier {rt.tier} | {rt.model} | effort {rt.effort}")
    for r in rt.reasons:
        print(f"  - {r}")
    if out.assembly is not None:
        a = out.assembly
        print(f"\n=== CONTEXT BUDGET: {a.used}/{a.budget} tokens ===")
        for d in a.decisions:
            print(f"  t{d['tier']} {d['key']:<20} {d['kept']:<16} {d['tokens']:>4}  {clip(d['why'], 70, out.trace.redactor)}")
        if a.note:
            print(f"  note: {a.note}")
    print("\n=== MEMORY WRITES ===")
    for w in out.memory_ops:
        print(f"  {w['op']:<18} {str(w['id']):<4} {clip(w['why'], 90, out.trace.redactor)}")
    print(f"\ntrace: {od / 'trace.md'} | prompt: {od / 'prompt.txt'} | memory: {od / 'memory_after.json'}")


if __name__ == "__main__":
    main()
