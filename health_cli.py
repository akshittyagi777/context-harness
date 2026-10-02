#!/usr/bin/env python3
"""Personal health harness CLI.

  python3 health_cli.py mock --out mock_exports
  python3 health_cli.py import whoop|garmin|epic|memory <path>
  python3 health_cli.py sync whoop|garmin|epic
  python3 health_cli.py status | audit | egress-check
  python3 health_cli.py turn "message" [--ts ISO] [--live|--stub] [--budget N] [--mode local|remote]
  python3 health_cli.py brief [--ts ISO]
  python3 health_cli.py regime add START END KIND LABEL | list
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness.clip import clip  # noqa: E402
from harness.context import BudgetTooSmall  # noqa: E402
from harness.memory import MemoryStore  # noqa: E402
from domains.health import runner  # noqa: E402
from domains.health.baselines import Baselines, add_regime, load_regimes  # noqa: E402
from domains.health.canonical import SIGNALS, SOURCE_LABEL, WearableStore, date_range, load_ledger  # noqa: E402
from domains.health.config import ENV_PASSPHRASE, get_passphrase, load_config, resolve_data_dir  # noqa: E402
from domains.health.domain import DEFAULT_BUDGET, DEFAULT_TS  # noqa: E402
from domains.health.precedence import WearView, disagreement_summary, load_precedence  # noqa: E402
from domains.health.sources import IMPORTERS, SYNCERS, ExportRefused, NotConfigured  # noqa: E402
from domains.health.vault import Vault, VaultError  # noqa: E402


def cmd_mock(a, dd) -> int:
    from domains.health.mock.generate import generate
    t = generate(a.out)
    print(f"wrote mock exports to {a.out}/ (all data is invented)")
    for k, v in sorted(t.counts.items()):
        print(f"  {k}: {v}")
    return 0


def cmd_import(a, dd) -> int:
    if a.source == "memory":
        store = runner.load_memory(dd)
        seed = MemoryStore.load(a.path)
        have = {r.id for r in store.records}
        added = [r for r in seed.records if r.id not in have]
        store.records.extend(added)
        runner.save_memory(dd, store)
        print(f"memory: added {len(added)} record(s), skipped {len(seed.records) - len(added)} already present")
        return 0
    imp = IMPORTERS[a.source]
    try:
        if not imp.can_import(a.path):
            print(f"{a.path} does not look like a {a.source} export", file=sys.stderr)
            return 2
        res = imp.import_path(a.path, dd)
    except (VaultError, ExportRefused) as exc:        # ExportRefused covers ExportTooLarge: archive limits apply at open
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(res.summary())
    return 0


def cmd_sync(a, dd) -> int:
    try:
        SYNCERS[a.source].sync(None, dd)
    except NotConfigured as exc:
        print(f"not configured: {exc}")
        print("would need:")
        for r in SYNCERS[a.source].requirements():
            print(f"  - {r}")
    return 0


def _coverage(store: WearableStore, src: str) -> tuple[str, int, list[str]]:
    dates = [r.date for r in store.night_rows(src)]
    if not dates:
        return "none", 0, []
    have = set(dates)
    return f"{dates[0]}..{dates[-1]}", len(dates), [d for d in date_range(dates[0], dates[-1]) if d not in have]


def cmd_status(a, dd) -> int:
    store = WearableStore.load(dd)
    print(f"data dir: {dd}")
    print("\nWearables")
    ranges = {}
    for src in store.sources():
        rng, n, gaps = _coverage(store, src)
        ranges[src] = rng
        print(f"  {SOURCE_LABEL.get(src, src):<7} nights {n:>4}  {rng}  gaps {len(gaps)}"
              + (f" ({', '.join(gaps[:6])}{' ...' if len(gaps) > 6 else ''})" if gaps else ""))
    srcs = [s for s in store.sources() if ranges.get(s) not in (None, "none")]
    if len(srcs) >= 2:
        starts = [ranges[s].split("..")[0] for s in srcs]
        ends = [ranges[s].split("..")[1] for s in srcs]
        both = set.intersection(*({r.date for r in store.night_rows(s)} for s in srcs))
        print(f"  overlap window {max(starts)}..{min(ends)}, {len(both)} night(s) with every source")
    v = Vault(dd)
    print("\nEHR (local only)")
    if not v.exists():
        print("  no EHR store (nothing imported)")
    else:
        led = [e for e in load_ledger(dd) if e["source"] == "epic"]
        print(f"  encrypted store present ({v.path.name}); passphrase {'set' if get_passphrase() else 'NOT set'} in {ENV_PASSPHRASE}")
        for e in led[-3:]:
            print(f"  import {e['ts']}: {e['path']} added {e['added'] or 0}")
        print(f"  access log entries: {len(v.read_access_log())}")
    regs = load_regimes(dd)
    if regs:
        print("\nRegimes (excluded from baselines and trends)")
        for r in regs:
            print(f"  {r['start']}..{r['end']} {r['kind']} ({r['label']})")
    print("\nLast imports")
    for e in [e for e in load_ledger(dd) if e["source"] != "epic"][-4:]:
        print(f"  {e['ts']} {e['source']}: {e['path']} added {e['added'] or 0}, skipped {e['skipped'] or 0}, {e['n_anomalies']} anomaly(ies)")
    return 0


def cmd_audit(a, dd) -> int:
    store = WearableStore.load(dd)
    cfg = load_config(dd)
    prec = load_precedence(dd)
    series = store.series()
    view = WearView(series, prec)
    print("Signal map: signal x source x granularity x coverage x overlap (precedence from precedence.json)\n")
    print(f"{'signal':<15}{'precedence':<18}{'source':<8}{'granularity':<13}{'days':>6}  {'range':<24}overlap")
    gran = {"sleep_duration": "per night", "sleep_stages": "per night", "resting_hr": "per night", "hrv": "per night",
            "recovery": "per day", "steps": "per day", "activity": "per workout", "stress": "per day",
            "body_battery": "per day", "spo2": "per night"}
    for sig, (table, field, _) in SIGNALS.items():
        per_src: dict[str, set[str]] = {}
        for src in ("whoop", "garmin"):
            if table == "workouts":
                ds = {w["date"] for w in series["workouts"] if w["source"] == src}
            else:
                ds = {r["date"] for r in series[table] if r["source"] == src and r.get(field) is not None}
            per_src[src] = ds
        both = per_src["whoop"] & per_src["garmin"]
        for src, ds in per_src.items():
            if not ds:
                continue
            rng = f"{min(ds)}..{max(ds)}"
            ov = f"{len(both)} day(s) with both" if both else "-"
            print(f"{sig:<15}{'>'.join(prec.get(sig, [])):<18}{src:<8}{gran[sig]:<13}{len(ds):>6}  {rng:<24}{ov}")
    last = view.last_night()
    if last:
        base = Baselines(series["nights"], load_regimes(dd), cfg)
        s = disagreement_summary(view, base, last, cfg)
        print(f"\n14-day disagreement summary as of {last}: {' vs '.join(SOURCE_LABEL.get(x, x) for x in s['sources'])}")
        print(f"  nights compared {s['nights_compared']}, mean sleep difference {s['mean_sleep_diff_min']} min (first minus second)")
        print(f"  over the {cfg['sleep_disagreement_min']} min threshold: {s['sleep_over_nights'] or 'none'}")
        print(f"  resting HR differing by more than {cfg['rhr_disagreement_bpm']} bpm: {s['rhr_over_nights'] or 'none'}")
        print(f"  HRV direction mismatch (each against its own baseline): {s['hrv_mismatch_nights'] or 'none'}")
        print("  HRV values are never compared across sources (different methods); disagreement is reported, never averaged.")
    return 0


def _print_turn(out, show_prompt: bool) -> None:
    rt = out.route
    red = out.trace.redactor          # the boundary's: free text is redacted before the columns below clip it
    print(f"=== RESPONSE ({out.trace.sections['mode']['kind']}, assembly mode {out.mode}) ===\n\n{out.response}")
    print(f"\n=== ROUTING ===\ntier {rt.tier} | {rt.model} | effort {rt.effort}")
    for r in rt.reasons:
        print(f"  - {r}")
    if out.assembly is not None:
        a = out.assembly
        print(f"\n=== CONTEXT BUDGET: {a.used}/{a.budget} tokens ===")
        for d in a.decisions:
            print(f"  t{d['tier']} {d['key']:<26} {d['kept']:<14} {d['tokens']:>4}  {clip(d['why'], 60, red)}")
        for e in a.excluded:
            print(f"  -- {e['key']:<26} {'EXCLUDED':<14} {e['tokens']:>4}  {e['why']}")
        if a.note:
            print(f"  note: {a.note}")
    if out.trace.sections.get("send_blocked"):
        print(f"\n=== BLOCKED ===\n{out.trace.sections['send_blocked']['reason']} (templated reply used; nothing was sent)")
    print("\n=== MEMORY WRITES ===")
    for w in out.memory_ops:
        print(f"  {w['op']:<18} {str(w['id']):<18} {clip(w['why'], 80, red)}")
    if show_prompt:
        if out.mode == "remote" and out.model_called:
            print(f"\n=== PROMPT (exactly what would be sent) ===\n--- system ---\n{out.system}\n--- user ---\n{out.prompt}")
        else:
            print("\n(prompt not shown: local assembly may contain EHR content; use --mode remote to see the sendable prompt)")


def cmd_turn(a, dd) -> int:
    mode = a.mode or ("remote" if a.live else "local")
    try:
        out = runner.run(dd, a.message, a.ts, live=a.live, mode=mode, budget=a.budget)
    except BudgetTooSmall as exc:
        print(f"budget {exc.budget} is below the required context floor ({exc.floor} tokens); raise --budget", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_turn(out, a.show_prompt)
    print(f"\ntrace: {dd / 'traces' / (out.trace.turn_id + '.md')}")
    return 0


def cmd_brief(a, dd) -> int:
    try:
        out = runner.run(dd, None, a.ts, live=False, mode=a.mode, budget=a.budget, trigger="morning_brief")
    except LookupError as exc:
        print(exc, file=sys.stderr)
        return 2
    _print_turn(out, False)
    return 0


def cmd_egress(a, dd) -> int:
    r = runner.egress_check(dd, None, a.ts, a.budget)
    store = "EHR store present" if r["ehr_store"] else "no EHR store (textual identifier patterns only)"
    print(f"remote prompt: {r['prompt_tokens']} tokens, route {r['route']}, {len(r['excluded'])} ehr candidate(s) excluded "
          f"({', '.join(r['excluded']) or 'none'}), {r['scrubbed_dates']} date(s) coarsened, "
          f"{sum(r['scrubbed_terms'].values())} term(s) scrubbed {r['scrubbed_terms'] or ''}; {store}")
    if r["passed"]:
        print("PASS")
        return 0
    print("BLOCKED: categories " + ", ".join(r["categories"]))
    return 1


def cmd_regime(a, dd) -> int:
    if a.action == "add":
        ok = add_regime(dd, a.start, a.end, a.kind, a.label)
        print("regime added" if ok else "regime already present")
    else:
        for r in load_regimes(dd):
            print(f"{r['start']}..{r['end']} {r['kind']} ({r['label']}, {r['origin']})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help="data directory (default $HEALTH_HARNESS_DATA or ~/.health-harness)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("mock"); p.add_argument("--out", default="mock_exports"); p.set_defaults(fn=cmd_mock)
    p = sp.add_parser("import"); p.add_argument("source", choices=["whoop", "garmin", "epic", "memory"]); p.add_argument("path"); p.set_defaults(fn=cmd_import)
    p = sp.add_parser("sync"); p.add_argument("source", choices=["whoop", "garmin", "epic"]); p.set_defaults(fn=cmd_sync)
    sp.add_parser("status").set_defaults(fn=cmd_status)
    sp.add_parser("audit").set_defaults(fn=cmd_audit)
    p = sp.add_parser("turn"); p.add_argument("message", nargs="?", default=None); p.add_argument("--ts", default=None)
    g = p.add_mutually_exclusive_group(); g.add_argument("--live", action="store_true"); g.add_argument("--stub", action="store_true")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET); p.add_argument("--mode", choices=["local", "remote"], default=None)
    p.add_argument("--show-prompt", action="store_true"); p.set_defaults(fn=cmd_turn)
    p = sp.add_parser("brief"); p.add_argument("--ts", default=None); p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--mode", choices=["local", "remote"], default="local"); p.set_defaults(fn=cmd_brief)
    p = sp.add_parser("egress-check"); p.add_argument("--ts", default=None); p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.set_defaults(fn=cmd_egress)
    p = sp.add_parser("regime"); p.add_argument("action", choices=["add", "list"]); p.add_argument("start", nargs="?")
    p.add_argument("end", nargs="?"); p.add_argument("kind", nargs="?", default="travel"); p.add_argument("label", nargs="?", default="user marked")
    p.set_defaults(fn=cmd_regime)
    a = ap.parse_args(argv)
    dd = None if a.cmd == "mock" else resolve_data_dir(a.data_dir)       # `mock` touches no data dir
    return a.fn(a, dd)


if __name__ == "__main__":
    sys.exit(main())
