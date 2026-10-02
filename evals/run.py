"""Run the eval cases of the lab domain, the health domain, or both.

  python3 -m evals.run [--domain lab|health|all] [--live] [--judge] [--only NAME]

`--domain` defaults to all. Stub mode needs no network and no credentials. Lab cases run on a fresh
copy of the lab seed. Health cases run over mock data: the mock exports are generated and imported
(the EHR into the encrypted vault, with a throwaway passphrase unless HEALTH_HARNESS_PASSPHRASE is
set), then each case runs on a private copy of that data directory. Prints one table per domain and
the check counts, exits non-zero if any check fails, writes runs/evals.json. `--judge` needs `--live`;
for the health domain it grades only cases that never touched the EHR, and sends through the same
egress guard as a real turn.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.context import BudgetTooSmall  # noqa: E402
from harness.domain import Turn  # noqa: E402
from harness.memory import MemoryStore  # noqa: E402
from harness.turn import Boundary, TurnResult, run_turn  # noqa: E402
from providers import anthropic as provider  # noqa: E402

DOMAINS = ("lab", "health")


# ------------------------------------------------------------------------- lab

def run_lab_case(case, live: bool = False) -> TurnResult | None:
    from domains.lab import DOMAIN
    store = MemoryStore.load(DOMAIN.seed_path)
    if case.setup:
        case.setup(store)                      # prior turns always use the stub: they only seed memory
    sc = DOMAIN.build_scenario(datetime.fromisoformat(case.ts))
    if case.incoming:
        sc = replace(sc, incoming=Turn(sc.incoming.ts, "user", case.incoming, sc.incoming.session))
    trigger = None
    if case.trigger:
        trigger = DOMAIN.triggers[case.trigger](sc, sc.now)
        if trigger is None:
            return None
    boundary = Boundary("remote", sender=provider.send) if live else None
    return run_turn(DOMAIN, sc, store, case.budget, live=live, trigger=trigger, boundary=boundary)


# ---------------------------------------------------------------------- health

def build_workspace(tmp: Path) -> Path:
    """Mock exports, imported into a fresh data directory. Returns the data directory."""
    from domains.health.mock.generate import generate
    from domains.health.sources import IMPORTERS
    generate(tmp / "mock")
    dd = tmp / "data"
    dd.mkdir(mode=0o700)
    for src in ("whoop", "garmin", "epic"):
        IMPORTERS[src].import_path(tmp / "mock" / src, dd)
    mem = MemoryStore.load(dd / "memory.json")
    mem.records.extend(MemoryStore.load(tmp / "mock" / "memory_seed.json").records)
    mem.save(dd / "memory.json")
    return dd


def run_health_case(case, base: Path, tmp: Path, live: bool = False, sender=None) -> TurnResult | None:
    from domains.health import runner
    from domains.health.domain import build_domain
    from domains.health.egress import EgressBlocked, Guard
    dd = tmp / f"case_{case.name}"
    shutil.copytree(base, dd)
    dd.chmod(0o700)
    store = MemoryStore.load(dd / "memory.json")
    if case.setup:
        case.setup(dd, store)
    try:
        with contextlib.ExitStack() as stack:
            if case.scrub_off:      # the guard must still block when the scrub misses everything
                stack.enter_context(mock.patch.object(Guard, "scrub_detail", lambda self, t: (t, 0, {})))
            out = runner.run(dd, case.incoming, case.ts, live=live, mode=case.mode, budget=case.budget, trigger=case.trigger,
                             persist=True, store=store, sender=sender)
    except LookupError:
        return None
    guard = Guard(dd)
    out.dd, out.guard_obj = dd, guard
    try:
        guard.check_texts([out.system, out.prompt])
        out.guard = []
    except EgressBlocked as exc:
        out.guard = list(exc.categories)
    out.nights = build_domain(dd).build_scenario(datetime.fromisoformat(case.ts)).series["nights"]
    return out


# ---------------------------------------------------------------------- shared

def evaluate(case, result: TurnResult | None) -> list[dict]:
    out = []
    for desc, fn in case.checks:
        try:
            ok = bool(fn(result))
        except Exception:                      # a missing result or section is a failed check
            ok = False
        out.append({"check": desc, "passed": ok})
    return out


def _fmt_judge(domain: str, j: dict | None) -> str:
    if not j:
        return "-"
    if "judge_error" in j:
        return "judge_error"
    return (f"g{j.get('grounding')} t{j.get('tension_handling')} c{j.get('constraint_respected')}" if domain == "lab"
            else f"g{j.get('grounding')} c{j.get('constraint_respected')}")


def _budget_error(domain: str, case, exc: BudgetTooSmall) -> int:
    if domain == "lab":
        print(f"case {case.name} needs a budget of at least {exc.floor} tokens (it was given {exc.budget}); "
              f"raise the case's budget in evals/lab_cases.py", file=sys.stderr)
    else:
        print(f"case {case.name} needs a budget of at least {exc.floor} tokens (given {exc.budget}); "
              f"raise it in evals/health_cases.py", file=sys.stderr)
    return 2


def run_domain(domain: str, args) -> tuple[list[dict], int] | int:
    """Records for every selected case of one domain, or an exit code (2) if a case budget is too small."""
    from evals.judge import judge_case, judge_pair
    if domain == "lab":
        from evals.lab_cases import CASES
    else:
        from evals.health_cases import CASES
    cases = [c for c in CASES if not args.only or c.name == args.only]
    records: list[dict] = []
    if not cases:
        return records, 0
    results: dict[str, TurnResult | None] = {}
    with contextlib.ExitStack() as stack:
        if domain == "health":
            os.environ.setdefault("HEALTH_HARNESS_PASSPHRASE", "eval-throwaway-passphrase")
            os.environ.pop("HEALTH_HARNESS_DATA", None)
            tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="health-evals-")))
            base = build_workspace(tmp)
        for case in cases:
            try:
                res = run_lab_case(case, args.live) if domain == "lab" else run_health_case(case, base, tmp, live=args.live)
            except BudgetTooSmall as exc:
                return _budget_error(domain, case, exc)
            results[case.name] = res
            rec = {"domain": domain, "case": case.name, "route": res.route.tier if res else None,
                   "checks": evaluate(case, res), "response": res.response if res else None, "judge": None}
            if domain == "lab":
                rec["model"] = res.route.model if res else None
            else:
                rec["mode"] = case.mode
            judgeable = res is not None and res.model_called and (domain == "lab" or (not case.ehr and res.mode == "remote"))
            if args.judge and judgeable:
                if domain == "lab":
                    sender = provider.send
                else:
                    from domains.health.model_boundary import make_sender
                    sender = make_sender(res.guard_obj)
                rec["judge"] = judge_case(domain, res.system, res.prompt, res.response, sender, case.rubric)
                partner = results.get(case.pair_with) if case.pair_with else None
                if partner is not None and partner.model_called:
                    rec["judge_pair"] = judge_pair(case.rubric or "", (partner.prompt, partner.response),
                                                   (res.prompt, res.response), sender)
            records.append(rec)
    return records, 0


def print_table(domain: str, records: list[dict]) -> int:
    width = 22 if domain == "lab" else 36
    print(f"\n[{domain}]\n{'case':<{width}} {'route':<7} {'checks':<8} judge")
    failed = 0
    for rec in records:
        ok = sum(c["passed"] for c in rec["checks"])
        failed += len(rec["checks"]) - ok
        extra = _fmt_judge(domain, rec["judge"])
        if "judge_pair" in rec:
            jp = rec["judge_pair"]
            extra += " pair:" + ("error" if "judge_error" in jp else f"same={jp.get('same_recommendation')}")
        print(f"{rec['case']:<{width}} {str(rec['route']):<7} {ok}/{len(rec['checks']):<6} {extra}")
        for c in rec["checks"]:
            if not c["passed"]:
                print(f"    FAILED: {c['check']}")
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=[*DOMAINS, "all"], default="all", help="which domain's cases to run (default all)")
    ap.add_argument("--live", action="store_true", help="use the live model (health: through the egress guard)")
    ap.add_argument("--judge", action="store_true", help="grade responses with the judge (requires --live)")
    ap.add_argument("--only", default=None, help="run one case by name")
    args = ap.parse_args()
    if args.judge and not args.live:
        sys.exit("--judge requires --live")
    print(f"mode: {'live' if args.live else 'stub'} ({'forced by --live' if args.live else 'default; pass --live for a live run'})")
    if args.live and not provider.available():
        sys.exit("live mode needs the anthropic package: pip install -r requirements.txt")

    selected = DOMAINS if args.domain == "all" else (args.domain,)
    all_records: list[dict] = []
    summary: list[str] = []
    failed_total = 0
    for domain in selected:
        out = run_domain(domain, args)
        if isinstance(out, int):
            return out
        records, _ = out
        if not records:
            continue
        failed = print_table(domain, records)
        total = sum(len(r["checks"]) for r in records)
        summary.append(f"{domain}: {len(records)} cases, {total - failed}/{total} checks passed")
        failed_total += failed
        all_records += records
    if not all_records:
        sys.exit(f"no case named {args.only}")
    out_dir = ROOT / "runs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "evals.json").write_text(json.dumps(all_records, indent=2, ensure_ascii=False))
    print("\n" + "\n".join(summary))
    print(f"{'all checks passed' if not failed_total else f'{failed_total} check(s) failed'}; wrote {out_dir / 'evals.json'}")
    return 1 if failed_total else 0


if __name__ == "__main__":
    sys.exit(main())
