"""One turn: safety pre-check, resolve, read, assemble, route, call, post-check, write.

Reactive when `trigger` is None (the user spoke); proactive otherwise (the assistant
speaks first). Same pipeline either way, and every stage writes to the trace.

Boundary. `Boundary.mode` is `local` (every origin may be rendered; no model API may be called) or
`remote` (ehr-origin records and candidates never reach assembly, recall or the prompt). A live call
needs remote mode plus a sender. `preflight` sees the rendered (system, prompt) before any model call
and may raise SendBlocked; `scrub` rewrites rendered text in remote mode (e.g. coarsen dates).
`scrub_detail`, when set, replaces `scrub` and also reports counts by category for terms it replaced with
placeholders. In remote mode the pre-check also runs on the scrubbed incoming text (flags are unioned), the
stub sees only scrubbed text, and the trace records what was scrubbed (counts, never the terms) and whether the
guard passed.

Local responders. A pre-check flag can route to deterministic local code instead of the model
(the emergency short-circuit is the first of them).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable

from .clip import clip
from .context import assemble_with_note, build_candidates, render
from .domain import Domain, LocalReply, Scenario, Tension, Trigger, Turn
from .memory import MemoryOp, MemoryRecord, MemoryStore, overlap_count, tokens
from .model import ModelResult, Preflight, Sender, ShippedPinned, StubInputs, call_model
from .resolve import Resolved, resolve
from .router import Route, emergency_route, escalate, route
from .safety import PostCheck, PreCheck, escalation_summary, post_check, pre_check, templated_fallback
from .tokens import count_tokens
from .tools import make_executor
from .trace import Trace


@dataclass
class Boundary:
    mode: str = "local"                                   # local | remote
    sender: Sender | None = None                          # the only way a live call leaves the process
    preflight: Preflight | None = None
    scrub: Callable[[str], tuple[str, int]] | None = None  # remote only; returns (text, n_changes)
    scrub_detail: Callable[[str], tuple[str, int, dict[str, int]]] | None = None   # remote only; (text, n_dates, counts by category)
    redact: Callable[[str], str] | None = None             # both modes; applied to every free-text value before it is clipped (memory writes, trace, fallback, rules)


@dataclass
class TurnResult:
    response: str
    system: str
    prompt: str
    trace: Trace
    assembly: Any
    route: Any
    resolved: Resolved
    pre: PreCheck
    post: PostCheck | None
    memory_ops: list[dict[str, Any]]
    fallback_used: bool = False
    model_called: bool = True
    trigger: Trigger | None = None
    store: MemoryStore | None = None
    local_kind: str | None = None
    mode: str = "local"


GATED_EPISODE_CONSTRAINT = "Earlier turn on a gated topic; details withheld."


def _gated_episode(store: MemoryStore, flags: list[str]) -> dict[str, Any]:
    """Record fields that carry a gate onto an episode, or {} when this turn opened none.
    A turn that opened a sensitive record's gate (a flag listed in its `reveal_on`) is itself
    sensitive: its episode takes that `reveal_on`, so the words that opened the gate are only
    replayed on a later turn that opens it again."""
    reveal: list[str] = []
    for r in store.active():
        if r.sensitivity == "high" and set(r.extra.get("reveal_on", [])) & set(flags):
            reveal += [f for f in r.extra["reveal_on"] if f not in reveal]
    if not reveal:
        return {}
    return {"sensitivity": "high", "extra": {"reveal_on": reveal, "constraint": GATED_EPISODE_CONSTRAINT}}


def _write_ops(domain: Domain, sc: Scenario, res: Resolved, store: MemoryStore, response: str,
               now: datetime, trigger: Trigger | None, tensions: list[Tension],
               flags: list[str], redact: Callable[[str], str] | None = None) -> list[MemoryOp]:
    """Memory ops for this turn, in order of trust:
      1. read-time repairs and domain-requested ops (e.g. reinforce a pattern)
      2. an episode of what happened, an open loop per tension, and closing any open
         loop the incoming message answers
      3. SEAM: model-extracted memories would be merged here. Constraints when built:
         source "inferred", confidence at most 0.6, never pinned, never sensitive,
         so a model's guess can never outrank a user statement or reach a gate.
    An episode written on a turn that opened a sensitive gate is gated the same way.
    Text is redacted before it is clipped: a clip first could split a denylisted word so redaction misses it.
    """
    ops = list(res.memory_ops)
    clean = redact or (lambda t: t)
    ts = now.isoformat()
    gated = _gated_episode(store, flags)
    n = int(store.next_id()[1:])

    def nid() -> str:
        nonlocal n
        n += 1
        return f"r{n - 1:02d}"

    said = " ".join(sc.incoming.text.split())
    if trigger is not None:
        content = f"{now.date()}: assistant initiated ({trigger.name}); said: \"{clip(clean(response), 110)}\""
        source, why = "inferred", "what the assistant initiated and said"
    else:
        content = f"{now.date()}: user said: \"{clip(clean(said), 140)}\""
        source, why = "user_stated", "what happened this turn, in the user's own words"
    ops.append(MemoryOp("add", MemoryRecord(
        id=nid(), type="episode", topic=f"episode:{now.date()}:{now.strftime('%H%M')}", content=content,
        created_at=ts, updated_at=ts, source=source, confidence=0.9, **gated), None, why))

    for t in tensions:
        old = f"'{t.old.content}' ({t.old.day})" if t.old else "the earlier plan"
        new = f"'{t.new.content}' ({t.new.day})" if t.new else "the current plan"
        ops.append(MemoryOp("add", MemoryRecord(
            id=nid(), type="open_loop", topic=f"open:{t.key}",
            content=f"Asked on {now.date()} whether {old} or {new} is the live plan. Unanswered.",
            created_at=ts, updated_at=ts, source="inferred", confidence=0.8), None,
            "a question about this tension was put to the user; the next turn must know it is pending"))

    if trigger is None:
        words = tokens(said)
        for r in store.active():
            if r.type == "open_loop" and overlap_count(words, tokens(r.content), domain.query_synonyms) >= 2:
                ops.append(MemoryOp("supersede", MemoryRecord(
                    id=nid(), type="episode", topic=r.topic.replace("open:", "closed:"),
                    content=f"Open loop closed {now.date()}: user answered \"{clip(clean(said), 100)}\"",
                    created_at=ts, updated_at=ts, source="user_stated", confidence=0.85, **gated), r.id,
                    "the incoming message answers a pending question"))
    return ops


def run_turn(domain: Domain, sc: Scenario, store: MemoryStore, budget: int, live: bool = False,
             trigger: Trigger | None = None, boundary: Boundary | None = None) -> TurnResult:
    b = boundary or Boundary()
    if b.mode not in ("local", "remote"):
        raise ValueError(f"unknown mode {b.mode!r}")
    if live and b.mode != "remote":
        raise ValueError("a live model call needs remote mode: local mode may render every origin")
    remote = b.mode == "remote"
    exclude_origins = ("ehr",) if remote else ()
    # In remote mode the whole turn reads a view of memory with ehr-origin records removed; writes
    # still go to the real store.
    view = MemoryStore([r for r in store.records if r.origin not in exclude_origins]) if remote else store
    now = sc.now
    if b.redact is not None:      # offered to domain rules, which clip user text into records and resolutions
        sc = replace(sc, profile={**sc.profile, "redact": b.redact})
    reactive = trigger is None
    if trigger is not None:
        session = max((t.session for t in sc.history), default=0) + 1
        sc = replace(sc, incoming=Turn(trigger.ts, "user", trigger.text, session))
        loops = [r.content for r in view.active() if r.type == "open_loop"]
        query = trigger.query + " " + " ".join(loops)
    else:
        query = sc.incoming.text
    turn_hash = hashlib.sha1(f"{sc.incoming.ts}|{sc.incoming.text}".encode()).hexdigest()[:6]
    trace = Trace(f"turn_{sc.incoming.ts[:10]}_{turn_hash}", redactor=b.redact)
    trace.set("mode", {"kind": "reactive" if reactive else f"proactive:{trigger.kind}",
                       "trigger": trigger.name if trigger else None,
                       "why": "the user sent a message" if reactive else trigger.why,
                       "memory_query": query, "assembly_mode": b.mode})
    if remote:
        held = [f"memory:{r.id}" for r in store.records if r.origin in exclude_origins]
        trace.set("remote_exclusions", {"memory_records": held, "why": "excluded: remote mode, origin ehr"})

    def scrub_text(text: str) -> tuple[str, int, dict[str, int]]:
        """What a remote model may see of `text`: (scrubbed text, dates coarsened, term counts by category)."""
        if b.scrub_detail is not None:
            return b.scrub_detail(text)
        if b.scrub is not None:
            out, n = b.scrub(text)
            return out, n, {}
        return text, 0, {}

    can_scrub = remote and (b.scrub_detail is not None or b.scrub is not None)
    term_counts: dict[str, int] = {}          # by category, never the terms

    # 1. Safety pre-check (deterministic, before any model call)
    pinned = [r for r in view.active() if r.pinned]
    pre = pre_check(domain, sc.incoming.text, pinned, sc)
    incoming_out = scrub_text(sc.incoming.text)[0] if can_scrub else sc.incoming.text    # all a remote stub or model sees of it
    if incoming_out != sc.incoming.text and pre.level != "emergency":
        # Redact, then guard: the pre-check also reads the scrubbed text, so a flag whose pattern matches a
        # placeholder (e.g. a medication on file) is raised. Flags are unioned; the original text still decides emergencies.
        extra = pre_check(domain, incoming_out, pinned, sc)
        if extra.level != "emergency":
            for f in extra.flags:
                if f not in pre.flags:
                    pre.flags.append(f)
                    pre.reasons += [r for r in extra.reasons if r.startswith(f"{f}:")] or [f"{f}: raised on the scrubbed text"]
            pre.level = "elevated" if pre.flags else pre.level
    if trigger is not None and pre.level != "emergency":
        for f in trigger.flags:
            pre.add_risk(f, f"trigger-supplied flag from {trigger.name}")
    trace.set("safety_pre", {"level": pre.level, "flags": pre.flags, "reasons": pre.reasons})
    trace.log("safety", f"pre-check level={pre.level}", flags=pre.flags)

    # 2. Resolve
    res = resolve(domain, sc, view, reactive)
    trace.set("resolutions", [r.__dict__ | {"loser_ids": list(r.loser_ids)} for r in res.resolutions])
    trace.set("tensions", [{"key": t.key, "why": t.why, "text": t.text} for t in res.tensions])
    trace.log("resolve", f"{len(res.resolutions)} resolutions, {len(res.tensions)} tensions, "
                         f"{len(res.memory_ops)} memory ops requested")

    # Local responders: a flag routes the turn to deterministic local code; the model is not called.
    # A responder registered under "*" is offered every reactive turn first and decides for itself.
    if pre.level != "emergency" and trigger is None:
        for flag in ("*", *pre.flags):
            fn = domain.local_responders.get(flag)
            reply: LocalReply | None = fn(sc.incoming.text, pre, sc) if fn else None
            if reply is not None:
                rt = Route("local", "local", None, [reply.note])
                trace.set("memory_read", [])
                trace.set("routing", {"tier": rt.tier, "model": rt.model, "effort": rt.effort, "reasons": rt.reasons})
                trace.set("model_call", {"skipped": reply.note, "calls": []})
                trace.set("local_response", {"kind": reply.kind, "flag": flag, "note": reply.note})
                trace.log("route", reply.note, flag=flag)
                ops = _write_ops(domain, sc, res, store, reply.text, now, trigger, [], pre.flags, b.redact)
                applied = store.apply(ops, now)
                trace.set("memory_write", applied)
                return TurnResult(reply.text, "", "", trace, None, rt, res, pre, None, applied,
                                  model_called=False, trigger=trigger, store=store, local_kind=reply.kind,
                                  mode=b.mode)

    # Emergency: deterministic reply, model skipped, escalation summary attached.
    if pre.level == "emergency":
        esc = escalation_summary(domain, pre, pinned, res.plans_active, res.values,
                                 [t.text for t in res.tensions])
        rt = emergency_route()
        response = domain.emergency_message
        trace.set("memory_read", [])
        trace.set("routing", {"tier": rt.tier, "model": rt.model, "effort": rt.effort, "reasons": rt.reasons})
        trace.set("model_call", {"skipped": "model skipped: emergency path", "calls": []})
        trace.set("escalation", esc)
        trace.log("safety", "model skipped: emergency path; escalation summary attached")
        ops = _write_ops(domain, sc, res, store, response, now, trigger, [], pre.flags, b.redact)
        applied = store.apply(ops, now)
        trace.set("memory_write", applied)
        return TurnResult(response, "", "", trace, None, rt, res, pre, None, applied,
                          model_called=False, trigger=trigger, store=store, local_kind="emergency",
                          mode=b.mode)

    # 3. Memory read
    ranked = view.read(query, now, domain.query_synonyms)
    trace.log("memory", f"ranked {len(ranked)} active records")

    # 4. Assemble under budget
    asm = assemble_with_note(build_candidates(domain, sc, ranked, res, pre), budget, b.mode)
    system, prompt = render(asm)
    scrubbed = 0
    if can_scrub:
        (system, n1, c1), (prompt, n2, c2) = scrub_text(system), scrub_text(prompt)
        scrubbed = n1 + n2
        for c in (c1, c2):
            for k, v in c.items():
                term_counts[k] = term_counts.get(k, 0) + v
        if scrubbed:
            trace.log("scrub", f"{scrubbed} date(s) coarsened to month precision for the remote prompt")
        trace.log("egress", f"scrubbed {sum(term_counts.values())} term(s) by category {term_counts}")
    keys = {p.candidate.key: p for p in asm.placed}
    gates = {g["id"]: g for g in asm.gates}
    losers = res.losers
    rows = []
    for m in ranked:
        rid = m["id"]
        if rid in losers:
            decision, why = "excluded", "superseded at read time by the resolver"
        elif f"pinned:{rid}" in keys:
            g = gates.get(rid)
            decision, why = ("in", "pinned" + (f"; gate {g['gate']}, shipped {g['shipped']}" if g else ""))
        elif m["type"] in ("plan", "preference"):
            decision, why = ("in (plans_prefs)" if "plans_prefs" in keys else "dropped"), "resolved plan/preference block"
        elif f"memory:{rid}" in keys:
            p = keys[f"memory:{rid}"]
            decision, why = "in", ("fit under budget at its tier" if not p.candidate.meta.get("gate") else
                                   f"fit under budget at its tier; gate {p.candidate.meta['gate']}, shipped {p.variant}")
        elif m["sensitivity"] == "high":
            decision, why = "withheld", "sensitive, gate closed, and no constraint line shipped (none on file, or dropped for budget)"
        else:
            decision, why = "dropped", "budget exhausted at its tier"
        rows.append({"id": rid, "type": m["type"], "score": m["score"], "decision": decision,
                     "why": f"{why}; rel {m['relevance']}, rec {m['recency']}, pinned {m['pinned']}"})
    trace.set("memory_read", rows)
    trace.set("context_assembly", {"budget": asm.budget, "used": asm.used, "headroom": asm.headroom,
                                   "decisions": asm.decisions, "gates": asm.gates, "note": asm.note,
                                   "note_tokens": asm.note_tokens, "mode": asm.mode,
                                   "excluded": asm.excluded, "scrubbed_dates": scrubbed})
    for ex in asm.excluded:
        trace.log("assemble", f"{ex['key']}: {ex['why']} ({ex['tokens']} tokens, not sent)")
    n_drop = sum(1 for d in asm.decisions if d["kept"] == "DROPPED")
    n_deg = sum(1 for d in asm.decisions
                if d["kept"] != "DROPPED" and d["kept"] != next(iter(d["variant_costs"])))
    trace.log("assemble", f"used {asm.used}/{budget} tokens; {n_drop} dropped, {n_deg} degraded")
    for g in asm.gates:
        trace.log("gate", f"sensitive record {g['id']}: gate {g['gate']}, shipped {g['shipped']}")

    # 5. Route
    n_loops = sum(1 for r in view.active() if r.type == "open_loop") if trigger and trigger.kind == "scheduled" else 0
    rt = route("" if trigger else sc.incoming.text, pre.flags, len(res.tensions), n_loops, trigger.kind if trigger else None)
    trace.log("route", f"{rt.tier} -> {rt.model}", reasons=rt.reasons)

    # 6. Call, post-check, one escalated retry, templated fallback
    stub = StubInputs(
        tensions=list(res.tensions), flags=list(pre.flags), risk_reasons=list(pre.risk_reasons),
        pinned=[ShippedPinned(p.candidate.meta["ids"][0], p.text[2:], bool(p.candidate.meta.get("sensitive")),
                              p.candidate.meta.get("gate") == "open" and p.variant == "full",
                              p.candidate.meta.get("constraint", ""), list(p.candidate.meta.get("forbid", [])),
                              p.candidate.meta.get("safe_note", ""))
                for k, p in keys.items() if k.startswith("pinned:")],
        notes=res.notes if "resolution_notes" in keys and keys["resolution_notes"].variant == "full" else [],
        open_loops=[r for r in view.active() if r.type == "open_loop" and f"memory:{r.id}" in keys],
        trigger=trigger, incoming=incoming_out, values=dict(res.values))
    recall_log: list[dict[str, Any]] = []
    base_executor = make_executor(view, domain, now, pre.flags, asm.placed_ids() | losers, recall_log,
                                  exclude_origins)
    recall_texts: list[str] = []      # what recall returned this turn: a source for the grounding check

    def executor(name: str, tool_input: dict[str, Any]) -> str:
        text = base_executor(name, tool_input)
        if can_scrub:                     # recall results leave the machine too: same scrub as the prompt
            text, _, c = scrub_text(text)
            for k, v in c.items():
                term_counts[k] = term_counts.get(k, 0) + v
        recall_texts.append(text)
        return text

    calls: list[dict[str, Any]] = []

    def attempt(r) -> ModelResult:
        before = len(recall_log)
        out = call_model(system, prompt, r, live=live, stub=stub, executor=executor, sender=b.sender,
                         preflight=b.preflight, stub_fn=domain.stub_reply)
        if can_scrub and out.mode == "stub":      # the stub only ever speaks from scrubbed inputs; scrub its words too
            out.text = scrub_text(out.text)[0]
        calls.append({"attempt": len(calls) + 1, "mode": out.mode, "model": out.model,
                      "latency_ms": out.latency_ms, "usage": out.usage, "stop": out.stop,
                      "error": out.error, "tool_calls": recall_log[before:]})
        return out

    must = [g for t in res.tensions for g in t.must_mention]
    if domain.post_requirements is not None:
        must += domain.post_requirements(pre, res)
    forbid = asm.forbid_terms()
    fb_reason = None
    post: PostCheck | None = None
    result = attempt(rt)
    if result.stop == "blocked":
        fb_reason = result.error or "blocked before sending"
        trace.set("send_blocked", {"reason": fb_reason})
    elif result.stop != "ok":
        fb_reason = "model refused the turn" if result.stop == "refusal" else f"API error: {result.error}"
    else:
        post = post_check(domain, result.text, pre, must, forbid, (system, prompt, *recall_texts))
        if not post.passed:
            trace.log("post_check", "failed; escalating one tier", failed=post.notes)
            rt = escalate(rt, f"post-check failed: {post.notes}")
            result = attempt(rt)
            if result.stop == "blocked":
                fb_reason = result.error or "blocked before sending"
                trace.set("send_blocked", {"reason": fb_reason})
            elif result.stop != "ok":
                fb_reason = "model refused the retry" if result.stop == "refusal" else f"API error: {result.error}"
            else:
                post = post_check(domain, result.text, pre, must, forbid, (system, prompt, *recall_texts))
                if not post.passed:
                    fb_reason = f"post-check failed after escalation retry: {post.notes}"
    if can_scrub:
        guard_passed = b.preflight is not None and not any(c["stop"] == "blocked" for c in calls)
        trace.set("egress", {"scrubbed_terms": sum(term_counts.values()), "by_category": dict(term_counts),
                             "guard": ("passed" if guard_passed else "blocked") if b.preflight is not None else "none configured"})
        trace.log("egress", ("guard passed" if guard_passed else f"guard blocked: {fb_reason}") if b.preflight is not None
                  else "no guard configured")
    trace.set("routing", {"tier": rt.tier, "model": rt.model, "effort": rt.effort, "reasons": rt.reasons})
    trace.set("model_call", {"calls": calls, "prompt_tokens": count_tokens(system) + count_tokens(prompt)})
    trace.set("safety_post", {"passed": post.passed, "checks": post.checks} if post else
              {"passed": False, "checks": {}})

    response = result.text
    if fb_reason:
        response = templated_fallback(sc.incoming.text, b.redact)
        trace.set("fallback", {"used": True, "reason": fb_reason, "original": result.text, "response": response})
        trace.log("safety", f"templated fallback used: {fb_reason}")

    # 7. Write path
    ops = _write_ops(domain, sc, res, store, response, now, trigger, res.tensions, pre.flags, b.redact)
    applied = store.apply(ops, now)
    trace.set("memory_write", applied)
    trace.log("memory", f"applied {len(applied)} ops")

    return TurnResult(response, system, prompt, trace, asm, rt, res, pre, post, applied,
                      fallback_used=bool(fb_reason), model_called=result.stop != "blocked", trigger=trigger,
                      store=store, mode=b.mode)
