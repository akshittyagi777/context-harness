"""Trace: the reasoning of a turn, not just its result.

Stages write structured sections plus an event log. Nothing in the pipeline makes a
keep, drop, degrade or route decision without a reason landing here. Rendered as JSON
for tooling and Markdown for people.

Redaction. A trace may be given a `redactor` (the boundary's, so the core stays domain-free). Every free-text value
is redacted before it is rendered, and so before any of it is clipped to a table cell: clipping first could cut a
denylisted word in half and leave a fragment no later pass would recognise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .clip import clip


@dataclass
class Trace:
    turn_id: str
    sections: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    redactor: Callable[[str], str] | None = None

    def set(self, key: str, value: Any) -> None:
        self.sections[key] = value

    def log(self, stage: str, msg: str, **data: Any) -> None:
        self.events.append({"stage": stage, "msg": msg, **data})

    def _clean(self, value: Any) -> Any:
        """A JSON-normalised copy of `value` with the redactor applied to every string (dict keys are structure,
        not text). Without a redactor it is the value itself."""
        if self.redactor is None:
            return value
        red = self.redactor

        def walk(v: Any) -> Any:
            if isinstance(v, str):
                return red(v)
            if isinstance(v, dict):
                return {k: walk(x) for k, x in v.items()}
            if isinstance(v, list):
                return [walk(x) for x in v]
            return v
        return walk(json.loads(json.dumps(value, default=str)))

    def to_json(self) -> str:
        return json.dumps(self._clean({"turn_id": self.turn_id, **self.sections, "events": self.events}),
                          indent=2, default=str, ensure_ascii=False)

    def to_markdown(self) -> str:
        s = self._clean(self.sections)
        events = self._clean(self.events)
        o: list[str] = [f"# Trace for turn `{self.turn_id}`", ""]
        mode = s.get("mode", {})
        o.append(f"mode: **{mode.get('kind')}** | trigger: {mode.get('trigger')} | {mode.get('why')}")
        o.append(f"memory query: `{clip(mode.get('memory_query', ''), 160)}`")
        o.append("")

        pre = s.get("safety_pre", {})
        o += ["## 1. Safety pre-check", "", f"- level: **{pre.get('level')}**  flags: {pre.get('flags')}"]
        o += [f"  - {r}" for r in pre.get("reasons", [])]
        o.append("")

        o += ["## 2. Resolutions", ""]
        for r in s.get("resolutions", []):
            o += [f"- **{r['topic']}**: {r['outcome']}", f"  - rule: {r['rule']}", f"  - evidence: {r['evidence']}"]
        for t in s.get("tensions", []):
            o.append(f"- tension `{t['key']}`: {t['why']}")
        o.append("")

        o += ["## 3. Memory read", "", "| id | type | score | decision | why |", "|---|---|---|---|---|"]
        for m in s.get("memory_read", []):
            o.append(f"| {m['id']} | {m['type']} | {m['score']:.2f} | {m['decision']} | {m['why']} |")
        o.append("")

        ca = s.get("context_assembly")
        o += ["## 4. Context assembly", ""]
        if ca is None:
            o.append("skipped: emergency path")
        else:
            o.append(f"budget **{ca['budget']}** tokens | used **{ca['used']}** | headroom **{ca['headroom']}**"
                     + (f" | missing-context line: {ca['note_tokens']} tokens" if ca.get("note") else ""))
            o += ["", "| tier | candidate | kept | tokens | variant costs | why |", "|---|---|---|---|---|---|"]
            for d in ca["decisions"]:
                alts = ", ".join(f"{k}={v}" for k, v in d["variant_costs"].items())
                o.append(f"| {d['tier']} | {d['key']} | **{d['kept']}** | {d['tokens']} | {alts} | {d['why']} |")
            for ex in ca.get("excluded", []):
                o.append(f"\n- `{ex['key']}`: {ex['why']} ({ex['tokens']} tokens, not sent)")
            if ca.get("scrubbed_dates"):
                o.append(f"\n- {ca['scrubbed_dates']} date(s) coarsened to month precision (remote mode)")
            for g in ca.get("gates", []):
                o.append(f"\n- sensitive gate for {g['id']}: **{g['gate']}**, shipped `{g['shipped']}`")
            if ca.get("note"):
                o.append(f"\n- note added to prompt: {ca['note']}")
        o.append("")

        r = s.get("routing", {})
        o += ["## 5. Routing", "", f"- tier **{r.get('tier')}**, model **{r.get('model')}**, effort {r.get('effort')}"]
        o += [f"  - {why}" for why in r.get("reasons", [])]
        o.append("")

        mc = s.get("model_call", {})
        o += ["## 6. Model call", ""]
        if mc.get("skipped"):
            o.append(f"- {mc['skipped']}")
        for c in mc.get("calls", []):
            line = (f"- attempt {c['attempt']}: mode {c['mode']}, model {c['model']}, "
                    f"latency {c['latency_ms']} ms, stop {c['stop']}")
            if c.get("error"):
                line += f", error: {c['error']}"
            o.append(line)
            if c.get("usage"):
                o.append(f"  - usage: {c['usage']}")
            for tc in c.get("tool_calls", []):
                o.append(f"  - recall: query={tc['query']!r} ids_requested={tc['ids_requested']} "
                         f"ids_returned={tc['ids_returned']}")
        if mc.get("prompt_tokens") is not None:
            o.append(f"- prompt tokens (estimate): {mc['prompt_tokens']}")
        o.append("")

        post = s.get("safety_post")
        o += ["## 7. Post-check", ""]
        o.append(f"- passed: **{post['passed']}**  checks: {post['checks']}" if post else "- not run")
        o.append("")

        lr = s.get("local_response")
        if lr:
            o += ["## Local response", "", f"- {lr['note']} (kind {lr['kind']}, flag {lr.get('flag')})", ""]
        sb = s.get("send_blocked")
        if sb:
            o += ["## Send blocked", "", f"- {sb['reason']}", ""]

        fb = s.get("fallback")
        if fb:
            o += ["## Fallback", "", f"- reason: {fb['reason']}", f"- original: {clip(fb.get('original', ''), 200)}",
                  f"- shipped: {fb['response']}", ""]

        esc = s.get("escalation")
        if esc:
            o += ["## Escalation", "", "```json", json.dumps(esc, indent=2, default=str, ensure_ascii=False),
                  "```", ""]

        o += ["## 8. Memory writes", "", "| op | id | type | content | why |", "|---|---|---|---|---|"]
        for w in s.get("memory_write", []):
            o.append(f"| {w['op']} | {w['id']} | {w.get('type', '')} | {clip(w.get('content', ''), 90)} | {w['why']} |")
        o += ["", "## Event log", ""]
        for e in events:
            extra = {k: v for k, v in e.items() if k not in ("stage", "msg")}
            o.append(f"- `{e['stage']}` {e['msg']}" + (f" `{json.dumps(extra, default=str, ensure_ascii=False)}`" if extra else ""))
        return "\n".join(o) + "\n"
