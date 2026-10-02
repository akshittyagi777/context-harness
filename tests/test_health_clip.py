"""Redact before clip: a denylisted word cut in half by a length limit must not survive as a fragment on disk."""
from __future__ import annotations

import re
import unittest

from tests import support
from tests.support import PASSPHRASE, data_copy, fixture

from harness.clip import clip
from harness.safety import templated_fallback
from harness.trace import Trace
from domains.health import runner
from domains.health.egress import Guard

TS = "2026-09-29T08:10:00"
TERM = "metoprolol"
# A change-of-plan message on a volatile day (crash language): the rules clip the user's own text into a plan record
# (160), an open loop (100) and a resolution (100); the trace's memory-write table clips each row's content to 90.
# No spaces, so nothing backs the clip off the word.
PREFIX = "cancel,wrecked,"


def _leaks(guard: Guard, data_dir, only: str | None = None) -> list[str]:
    """Every standalone token in traces, history and memory that is a prefix (4+ characters) of a denylist term (6+)."""
    dl = guard.denylist()
    terms = {t for phrases in dl.phrases.values() for t in phrases if len(t) >= 6} - set(dl.context_terms)
    prefixes = {t[:k] for t in terms for k in range(4, len(t) + 1)}
    rx = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(p) for p in sorted(prefixes, key=len, reverse=True)) + r")(?![a-z0-9])", re.I)
    found = []
    files = [*sorted((data_dir / "traces").glob("*")), data_dir / "history.json", data_dir / "memory.json"]
    for f in files:
        if f.exists() and (only is None or f.name == only):
            found += [f"{f.name}: {m.group(1)}" for m in rx.finditer(f.read_text())]
    return found


class ClipHelperTests(unittest.TestCase):
    def test_short_text_is_untouched_and_long_text_is_redacted_before_it_is_cut(self):
        red = lambda t: t.replace(TERM, "[redacted: medication]")
        self.assertEqual(clip("short metoprolol", 40, red), "short metoprolol")
        text = "x" * 12 + TERM + "!"
        self.assertNotIn("metop", clip(text, 18, red))

    def test_word_clip_backs_up_to_a_space(self):
        self.assertEqual(clip("alpha beta gamma", 12, word=True), "alpha beta")

    def test_fallback_redacts_before_clipping(self):
        red = lambda t: t.replace(TERM, "[redacted: medication]")
        for n in range(120, 141):
            msg = "x" * n + "," + TERM + " tonight and a few more words after it"
            self.assertNotIn("metop", templated_fallback(msg, red), n)

    def test_trace_redacts_every_value_before_it_clips(self):
        red = lambda t: t.replace(TERM, "[redacted: medication]")
        for n in range(140, 160):
            tr = Trace("t", redactor=red)
            tr.set("mode", {"kind": "reactive", "trigger": None, "why": "w", "memory_query": "x" * n + " " + TERM})
            tr.set("memory_write", [{"op": "add", "id": "r01", "type": "plan", "content": "x" * (n - 70) + TERM, "why": "w"}])
            tr.set("fallback", {"reason": "r", "original": "x" * (n + 40) + TERM, "response": "ok"})
            for out in (tr.to_markdown(), tr.to_json()):
                self.assertNotIn("metop", out.lower(), n)


class StraddleOnDiskTests(unittest.TestCase):
    guard = None

    def setUp(self):
        self.dd = data_copy()
        StraddleOnDiskTests.guard = StraddleOnDiskTests.guard or Guard(self.dd, PASSPHRASE)    # one scrypt for the class
        self.pristine = (self.dd / "memory.json").read_text()
        self.memory_leaks: list[str] = []

    def run_turn(self, message: str) -> None:
        (self.dd / "memory.json").write_text(self.pristine)       # a pending change stops the rules from making another
        runner.run(self.dd, message, TS, persist=True, guard=self.guard)
        self.memory_leaks += _leaks(self.guard, self.dd, only="memory.json")     # memory is reset every turn: scan each state

    def leaks(self) -> list[str]:
        return _leaks(self.guard, self.dd) + self.memory_leaks

    def test_the_guard_is_not_vacuous(self):
        self.run_turn(f"{PREFIX} {TERM} tonight")
        text = "".join(f.read_text() for f in (self.dd / "traces").glob("*.md"))
        self.assertIn("[redacted: medication]", text)
        self.assertEqual(self.leaks(), [])

    def test_the_leak_scan_does_catch_a_fragment(self):
        dd = data_copy()
        (dd / "traces").mkdir(exist_ok=True)
        (dd / "traces" / "t.md").write_text("memory query: `xxxx metopro`")
        self.assertEqual(_leaks(self.guard, dd), ["t.md: metopro"])

    def test_memory_query_straddling_the_160_clip(self):
        for n in range(140, 159):
            self.run_turn("x" * n + f" {TERM} tonight")
        self.assertEqual(self.leaks(), [])

    def test_memory_query_straddling_the_clip_with_no_space_before_the_word(self):
        for n in range(140, 159):
            self.run_turn("x" * n + f",{TERM} tonight")
        self.assertEqual(self.leaks(), [])

    def test_memory_write_rows_and_records_straddling_every_clip(self):
        for clip_at in (90, 100, 160):                 # trace row, open loop and resolution, plan record
            for n in range(clip_at - len(PREFIX) - 12, clip_at - len(PREFIX) + 2):
                self.run_turn(f"{PREFIX}" + "x" * n + f",{TERM} tonight")
        self.assertEqual(self.leaks(), [])


if __name__ == "__main__":
    unittest.main()
