"""Clipping that cannot split a word the redactor would have caught.

Rule for the whole codebase: free text that is clipped and then written to disk or rendered must be redacted
first. Redacting a clipped string is useless: "metopro" is no longer a denylisted word. `clip` therefore takes
the redactor and applies it before it cuts. The core knows no domain; the redactor is whatever the boundary
supplies (`Boundary.redact`).
"""
from __future__ import annotations

from typing import Callable

Redactor = Callable[[str], str]


def clip(text: str, n: int, redact: Redactor | None = None, *, word: bool = False) -> str:
    """At most `n` characters of `text`. A text that fits is returned unchanged (nothing is cut, so nothing can
    straddle the cut; whole values are redacted by whoever stores them). A longer one is redacted first, then cut;
    a cut may land inside a `[redacted: ...]` placeholder, which is harmless (a placeholder holds no term). With
    `word=True` the cut backs up to the last space."""
    text = str(text)
    if len(text) <= n:
        return text
    if redact is not None:
        text = redact(text)
        if len(text) <= n:
            return text
    out = text[:n]
    if word and " " in out:
        out = out.rsplit(" ", 1)[0]
    return out
