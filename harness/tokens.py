"""Token estimate used for per-candidate planning.

Seam: a production build would call a real token counter on the final prompt and
keep this heuristic only for planning, since the budget decision is made before the
prompt exists. The heuristic errs high: prose is about 3.8 characters per token,
while digits, punctuation and table glue tokenize at roughly 1.6 characters per token.
"""
from __future__ import annotations

import math
import re


def count_tokens(text: str) -> int:
    if not text:
        return 0
    dense = len(re.findall(r"[\d:.\-|/]", text))
    prose = len(text) - dense
    return max(1, math.ceil(prose / 3.8 + dense / 1.6))
