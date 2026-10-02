"""Typed memory: a flat list of dated, sourced records about one user.

Records are never deleted. A newer claim supersedes an older one and the chain is
kept (`supersedes` / `superseded_by`). Two *active* records on one topic are a
conflict that the resolver settles at read time.

Read path   MemoryStore.read(query, now, synonyms) scores every active record
            and returns the ranking with its breakdown.
Write path  MemoryStore.apply(ops, now) takes explicit ops: add, supersede,
            reinforce. An add that overlaps an active record of the same type
            becomes a reinforce instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

TYPES = ("plan", "preference", "fact", "constraint", "pattern", "episode", "open_loop")
SOURCES = ("user_stated", "instrument", "document", "inferred")
# Where a record or candidate came from. `ehr` content must never reach a remote model call.
ORIGINS = ("ehr", "wearable", "user", "derived_local", "system")

TYPE_PRIORITY = {
    "constraint": 1.0, "plan": 0.9, "open_loop": 0.9, "preference": 0.8,
    "pattern": 0.7, "episode": 0.5, "fact": 0.3,
}

_STOP = set(
    "the a an and or to of in on at for with is are was were be been i my me it its this that "
    "again just thoughts about have had has do don't dont im i'm i'll it's you your will would "
    "can could get got from into then than so if but not no yes".split()
)


def tokens(text: str) -> set[str]:
    """Lowercased content words. Two-character tokens survive only if they carry a digit."""
    out = set()
    for w in re.findall(r"[a-z0-9']+", text.lower()):
        if w in _STOP:
            continue
        if len(w) > 2 or (len(w) == 2 and any(ch.isdigit() for ch in w)):
            out.add(w)
    return out


def expand(words: set[str], synonyms: dict[str, set[str]] | None = None) -> set[str]:
    out = set(words)
    for w in words:
        out |= (synonyms or {}).get(w, set())
    return out


def overlap_count(user_words: set[str], record_words: set[str],
                  synonyms: dict[str, set[str]] | None = None) -> int:
    """How many of the user's own words reach the record, each counted once even when
    its synonym expansion also lands in the record."""
    target = expand(record_words, synonyms)
    return sum(1 for w in user_words if expand({w}, synonyms) & target)


@dataclass
class MemoryRecord:
    id: str
    type: str
    content: str
    created_at: str
    updated_at: str
    source: str
    confidence: float
    status: str = "active"
    supersedes: str | None = None
    superseded_by: str | None = None
    topic: str = ""
    pinned: bool = False
    sensitivity: str = "normal"
    evidence_count: int = 1
    extra: dict[str, Any] = field(default_factory=dict)
    origin: str = "user"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def day(self) -> str:
        return self.updated_at[:10]


@dataclass
class MemoryOp:
    op: str                          # add | supersede | reinforce
    record: MemoryRecord | None      # the new record for add / supersede
    target_id: str | None            # the old record for supersede / reinforce
    why: str
    by: str | None = None            # id of an existing winner (read-time repair)


class MemoryStore:
    def __init__(self, records: list[MemoryRecord]):
        self.records = records

    @classmethod
    def load(cls, path: str | Path) -> "MemoryStore":
        return cls([MemoryRecord.from_dict(d) for d in json.loads(Path(path).read_text())])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([asdict(r) for r in self.records], indent=2, ensure_ascii=False))

    def get(self, rid: str) -> MemoryRecord | None:
        return next((r for r in self.records if r.id == rid), None)

    def active(self) -> list[MemoryRecord]:
        return [r for r in self.records if r.status == "active"]

    def next_id(self) -> str:
        n = max((int(r.id[1:]) for r in self.records if r.id[1:].isdigit()), default=0) + 1
        return f"r{n:02d}"

    # ---------------------------------------------------------------- read path
    def read(self, query: str, now: datetime,
             synonyms: dict[str, set[str]] | None = None) -> list[dict[str, Any]]:
        """Rank active records: 0.45 relevance + 0.25 recency + 0.30 type priority,
        pinned floors at 0.75, then scaled by 0.6 + 0.4 * confidence."""
        q = expand(tokens(query), synonyms)
        ranked = []
        for r in self.active():
            hits = len(q & expand(tokens(r.content + " " + r.topic), synonyms))
            relevance = min(1.0, hits / 3.0)
            age_days = max(0.0, (now - datetime.fromisoformat(r.updated_at)).total_seconds() / 86400)
            recency = 1.0 / (1.0 + age_days / 7.0)
            prio = TYPE_PRIORITY.get(r.type, 0.4)
            score = 0.45 * relevance + 0.25 * recency + 0.30 * prio
            if r.pinned:
                score = max(score, 0.75)
            score *= 0.6 + 0.4 * r.confidence
            ranked.append({
                "id": r.id, "type": r.type, "score": round(score, 3),
                "relevance": round(relevance, 2), "recency": round(recency, 2),
                "type_priority": prio, "pinned": r.pinned, "confidence": r.confidence,
                "topic": r.topic, "sensitivity": r.sensitivity, "record": r,
            })
        ranked.sort(key=lambda d: -d["score"])
        return ranked

    def conflicts(self) -> dict[str, list[MemoryRecord]]:
        """Topics with more than one active record."""
        by_topic: dict[str, list[MemoryRecord]] = {}
        for r in self.active():
            by_topic.setdefault(r.topic, []).append(r)
        return {t: rs for t, rs in by_topic.items() if len(rs) > 1}

    # --------------------------------------------------------------- write path
    def apply(self, ops: list[MemoryOp], now: datetime) -> list[dict[str, Any]]:
        applied: list[dict[str, Any]] = []
        for op in ops:
            if op.op == "supersede":
                old = self.get(op.target_id or "")
                if old is None or old.status != "active":
                    applied.append({"op": "skip", "id": op.target_id, "type": "", "content": "",
                                    "why": "supersede target missing or already superseded"})
                    continue
                old.status = "superseded"
                new = op.record or (self.get(op.by) if op.by else None)
                if op.record is not None and self.get(op.record.id) is None:
                    self.records.append(op.record)
                if new is not None:
                    old.superseded_by = new.id
                    if new.supersedes is None:
                        new.supersedes = old.id
                applied.append({"op": "supersede", "id": old.id, "type": old.type,
                                "content": old.content, "why": op.why,
                                "by": new.id if new else None})
            elif op.op == "add":
                assert op.record is not None
                dup = self._find_duplicate(op.record)
                if dup is not None:
                    dup.evidence_count += 1
                    dup.updated_at = now.isoformat()
                    applied.append({"op": "dedupe->reinforce", "id": dup.id, "type": dup.type,
                                    "content": dup.content,
                                    "why": f"overlaps {dup.id}; {op.why}"})
                    continue
                self.records.append(op.record)
                applied.append({"op": "add", "id": op.record.id, "type": op.record.type,
                                "content": op.record.content, "why": op.why})
            elif op.op == "reinforce":
                r = self.get(op.target_id or "")
                if r is None:
                    applied.append({"op": "skip", "id": op.target_id, "type": "", "content": "",
                                    "why": "reinforce target missing"})
                    continue
                r.evidence_count += 1
                r.confidence = round(min(0.95, r.confidence + 0.1), 2)
                r.updated_at = now.isoformat()
                applied.append({"op": "reinforce", "id": r.id, "type": r.type, "content": r.content,
                                "why": f"{op.why} (confidence {r.confidence}, evidence {r.evidence_count})"})
        return applied

    def _find_duplicate(self, rec: MemoryRecord) -> MemoryRecord | None:
        a = tokens(rec.content)
        for r in self.active():
            if r.type != rec.type:
                continue
            b = tokens(r.content)
            if a and b and len(a & b) / len(a | b) > 0.6:
                return r
        return None
