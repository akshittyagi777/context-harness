"""Importer and Syncer protocols: the interface a future upload or sync UI calls.

A web handler can call `importer.import_path(path, data_dir)` or `syncer.sync(since, data_dir, credentials)`
without touching the CLI. Importers are idempotent: re-importing the same file adds nothing and a newer
export appends only new dates.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable


class NotConfigured(Exception):
    """A syncer cannot run: it says what it would need."""


@dataclass
class SourceInfo:
    name: str
    kind: str                      # wearable | ehr
    formats: list[str]
    description: str


@dataclass
class ImportResult:
    source: str
    added: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)      # informational lines (e.g. derived constraints)

    def add(self, kind: str, new: bool, n: int = 1) -> None:
        d = self.added if new else self.skipped
        d[kind] = d.get(kind, 0) + n

    @property
    def total_added(self) -> int:
        return sum(self.added.values())

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped.values())

    def summary(self) -> str:
        lines = [f"{self.source}: added {self.added or 0} | skipped {self.skipped or 0}"]
        lines += [f"  anomaly: {a}" for a in self.anomalies]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def ledger_entry(self, path: str | Path) -> dict[str, Any]:
        return {"ts": datetime.now().isoformat(timespec="seconds"), "source": self.source,
                "path": Path(path).name, "added": self.added, "skipped": self.skipped, "n_anomalies": len(self.anomalies),
                "files": [Path(f).name for f in self.files][:20]}


@runtime_checkable
class Importer(Protocol):
    info: SourceInfo

    def can_import(self, path: str | Path) -> bool: ...

    def import_path(self, path: str | Path, data_dir: Path, **kw: Any) -> ImportResult: ...


@runtime_checkable
class Syncer(Protocol):
    info: SourceInfo

    def requirements(self) -> list[str]: ...

    def sync(self, since: str | None, data_dir: Path, credentials: dict[str, Any] | None = None) -> ImportResult: ...


MAX_MEMBER_BYTES = 512 * 1024 * 1024     # one file inside an export zip, uncompressed
MAX_RATIO = 200                          # uncompressed / compressed; text exports compress ~10-30x
MAX_ARCHIVE_BYTES = 2 * 1024 ** 3        # every member's declared uncompressed size, summed
MAX_MEMBERS = 20_000                     # entries in one archive
_DRIVE = re.compile(r"^[A-Za-z]:")


class ExportRefused(ValueError):
    """An archive is not an export we will open: refused before anything in it is read."""


class ExportTooLarge(ExportRefused):
    """A zip, or one member of it, is too large or too compressed to be an export: refused before it is decompressed."""


def _unsafe_name(name: str) -> bool:
    """Absolute, drive-lettered or climbing out of the root. Nothing is extracted to disk, so this is defence in depth."""
    flat = name.replace("\\", "/")
    return flat.startswith("/") or bool(_DRIVE.match(flat)) or ".." in flat.split("/")


def check_archive(zf: zipfile.ZipFile) -> None:
    """Archive-level limits, from the central directory alone (nothing is decompressed). Raises ExportRefused/ExportTooLarge."""
    infos = zf.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ExportTooLarge(f"archive has {len(infos)} members, over the {MAX_MEMBERS} limit")
    for i in infos:
        if _unsafe_name(i.filename):
            raise ExportRefused(f"archive member name {i.filename[:80]!r} is absolute or contains '..'")
    total = sum(i.file_size for i in infos)
    if total > MAX_ARCHIVE_BYTES:
        raise ExportTooLarge(f"archive declares {total} bytes uncompressed in total, over the {MAX_ARCHIVE_BYTES} limit")


class Tree:
    """A directory or a zip, read through one interface. Names are posix paths relative to the root."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path) if self.path.is_file() and zipfile.is_zipfile(self.path) else None
        if self._zip is not None:
            try:
                check_archive(self._zip)
            except ExportRefused:
                self._zip.close()
                raise
        if self._zip is None and not self.path.is_dir():
            raise FileNotFoundError(f"{path} is neither a directory nor a zip file")

    def names(self) -> list[str]:
        if self._zip is not None:
            return sorted(n for n in self._zip.namelist() if not n.endswith("/"))
        return sorted(p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file())

    def read_bytes(self, name: str) -> bytes:
        if self._zip is not None:
            info = self._zip.getinfo(name)
            if info.file_size > MAX_MEMBER_BYTES or info.file_size > MAX_RATIO * max(info.compress_size, 1024):
                raise ExportTooLarge(f"{name}: {info.file_size} bytes from {info.compress_size} compressed; "
                                     "refused as too large or too compressed for an export")
            return self._zip.read(info)              # zipfile stops at the declared size and checks the CRC
        return (self.path / name).read_bytes()

    def read_text(self, name: str) -> str:
        return self.read_bytes(name).decode("utf-8-sig")

    def find(self, predicate) -> list[str]:
        return [n for n in self.names() if predicate(PurePosixPath(n))]

    def sha256(self) -> str:
        h = hashlib.sha256()
        for n in self.names():
            h.update(n.encode())
            h.update(self.read_bytes(n))
        return h.hexdigest()
