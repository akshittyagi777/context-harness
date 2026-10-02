"""Shared fixtures: the mock exports are generated and imported once per test process."""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASSPHRASE = "test-passphrase-not-secret"
os.environ["HEALTH_HARNESS_PASSPHRASE"] = PASSPHRASE
os.environ.pop("HEALTH_HARNESS_DATA", None)

from harness.memory import MemoryStore  # noqa: E402
from domains.health.sources import IMPORTERS  # noqa: E402
from domains.health.mock.generate import generate  # noqa: E402

_state: dict = {}


def _tmp() -> Path:
    d = Path(tempfile.mkdtemp(prefix="health-harness-test-"))
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    return d


def fixture() -> dict:
    """{'root', 'mock', 'data', 'truth'}: mock exports plus a data dir with every source imported."""
    if not _state:
        root = _tmp()
        truth = generate(root / "mock")
        data = root / "data"
        data.mkdir(mode=0o700)
        IMPORTERS["whoop"].import_path(root / "mock" / "whoop", data)
        IMPORTERS["garmin"].import_path(root / "mock" / "garmin", data)
        IMPORTERS["epic"].import_path(root / "mock" / "epic", data)
        store = MemoryStore.load(root / "mock" / "memory_seed.json")
        mem = MemoryStore.load(data / "memory.json")
        mem.records.extend(store.records)
        mem.save(data / "memory.json")
        _state.update(root=root, mock=root / "mock", data=data, truth=truth)
    return _state


def data_copy() -> Path:
    """A private copy of the imported data dir, for tests that write."""
    src = fixture()["data"]
    dst = _tmp() / "data"
    shutil.copytree(src, dst)
    dst.chmod(0o700)
    return dst


def empty_data() -> Path:
    d = _tmp() / "data"
    d.mkdir(mode=0o700)
    return d
