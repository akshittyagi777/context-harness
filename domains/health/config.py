"""Data directory, user configuration and the egress mode.

Data directory precedence: --data-dir flag, then HEALTH_HARNESS_DATA, then ~/.health-harness/.
It is created with mode 0700. Nothing under the repository ever holds real data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ENV_DATA_DIR = "HEALTH_HARNESS_DATA"
ENV_PASSPHRASE = "HEALTH_HARNESS_PASSPHRASE"
DEFAULT_DIR = Path("~/.health-harness")

DEFAULTS: dict[str, Any] = {
    "home_tz": "America/Los_Angeles",   # used only when an export carries no local timestamps
    "anchor_days": 90,                  # anchored baseline: the first N days of data from a source
    "anchor": {},                       # optional per-source override {"whoop": {"start": ..., "end": ...}}
    "rolling_days": 30,
    "sleep_disagreement_min": 45,
    "rhr_disagreement_bpm": 5,
}


def resolve_data_dir(arg: str | os.PathLike | None = None, create: bool = True) -> Path:
    raw = arg or os.environ.get(ENV_DATA_DIR) or DEFAULT_DIR
    path = Path(raw).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
    return path


def load_config(data_dir: Path) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    p = data_dir / "config.json"
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


def get_passphrase() -> str | None:
    return os.environ.get(ENV_PASSPHRASE) or None


def egress_mode(live: bool) -> str:
    """`remote` whenever a model API is involved; `local` for the stub (or a future on-device model)."""
    return "remote" if live else "local"
