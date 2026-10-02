"""The encrypted EHR store and its access log.

EHR content lives in one Fernet-encrypted file under the data directory. The key is derived from a
passphrase (HEALTH_HARNESS_PASSPHRASE) with hashlib.scrypt and a random salt stored next to the data.
Every read of the store appends one JSON line to `access.log`: timestamp, purpose, resource types
touched. Never values. The log is append-only by convention (the process only ever opens it for "a").
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import ENV_PASSPHRASE, get_passphrase

# store key -> FHIR resource type name (what the access log records)
RESOURCE_TYPES = {
    "patient": "Patient", "conditions": "Condition", "medications": "MedicationRequest",
    "allergies": "AllergyIntolerance", "labs": "Observation", "vitals": "Observation",
}
PURPOSES = ("import", "derive", "local answer", "local assembly", "egress guard", "status")
SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)


class VaultError(Exception):
    pass


class VaultLocked(VaultError):
    """No passphrase available (or it is wrong)."""


class VaultUnavailable(VaultError):
    """The `cryptography` package is not installed."""


def _fernet_cls():
    try:
        from cryptography.fernet import Fernet, InvalidToken
    except ImportError as exc:  # the vault is a requirement, not a nicety: no fallback encryption
        raise VaultUnavailable("EHR import needs the `cryptography` package: pip install cryptography") from exc
    return Fernet, InvalidToken


def empty_store() -> dict[str, Any]:
    return {"version": 1, "patient": None, "conditions": [], "medications": [], "allergies": [],
            "labs": [], "vitals": [], "sources": []}


class Vault:
    FILE = "ehr.fernet"
    SALT = "ehr.salt"
    LOG = "access.log"

    def __init__(self, data_dir: Path, passphrase: str | None = None,
                 clock: Callable[[], datetime] | None = None):
        self.dir = Path(data_dir)
        self._pass = passphrase if passphrase is not None else get_passphrase()
        self._clock = clock or datetime.now
        self._key: bytes | None = None

    # ------------------------------------------------------------- state
    @property
    def path(self) -> Path:
        return self.dir / self.FILE

    @property
    def log_path(self) -> Path:
        return self.dir / self.LOG

    def exists(self) -> bool:
        return self.path.exists()

    def unlocked(self) -> bool:
        """True when a read would succeed: a passphrase is set and (if a store exists) it decrypts."""
        if not self._pass:
            return False
        if not self.exists():
            return True
        try:
            self._load()
            self.log_access("unlock check", [])        # decrypting is a read, so it is logged
            return True
        except VaultError:
            return False

    def require_passphrase(self) -> None:
        if not self._pass:
            raise VaultLocked(f"EHR content is only ever stored encrypted. Set {ENV_PASSPHRASE} to a passphrase "
                              f"and run again; without it the EHR import refuses to run.")
        _fernet_cls()

    # ------------------------------------------------------------- crypto
    def _derive(self, create: bool) -> bytes:
        if self._key is not None:
            return self._key
        salt_path = self.dir / self.SALT
        if salt_path.exists():
            salt = salt_path.read_bytes()
        elif create:
            salt = os.urandom(16)
            salt_path.write_bytes(salt)
            salt_path.chmod(0o600)
        else:
            raise VaultError("no salt on disk")
        raw = hashlib.scrypt(self._pass.encode("utf-8"), salt=salt, **SCRYPT)
        self._key = base64.urlsafe_b64encode(raw)
        return self._key

    def _load(self) -> dict[str, Any]:
        self.require_passphrase()
        Fernet, InvalidToken = _fernet_cls()
        if not self.exists():
            return empty_store()
        try:
            blob = Fernet(self._derive(create=False)).decrypt(self.path.read_bytes())
        except InvalidToken as exc:
            raise VaultLocked("wrong passphrase, or the store is corrupted") from exc
        return json.loads(blob)

    # ------------------------------------------------------------- access
    def log_access(self, purpose: str, types: Iterable[str]) -> None:
        entry = {"ts": self._clock().isoformat(timespec="seconds"), "purpose": purpose,
                 "resource_types": sorted(set(types))}
        new = not self.log_path.exists()
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        if new:
            self.log_path.chmod(0o600)

    def read(self, purpose: str, keys: Iterable[str] | None = None) -> dict[str, Any]:
        """Decrypt and return the requested keys of the store (all when `keys` is None), logging the
        access. Returns an empty store when nothing was imported yet (and logs nothing: no EHR was read)."""
        data = self._load()
        if not self.exists():
            return data
        wanted = list(RESOURCE_TYPES) if keys is None else list(keys)
        self.log_access(purpose, [RESOURCE_TYPES[k] for k in wanted if k in RESOURCE_TYPES])
        return {k: v for k, v in data.items() if k in wanted or k in ("version",)}

    def write(self, purpose: str, data: dict[str, Any]) -> None:
        self.require_passphrase()
        Fernet, _ = _fernet_cls()
        key = self._derive(create=True)
        self.path.write_bytes(Fernet(key).encrypt(json.dumps(data).encode("utf-8")))
        self.path.chmod(0o600)

    def update(self, purpose: str, fn: Callable[[dict[str, Any]], Any]) -> Any:
        """Read-modify-write under one logged access (all resource types are touched by an import)."""
        data = self._load()
        out = fn(data)
        self.write(purpose, data)
        self.log_access(purpose, RESOURCE_TYPES.values())
        return out

    def read_access_log(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [json.loads(l) for l in self.log_path.read_text().splitlines() if l.strip()]
