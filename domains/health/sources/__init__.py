"""Registry of importers and syncers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ExportRefused, ExportTooLarge, ImportResult, Importer, NotConfigured, SourceInfo, Syncer
from .epic_api import EpicSyncer
from .epic_ccda import EpicCcdaImporter
from .epic_fhir import EpicFhirImporter
from .garmin_api import GarminSyncer
from .garmin_export import GarminExportImporter
from .whoop_api import WhoopSyncer
from .whoop_export import WhoopExportImporter


class EpicImporter:
    """Epic: FHIR searchset bundles are primary; a C-CDA CCD (XDM zip, directory, or .xml) is secondary."""
    info = SourceInfo("epic", "ehr", ["FHIR R4 Bundle JSON (files or directory)", "C-CDA CCD XML"], "Epic EHR, local-only")
    fhir, ccda = EpicFhirImporter(), EpicCcdaImporter()

    def can_import(self, path: str | Path) -> bool:
        return self.fhir.can_import(path) or self.ccda.can_import(path)

    def import_path(self, path: str | Path, data_dir: Path, **kw: Any) -> ImportResult:
        """Import, then refresh the derived constraints locally (the only thing a remote model ever learns of the EHR)."""
        res = (self.fhir if self.fhir.can_import(path) else self.ccda).import_path(path, data_dir, **kw)
        from ..ehr.derive import run_derive
        from ..vault import Vault
        derived = run_derive(Path(data_dir), Vault(Path(data_dir), kw.get("passphrase")))
        res.notes.append("derived constraints (local, template ids): " + (", ".join(derived["templates"]) or "none"))
        return res


IMPORTERS: dict[str, Importer] = {"whoop": WhoopExportImporter(), "garmin": GarminExportImporter(), "epic": EpicImporter()}
SYNCERS: dict[str, Syncer] = {"whoop": WhoopSyncer(), "garmin": GarminSyncer(), "epic": EpicSyncer()}

__all__ = ["IMPORTERS", "SYNCERS", "ExportRefused", "ExportTooLarge", "ImportResult", "NotConfigured", "SourceInfo", "EpicImporter"]
