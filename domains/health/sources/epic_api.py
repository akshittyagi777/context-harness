"""Epic FHIR R4 patient-access API (any health system on Epic): syncer skeleton. NOT implemented.

The real flow, as found on 2026-10-01 (health_formats.md section 3.2):
  1. Register a patient-facing app at fhir.epic.com (audience Patients, SMART standalone launch, read-only
     US Core / USCDI R4 APIs, redirect URI; a public client or private_key_jwt with a hosted JWKS).
  2. Endpoint discovery: Epic's bundle at https://open.epic.com/Endpoints/R4 lists each organisation's
     FHIR R4 base URL; read `.well-known/smart-configuration` for
     the authorization and token endpoints (it advertises code_challenge_methods S256 and
     launch-standalone / context-standalone-patient).
  3. Authorize with response_type=code, client_id, redirect_uri, state, scope, aud=<FHIR base>,
     code_challenge, code_challenge_method=S256. The patient signs in to MyChart and approves.
  4. POST the token endpoint (authorization_code, code_verifier). The response carries `patient` (the
     FHIR Patient id), `access_token`, and optionally `refresh_token`.
  5. GET {base}/Patient/{patient}, Condition?patient=, MedicationRequest?patient=,
     AllergyIntolerance?patient=, Observation?patient=&category=laboratory (Observation needs `category`
     or `code`), Observation?patient=&category=vital-signs. Each response is a searchset Bundle:
     exactly what `epic_fhir.py` imports.
Unconfirmed: the exact scope strings your organisation accepts, whether it approves third-party apps, whether
Epic extras (local OID codes) appear. Whatever a sync fetches goes through the same vault as an import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ImportResult, NotConfigured, SourceInfo

INFO = SourceInfo("epic-api", "ehr", ["Epic FHIR R4 (SMART standalone launch, PKCE S256)"], "Epic patient-access API (not implemented)")


class EpicSyncer:
    info = INFO

    def requirements(self) -> list[str]:
        return ["a patient-facing app registered at fhir.epic.com (SMART standalone launch, US Core read scopes)",
                "your organisation's FHIR R4 base URL, from Epic's endpoint directory https://open.epic.com/Endpoints/R4 "
                "(discover the OAuth endpoints from .well-known/smart-configuration)",
                "OAuth 2.0 authorization code with PKCE S256; Observation searches need `category` or `code`",
                "a client id approved by the health system (unconfirmed for your organisation)"]

    def sync(self, since: str | None, data_dir: Path, credentials: dict[str, Any] | None = None) -> ImportResult:
        raise NotConfigured("Epic sync is not implemented. It would need: " + "; ".join(self.requirements()))
