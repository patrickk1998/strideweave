"""Canonical private identities shared across verification-store boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from ..model import VerificationReport
from .base import VerificationStoreError


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _thaw(value), allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _todo_provenance_digest(report: VerificationReport) -> str:
    """Return the outcome-independent digest used to match todo observations."""

    header = report.header
    if header is None:
        raise VerificationStoreError("verification report has no v3 header")
    return _digest(
        {
            "bundle_id": header.compilation_bundle.bundle_id,
            "oracle_profile": header.oracle_profile,
            "oracle_reference_ids": [
                value["oracle_reference_id"] for value in header.oracle_references
            ],
            "report_schema": report.schema_version,
            "selected_target_profile": header.selected_target_profile,
            "tolerance_policy_ids": [
                value["tolerance_policy_id"] for value in header.tolerance_policies
            ],
            "verification_spec_id": header.verification_spec["verification_spec_id"],
        }
    )
