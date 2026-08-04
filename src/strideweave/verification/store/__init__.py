"""Internal local persistence boundary for kernel verification evidence."""

from .base import EvidenceStore, SQLStatement, SQLValue, VerificationStoreError
from .dolt import DoltEvidenceStore, default_store_path
from .publication import (
    PublishResult,
    RefreshResult,
    publish_evidence,
    refresh_evidence,
)
from .querying import (
    IdentityDifference,
    MissingRequirement,
    RunStaleness,
    StatusObservation,
    query_stale,
    query_status,
    query_todo,
)
from .recording import RecordResult, record_report

__all__ = [
    "DoltEvidenceStore",
    "EvidenceStore",
    "IdentityDifference",
    "MissingRequirement",
    "PublishResult",
    "RecordResult",
    "RefreshResult",
    "RunStaleness",
    "SQLStatement",
    "SQLValue",
    "StatusObservation",
    "VerificationStoreError",
    "default_store_path",
    "publish_evidence",
    "query_stale",
    "query_status",
    "query_todo",
    "record_report",
    "refresh_evidence",
]
