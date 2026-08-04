"""Internal local persistence boundary for kernel verification evidence."""

from .base import EvidenceStore, SQLStatement, SQLValue, VerificationStoreError
from .dolt import DoltEvidenceStore, default_store_path
from .recording import RecordResult, record_report

__all__ = [
    "DoltEvidenceStore",
    "EvidenceStore",
    "RecordResult",
    "SQLStatement",
    "SQLValue",
    "VerificationStoreError",
    "default_store_path",
    "record_report",
]
