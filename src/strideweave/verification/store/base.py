"""Backend-neutral verification-store primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

SQLValue: TypeAlias = str | int | float | bool | bytes | datetime | None


class VerificationStoreError(RuntimeError):
    """Report an unavailable, incompatible, or inconsistent verification store."""


@dataclass(frozen=True, slots=True)
class SQLStatement:
    """One internal SQL template plus values encoded by the store adapter.

    Templates are repository-owned SQL and use one ``?`` for each value. User
    data is always supplied through ``parameters`` rather than interpolated into
    the template by a caller.
    """

    template: str
    parameters: tuple[SQLValue, ...] = ()


class EvidenceStore(ABC):
    """Backend-neutral interface for the local verification evidence store."""

    @abstractmethod
    def initialize(self) -> None:
        """Create or validate the store and apply checked migrations."""

    @abstractmethod
    def execute_transaction(self, statements: Sequence[SQLStatement]) -> None:
        """Execute all statements atomically or leave the store unchanged."""

    @abstractmethod
    def query(self, statement: SQLStatement) -> tuple[Mapping[str, object], ...]:
        """Return immutable row mappings for one read-only statement."""
