"""Backend-neutral verification-store primitives."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, TypeAlias

SQLValue: TypeAlias = str | int | float | bool | bytes | datetime | None

DIAGNOSTIC_CLOSURE_INPUT_KINDS: Final = frozenset(
    {"build_input", "generated_header", "header", "source"}
)


def is_diagnostic_closure_input(input_kind: object) -> bool:
    """Report whether one closure member is stored as an individual row.

    Closure identity is always complete: the content-addressed ``closure_id``
    and the stored closure descriptor cover every transitive input, including
    the external SDK, standard-library, and compiler headers. Changing any of
    them changes the closure, so evidence stays fail-closed either way.

    Normalized member rows are deliberately narrower. They exist to name
    *which* input changed, and only project- and build-owned inputs are
    actionable that way, while external headers are both unactionable and the
    large majority of members. Recording, publication, and staleness therefore
    share this one rule, so a store never contains, exports, or compares a
    member row that another path would have skipped.

    Args:
        input_kind: Closure input kind recorded by the provenance provider.

    Returns:
        ``True`` for project- and build-owned members, ``False`` otherwise.

    Examples:
        >>> is_diagnostic_closure_input("header")
        True
        >>> is_diagnostic_closure_input("external_header")
        False
    """

    return input_kind in DIAGNOSTIC_CLOSURE_INPUT_KINDS


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


def sql_literal(value: SQLValue) -> str:
    """Return the canonical fully rendered form of one statement value.

    Adapters send values as bound parameters rather than statement text. This
    rendering is the store's shared size model: callers that must keep a
    generated statement inside a byte budget measure it against this form,
    which is never smaller than what the wire protocol carries.
    """

    if value is None:
        return "NULL"
    if type(value) is bool:
        return "TRUE" if value else "FALSE"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("verification-store SQL values must be finite")
        return repr(value)
    if type(value) is datetime:
        if value.tzinfo is None:
            raise ValueError("verification-store timestamps must include a timezone")
        normalized = value.astimezone(timezone.utc).replace(tzinfo=None)
        value = normalized.isoformat(timespec="microseconds")
    if type(value) is str:
        encoded = value.encode("utf-8").hex()
        return f"CONVERT(0x{encoded} USING utf8mb4)"
    if type(value) is bytes:
        return f"0x{value.hex()}"
    raise TypeError(f"unsupported verification-store SQL value {type(value).__name__}")


def render_statement(statement: SQLStatement) -> str:
    """Return one statement with every value rendered into its text.

    Args:
        statement: Statement whose placeholder count must match its values.

    Returns:
        The rendered statement used as the store's byte-budget size model.
    """

    pieces = statement.template.split("?")
    if len(pieces) - 1 != len(statement.parameters):
        raise ValueError(
            "verification-store SQL placeholder count does not match values"
        )
    rendered = [pieces[0]]
    for value, suffix in zip(statement.parameters, pieces[1:], strict=True):
        rendered.extend((sql_literal(value), suffix))
    return "".join(rendered)


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
