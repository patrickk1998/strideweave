"""Non-Dolt evidence stores used where the store itself is not under test.

``record_report`` validates a report against the installed native build and
then hands one atomic statement sequence to whatever store it was given. The
double below is that boundary and nothing more, so the native construction,
binding, reconciliation, and rejection paths can be exercised in the unmarked,
sanitizer-instrumented suite without a Dolt runtime, while the marked suites
prove what a real Dolt store does with the statements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from strideweave.verification.store import EvidenceStore, SQLStatement


class RecordingEvidenceStore(EvidenceStore):
    """Capture the exact statements a caller asks the store to apply.

    Args:
        path: Optional location the store would occupy, reported unchanged so
            callers that print or compare it behave as they would for a real
            store.

    Examples:
        >>> store = RecordingEvidenceStore()
        >>> store.execute_transaction((SQLStatement("SELECT 1"),))
        >>> len(store.transactions)
        1
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path("in-memory-evidence")
        self.initialized = 0
        self.transactions: list[tuple[SQLStatement, ...]] = []

    def initialize(self) -> None:
        """Record that the caller asked to create or validate the store."""

        self.initialized += 1

    def execute_transaction(self, statements: Sequence[SQLStatement]) -> None:
        """Capture one atomic statement sequence without applying it."""

        self.transactions.append(tuple(statements))

    def query(self, statement: SQLStatement) -> tuple[Mapping[str, object], ...]:
        """Refuse reads, which no caller of this double is meant to perform."""

        raise AssertionError("this store double answers no queries")

    def tables(self) -> dict[str, int]:
        """Return how many rows each table received across every transaction.

        Returns:
            Mapping from table name to the number of insert statements applied.
        """

        counts: dict[str, int] = {}
        for transaction in self.transactions:
            for statement in transaction:
                _, _, remainder = statement.template.partition("INTO ")
                table = remainder.partition(" ")[0]
                if table:
                    counts[table] = counts.get(table, 0) + 1
        return counts
