"""Pytest isolation for executable examples in the StrideWeave package."""

from collections.abc import Iterator

import pytest

from strideweave.carriers.move.ops import _MOVE_OPERATIONS


@pytest.fixture(autouse=True)
def _restore_move_operations() -> Iterator[None]:
    """Restore the process-global move registry after each source doctest."""
    snapshot = _MOVE_OPERATIONS.copy()
    try:
        yield
    finally:
        _MOVE_OPERATIONS.clear()
        _MOVE_OPERATIONS.update(snapshot)
