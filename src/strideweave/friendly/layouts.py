"""Compact layout builders for flat multi-mode tensors.

These helpers construct contiguous ``Layout`` objects from mode extents so
callers do not have to hand-compute stride patterns. They cover the common
flat (non-hierarchical) case; hierarchical layouts are still built directly
from ``Shape`` and ``Stride`` trees.
"""

from __future__ import annotations

from ..layout import Layout, Shape, Stride


def _require_extents(extents: tuple[int, ...]) -> None:
    if not extents:
        raise ValueError("at least one extent is required")
    for extent in extents:
        if not isinstance(extent, int) or extent <= 0:
            raise ValueError("extents must be positive integers")


def column_major(*extents: int) -> Layout:
    """Build a compact column-major layout from flat mode extents.

    The first mode is fastest-varying: strides are the exclusive running
    product of the extents, e.g. extents ``(2, 3, 4)`` give strides
    ``(1, 2, 6)``. This matches the convention used throughout the strideweave
    test suite and ``strideweave.nn``.

    Args:
        extents: One or more positive mode extents.

    Returns:
        Contiguous column-major ``Layout`` over the extents.

    Examples:
        >>> from strideweave import Layout, Shape, Stride
        >>> from sw.friendly import column_major
        >>> column_major(2, 3) == Layout(Shape([2, 3]), Stride([1, 2]))
        True
    """

    _require_extents(extents)
    strides = []
    running = 1
    for extent in extents:
        strides.append(running)
        running *= extent
    return Layout(Shape(list(extents)), Stride(strides))


def row_major(*extents: int) -> Layout:
    """Build a compact row-major layout from flat mode extents.

    The last mode is fastest-varying: strides are the exclusive running
    product of the extents from the right, e.g. extents ``(2, 3, 4)`` give
    strides ``(12, 4, 1)``.

    Args:
        extents: One or more positive mode extents.

    Returns:
        Contiguous row-major ``Layout`` over the extents.

    Examples:
        >>> from strideweave import Layout, Shape, Stride
        >>> from sw.friendly import row_major
        >>> row_major(2, 3) == Layout(Shape([2, 3]), Stride([3, 1]))
        True
    """

    _require_extents(extents)
    strides = [1] * len(extents)
    running = 1
    for position in range(len(extents) - 1, -1, -1):
        strides[position] = running
        running *= extents[position]
    return Layout(Shape(list(extents)), Stride(strides))
