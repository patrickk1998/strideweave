"""Ergonomic tensor factories over the CPU carrier.

Every factory allocates fresh ``CPU`` storage, builds a compact column-major
layout (unless one is supplied), and fills it through the layout, replacing
the alloc-loop-layout boilerplate otherwise needed for each tensor. Other
carriers are still constructed directly from primitives.
"""

from __future__ import annotations

import random
from typing import Any

from ..carriers import CPU
from ..carriers.dtype import DType
from ..layout import Layout
from ..tensor import Tensor
from .layouts import column_major


def _infer_extents(values: Any) -> list[int]:
    extents: list[int] = []
    level = values
    while isinstance(level, (list, tuple)):
        if len(level) == 0:
            raise ValueError("values must not contain empty levels")
        extents.append(len(level))
        level = level[0]
    if not extents:
        raise ValueError("values must be a non-empty list or tuple")
    return extents


def _flatten_row_major(values: Any, extents: list[int], depth: int = 0) -> list[Any]:
    if depth == len(extents):
        return [values]
    if not isinstance(values, (list, tuple)) or len(values) != extents[depth]:
        raise ValueError("values must be rectangular across every level")
    flattened: list[Any] = []
    for element in values:
        flattened.extend(_flatten_row_major(element, extents, depth + 1))
    return flattened


def _new_cpu_tensor(layout: Layout, dtype: DType) -> Tensor:
    return Tensor(CPU(layout.cosize, dtype=dtype), 0, layout)


def _fill_logical(tensor: Tensor, values: list[Any]) -> None:
    layout = tensor.layout
    for i, value in enumerate(values):
        tensor.carrier[tensor.offset + layout.index(i)] = value


def tensor(
    values: Any,
    *,
    layout: Layout | None = None,
    dtype: DType = DType.Float32,
) -> Tensor:
    """Create a CPU tensor from (possibly nested) Python values.

    Nested lists infer one flat mode per nesting level, with the outermost
    list as the first mode — ``[[1, 2], [3, 4]]`` becomes a ``[2, 2]`` tensor
    with ``result[i, j]`` equal to ``values[i][j]`` — stored in a compact
    column-major layout. When an explicit ``layout`` is given, ``values``
    must be a flat list in logical index order for that layout.

    Args:
        values: Non-empty rectangular nested lists or tuples of numbers, or a
            flat list when ``layout`` is provided.
        layout: Optional layout overriding extent inference.
        dtype: CPU carrier value type, ``Float32`` or ``Int32``.

    Returns:
        Tensor over fresh CPU storage holding the values.

    Examples:
        >>> from sw.friendly import tensor
        >>> t = tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> t[1, 0]
        3.0
    """

    if layout is not None:
        values = list(values)
        if len(values) != layout.size:
            raise ValueError("values length must equal the layout logical size")
        result = _new_cpu_tensor(layout, dtype)
        _fill_logical(result, values)
        return result

    extents = _infer_extents(values)
    flattened = _flatten_row_major(values, extents)
    result = _new_cpu_tensor(column_major(*extents), dtype)
    layout = result.layout
    # Flattened values are in row-major (outermost-first) order; write each
    # through its logical coordinate rather than the logical index, which
    # enumerates column-major.
    coordinate = [0] * len(extents)
    for value in flattened:
        result.carrier[result.offset + layout.index(tuple(coordinate))] = value
        for axis in range(len(extents) - 1, -1, -1):
            coordinate[axis] += 1
            if coordinate[axis] < extents[axis]:
                break
            coordinate[axis] = 0
    return result


def full(*extents: int, value: float, dtype: DType = DType.Float32) -> Tensor:
    """Create a CPU tensor with every logical element set to one value.

    Args:
        extents: One or more positive mode extents.
        value: Value assigned to every logical element.
        dtype: CPU carrier value type, ``Float32`` or ``Int32``.

    Returns:
        Column-major tensor over fresh CPU storage filled with ``value``.

    Examples:
        >>> from sw.friendly import full
        >>> full(2, 2, value=7.0)[1, 1]
        7.0
    """

    result = _new_cpu_tensor(column_major(*extents), dtype)
    _fill_logical(result, [value] * result.layout.size)
    return result


def zeros(*extents: int, dtype: DType = DType.Float32) -> Tensor:
    """Create a zero-filled column-major CPU tensor.

    Args:
        extents: One or more positive mode extents.
        dtype: CPU carrier value type, ``Float32`` or ``Int32``.

    Returns:
        Column-major tensor over fresh zero-initialized CPU storage.

    Examples:
        >>> from sw.friendly import zeros
        >>> zeros(2, 3)[1, 2]
        0.0
    """

    return _new_cpu_tensor(column_major(*extents), dtype)


def ones(*extents: int, dtype: DType = DType.Float32) -> Tensor:
    """Create a one-filled column-major CPU tensor.

    Args:
        extents: One or more positive mode extents.
        dtype: CPU carrier value type, ``Float32`` or ``Int32``.

    Returns:
        Column-major tensor over fresh CPU storage filled with ones.

    Examples:
        >>> from sw.friendly import ones
        >>> ones(3)[2]
        1.0
    """

    value = 1 if dtype is DType.Int32 else 1.0
    return full(*extents, value=value, dtype=dtype)


def arange(count: int, *, dtype: DType = DType.Float32) -> Tensor:
    """Create a one-mode CPU tensor holding ``0, 1, ..., count - 1``.

    Args:
        count: Positive number of elements.
        dtype: CPU carrier value type, ``Float32`` or ``Int32``.

    Returns:
        One-mode tensor with ascending values.

    Examples:
        >>> from sw.friendly import arange
        >>> arange(4)[3]
        3.0
    """

    result = _new_cpu_tensor(column_major(count), dtype)
    values = range(count) if dtype is DType.Int32 else [float(i) for i in range(count)]
    _fill_logical(result, list(values))
    return result


def rand(*extents: int, rng: random.Random | None = None) -> Tensor:
    """Create a CPU Float32 tensor of uniform samples from ``[0, 1)``.

    Args:
        extents: One or more positive mode extents.
        rng: Random generator supplying the samples; pass a seeded
            ``random.Random`` for reproducibility.

    Returns:
        Column-major tensor of independent uniform samples.

    Examples:
        >>> import random
        >>> from sw.friendly import rand
        >>> 0.0 <= rand(2, 2, rng=random.Random(0))[0, 0] < 1.0
        True
    """

    rng = rng if rng is not None else random.Random()
    result = _new_cpu_tensor(column_major(*extents), DType.Float32)
    _fill_logical(result, [rng.random() for _ in range(result.layout.size)])
    return result


def randn(*extents: int, rng: random.Random | None = None) -> Tensor:
    """Create a CPU Float32 tensor of standard normal samples.

    Args:
        extents: One or more positive mode extents.
        rng: Random generator supplying the samples; pass a seeded
            ``random.Random`` for reproducibility.

    Returns:
        Column-major tensor of independent standard normal samples.

    Examples:
        >>> import random
        >>> from sw.friendly import randn
        >>> isinstance(randn(3, rng=random.Random(0))[0], float)
        True
    """

    rng = rng if rng is not None else random.Random()
    result = _new_cpu_tensor(column_major(*extents), DType.Float32)
    _fill_logical(result, [rng.gauss(0.0, 1.0) for _ in range(result.layout.size)])
    return result
