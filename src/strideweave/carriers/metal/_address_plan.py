"""Immutable logical-to-physical address plans for Metal kernels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from operator import index as operator_index
from typing import Literal

from ...layout import Layout, Shape, Stride

type AddressTree = (
    tuple[Literal["leaf"], int, int] | tuple[Literal["node"], tuple[AddressTree, ...]]
)
type AddressPlanKey = tuple[
    Literal["strideweave.metal.address-plan.v1"], int, AddressTree
]


@dataclass(frozen=True, slots=True)
class AddressPlan:
    """Frozen address map and exact structural cache identity for one layout."""

    key: AddressPlanKey
    offset: int
    logical_size: int
    cosize: int
    addresses: tuple[int, ...]
    is_injective: bool


def _snapshot_tree(
    shape_level: Sequence[object], stride_level: Sequence[object]
) -> AddressTree:
    children: list[AddressTree] = []
    for extent, stride in zip(shape_level, stride_level, strict=True):
        if isinstance(extent, int):
            if not isinstance(stride, int):
                raise ValueError("Shape and stride hierarchy do not match")
            children.append(("leaf", extent, stride))
            continue
        if isinstance(stride, int):
            raise ValueError("Shape and stride hierarchy do not match")
        if not isinstance(extent, Sequence) or not isinstance(stride, Sequence):
            raise ValueError("Shape and stride hierarchy do not match")
        children.append(_snapshot_tree(extent, stride))
    return ("node", tuple(children))


def _tree_values(tree: AddressTree) -> tuple[object, object]:
    if tree[0] == "leaf":
        return tree[1], tree[2]
    shape: list[object] = []
    stride: list[object] = []
    for child in tree[1]:
        child_shape, child_stride = _tree_values(child)
        shape.append(child_shape)
        stride.append(child_stride)
    return shape, stride


@lru_cache(maxsize=512)
def _build_address_plan(key: AddressPlanKey) -> AddressPlan:
    _, offset, tree = key
    shape_values, stride_values = _tree_values(tree)
    layout = Layout(Shape(shape_values), Stride(stride_values))
    addresses = tuple(
        offset + layout.index(logical_index) for logical_index in range(layout.size)
    )
    return AddressPlan(
        key=key,
        offset=offset,
        logical_size=layout.size,
        cosize=layout.cosize,
        addresses=addresses,
        is_injective=layout.is_injective,
    )


def address_plan(layout: Layout, *, offset: int = 0) -> AddressPlan:
    """Return the cached immutable address plan for an exact layout snapshot."""
    if not isinstance(layout, Layout):
        raise TypeError("layout must be a Layout")
    normalized_offset = operator_index(offset)
    if normalized_offset < 0:
        raise ValueError("address-plan offset must be non-negative")
    tree = _snapshot_tree(layout.shape.top_level, layout.stride.top_level)
    key: AddressPlanKey = (
        "strideweave.metal.address-plan.v1",
        normalized_offset,
        tree,
    )
    return _build_address_plan(key)


__all__: list[str] = []
