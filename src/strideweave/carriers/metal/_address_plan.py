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


def _validated_tree(value: object) -> AddressTree:
    if type(value) is not tuple or not value or type(value[0]) is not str:
        raise RuntimeError("Metal specialization contains an invalid address-plan tree")
    tag = value[0]
    if tag == "leaf":
        if (
            len(value) != 3
            or type(value[1]) is not int
            or value[1] < 1
            or type(value[2]) is not int
            or value[2] < 0
        ):
            raise RuntimeError(
                "Metal specialization contains an invalid address-plan leaf"
            )
        return value
    if tag == "node":
        if len(value) != 2 or type(value[1]) is not tuple:
            raise RuntimeError(
                "Metal specialization contains an invalid address-plan node"
            )
        return ("node", tuple(_validated_tree(child) for child in value[1]))
    raise RuntimeError(
        f"Metal specialization contains unknown address-plan tag {tag!r}"
    )


def _materialize_address_plan(key: AddressPlanKey) -> AddressPlan:
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


@lru_cache(maxsize=512)
def _build_address_plan(key: AddressPlanKey) -> AddressPlan:
    return _materialize_address_plan(key)


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


def address_plan_from_key(value: object, *, cache: bool = True) -> AddressPlan:
    """Rebuild an immutable address plan from one canonical recipe key."""
    if (
        type(value) is not tuple
        or len(value) != 3
        or value[0] != "strideweave.metal.address-plan.v1"
        or type(value[1]) is not int
        or value[1] < 0
    ):
        raise RuntimeError("Metal specialization contains an invalid address-plan key")
    key: AddressPlanKey = (
        "strideweave.metal.address-plan.v1",
        value[1],
        _validated_tree(value[2]),
    )
    try:
        return _build_address_plan(key) if cache else _materialize_address_plan(key)
    except (OverflowError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Metal specialization contains an invalid address-plan tree"
        ) from exc


__all__: list[str] = []
