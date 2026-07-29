"""Validated internal storage representation for one logical Tensor."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..carriers import (
    Carrier,
    CompoundDType,
    DType,
    RepresentationValidationContext,
)
from .layout import Layout, Shape


@dataclass(frozen=True, slots=True)
class Subtensor:
    """One carrier-backed plane and its physical placement."""

    dtype: DType
    carrier: Carrier
    offset: int
    layout: Layout


@dataclass(frozen=True, slots=True)
class _ValidatedRepresentationContext:
    """Concrete frozen context supplied to public representation rules."""

    logical_dtype: DType
    storage_dtypes: tuple[DType, ...]
    placement_layouts: tuple[Layout, ...]
    adjacent_layouts: tuple[Layout, ...]
    level_shapes: tuple[Shape, ...]


@dataclass(frozen=True, slots=True, init=False)
class TensorRepresentation:
    """Validated ordered subtensors representing one logical dtype.

    Placement layout ``L_i`` belongs to subtensor ``i`` and maps level
    coordinate ``c_i`` to that carrier. Adjacent layout ``S_i`` has
    ``L_i.shape`` as its domain and maps a coordinate in ``c_i`` to an integer
    that is decoded in ``L_(i+1).shape``. The same :class:`Layout` type serves
    both roles; their positions in this object distinguish their semantics.
    """

    logical_dtype: DType
    subtensors: tuple[Subtensor, ...]
    adjacent_layouts: tuple[Layout, ...]

    def __init__(
        self,
        logical_dtype: DType,
        subtensors: Iterable[Subtensor],
        adjacent_layouts: Iterable[Layout] = (),
    ) -> None:
        try:
            canonical_subtensors = tuple(subtensors)
        except TypeError as error:
            raise TypeError(
                "subtensors must be an iterable of Subtensor values"
            ) from error
        try:
            canonical_adjacent = tuple(adjacent_layouts)
        except TypeError as error:
            raise TypeError(
                "adjacent_layouts must be an iterable of Layout values"
            ) from error

        object.__setattr__(self, "logical_dtype", logical_dtype)
        object.__setattr__(self, "subtensors", canonical_subtensors)
        object.__setattr__(self, "adjacent_layouts", canonical_adjacent)

        self._validate_universal()
        context: RepresentationValidationContext = _ValidatedRepresentationContext(
            logical_dtype=logical_dtype,
            storage_dtypes=tuple(subtensor.dtype for subtensor in canonical_subtensors),
            placement_layouts=tuple(
                subtensor.layout for subtensor in canonical_subtensors
            ),
            adjacent_layouts=canonical_adjacent,
            level_shapes=tuple(
                subtensor.layout.shape for subtensor in canonical_subtensors
            ),
        )
        for rule in logical_dtype.representation_rules:
            rule.validate(context)

    @property
    def primary(self) -> Subtensor:
        """Return subtensor zero for the direct conventional-tensor fast path."""
        return self.subtensors[0]

    @property
    def is_single_subtensor(self) -> bool:
        """Return whether this representation has no compound traversal."""
        return len(self.subtensors) == 1

    def _version_token(self) -> tuple[tuple[int, int], ...]:
        """Snapshot every unique constituent carrier's version in level order."""
        seen: set[int] = set()
        versions: list[tuple[int, int]] = []
        for subtensor in self.subtensors:
            identity = id(subtensor.carrier)
            if identity in seen:
                continue
            seen.add(identity)
            versions.append((identity, subtensor.carrier.version))
        return tuple(versions)

    def _validate_universal(self) -> None:
        dtype = self.logical_dtype
        if not isinstance(dtype, DType):
            raise TypeError(
                f"logical_dtype must be a DType, not {type(dtype).__name__}"
            )
        for position, subtensor in enumerate(self.subtensors):
            if not isinstance(subtensor, Subtensor):
                raise TypeError(
                    f"subtensors[{position}] must be a Subtensor, not "
                    f"{type(subtensor).__name__}"
                )

        expected_dtypes = _storage_schema(dtype)
        if len(self.subtensors) != len(expected_dtypes):
            raise ValueError(
                f"{dtype.name} representation requires {len(expected_dtypes)} "
                f"subtensors, one for each storage dtype, but received "
                f"{len(self.subtensors)}"
            )
        if len(self.adjacent_layouts) != len(self.subtensors) - 1:
            raise ValueError(
                f"{dtype.name} representation with {len(self.subtensors)} "
                f"subtensors requires {len(self.subtensors) - 1} adjacent "
                f"layouts, but received {len(self.adjacent_layouts)}"
            )

        carrier_class: type[Carrier] | None = None
        for level, (subtensor, expected_dtype) in enumerate(
            zip(self.subtensors, expected_dtypes, strict=True)
        ):
            _validate_subtensor(dtype, level, subtensor, expected_dtype)
            current_class = type(subtensor.carrier)
            if carrier_class is None:
                carrier_class = current_class
            elif current_class is not carrier_class:
                raise TypeError(
                    f"{dtype.name} representation carriers must have one exact "
                    f"class; level 0 uses {carrier_class.__name__} while level "
                    f"{level} uses {current_class.__name__}"
                )

        for level, adjacent in enumerate(self.adjacent_layouts):
            if not isinstance(adjacent, Layout):
                raise TypeError(
                    f"adjacent_layouts[{level}] must be a Layout, not "
                    f"{type(adjacent).__name__}"
                )
            source_shape = self.subtensors[level].layout.shape
            target_shape = self.subtensors[level + 1].layout.shape
            if adjacent.shape != source_shape:
                raise ValueError(
                    f"{dtype.name} adjacent layout at level {level} must have "
                    "the source placement shape as its domain"
                )
            if adjacent.cosize > target_shape.logical_size:
                raise ValueError(
                    f"{dtype.name} adjacent layout at level {level} reaches "
                    f"decoded target index {adjacent.cosize - 1}, outside level "
                    f"{level + 1} shape cardinality {target_shape.logical_size}"
                )


def _storage_schema(dtype: DType) -> tuple[DType, ...]:
    """Return the exact ordered carrier dtypes required by ``dtype``."""
    if isinstance(dtype, CompoundDType):
        return dtype.simple_types
    if dtype.is_simple() or dtype.is_opaque_storage():
        return (dtype,)
    raise ValueError(
        f"{dtype.name} is an abstract dtype category with no Tensor storage schema"
    )


def _validate_subtensor(
    logical_dtype: DType,
    level: int,
    subtensor: Subtensor,
    expected_dtype: DType,
) -> None:
    """Validate one plane before any optional representation rule runs."""
    if not isinstance(subtensor.dtype, DType):
        raise TypeError(
            f"{logical_dtype.name} subtensor {level} dtype must be a DType, not "
            f"{type(subtensor.dtype).__name__}"
        )
    if subtensor.dtype is not expected_dtype:
        raise ValueError(
            f"{logical_dtype.name} subtensor {level} must use storage dtype "
            f"{expected_dtype.name}, not {subtensor.dtype.name}"
        )
    if not isinstance(subtensor.carrier, Carrier):
        raise TypeError(
            f"{logical_dtype.name} subtensor {level} carrier must be a Carrier, "
            f"not {type(subtensor.carrier).__name__}"
        )
    carrier_dtype = subtensor.carrier.dtype()
    if carrier_dtype is not subtensor.dtype:
        actual = getattr(carrier_dtype, "name", type(carrier_dtype).__name__)
        raise ValueError(
            f"{logical_dtype.name} subtensor {level} carrier dtype must be "
            f"identical to {subtensor.dtype.name}, not {actual}"
        )
    if type(subtensor.offset) is not int:
        raise TypeError(
            f"{logical_dtype.name} subtensor {level} offset must be an int, not "
            f"{type(subtensor.offset).__name__}"
        )
    if subtensor.offset < 0:
        raise ValueError(
            "Tensor offset must be non-negative: "
            f"{logical_dtype.name} subtensor {level} offset must be non-negative"
        )
    if not isinstance(subtensor.layout, Layout):
        raise TypeError(
            f"{logical_dtype.name} subtensor {level} placement must be a Layout, "
            f"not {type(subtensor.layout).__name__}"
        )
    carrier_size = subtensor.carrier.size()
    if carrier_size < 0:
        raise ValueError(
            f"{logical_dtype.name} subtensor {level} carrier size must be non-negative"
        )
    if (
        subtensor.offset > carrier_size
        or subtensor.layout.cosize > carrier_size - subtensor.offset
    ):
        raise ValueError(
            "Tensor storage exceeds carrier size: "
            f"{logical_dtype.name} subtensor {level} placement storage exceeds "
            "carrier size"
        )


__all__ = [
    "Subtensor",
    "TensorRepresentation",
]
