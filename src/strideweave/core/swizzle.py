from __future__ import annotations

from dataclasses import dataclass

from .index_map import IndexMap, _compose_generic
from .layout import Shape


def _normalize_stages(
    stages: tuple[SwizzleStage, ...],
) -> tuple[SwizzleStage, ...]:
    normalized: list[SwizzleStage] = []
    for stage in stages:
        if normalized and normalized[-1] == stage:
            normalized.pop()
        else:
            normalized.append(stage)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class SwizzleStage:
    """Describe one immutable directed XOR field transform.

    The source field remains unchanged, so applying one stage twice is the
    identity. Positive shifts XOR the upper field into the lower field;
    negative shifts XOR the lower field into the upper field.

    Args:
        bits: Positive width of both bit fields.
        base: Non-negative bit position of the lower field.
        shift: Non-zero signed separation and XOR direction of the fields.

    Examples:
        >>> import strideweave as sw
        >>> sw.SwizzleStage(bits=1, base=0, shift=1)
        SwizzleStage(bits=1, base=0, shift=1)
    """

    bits: int
    base: int
    shift: int

    def __post_init__(self) -> None:
        for name, value in (
            ("bits", self.bits),
            ("base", self.base),
            ("shift", self.shift),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")

        if self.bits <= 0:
            raise ValueError("bits must be positive")
        if self.base < 0:
            raise ValueError("base must be non-negative")
        if self.shift == 0:
            raise ValueError("shift must be non-zero")
        if abs(self.shift) < self.bits:
            raise ValueError("swizzle stage fields must not overlap")


class Swizzle(IndexMap):
    """Apply an immutable sequence of XOR stages over a binary index space.

    The domain size must be a power of two. Stages run in argument order, and
    the result is a flat integer index in the same fixed-size codomain.

    Args:
        shape: Hierarchical domain whose size is a power of two.
        stages: Zero or more ``SwizzleStage`` transforms in evaluation order.

    Examples:
        >>> import strideweave as sw
        >>> stage = sw.SwizzleStage(bits=2, base=0, shift=2)
        >>> swizzle = sw.Swizzle(sw.Shape(16), stage)
        >>> swizzle(0b1101) == 0b1110
        True
    """

    _stages: tuple[SwizzleStage, ...]

    def __init__(self, shape: Shape, *stages: SwizzleStage) -> None:
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape")

        canonical_stages = tuple(stages)
        if any(not isinstance(stage, SwizzleStage) for stage in canonical_stages):
            raise TypeError("every stage must be a SwizzleStage")

        size = shape.size
        if size & (size - 1):
            raise ValueError("Swizzle domain size must be a power of two")

        bit_width = (size - 1).bit_length()
        for stage in canonical_stages:
            if stage.base + abs(stage.shift) + stage.bits > bit_width:
                raise ValueError("swizzle stage fields exceed the domain bit width")

        super().__init__(shape, shape.size, True)
        object.__setattr__(self, "_stages", canonical_stages)

    @property
    def stages(self) -> tuple[SwizzleStage, ...]:
        return self._stages

    def _index_ordinal(self, index: int) -> int:
        result = index
        for stage in self.stages:
            field_mask = (1 << stage.bits) - 1
            if stage.shift > 0:
                source = (result >> (stage.base + stage.shift)) & field_mask
                result ^= source << stage.base
            else:
                source = (result >> stage.base) & field_mask
                result ^= source << (stage.base - stage.shift)
        return result

    def _compose(self, inner: IndexMap) -> IndexMap:
        if isinstance(inner, Swizzle) and inner.size == self.size:
            return Swizzle(
                inner.shape,
                *_normalize_stages((*inner.stages, *self.stages)),
            )
        return _compose_generic(self, inner)

    def _composition_is_identity(self) -> bool:
        return not _normalize_stages(self.stages)
