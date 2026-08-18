from dataclasses import dataclass

from .index_map import IndexMap, _compose_generic
from .layout import Shape


@dataclass(frozen=True, slots=True)
class SwizzleStage:
    """Immutable metadata for one directed XOR field transform."""

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
    """Immutable sequence of directed XOR stages over a fixed index space."""

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
        return _compose_generic(self, inner)
