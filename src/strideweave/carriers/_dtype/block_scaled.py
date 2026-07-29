"""Block-scaled dtype values and compound descriptor implementation."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import _BLOCK_SCALED_OWNED, _ContractSpec, _declare_contract
from .model import CompoundDType, DType, SimpleDType, _referenced_structure
from .representation_rule import LevelExtent
from .structure import Whole, WholeExtent, _encoded_leaf


@dataclass(frozen=True, slots=True)
class Level:
    """One scale level of a block-scaled dtype.

    Args:
        scale: Simple dtype of the scale values stored at this level.
        block: Extent of one block, measured in the *previous* level's
            coordinate space, or :data:`Whole` for a single scale covering the
            entire tensor. ``Whole`` is only valid on the final level.

    Examples:
        >>> import strideweave as sw
        >>> sw.DType.MXFP8_E4M3.levels[0]
        Level(scale=SimpleDType('E8M0'), block=32)
    """

    scale: SimpleDType
    block: int | WholeExtent

    def __post_init__(self) -> None:
        if not isinstance(self.scale, SimpleDType):
            raise TypeError("A block-scaled level scale must be a SimpleDType")
        if self.block is Whole:
            return
        if isinstance(self.block, bool) or not isinstance(self.block, int):
            raise TypeError("A block extent must be an integer or Whole")
        if self.block < 1:
            raise ValueError("A block extent must be a positive integer or Whole")
        # Keep an exact ``int``: the extent takes part in a block-scaled dtype's
        # structure, which is hashed as a registry key and compared across
        # processes, so a subclass instance must not reach it.
        object.__setattr__(self, "block", int(self.block))

    def is_whole(self) -> bool:
        """Return whether this level carries a single scale for the tensor."""
        return self.block is Whole


@dataclass(frozen=True, slots=True)
class SymbolicBits:
    """Bits per element of a dtype whose final scale level is :data:`Whole`.

    A ``Whole`` level stores one scale for the whole tensor, so its cost per
    element depends on the element count and cannot be a plain number.

    Args:
        constant: Bits per element contributed by the element encoding and
            every concretely blocked level.
        whole_scale_bits: Width of the single ``Whole`` scale, in bits.

    Examples:
        >>> import strideweave as sw
        >>> sw.DType.NVFP4.bits_per_element.evaluate(1024)
        4.53125
    """

    constant: float
    whole_scale_bits: int

    def evaluate(self, element_count: int) -> float:
        """Return the concrete bits per element for ``element_count`` elements."""
        if isinstance(element_count, bool) or not isinstance(element_count, int):
            raise TypeError("An element count must be an integer")
        if element_count < 1:
            raise ValueError("An element count must be positive")
        return self.constant + self.whole_scale_bits / element_count


class BlockScaledDType(CompoundDType, abstract=False):
    """Simple element encoding plus a linear chain of simple scale levels.

    The reconstructed value of an element is its encoded value multiplied by one
    scale from each level. Level extents are relative to the level below, so
    every level coarsens the grouping of the level under it. Descriptors are
    unique by structure: constructing a second descriptor with the same element
    and levels is rejected, so equal representations are the same object. Its
    ``simple_types`` are derived rather than supplied: the element followed by
    each level's scale, in that order.

    Args:
        name: Unique registered name for this format, such as ``"MXFP4"``.
        element: Simple dtype of the encoded elements.
        levels: Ordered scale levels, outermost last. At least one is required,
            and only the final level may use :data:`Whole`.

    Examples:
        >>> import strideweave as sw
        >>> sw.DType.MXFP8_E4M3.simple_types
        (SimpleDType('E4M3'), SimpleDType('E8M0'))
        >>> sw.DType.MXFP8_E4M3.num_carriers
        2
    """

    __slots__ = ("_element", "_levels")

    def __init__(
        self, name: str, *, element: SimpleDType, levels: tuple[Level, ...]
    ) -> None:
        if not isinstance(element, SimpleDType):
            raise TypeError("A block-scaled element dtype must be a SimpleDType")
        levels = tuple(levels)
        if not levels:
            raise ValueError("A block-scaled dtype requires at least one scale level")
        for position, level in enumerate(levels):
            if not isinstance(level, Level):
                raise TypeError("Block-scaled levels must be Level descriptors")
            if level.is_whole() and position != len(levels) - 1:
                raise ValueError("Only the final block-scaled level may use Whole")
        # The planes of a block-scaled representation are derived, not supplied:
        # the element followed by each level's scale, in that order.
        super().__init__(
            name,
            supertype=DType.Any,
            simple_types=(element, *(level.scale for level in levels)),
            representation_rules=tuple(
                LevelExtent(position, level.block)
                for position, level in enumerate(levels)
            ),
        )
        self._element = element
        self._levels = levels

    @property
    def element(self) -> SimpleDType:
        """Return the simple dtype of the encoded elements."""
        return self._element

    @property
    def levels(self) -> tuple[Level, ...]:
        """Return the scale levels, innermost first."""
        return self._levels

    @property
    def num_axes(self) -> int:
        """Return how many blocking axes a tensor of this dtype must be given."""
        return sum(1 for level in self._levels if not level.is_whole())

    @property
    def bits_per_element(self) -> float | SymbolicBits:
        """Return the stored bits per logical element.

        A format whose final level is :data:`Whole` returns
        :class:`SymbolicBits`, because that level's cost depends on the tensor's
        element count.
        """
        total = float(self._element.bits)
        denominator = 1
        for level in self._levels:
            if level.is_whole():
                return SymbolicBits(total, level.scale.bits)
            assert isinstance(level.block, int)
            denominator *= level.block
            total += level.scale.bits / denominator
        return total


def _block_scaled_layer(dtype: BlockScaledDType) -> tuple[object, ...]:
    """Return the block-scaled contract's own fragment of a structure."""
    return (
        _encoded_leaf("BlockScaledDType"),
        _referenced_structure(dtype._element),
        tuple(
            (_referenced_structure(level.scale), _encoded_leaf(level.block))
            for level in dtype._levels
        ),
    )


def _block_scaled_conflict(claimed: DType) -> str:
    """Return the message rejecting a duplicate of ``claimed``'s representation.

    Block-scaled descriptors are keyed on their whole structure, whose
    discriminators are contract names, so a subclass that adds no structure of
    its own describes an already claimed representation and is rejected instead
    of becoming a second identity for it. A subclass that does add structure
    through :meth:`DType.structure_extension` claims a representation of its
    own.
    """
    return (
        f"Block-scaled dtype {claimed.name!r} already describes this "
        "element and level chain"
    )


_declare_contract(
    BlockScaledDType,
    _ContractSpec(
        _BLOCK_SCALED_OWNED,
        layer=_block_scaled_layer,
        structural_conflict=_block_scaled_conflict,
    ),
)
