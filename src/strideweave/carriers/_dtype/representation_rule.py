"""Immutable declarative rules for logical Tensor representations."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Protocol, Self

from ...core.layout import Layout, Shape
from .model import DType
from .structure import Whole, WholeExtent, _encoded_leaf, _encoded_structure


class RepresentationValidationContext(Protocol):
    """Describe the read-only representation facts available to rules.

    The framework supplies an object satisfying this protocol only after its
    universal dtype, carrier, placement, and adjacent-layout checks succeed.
    Rule authors inspect the logical dtype, ordered storage dtypes, physical
    placement layouts, adjacent grouping layouts, and derived level shapes.
    Every sequence is an immutable tuple; rules do not construct or mutate the
    context.

    Examples:
        >>> import strideweave as sw
        >>> class TwoPlanes(sw.RepresentationRule):
        ...     __slots__ = ()
        ...
        ...     def validate(
        ...         self, context: sw.RepresentationValidationContext
        ...     ) -> None:
        ...         if len(context.storage_dtypes) != 2:
        ...             raise ValueError("TwoPlanes requires two storage planes")
    """

    @property
    def logical_dtype(self) -> DType:
        """Return the authoritative logical dtype being represented."""
        ...

    @property
    def storage_dtypes(self) -> tuple[DType, ...]:
        """Return the ordered carrier-storage dtype schema."""
        ...

    @property
    def placement_layouts(self) -> tuple[Layout, ...]:
        """Return the physical placement layout for every storage plane."""
        ...

    @property
    def adjacent_layouts(self) -> tuple[Layout, ...]:
        """Return the ordered layouts mapping each level to the next."""
        ...

    @property
    def level_shapes(self) -> tuple[Shape, ...]:
        """Return the logical coordinate shape at every storage level."""
        ...


class _RepresentationRuleType(ABCMeta):
    """Finalize one rule after its most-derived constructor has completed."""

    def __call__(cls, *args: object, **kwargs: object) -> Any:
        rule = super().__call__(*args, **kwargs)
        extension = rule.structure_extension()
        if type(extension) is not tuple:
            raise TypeError(
                f"{cls.__name__}.structure_extension must return a tuple, "
                f"not {type(extension).__name__}"
            )
        structure = (
            _encoded_leaf(cls.__module__),
            _encoded_leaf(cls.__qualname__),
            _encoded_structure(rule, extension),
        )
        object.__setattr__(rule, "_structure", structure)
        object.__setattr__(rule, "_finalized", True)
        return rule


class RepresentationRule(metaclass=_RepresentationRuleType):
    """Add an immutable, reusable constraint to a compound representation.

    A rule contributes canonical structure to its owning
    :class:`CompoundDType` and validates only after the representation's
    universal dtype, carrier, placement, and adjacent-layout checks succeed.
    Implementations initialize their fields normally, describe those fields
    through :meth:`structure_extension`, and inspect the read-only validation
    context in :meth:`validate`.

    Rules are declarative constraints. They must not mutate carriers, layouts,
    descriptors, or other external state while validating.

    Examples:
        >>> import strideweave as sw
        >>> class OneLevel(sw.RepresentationRule):
        ...     __slots__ = ()
        ...
        ...     def validate(
        ...         self, context: sw.RepresentationValidationContext
        ...     ) -> None:
        ...         if len(context.adjacent_layouts) != 1:
        ...             raise ValueError("OneLevel requires one adjacent level")
        >>> rule = OneLevel()
        >>> rule.structure() == rule.structure()
        True
    """

    __slots__ = ("_finalized", "_structure")

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Keep framework-owned rule identity and immutability behavior fixed."""
        super().__init_subclass__(**kwargs)
        owned = {
            "__copy__",
            "__deepcopy__",
            "__delattr__",
            "__setattr__",
            "_finalized",
            "_structure",
            "structure",
        }
        shadowed = sorted(owned.intersection(vars(cls)))
        if shadowed:
            raise TypeError(
                f"{cls.__name__} must not redefine {', '.join(shadowed)}: "
                "representation-rule structure and immutability are owned by "
                "the dtype model"
            )

    def structure_extension(self) -> tuple[object, ...]:
        """Return immutable state that distinguishes this rule.

        The rule class identity is included by the framework. Implementations
        return only their instance-specific fields as exact strings, numbers,
        ``None``, :data:`Whole`, or tuples of those.
        """
        return ()

    def structure(self) -> tuple[object, ...]:
        """Return the canonical structure recorded for this rule."""
        return self._structure

    @abstractmethod
    def validate(self, context: RepresentationValidationContext) -> None:
        """Validate a universally valid representation context."""

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_finalized", False):
            raise AttributeError(
                f"{type(self).__name__} representation rules are immutable"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_finalized", False):
            raise AttributeError(
                f"{type(self).__name__} representation rules are immutable"
            )
        object.__delattr__(self, name)

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self


class LevelExtent(RepresentationRule):
    """Require one adjacent level map to group a declared number of coordinates.

    The rule targets adjacent layout ``S_level``. For an integer extent, its
    source cardinality must be exactly ``extent`` times its target cardinality,
    and every decoded target coordinate must have exactly ``extent`` source
    coordinates mapped to it. :data:`Whole` instead requires one target
    coordinate receiving the whole source level. Physical placement layouts do
    not affect this check.

    Args:
        level: Zero-based adjacent edge to validate.
        extent: Positive number of source coordinates per target coordinate, or
            :data:`Whole` for one target covering the complete source level.

    Examples:
        >>> import strideweave as sw
        >>> rule = sw.LevelExtent(0, 32)
        >>> rule.level
        0
        >>> rule.extent
        32
    """

    __slots__ = ("_extent", "_level")

    def __init__(self, level: int, extent: int | WholeExtent) -> None:
        if type(level) is not int:
            raise TypeError("A LevelExtent level must be an integer")
        if level < 0:
            raise ValueError("A LevelExtent level must be non-negative")
        if extent is not Whole:
            if type(extent) is not int:
                raise TypeError("A LevelExtent extent must be an integer or Whole")
            if extent < 1:
                raise ValueError("A LevelExtent extent must be positive or Whole")
        self._level = level
        self._extent = extent

    @property
    def level(self) -> int:
        """Return the zero-based adjacent level edge this rule validates."""
        return self._level

    @property
    def extent(self) -> int | WholeExtent:
        """Return the required group extent or :data:`Whole`."""
        return self._extent

    def structure_extension(self) -> tuple[object, ...]:
        """Return the target level and extent as canonical rule state."""
        return (self._level, self._extent)

    def __reduce__(self) -> tuple[object, ...]:
        """Reconstruct this immutable rule from its public constructor state."""
        return (LevelExtent, (self._level, self._extent))

    def validate(self, context: RepresentationValidationContext) -> None:
        """Validate grouping cardinality on this rule's adjacent level."""
        prefix = f"{context.logical_dtype.name} LevelExtent rule at level {self._level}"
        if self._level >= len(context.adjacent_layouts):
            raise ValueError(
                f"{prefix} has no adjacent layout; the representation has "
                f"{len(context.adjacent_layouts)} adjacent levels"
            )

        source_size = context.level_shapes[self._level].logical_size
        target_size = context.level_shapes[self._level + 1].logical_size
        adjacent = context.adjacent_layouts[self._level]
        if self._extent is Whole:
            if target_size != 1:
                raise ValueError(
                    f"{prefix} requires Whole to map the complete source level "
                    f"to one target coordinate, not {target_size}"
                )
            uniform_extent = adjacent.uniform_preimage_extent(
                context.level_shapes[self._level + 1]
            )
            if uniform_extent != source_size:
                raise ValueError(
                    f"{prefix} requires Whole to map every source coordinate "
                    "uniformly to the sole target coordinate"
                )
            return

        extent = self._extent
        assert isinstance(extent, int)
        if source_size % extent != 0:
            raise ValueError(
                f"{prefix} requires source cardinality {source_size} to be "
                f"divisible by extent {extent}"
            )
        expected_target_size = source_size // extent
        if target_size != expected_target_size:
            raise ValueError(
                f"{prefix} requires target cardinality {expected_target_size} "
                f"for source cardinality {source_size} and extent {extent}, "
                f"not {target_size}"
            )

        uniform_extent = adjacent.uniform_preimage_extent(
            context.level_shapes[self._level + 1]
        )
        if uniform_extent is None:
            raise ValueError(
                f"{prefix} requires every target coordinate to group {extent} "
                "source coordinates, but the adjacent layout overlaps targets "
                "or leaves holes"
            )
        if uniform_extent != extent:
            raise ValueError(
                f"{prefix} requires every target coordinate to group {extent} "
                f"source coordinates, not {uniform_extent}"
            )


def _canonical_rules(owner: type, rules: object) -> tuple[RepresentationRule, ...]:
    """Copy and validate a compound descriptor's rule sequence."""
    try:
        canonical = tuple(rules)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(
            f"{owner.__name__} representation_rules must be an iterable of "
            "RepresentationRule values"
        ) from error
    for position, rule in enumerate(canonical):
        if not isinstance(rule, RepresentationRule):
            raise TypeError(
                f"{owner.__name__} representation_rules[{position}] must be a "
                f"RepresentationRule, not {type(rule).__name__}"
            )
    return canonical


def _rule_structure(rule: RepresentationRule) -> tuple[object, ...]:
    """Return framework-recorded rule structure without virtual dispatch."""
    return rule._structure


__all__ = [
    "LevelExtent",
    "RepresentationRule",
    "RepresentationValidationContext",
]
