from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .layout import Shape


class IndexMap(ABC):
    """Represent an immutable shaped-domain map into a flat integer codomain.

    Concrete subclasses map coordinates in a hierarchical ``Shape`` to integer
    indices below ``codomain_size``. ``IndexMap`` is the extension contract;
    construct a concrete subclass such as ``Layout`` or ``Permutation``.
    Only ``Layout`` values may describe Tensor placement.

    Args:
        shape: Hierarchical domain accepted by the map.
        codomain_size: Positive exclusive upper bound for every result index.
        is_injective: ``True`` when injectivity is proven, ``False`` when a
            collision is known, or ``None`` when it is unknown.

    Examples:
        >>> import strideweave as sw
        >>> layout = sw.Layout(sw.Shape(2), sw.Stride(1))
        >>> isinstance(layout, sw.IndexMap)
        True
    """

    _codomain_size: int
    _is_injective: bool | None
    _shape: Shape

    def __init__(
        self,
        shape: Shape,
        codomain_size: int,
        is_injective: bool | None,
    ) -> None:
        from .layout import Shape

        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape")
        if not isinstance(codomain_size, int):
            raise TypeError("codomain_size must be an integer")
        if codomain_size <= 0:
            raise ValueError("codomain_size must be positive")
        if is_injective is not None and not isinstance(is_injective, bool):
            raise TypeError("is_injective must be True, False, or None")

        object.__setattr__(self, "_shape", shape)
        object.__setattr__(self, "_codomain_size", codomain_size)
        object.__setattr__(self, "_is_injective", is_injective)

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def size(self) -> int:
        return self.shape.size

    @property
    def codomain_size(self) -> int:
        return self._codomain_size

    @property
    def is_injective(self) -> bool | None:
        return self._is_injective

    @abstractmethod
    def _index_ordinal(self, index: int) -> int:
        """Evaluate a normalized domain ordinal."""

    def _encode_key(self, key: Any) -> int:
        return self.shape.encode(key)

    def index(self, key: Any) -> int:
        """Evaluate this map at one scalar or hierarchical coordinate."""

        result = self._index_ordinal(self._encode_key(key))
        if not isinstance(result, int):
            raise TypeError("IndexMap evaluation must return an integer")
        if result < 0 or result >= self.codomain_size:
            raise ValueError("IndexMap result is outside its declared codomain")
        return result

    def __call__(self, key: Any) -> int:
        return self.index(key)

    @abstractmethod
    def _compose(self, inner: IndexMap) -> IndexMap:
        """Lower a compatible composition to a concrete result."""

    def _composition_injectivity(self) -> bool | None:
        """Return injectivity metadata usable without evaluating the domain."""

        return self.is_injective

    def _composition_is_identity(self) -> bool:
        """Return whether structure proves this map is identity."""

        return False

    def compose(self, inner: IndexMap) -> IndexMap:
        """Compose this outer map with one inner map after validating bounds."""

        if not isinstance(inner, IndexMap):
            raise TypeError("inner must be an IndexMap")
        if inner.codomain_size > self.size:
            raise ValueError("inner codomain exceeds the outer map domain")
        if inner._composition_is_identity() and self.shape == inner.shape:
            return self
        if (
            self._composition_is_identity()
            and inner.codomain_size == self.codomain_size
        ):
            return inner
        result = self._compose(inner)
        if not isinstance(result, IndexMap):
            raise TypeError("IndexMap composition must return an IndexMap")
        return result

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("IndexMap is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("IndexMap is immutable")


class _ComposedIndexMap(IndexMap):
    """Private immutable inner-to-outer composition chain."""

    _maps: tuple[IndexMap, ...]

    def __init__(self, maps: tuple[IndexMap, ...]) -> None:
        if len(maps) < 2:
            raise ValueError("a generic composition requires at least two maps")

        innermost_injectivity = maps[0]._composition_injectivity()
        if innermost_injectivity is False:
            is_injective: bool | None = False
        elif all(map_._composition_injectivity() is True for map_ in maps):
            is_injective = True
        else:
            is_injective = None

        super().__init__(maps[0].shape, maps[-1].codomain_size, is_injective)
        object.__setattr__(self, "_maps", maps)

    def _index_ordinal(self, index: int) -> int:
        result = index
        for map_ in self._maps:
            result = map_.index(result)
        return result

    def _compose(self, inner: IndexMap) -> IndexMap:
        return _compose_generic(self, inner)


def _compose_generic(outer: IndexMap, inner: IndexMap) -> IndexMap:
    inner_maps = inner._maps if isinstance(inner, _ComposedIndexMap) else (inner,)
    outer_maps = outer._maps if isinstance(outer, _ComposedIndexMap) else (outer,)
    maps = [*inner_maps, *outer_maps]
    result_shape = maps[0].shape
    result_codomain_size = maps[-1].codomain_size

    position = 0
    while position < len(maps) and len(maps) > 1:
        map_ = maps[position]
        if not map_._composition_is_identity():
            position += 1
            continue
        if position == 0 and maps[1].shape != result_shape:
            position += 1
            continue
        if position == len(maps) - 1 and (
            maps[-2].codomain_size != result_codomain_size
        ):
            position += 1
            continue
        maps.pop(position)
        position = max(0, position - 1)

    if len(maps) == 1:
        return maps[0]
    return _ComposedIndexMap(tuple(maps))
