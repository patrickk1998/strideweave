from collections.abc import Sequence

from .index_map import IndexMap, _compose_generic
from .layout import Shape


class Permutation(IndexMap):
    """Immutable explicit lookup from a one-mode domain."""

    _values: tuple[int, ...]

    def __init__(self, values: Sequence[int], codomain_size: int) -> None:
        if not isinstance(values, Sequence):
            raise TypeError("values must be a sequence of integers")
        copied_values = tuple(values)
        if not copied_values:
            raise ValueError("values must contain at least one entry")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in copied_values
        ):
            raise TypeError("values must contain only integers")
        if isinstance(codomain_size, bool):
            raise TypeError("codomain_size must be an integer")

        super().__init__(Shape(len(copied_values)), codomain_size, True)

        if any(value < 0 for value in copied_values):
            raise ValueError("values must be non-negative")
        if any(value >= codomain_size for value in copied_values):
            raise ValueError("values must be smaller than codomain_size")
        if len(set(copied_values)) != len(copied_values):
            raise ValueError("values must be pairwise distinct")

        object.__setattr__(self, "_values", copied_values)

    @property
    def values(self) -> tuple[int, ...]:
        return self._values

    def _index_ordinal(self, index: int) -> int:
        return self.values[index]

    def _compose(self, inner: IndexMap) -> IndexMap:
        return _compose_generic(self, inner)
