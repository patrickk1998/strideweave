from collections.abc import Sequence

from .index_map import IndexMap, _compose_generic
from .layout import Shape


class Permutation(IndexMap):
    """Map a one-mode domain through an immutable sparse lookup table.

    Results are flat integer indices. The explicit codomain bound is preserved
    even when positions above the largest lookup value are unused.

    Args:
        values: Non-empty sequence of distinct, non-negative result indices.
        codomain_size: Positive exclusive upper bound for every result index.

    Examples:
        >>> import strideweave as sw
        >>> permutation = sw.Permutation([4, 3], codomain_size=10)
        >>> permutation(0), permutation(1)
        (4, 3)
    """

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
        if isinstance(inner, Permutation):
            return Permutation(
                tuple(self.values[value] for value in inner.values),
                self.codomain_size,
            )
        return _compose_generic(self, inner)

    def _composition_is_identity(self) -> bool:
        return self.codomain_size == self.size and self.values == tuple(
            range(self.size)
        )
