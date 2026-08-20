from __future__ import annotations

from .index_map import IndexMap, _compose_generic
from .layout import Shape


class Product(IndexMap):
    """Pack two or more ordered index maps as one Cartesian map.

    Each child supplies one domain mode. Child results are encoded once in the
    Product's hierarchical ``target_shape`` using first-mode-fastest order.

    Args:
        children: Two or more ``IndexMap`` values in target packing order.

    Examples:
        >>> import strideweave as sw
        >>> first = sw.Permutation([4, 3], 10)
        >>> second = sw.Permutation([2, 9], 15)
        >>> product = sw.Product(first, second)
        >>> product.target_shape == sw.Shape(10, 15), product((0, 0))
        (True, 24)
    """

    _children: tuple[IndexMap, ...]
    _target_shape: Shape

    def __init__(self, *children: IndexMap) -> None:
        if len(children) < 2:
            raise ValueError("Product requires at least two children")
        if any(not isinstance(child, IndexMap) for child in children):
            raise TypeError("Product children must be IndexMap values")

        owned_children = tuple(children)
        shape = Shape(
            *(
                int(child.shape)
                if len(child.shape) == 1 and child.shape.is_int
                else child.shape.top_level
                for child in owned_children
            ),
        )
        target_shape = Shape(
            *(
                child.target_shape
                if isinstance(child, Product)
                else child.codomain_size
                for child in owned_children
            ),
        )
        if any(child.is_injective is False for child in owned_children):
            is_injective: bool | None = False
        elif all(child.is_injective is True for child in owned_children):
            is_injective = True
        else:
            is_injective = None

        super().__init__(shape, target_shape.size, is_injective)
        object.__setattr__(self, "_children", owned_children)
        object.__setattr__(self, "_target_shape", target_shape)

    @property
    def children(self) -> tuple[IndexMap, ...]:
        return self._children

    @property
    def target_shape(self) -> Shape:
        return self._target_shape

    def _index_ordinal(self, index: int) -> int:
        coordinate = self.shape.decode(index)
        child_results = tuple(
            child(child_coordinate)
            for child, child_coordinate in zip(
                self.children,
                coordinate,
                strict=True,
            )
        )
        return self.target_shape.encode(child_results)

    def _compose(self, inner: IndexMap) -> IndexMap:
        if not isinstance(inner, Product):
            return _compose_generic(self, inner)
        if not self._composition_tree_aligns(inner):
            return _compose_generic(self, inner)
        return Product(
            *(
                outer_child.compose(inner_child)
                for outer_child, inner_child in zip(
                    self.children,
                    inner.children,
                    strict=True,
                )
            )
        )

    def _composition_tree_aligns(self, inner: Product) -> bool:
        if len(self.children) != len(inner.children):
            return False
        for outer_child, inner_child in zip(
            self.children,
            inner.children,
            strict=True,
        ):
            if isinstance(outer_child, Product) and isinstance(inner_child, Product):
                if not outer_child._composition_tree_aligns(inner_child):
                    return False
                continue
            if isinstance(outer_child, Product) or isinstance(inner_child, Product):
                return False
            if inner_child.codomain_size != outer_child.size:
                return False
        return True

    def _composition_is_identity(self) -> bool:
        return all(child._composition_is_identity() for child in self.children)
