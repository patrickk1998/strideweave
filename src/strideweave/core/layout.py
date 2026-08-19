from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from math import gcd
from typing import Any, Self, cast

from .index_map import IndexMap, _compose_generic

type Tiler = Sequence[Layout]

_MISSING_COMPOSE_OPERAND: Any = object()


@dataclass(frozen=True)
class _NodeLeaf:
    id: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, int):
            raise TypeError("Node leaf id must be an integer")
        if self.id < 0:
            raise ValueError("Node leaf id must be non-negative")

    def __repr__(self) -> str:
        return f"Node.id({self.id})"


class Node(Enum):
    """Tree markers used to describe layout leaves, nesting, and source ids."""

    Leaf = 1
    Push = 2
    Pop = 3

    @staticmethod
    def id(id_: int) -> _NodeLeaf:
        return _NodeLeaf(id_)


def _is_leaf_marker(value: object) -> bool:
    return value == Node.Leaf or isinstance(value, _NodeLeaf)


class Tree(tuple):
    """Immutable hierarchical tree used to describe layout structure."""

    size: int
    depth: int

    def __new__(cls, *args: int | Node | _NodeLeaf | Tree) -> Self:
        normalized_iterable, depth, size = Tree.norm(args)
        obj = super().__new__(cls, normalized_iterable)
        object.__setattr__(obj, "size", size)
        object.__setattr__(obj, "depth", depth)
        return obj

    @staticmethod
    def norm(
        level: Iterable[int | Node | _NodeLeaf | Tree],
    ) -> tuple[list[Node | _NodeLeaf | Tree], int, int]:
        normalized = []
        max_depth = 0
        size = 0
        for el in level:
            if isinstance(el, int):
                normalized.append(Node.Leaf)
                size += 1
            elif _is_leaf_marker(el):
                normalized.append(el)
                size += 1
            elif isinstance(el, Tree):
                max_depth = el.depth if el.depth > max_depth else max_depth
                size += el.size
                normalized.append(el)
            else:
                raise ValueError(
                    "Tree can only accept integers or Node.Leaf for leaf markers, "
                    "and Tree objects for subtrees."
                )
        return normalized, max_depth + 1, size

    @staticmethod
    def get_recipe(t: Tree) -> list[Node]:
        recipe = []
        for el in t:
            if _is_leaf_marker(el):
                recipe.append(Node.Leaf)
            else:
                recipe.append(Node.Push)
                recipe = [*recipe, *Tree.get_recipe(el)]
                recipe.append(Node.Pop)
        return recipe

    @property
    def recipe(self) -> list[Node]:
        return Tree.get_recipe(self)

    @staticmethod
    def bake(itr_able: Iterable[Any], recipe: Iterable[Node]) -> list[Any]:
        stack: list[list[Any]] = [[]]
        itr = iter(itr_able)
        for instr in recipe:
            if instr == Node.Leaf:
                try:
                    to_append = next(itr)
                except StopIteration as exc:
                    raise ValueError(
                        "Iterable object to bake does not match recipe length"
                    ) from exc
                stack[-1].append(to_append)
            if instr == Node.Push:
                stack.append([])
            if instr == Node.Pop:
                lower_level = stack.pop()
                stack[-1].append(lower_level)
        return stack[0]

    @staticmethod
    def bake_tree(recipe: Iterable[Node]) -> Tree:
        stack: list[list[Node | Tree]] = [[]]
        for instr in recipe:
            if instr == Node.Leaf:
                stack[-1].append(Node.Leaf)
            if instr == Node.Push:
                stack.append([])
            if instr == Node.Pop:
                lower_level = stack.pop()
                stack[-1].append(Tree(*lower_level))
        return Tree(*stack[0])

    def reshape(self, itr_able: Iterable[Any]) -> list[Any]:
        return Tree.bake(itr_able, self.recipe)


class _ShapeLevel(tuple):
    logical_size: int

    def __new__(cls, iterable: Iterable[int | _ShapeLevel] = ()) -> Self:
        logical_size = 1
        for el in iterable:
            if not isinstance(el, int) and not isinstance(el, _ShapeLevel):
                raise ValueError(
                    "_ShapeLevel can only contain integer or _ShapeLevel elements"
                )
            if isinstance(el, int):
                logical_size *= el
            if isinstance(el, _ShapeLevel):
                logical_size *= el.logical_size
        obj = super().__new__(cls, iterable)
        object.__setattr__(obj, "logical_size", logical_size)
        return obj

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("object is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("object is immutable")


class Shape:
    """Hierarchical positive-integer tensor shape."""

    top_level: _ShapeLevel
    depth: int
    logical_size: int

    def __init__(self, *items: Any):
        if len(items) == 0:
            iterable = ()
        elif len(items) == 1:
            iterable = items[0]
        else:
            iterable = items

        if isinstance(iterable, int):
            iterable = [iterable]
        normalized, depth = Shape.normalize_input(iterable, 0)
        object.__setattr__(self, "top_level", normalized)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "logical_size", self.top_level.logical_size)

    def __getitem__(self, key: Any) -> Shape:
        return Shape(self.top_level[key])

    def __len__(self) -> int:
        return len(self.top_level)

    def __eq__(self, other: object):
        if not isinstance(other, Shape):
            return NotImplemented
        return self.top_level == other.top_level

    @property
    def is_int(self) -> bool:
        if len(self) <= 1 and self.depth <= 1:
            return True
        return False

    def __int__(self) -> int:
        if not self.is_int:
            raise ValueError("Shape can not be represented as a integer")
        return self.top_level[0]

    @property
    def size(self) -> int:
        return self.logical_size

    @staticmethod
    def _decode_level(level: _ShapeLevel, index: int) -> tuple[Any, ...]:
        coordinate = []
        for element in level:
            element_size = element if isinstance(element, int) else element.logical_size
            element_index = index % element_size
            index //= element_size
            coordinate.append(
                element_index
                if isinstance(element, int)
                else Shape._decode_level(element, element_index)
            )
        return tuple(coordinate)

    def decode(self, index: int) -> tuple[Any, ...]:
        """Decode a flat ordinal into this shape's hierarchical coordinate.

        Args:
            index: Non-negative first-mode-fastest ordinal in this shape.

        Returns:
            Nested tuple coordinate congruent with this shape's hierarchy.

        Examples:
            >>> Shape(5, [20, 4]).decode(214)
            (4, (2, 2))
        """

        if not isinstance(index, int):
            raise TypeError("Shape index must be an integer")
        if index < 0 or index >= self.size:
            raise ValueError("Index is not in domain of shape")
        return Shape._decode_level(self.top_level, index)

    @staticmethod
    def _encode_level(level: _ShapeLevel, coordinate: Any) -> int:
        if isinstance(coordinate, int):
            if coordinate < 0 or coordinate >= level.logical_size:
                raise ValueError("Coordinate is not in domain of shape")
            return coordinate
        if not isinstance(coordinate, tuple | list):
            raise TypeError("Coordinate must be an integer, tuple, or list")
        if len(coordinate) != len(level):
            raise ValueError("Coordinate does not match shape hierarchy")

        index = 0
        multiplier = 1
        for element, component in zip(level, coordinate, strict=True):
            if isinstance(element, int):
                if not isinstance(component, int):
                    if isinstance(component, tuple | list):
                        raise ValueError("Coordinate does not match shape hierarchy")
                    raise TypeError(
                        "Coordinate must contain only integers, tuples, or lists"
                    )
                if component < 0 or component >= element:
                    raise ValueError("Coordinate is not in domain of shape")
                component_index = component
                element_size = element
            else:
                component_index = Shape._encode_level(element, component)
                element_size = element.logical_size
            index += multiplier * component_index
            multiplier *= element_size
        return index

    def encode(self, coordinate: Any) -> int:
        """Encode a hierarchical coordinate as a flat ordinal.

        An integer may replace the complete coordinate or any nested subtree.

        Args:
            coordinate: Integer, tuple, or list coordinate in this shape.

        Returns:
            First-mode-fastest ordinal corresponding to ``coordinate``.

        Examples:
            >>> shape = Shape(5, [20, 4])
            >>> shape.encode((4, (2, 2)))
            214
            >>> shape.encode((4, 42))
            214
        """

        return Shape._encode_level(self.top_level, coordinate)

    @staticmethod
    def normalize_input(input_: Any, depth: int) -> tuple[_ShapeLevel, int]:
        current_level: list[int | _ShapeLevel] = []
        max_sublevel_depth = 0
        for el in input_:
            if not isinstance(el, int) and not Shape.is_iterable(el):
                raise ValueError(
                    "Shape contains an element that is not an integer or a iterable"
                )
            if isinstance(el, int) or (isinstance(el, Shape) and el.is_int):
                el = int(el)
                if el < 1:
                    raise ValueError("Dimension shape must not be less than 1")
                current_level.append(el)
            elif Shape.is_iterable(el):
                lower_level, this_depth = Shape.normalize_input(el, depth)
                current_level.append(lower_level)
                max_sublevel_depth = (
                    this_depth
                    if this_depth > max_sublevel_depth
                    else max_sublevel_depth
                )
            else:
                raise ValueError("Shape element is not a iterable or an integer")
        return _ShapeLevel(current_level), depth + 1 + max_sublevel_depth

    @staticmethod
    def concat(shape1: Shape, shape2: Shape) -> Shape:
        top_level = []
        for shape in shape1.top_level:
            top_level.append(shape)
        for shape in shape2.top_level:
            top_level.append(shape)
        return Shape(top_level)

    @staticmethod
    def append(shape1: Shape, shape2: Shape) -> Shape:
        if shape2.is_int:
            return Shape.concat(shape1, shape2)

        top_level = []
        for shape in shape1.top_level:
            top_level.append(shape)
        top_level.append(shape2)
        return Shape(top_level)

    @staticmethod
    def is_iterable(obj: Any) -> bool:
        try:
            iter(obj)
            try:
                len(obj)
                return True
            except TypeError:
                return False
        except TypeError:
            return False

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("Shape is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("Shape is immutable")

    def __repr__(self) -> str:
        return f"Shape<{self.top_level!r}>"

    def __str__(self) -> str:
        return f"Shape<{self.top_level}>"


class _StrideLevel(tuple):
    def __new__(cls, iterable: Iterable[int | _StrideLevel] = ()) -> Self:
        for el in iterable:
            if not isinstance(el, int) and not isinstance(el, _StrideLevel):
                raise ValueError(
                    "_ShapeLevel can only contain integer or _ShapeLevel elements"
                )
        obj = super().__new__(cls, iterable)
        return obj

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("object is immutable")


class Stride:
    """Hierarchical non-negative tensor stride profile."""

    top_level: _StrideLevel
    depth: int

    def __init__(self, *items: Any):
        if len(items) == 0:
            iterable = ()
        elif len(items) == 1:
            iterable = items[0]
        else:
            iterable = items

        if isinstance(iterable, int):
            iterable = [iterable]
        normalized, depth = Stride.normalize_input(iterable)
        object.__setattr__(self, "top_level", normalized)
        object.__setattr__(self, "depth", depth)

    def __getitem__(self, key: Any) -> Stride:
        return Stride(self.top_level[key])

    def __len__(self) -> int:
        return len(self.top_level)

    def __eq__(self, other: object):
        if not isinstance(other, Stride):
            return NotImplemented
        return self.top_level == other.top_level

    @property
    def is_int(self) -> bool:
        if len(self) <= 1 and self.depth <= 1:
            return True
        return False

    def __int__(self) -> int:
        if not self.is_int:
            raise ValueError("Shape can not be represented as a integer")
        return self.top_level[0]

    @staticmethod
    def normalize_input(input_: Any) -> tuple[_StrideLevel, int]:
        max_sublevel_depth = 0
        current_level: list[int | _StrideLevel] = []
        for el in input_:
            if not isinstance(el, int) and not Stride.is_iterable(el):
                raise ValueError(
                    "Stride contains an element that is not an integer or a iterable"
                )
            if isinstance(el, int) or (isinstance(el, Stride) and el.is_int):
                el = int(el)
                if el < 0:
                    raise ValueError("Stride value must not be negative")
                current_level.append(el)
            elif Stride.is_iterable(el):
                lower_level, this_depth = Stride.normalize_input(el)
                max_sublevel_depth = (
                    this_depth
                    if this_depth > max_sublevel_depth
                    else max_sublevel_depth
                )
                current_level.append(lower_level)
            else:
                raise ValueError("Stride element is not a iterable or an integer")
        return _StrideLevel(current_level), max_sublevel_depth + 1

    @staticmethod
    def is_iterable(obj: Any) -> bool:
        try:
            iter(obj)
            try:
                len(obj)
                return True
            except TypeError:
                return False
        except TypeError:
            return False

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("Stride is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("Stride is immutable")

    def __repr__(self) -> str:
        return f"Stride<{self.top_level!r}>"

    def __str__(self) -> str:
        return f"Stride<{self.top_level}>"

    @staticmethod
    def concat(stride1: Stride, stride2: Stride) -> Stride:
        top_level = []
        for stride in stride1.top_level:
            top_level.append(stride)
        for stride in stride2.top_level:
            top_level.append(stride)
        return Stride(top_level)

    @staticmethod
    def append(stride1: Stride, stride2: Stride) -> Stride:
        if stride2.is_int:
            return Stride.concat(stride1, stride2)

        top_level = []
        for stride in stride1.top_level:
            top_level.append(stride)
        top_level.append(stride2)
        return Stride(top_level)


class LayoutIterable:
    def __init__(self, layout: Layout):
        self.layout = layout
        self.position = 0

    def __iter__(self) -> LayoutIterable:
        return self

    def __next__(self) -> Layout:
        if self.position >= len(self.layout):
            raise StopIteration
        value = self.layout[self.position]
        self.position += 1
        return value


class Layout(IndexMap):
    """Hierarchical shape and stride pair for logical-to-physical indexing."""

    _cache: Any
    stride: Stride

    def __init__(self, shape: Shape, stride: Stride):
        if not isinstance(shape, Shape):
            raise ValueError("shape input must be a Shape object")
        if not isinstance(stride, Stride):
            raise ValueError("stride input must be a Stride object")
        if not Layout.check_tree(shape.top_level, stride.top_level):
            raise ValueError("Shape and Stride do not match in Structure")
        object.__setattr__(self, "_shape", shape)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(
            self, "_cache", import_module("strideweave._index")._LayoutCache(self)
        )
        super().__init__(shape, self._cache.cosize, None)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("Layout is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("Layout is immutable")

    @staticmethod
    def check_tree(shape: _ShapeLevel, stride: _StrideLevel) -> bool:
        if len(shape) != len(stride):
            return False
        for sh, st in zip(shape, stride, strict=True):
            if isinstance(sh, int) and isinstance(st, int):
                continue
            if isinstance(sh, _ShapeLevel) and isinstance(st, _StrideLevel):
                if not Layout.check_tree(sh, st):
                    return False
                continue
            return False
        return True

    @staticmethod
    def expand_int(key: int, shape: _ShapeLevel) -> list[int]:
        if key < 0 or key >= shape.logical_size:
            raise ValueError("Key is not in domain of shape")

        cord = []
        for el in shape:
            if isinstance(el, int):
                lsize = el
            else:
                lsize = el.logical_size
            cord.append(key % lsize)
            key //= lsize
        return cord

    @staticmethod
    def get_index(layout: Layout, key: Any) -> int:
        if not isinstance(layout, Layout):
            raise ValueError("layout input must be a Layout object")
        return layout.index(key)

    def _encode_key(self, key: Any) -> int:
        try:
            return super()._encode_key(key)
        except ValueError as error:
            raise ValueError("Key is not in domain of shape") from error

    def _index_ordinal(self, index: int) -> int:
        return self._cache.get_index(index)

    @staticmethod
    def _get_index_levels(shape: _ShapeLevel, stride: _StrideLevel, key: Any) -> int:
        if len(shape) != len(stride):
            raise ValueError("Shape and Stride Lengths do not match")

        if isinstance(key, int):
            curr_key = Layout.expand_int(key, shape)
        else:
            curr_key = key

        idx = 0
        for sh, stride_value, k in zip(shape, stride, curr_key, strict=True):
            if isinstance(sh, int):
                if k < 0 or k >= sh:
                    raise ValueError("Key is not in domain of shape")
                idx += stride_value * k
            else:
                idx += Layout._get_index_levels(sh, stride_value, k)

        return idx

    def __eq__(self, other: object):
        if not isinstance(other, Layout):
            return NotImplemented
        return other.shape == self.shape and other.stride == self.stride

    def __len__(self) -> int:
        return len(self.shape.top_level)

    def __copy__(self) -> Layout:
        return Layout(self.shape, self.stride)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Layout:
        return Layout(self.shape, self.stride)

    def __getitem__(self, key: Any) -> Layout:
        return Layout(self.shape[key], self.stride[key])

    @property
    def is_leaf(self) -> bool:
        return self.shape.is_int and self.stride.is_int

    @property
    def size(self) -> int:
        return self.shape.size

    @staticmethod
    def _profile_for_shape(shape: Shape) -> list[Node]:
        def shape_tree(level: _ShapeLevel) -> Tree:
            return Tree(
                *(
                    Node.Leaf if isinstance(element, int) else shape_tree(element)
                    for element in level
                )
            )

        return shape_tree(shape.top_level).recipe

    @property
    def profile(self) -> list[Node]:
        """Return the recipe describing this layout's hierarchical shape tree.

        The profile records only leaf positions and nesting. Extents and
        strides do not contribute, so layouts with the same hierarchy share a
        profile even when their values differ.

        Syntax:
            ``Node.Leaf`` marks a leaf, while ``Node.Push`` and ``Node.Pop``
            delimit a nested mode.

        Semantics:
            The returned recipe is structural and independent of shape extents
            and stride values.

        Mode assumptions:
            Every hierarchical mode is preserved; the profile never flattens
            or reorders modes.

        Returns:
            Fresh list containing the layout's structural recipe.

        Examples:
            >>> from strideweave import Layout, Node, Shape, Stride
            >>> layout = Layout(Shape([2, [3, 4]]), Stride([1, [2, 6]]))
            >>> layout.profile
            [<Node.Leaf: 1>, <Node.Push: 2>, <Node.Leaf: 1>, <Node.Leaf: 1>, <Node.Pop: 3>]
        """

        return Layout._profile_for_shape(self.shape)

    @property
    def is_injective(self) -> bool:
        """Report whether logical coordinates map to distinct physical offsets.

        Syntax:
            Injectivity is evaluated over the complete hierarchical layout,
            including leaves nested inside modes.

        Semantics:
            Returns ``False`` for broadcast modes with extent greater than one
            and stride zero, and for any other collision such as overlapping
            non-zero strides. Strided layouts with holes remain injective when
            all addressed offsets are distinct.

        Mode assumptions:
            Shape hierarchy affects coordinate structure but not the collision
            definition; every leaf contributes its extent and stride.

        Returns:
            ``True`` exactly when no two logical coordinates share an offset.

        Examples:
            >>> from strideweave import Layout, Shape, Stride
            >>> Layout(Shape([2, 3]), Stride([1, 4])).is_injective
            True
            >>> Layout(Shape([4, 2]), Stride([0, 1])).is_injective
            False
        """
        if self.size > self.cosize:
            return False

        span = 1
        for extent, stride in sorted(self.infix(), key=lambda mode: mode[1]):
            if extent == 1:
                continue
            if stride == 0:
                return False
            if stride < span:
                break
            span += (extent - 1) * stride
        else:
            return True

        offsets: set[int] = set()
        for logical_index in range(self.size):
            offset = self._cache.get_index(logical_index)
            if offset in offsets:
                return False
            offsets.add(offset)
        return True

    @property
    def _has_only_broadcast_aliasing(self) -> bool:
        if self.is_injective:
            return False

        def collapse_broadcasts(
            shape: _ShapeLevel, stride: _StrideLevel
        ) -> tuple[list[Any], list[Any]]:
            collapsed_shape: list[Any] = []
            collapsed_stride: list[Any] = []
            for extent, stride_value in zip(shape, stride, strict=True):
                if isinstance(extent, int):
                    assert isinstance(stride_value, int)
                    collapsed_shape.append(
                        1 if stride_value == 0 and extent > 1 else extent
                    )
                    collapsed_stride.append(stride_value)
                    continue
                assert isinstance(stride_value, _StrideLevel)
                child_shape, child_stride = collapse_broadcasts(extent, stride_value)
                collapsed_shape.append(child_shape)
                collapsed_stride.append(child_stride)
            return collapsed_shape, collapsed_stride

        shape, stride = collapse_broadcasts(
            self.shape.top_level,
            self.stride.top_level,
        )
        return Layout(Shape(shape), Stride(stride)).is_injective

    def broadcast_to(self, target: Shape) -> Layout:
        """Widen singleton leaves to a structurally congruent target shape.

        Syntax:
            ``target`` uses StrideWeave's hierarchical ``Shape`` syntax and
            must have the same shape-tree profile as this layout.

        Semantics:
            A source leaf of extent one may widen to the corresponding target
            extent by changing only its stride to zero. Equal extents preserve
            their existing strides. Every other extent change is refused.

        Mode assumptions:
            Leaves match by their position in the hierarchy. No flattening,
            rank alignment, insertion, removal, or reordering is performed.

        Args:
            target: Hierarchical shape to which singleton leaves are widened.

        Returns:
            New layout with ``target`` as its shape and widened leaves using
            stride zero.

        Examples:
            >>> from strideweave import Layout, Shape, Stride
            >>> layout = Layout(Shape([2, 1]), Stride([1, 2]))
            >>> layout.broadcast_to(Shape([2, 3]))
            Layout( Shape<(2, 3)>, Stride<(1, 0)>)
        """
        if not isinstance(target, Shape):
            raise TypeError("Layout broadcast target must be a Shape")

        target_profile = Layout._profile_for_shape(target)
        if self.profile != target_profile:
            raise ValueError(
                "Layout broadcast target must have the same shape profile: "
                f"{self.profile!r} != {target_profile!r}"
            )

        def widen(
            source_shape: _ShapeLevel,
            source_stride: _StrideLevel,
            target_shape: _ShapeLevel,
        ) -> tuple[list[Any], list[Any]]:
            widened_shape: list[Any] = []
            widened_stride: list[Any] = []
            for source_extent, stride, target_extent in zip(
                source_shape, source_stride, target_shape, strict=True
            ):
                if isinstance(source_extent, int):
                    assert isinstance(stride, int)
                    assert isinstance(target_extent, int)
                    if source_extent == target_extent:
                        widened_shape.append(source_extent)
                        widened_stride.append(stride)
                    elif source_extent == 1:
                        widened_shape.append(target_extent)
                        widened_stride.append(0)
                    else:
                        raise ValueError(
                            "Layout broadcast can only widen an extent-1 leaf: "
                            f"{source_extent} cannot become {target_extent}"
                        )
                    continue

                assert isinstance(stride, _StrideLevel)
                assert isinstance(target_extent, _ShapeLevel)
                child_shape, child_stride = widen(source_extent, stride, target_extent)
                widened_shape.append(child_shape)
                widened_stride.append(child_stride)
            return widened_shape, widened_stride

        shape, stride = widen(
            self.shape.top_level,
            self.stride.top_level,
            target.top_level,
        )
        return Layout(Shape(shape), Stride(stride))

    @property
    def cosize(self) -> int:
        """Physical storage size required to materialize this layout.

        A layout maps ``size`` logical elements onto physical offsets through
        its strides; ``cosize`` is one past the largest offset any logical
        coordinate reaches, i.e. the smallest buffer that can back the layout.
        For a compact (contiguous) layout it equals ``size``; strided or
        hierarchical layouts may require more. Use this when allocating storage
        for a tensor over a given layout.

        Returns:
            Number of physical slots the layout addresses.

        Examples:
            >>> from strideweave import Layout, Shape, Stride
            >>> Layout(Shape([2, 3]), Stride([1, 2])).cosize
            6
        """
        return self._cache.cosize

    def uniform_preimage_extent(self, target_shape: Shape) -> int | None:
        """Prove a uniform surjection onto a target coordinate space.

        The proof inspects only the hierarchical shape/stride leaves. Positive
        strides must form a compact mixed-radix enumeration of ``target_shape``;
        zero-stride modes contribute uniform replication. Singleton modes are
        ignored because their coordinate is always zero. No logical coordinate
        is enumerated.

        Args:
            target_shape: Shape whose flattened coordinate indices form the
                intended codomain.

        Returns:
            Number of source coordinates mapping to each target coordinate when
            the layout is a provably uniform surjection, otherwise ``None``.

        Examples:
            >>> from strideweave import Layout, Shape, Stride
            >>> grouping = Layout(Shape([4, 3]), Stride([0, 1]))
            >>> grouping.uniform_preimage_extent(Shape(3))
            4
        """
        if not isinstance(target_shape, Shape):
            raise TypeError("target_shape must be a Shape")

        replication = 1
        positive_modes: list[tuple[int, int]] = []
        for mode_shape, mode_stride in self.infix():
            if mode_shape == 1:
                continue
            if mode_stride == 0:
                replication *= mode_shape
            else:
                positive_modes.append((mode_stride, mode_shape))

        covered = 1
        for mode_stride, mode_shape in sorted(positive_modes):
            if mode_stride != covered:
                return None
            covered *= mode_shape

        if covered != target_shape.logical_size:
            return None
        return replication

    @staticmethod
    def flatten_layout(layout: Layout) -> tuple[Layout, list[Node]]:
        flat = Layout(Shape(), Stride())
        recipe = []
        for el in layout:
            if el.is_leaf:
                flat = Layout.concat(flat, el)
                recipe.append(Node.Leaf)
            else:
                recipe.append(Node.Push)
                lower_layout, lower_recipe = Layout.flatten_layout(el)
                recipe = [*recipe, *lower_recipe]
                flat = Layout.concat(flat, lower_layout)
                recipe.append(Node.Pop)
        return flat, recipe

    @staticmethod
    def concat(l1: Layout, l2: Layout) -> Layout:
        concat_shape = Shape.concat(l1.shape, l2.shape)
        concat_stride = Stride.concat(l1.stride, l2.stride)
        return Layout(concat_shape, concat_stride)

    def __add__(self, layout: Layout) -> Layout:
        return Layout.concat(self, layout)

    @staticmethod
    def append(l1: Layout, l2: Layout) -> Layout:
        append_shape = Shape.append(l1.shape, l2.shape)
        append_stride = Stride.append(l1.stride, l2.stride)
        return Layout(append_shape, append_stride)

    def __iter__(self) -> Iterator[Layout]:
        return LayoutIterable(self)

    @staticmethod
    def coalesce(layout: Layout) -> Layout:
        traversal = layout.infix()
        if len(traversal) == 1:
            return copy.copy(layout)
        i = 1
        prefix = [traversal[0]]
        prefix_i = 0
        while i < len(traversal):
            candidate = traversal[i]
            if candidate[0] == 1:
                i += 1
                continue
            if prefix[prefix_i][0] == 1:
                prefix[prefix_i] = candidate
                i += 1
                continue
            coalesced_stride = prefix[prefix_i][0] * prefix[prefix_i][1]
            if candidate[1] == coalesced_stride:
                new_mode = (candidate[0] * prefix[prefix_i][0], prefix[prefix_i][1])
                prefix[prefix_i] = new_mode
                i += 1
                continue
            prefix.append(traversal[i])
            prefix_i += 1
            i += 1
        return Layout(Shape([sh for sh, _ in prefix]), Stride([st for _, st in prefix]))

    @staticmethod
    def coalesce_by_mode(layout: Layout, profile: Tree) -> Layout:
        extracted = Layout.extract_profile(layout, profile)
        coalesced = [Layout.coalesce(subl) for subl in extracted]
        coalesced_shape = [subl.shape for subl in coalesced]
        coalesced_stride = [subl.stride for subl in coalesced]
        new_shape = Tree.bake(coalesced_shape, profile.recipe)
        new_stride = Tree.bake(coalesced_stride, profile.recipe)
        return Layout(Shape(new_shape), Stride(new_stride))

    @staticmethod
    def extract_profile(layout: Layout, profile: Tree | None = None) -> list[Layout]:
        if profile is None:
            return Layout._prefix_layout_leaves(layout)

        extracted = []
        if len(layout) != len(profile):
            raise ValueError("layout and tree profile do not match")
        for node, marker in zip(layout, profile, strict=True):
            if _is_leaf_marker(marker):
                extracted.append(node)
            if isinstance(marker, Tree):
                extracted = [*extracted, *Layout.extract_profile(node, marker)]
        return extracted

    @staticmethod
    def _prefix_layout_leaves(layout: Layout) -> list[Layout]:
        if layout.is_leaf:
            return [copy.copy(layout)]

        extracted = []
        for node in layout:
            extracted = [*extracted, *Layout._prefix_layout_leaves(node)]
        return extracted

    @staticmethod
    def _default_selection_tree(layout: Layout) -> Tree:
        if layout.is_leaf:
            return Tree(Node.Leaf)

        selection = []
        for node in layout:
            if node.is_leaf:
                selection.append(Node.Leaf)
            else:
                selection.append(Layout._default_selection_tree(node))
        return Tree(*selection)

    @staticmethod
    def rearrange(
        layout: Layout, output: Tree, selection: Tree | None = None
    ) -> Layout:
        if not isinstance(output, Tree):
            raise ValueError("output must be a Tree")

        extracted = Layout.extract_profile(layout, selection)
        used_ids: list[int] = []
        rearranged = Layout._rearrange_from_tree(output, extracted, used_ids)
        Layout._validate_rearrange_ids(used_ids, extracted)
        return rearranged

    @staticmethod
    def _rearrange_from_tree(
        output: Tree, extracted: list[Layout], used_ids: list[int]
    ) -> Layout:
        rearranged = Layout.empty()
        for marker in output:
            if isinstance(marker, _NodeLeaf):
                if marker.id >= len(extracted):
                    raise ValueError("Layout rearrange id is out of range")
                child = extracted[marker.id]
                used_ids.append(marker.id)
            elif marker == Node.Leaf:
                child = Layout(Shape(1), Stride(0))
            elif isinstance(marker, Tree):
                child = Layout._rearrange_from_tree(marker, extracted, used_ids)
            else:
                raise ValueError("output tree contains an invalid marker")
            rearranged = Layout.append(rearranged, child)
        return rearranged

    @staticmethod
    def _validate_rearrange_ids(used_ids: list[int], extracted: list[Layout]) -> None:
        seen = set()
        for id_ in used_ids:
            if id_ in seen:
                raise ValueError("Layout rearrange ids must not be duplicated")
            seen.add(id_)

        missing_non_singleton_ids = [
            id_
            for id_, layout in enumerate(extracted)
            if id_ not in seen and layout.shape.logical_size != 1
        ]
        if missing_non_singleton_ids:
            raise ValueError("Layout rearrange ids must include every extracted layout")

    @staticmethod
    def reverse_rearrange(output: Tree, selection: Tree) -> tuple[Tree, Tree]:
        if not isinstance(output, Tree):
            raise ValueError("output must be a Tree")
        if not isinstance(selection, Tree):
            raise ValueError("selection must be a Tree")

        source_to_output: dict[int, int] = {}
        reverse_selection, output_leaf_count = Layout._strip_rearrange_ids(
            output, source_to_output, 0
        )
        for source_id in source_to_output:
            if source_id >= selection.size:
                raise ValueError("Layout rearrange id is out of range")

        reverse_output, source_count = Layout._invert_rearrange_selection(
            selection, source_to_output, 0
        )
        if source_count != selection.size:
            raise ValueError("selection tree is inconsistent")
        if output_leaf_count != reverse_selection.size:
            raise ValueError("output tree is inconsistent")
        return reverse_output, reverse_selection

    @staticmethod
    def _strip_rearrange_ids(
        output: Tree, source_to_output: dict[int, int], output_id: int
    ) -> tuple[Tree, int]:
        stripped = []
        for marker in output:
            if isinstance(marker, Tree):
                child, output_id = Layout._strip_rearrange_ids(
                    marker, source_to_output, output_id
                )
                stripped.append(child)
            elif isinstance(marker, _NodeLeaf):
                if marker.id in source_to_output:
                    raise ValueError("Layout rearrange ids must not be duplicated")
                source_to_output[marker.id] = output_id
                stripped.append(Node.Leaf)
                output_id += 1
            elif marker == Node.Leaf:
                stripped.append(Node.Leaf)
                output_id += 1
            else:
                raise ValueError("output tree contains an invalid marker")
        return Tree(*stripped), output_id

    @staticmethod
    def _invert_rearrange_selection(
        selection: Tree, source_to_output: dict[int, int], source_id: int
    ) -> tuple[Tree, int]:
        inverted = []
        for marker in selection:
            if isinstance(marker, Tree):
                child, source_id = Layout._invert_rearrange_selection(
                    marker, source_to_output, source_id
                )
                inverted.append(child)
            elif _is_leaf_marker(marker):
                if source_id in source_to_output:
                    inverted.append(Node.id(source_to_output[source_id]))
                else:
                    inverted.append(Node.Leaf)
                source_id += 1
            else:
                raise ValueError("selection tree contains an invalid marker")
        return Tree(*inverted), source_id

    @staticmethod
    def permute(layout: Layout, *order: Any) -> Layout:
        normalized_order = Layout._normalize_permute_order(order, len(layout))
        output = Tree(*(Node.id(dim) for dim in normalized_order))
        selection = Tree(*(Node.Leaf for _ in range(len(layout))))
        return Layout.rearrange(layout, output, selection)

    @staticmethod
    def _normalize_permute_order(order: tuple[Any, ...], rank: int) -> tuple[int, ...]:
        if len(order) == 1 and not isinstance(order[0], int):
            try:
                order = tuple(order[0])
            except TypeError:
                pass

        for dim in order:
            if type(dim) is not int:
                raise TypeError("Permutation dimensions must be integers")

        normalized_order = tuple(order)
        expected = set(range(rank))
        if len(normalized_order) != rank or set(normalized_order) != expected:
            raise ValueError("Permutation dimensions must reorder every layout mode")
        return normalized_order

    @property
    def depth(self) -> int:
        return self.shape.depth

    def __repr__(self) -> str:
        return f"Layout( {self.shape!r}, {self.stride!r})"

    def __str__(self) -> str:
        return f"Layout( {self.shape}, {self.stride})"

    @staticmethod
    def empty() -> Layout:
        return Layout(Shape(), Stride())

    @staticmethod
    def _infix_traversal(
        shape: _ShapeLevel, stride: _StrideLevel
    ) -> list[tuple[int, int]]:
        traversal = []
        for sh, st in zip(shape, stride, strict=True):
            if isinstance(sh, int):
                traversal.append((sh, st))
            else:
                traversal = [*traversal, *Layout._infix_traversal(sh, st)]
        return traversal

    def infix(self) -> list[tuple[int, int]]:
        return Layout._infix_traversal(self.shape.top_level, self.stride.top_level)

    @staticmethod
    def choose(A: Layout, d: int) -> Layout:
        if A.shape.logical_size % d != 0:
            raise ValueError(f"Can not choose the {d}-th element of Layout {A}")
        new_stride = []
        new_shape = []
        d_remaining = d
        for el in A:
            cur_shape = int(el.shape)
            if d_remaining == 1:
                new_shape.append(cur_shape)
                new_stride.append(int(el.stride))
                continue
            if cur_shape > d_remaining:
                if cur_shape % d_remaining != 0:
                    raise ValueError(
                        f"Can not choose the {d}-th element of Layout {A}, "
                        f"{cur_shape} can not be reduced by {d_remaining}"
                    )
                new_stride.append(d_remaining * int(el.stride))
                new_shape.append(cur_shape // d_remaining)
                d_remaining = 1
            else:
                if d_remaining % cur_shape != 0:
                    raise ValueError(
                        f"Can not choose the {d}-th element of Layout {A}, "
                        f"{cur_shape} can not be reduced by {d_remaining}"
                    )
                new_stride.append(d_remaining * int(el.stride))
                new_shape.append(1)
                d_remaining = d_remaining // cur_shape
        return Layout(Shape(new_shape), Stride(new_stride))

    @staticmethod
    def modout(A: Layout, s: int) -> Layout:
        new_shape = []
        cur = 1
        for el in A:
            if cur == s:
                new_shape.append(1)
                continue
            if int(el.shape) * cur <= s:
                new_shape.append(int(el.shape))
                cur = int(el.shape) * cur
            else:
                if s % cur != 0:
                    raise ValueError(
                        f"Shape divisibility condition not met for {A} and {s}"
                    )
                new_shape.append(s // cur)
                cur = s
        return Layout(Shape(new_shape), A.stride)

    @staticmethod
    def compose_layouts(A: Layout, B: Layout) -> Layout:
        layout = Layout(Shape([]), Stride([]))
        for b in B:
            if b.is_leaf:
                if int(b.shape) == 1:
                    layout = Layout.append(layout, Layout(b.shape, Stride(0)))
                elif A.is_leaf:
                    layout = Layout.append(
                        layout, Layout(b.shape, Stride(int(b.stride) * int(A.stride)))
                    )
                else:
                    layout = Layout.append(
                        layout,
                        Layout.coalesce(
                            Layout.modout(Layout.choose(A, int(b.stride)), int(b.shape))
                        ),
                    )
            else:
                layout = Layout.append(layout, Layout.compose_layouts(A, b))
        if len(layout) == 1:
            return layout[0]
        return layout

    @staticmethod
    def _ordinal_scale(layout: Layout) -> int | None:
        radix = 1
        scale: int | None = None
        for extent, stride in layout.infix():
            if extent > 1:
                if scale is None:
                    scale = stride
                elif stride != scale * radix:
                    return None
            radix *= extent
        return 0 if scale is None else scale

    @staticmethod
    def _scale_stride_level(level: _StrideLevel, scale: int) -> list[Any]:
        return [
            element * scale
            if isinstance(element, int)
            else Layout._scale_stride_level(element, scale)
            for element in level
        ]

    @staticmethod
    def _is_structurally_compact(layout: Layout) -> bool:
        covered = 1
        for extent, stride in sorted(layout.infix(), key=lambda mode: mode[1]):
            if extent == 1:
                continue
            if stride != covered:
                return False
            covered *= extent
        return covered == layout.size

    @staticmethod
    def _legacy_choose_shapes(A: Layout, stride: int) -> list[int] | None:
        if stride <= 0 or A.size % stride != 0:
            return None

        chosen_shapes: list[int] = []
        remaining = stride
        for element in A:
            extent = int(element.shape)
            if remaining == 1:
                chosen_shapes.append(extent)
            elif extent > remaining:
                if extent % remaining != 0:
                    return None
                chosen_shapes.append(extent // remaining)
                remaining = 1
            else:
                if remaining % extent != 0:
                    return None
                chosen_shapes.append(1)
                remaining //= extent
        return chosen_shapes if remaining == 1 else None

    @staticmethod
    def _legacy_leaf_is_representable(A: Layout, leaf: Layout) -> bool:
        extent = int(leaf.shape)
        if extent == 1 or A.is_leaf:
            return True

        chosen_shapes = Layout._legacy_choose_shapes(A, int(leaf.stride))
        if chosen_shapes is None:
            return False

        covered = 1
        for chosen_extent in chosen_shapes:
            if covered == extent:
                continue
            if chosen_extent * covered <= extent:
                covered *= chosen_extent
            else:
                if extent % covered != 0:
                    return False
                covered = extent
        return covered == extent

    @staticmethod
    def _legacy_lowering_is_representable(A: Layout, B: Layout) -> bool:
        if any(not element.is_leaf for element in A):
            return False

        return all(
            Layout._legacy_leaf_is_representable(A, leaf)
            for leaf in Layout._prefix_layout_leaves(B)
        )

    @staticmethod
    def _shape_refines_inner_coordinates(
        inner: _ShapeLevel,
        candidate: _ShapeLevel,
    ) -> bool:
        if len(inner) != len(candidate):
            return False
        for inner_element, candidate_element in zip(inner, candidate, strict=True):
            if isinstance(inner_element, int):
                candidate_size = (
                    candidate_element
                    if isinstance(candidate_element, int)
                    else candidate_element.logical_size
                )
                if candidate_size != inner_element:
                    return False
            elif isinstance(candidate_element, int) or not (
                Layout._shape_refines_inner_coordinates(
                    inner_element,
                    candidate_element,
                )
            ):
                return False
        return True

    def compose(  # type: ignore[reportIncompatibleMethodOverride]
        A: Layout,  # type: ignore[reportSelfClsParameterName]
        B: Layout | Shape | Tiler = _MISSING_COMPOSE_OPERAND,
        *,
        inner: IndexMap | Shape | Tiler = _MISSING_COMPOSE_OPERAND,
    ) -> IndexMap:
        if not isinstance(A, Layout):
            raise TypeError("A must be a Layout")
        if B is _MISSING_COMPOSE_OPERAND:
            if inner is _MISSING_COMPOSE_OPERAND:
                raise TypeError("compose() missing required inner operand")
            operand = inner
        elif inner is not _MISSING_COMPOSE_OPERAND:
            raise TypeError("compose() received both B and inner")
        else:
            operand = B

        if isinstance(operand, IndexMap):
            return IndexMap.compose(A, operand)

        tiler: Sequence[Layout]
        if isinstance(operand, Shape):
            tiler = tuple(Layout(element, Stride(1)) for element in operand)
        elif isinstance(operand, Sequence):
            if any(not isinstance(tile, Layout) for tile in operand):
                raise TypeError("B must contain only Layout values")
            tiler = operand
        else:
            raise TypeError("B must be an IndexMap, Shape, or Tiler")
        if len(tiler) > len(A):
            raise ValueError("B has more tiles than A has top-level modes")

        result = Layout(Shape(), Stride())
        for A_el, tile in zip(A[0 : len(tiler)], tiler, strict=True):
            to_append = Layout.compose_layouts(A_el, tile)
            result = Layout.append(result, to_append)
        for A_el in A[len(tiler) :]:
            result = Layout.append(result, A_el)
        return result

    def _compose(self, inner: IndexMap) -> IndexMap:
        if not isinstance(inner, Layout):
            return _compose_generic(self, inner)

        scale = Layout._ordinal_scale(self)
        if scale is not None:
            return Layout(
                inner.shape,
                Stride(Layout._scale_stride_level(inner.stride.top_level, scale)),
            )

        if Layout._is_structurally_compact(
            inner
        ) and Layout._legacy_lowering_is_representable(self, inner):
            candidate = Layout.compose_layouts(self, inner)
            if Layout._shape_refines_inner_coordinates(
                inner.shape.top_level,
                candidate.shape.top_level,
            ):
                return candidate

        return _compose_generic(self, inner)

    def _composition_injectivity(self) -> bool | None:
        if self.size > self.cosize:
            return False

        modes = [mode for mode in self.infix() if mode[0] > 1]
        if any(stride == 0 for _, stride in modes):
            return False

        for position, (left_extent, left_stride) in enumerate(modes):
            for right_extent, right_stride in modes[position + 1 :]:
                stride_gcd = gcd(left_stride, right_stride)
                if (
                    right_stride // stride_gcd < left_extent
                    and left_stride // stride_gcd < right_extent
                ):
                    return False

        span = 1
        for extent, stride in sorted(modes, key=lambda mode: mode[1]):
            if stride == span - 1:
                return False
            if stride < span:
                return None
            span += (extent - 1) * stride
        return True

    @staticmethod
    def leaf(s: int, d: int) -> Layout:
        return Layout(Shape(s), Stride(d))

    @staticmethod
    def complement(A: Layout, cotarget: int) -> Layout:
        if not A.is_injective:
            raise ValueError(f"Layout {A}, overlaps with itself")

        traversal = sorted(A.infix(), key=lambda x: x[1])
        shape = []
        stride = []

        if traversal[0][1] != 1:
            shape.append(traversal[0][1])
            stride.append(1)

        for i in range(len(traversal) - 1):
            if traversal[i][0] * traversal[i][1] == traversal[i + 1][1]:
                continue
            new_stride = traversal[i][0] * traversal[i][1]
            if new_stride >= traversal[i + 1][1]:
                raise ValueError(f"Layout {A}, overlaps with itself")
            if traversal[i + 1][1] % new_stride != 0:
                raise ValueError(f"Layout {A} is incongruent")
            new_shape = traversal[i + 1][1] // new_stride
            shape.append(new_shape)
            stride.append(new_stride)

        if traversal[-1][0] * traversal[-1][1] != cotarget:
            new_stride = traversal[-1][0] * traversal[-1][1]
            if new_stride >= cotarget:
                raise ValueError(f"Layout {A} is larger than cotarget {cotarget}")
            if cotarget % new_stride != 0:
                raise ValueError(f"Layout {A} is incongruent with cotarget {cotarget}")
            new_shape = cotarget // new_stride
            shape.append(new_shape)
            stride.append(new_stride)

        return Layout(Shape(shape), Stride(stride))

    @staticmethod
    def make_layout(*args: Layout) -> Layout:
        new_layout = Layout(Shape(), Stride())
        for a in args:
            if not isinstance(a, Layout):
                raise ValueError(
                    "function make_layout only accepts Layouts as argument"
                )
            new_layout = Layout.append(new_layout, a)
        return new_layout

    @staticmethod
    def divide(A: Layout, B: Layout) -> Layout:
        inner = Layout.make_layout(B, Layout.complement(B, A.size))
        return Layout.compose_layouts(A, inner)

    @staticmethod
    def divide_tiler(A: Layout, B: Tiler) -> Layout:
        tiler = []
        for a, b in zip(A[0 : len(B)], B, strict=True):
            tiler.append(Layout.make_layout(b, Layout.complement(b, a.size)))
        return cast(Layout, Layout.compose(A, tiler))

    @staticmethod
    def zipped_divide(A: Layout, B: Tiler) -> Layout:
        tiler = []
        for a, b in zip(A[0 : len(B)], B, strict=True):
            tiler.append(Layout.make_layout(b, Layout.complement(b, a.size)))
        unzipped = cast(Layout, Layout.compose(A, tiler))
        zipped = Layout.empty()
        tiles = []
        rest = []
        for uz in unzipped[0 : len(B)]:
            tiles.append(uz[0])
            rest.append(uz[1])
        zipped = Layout.append(zipped, Layout.make_layout(*tiles))
        zipped = Layout.append(zipped, Layout.make_layout(*rest))
        zipped = Layout.concat(zipped, unzipped[len(B) :])
        return zipped
