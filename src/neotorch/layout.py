from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any


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

    def __new__(cls, *args: int | Node | _NodeLeaf | Tree) -> Tree:
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

    def __new__(cls, iterable: Iterable[int | _ShapeLevel] = ()) -> _ShapeLevel:
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

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "top_level":
            raise AttributeError("top_level field is immutable")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return f"Shape<{self.top_level!r}>"

    def __str__(self) -> str:
        return f"Shape<{self.top_level}>"


class _StrideLevel(tuple):
    def __new__(cls, iterable: Iterable[int | _StrideLevel] = ()) -> _StrideLevel:
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

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "top_level":
            raise AttributeError("top_level field is immutable")
        object.__setattr__(self, name, value)

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


class Layout:
    """Hierarchical shape and stride pair for logical-to-physical indexing."""

    _cache: Any

    def __init__(self, shape: Shape, stride: Stride):
        if not isinstance(shape, Shape):
            raise ValueError("shape input must be a Shape object")
        if not isinstance(stride, Stride):
            raise ValueError("stride input must be a Stride object")
        if not Layout.check_tree(shape.top_level, stride.top_level):
            raise ValueError("Shape and Stride do not match in Structure")
        self.shape = shape
        self.stride = stride
        self._cache = import_module("neotorch._index")._LayoutCache(self)

    @staticmethod
    def check_tree(shape: _ShapeLevel, stride: _StrideLevel) -> bool:
        if len(shape) != len(stride):
            return False
        for sh, st in zip(shape, stride):
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
        return layout._cache.get_index(key)

    @staticmethod
    def _get_index_levels(shape: _ShapeLevel, stride: _StrideLevel, key: Any) -> int:
        if len(shape) != len(stride):
            raise ValueError("Shape and Stride Lengths do not match")

        if isinstance(key, int):
            curr_key = Layout.expand_int(key, shape)
        else:
            curr_key = key

        idx = 0
        for sh, stride_value, k in zip(shape, stride, curr_key):
            if isinstance(sh, int):
                if k < 0 or k >= sh:
                    raise ValueError("Key is not in domain of shape")
                idx += stride_value * k
            else:
                idx += Layout._get_index_levels(sh, stride_value, k)

        return idx

    def index(self, key: Any) -> int:
        return Layout.get_index(self, key)

    def __eq__(self, other: object):
        if not isinstance(other, Layout):
            return NotImplemented
        return other.shape == self.shape and other.stride == self.stride

    def __len__(self) -> int:
        return len(self.shape.top_level)

    def __call__(self, key: int | Any) -> int:
        return Layout.get_index(self, key)

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
        for node, marker in zip(layout, profile):
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
        for sh, st in zip(shape, stride):
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
                if A.is_leaf:
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
                layout = Layout.append(layout, Layout.compose(A, b))
        if len(layout) == 1:
            return layout[0]
        return layout

    @staticmethod
    def compose(A: Layout, B: Layout | Shape | Any) -> Layout:
        if isinstance(B, Layout):
            return Layout.compose_layouts(A, B)
        tiler = []
        if isinstance(B, Shape):
            for el in B:
                tiler.append(Layout(el, Stride(1)))
        else:
            tiler = B

        result = Layout(Shape(), Stride())
        for A_el, tile in zip(A[0 : len(tiler)], tiler):
            to_append = Layout.compose(A_el, tile)
            result = Layout.append(result, to_append)
        for A_el in A[len(tiler) :]:
            result = Layout.append(result, A_el)
        return result

    @staticmethod
    def leaf(s: int, d: int) -> Layout:
        return Layout(Shape(s), Stride(d))

    @staticmethod
    def complement(A: Layout, cotarget: int) -> Layout:
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
        return Layout.compose(A, Layout.make_layout(B, Layout.complement(B, A.size)))

    @staticmethod
    def divide_tiler(A: Layout, B: list[Layout]) -> Layout:
        tiler = []
        for a, b in zip(A[0 : len(B)], B):
            tiler.append(Layout.make_layout(b, Layout.complement(b, a.size)))
        return Layout.compose(A, tiler)

    @staticmethod
    def zipped_divide(A: Layout, B: list[Layout]) -> Layout:
        tiler = []
        for a, b in zip(A[0 : len(B)], B):
            tiler.append(Layout.make_layout(b, Layout.complement(b, a.size)))
        unzipped = Layout.compose(A, tiler)
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
