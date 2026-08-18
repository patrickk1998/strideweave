from typing import Any, cast

import pytest

from strideweave.core.index_map import IndexMap, _compose_generic
from strideweave.core.layout import Shape


class _TestIndexMap(IndexMap):
    _values: tuple[int, ...]

    def __init__(
        self,
        shape: Shape,
        codomain_size: int,
        is_injective: bool | None,
        values: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(shape, codomain_size, is_injective)
        object.__setattr__(
            self,
            "_values",
            tuple(range(shape.size)) if values is None else values,
        )

    def _index_ordinal(self, index: int) -> int:
        return self._values[index]

    def _compose(self, inner: IndexMap) -> IndexMap:
        values = tuple(self.index(inner.index(index)) for index in range(inner.size))
        if inner.is_injective is False:
            is_injective = False
        elif self.is_injective is True and inner.is_injective is True:
            is_injective = True
        else:
            is_injective = None
        return _TestIndexMap(
            inner.shape,
            self.codomain_size,
            is_injective,
            values,
        )


class _GenericTestIndexMap(_TestIndexMap):
    def _compose(self, inner: IndexMap) -> IndexMap:
        return _compose_generic(self, inner)


class _CountingGenericTestIndexMap(_GenericTestIndexMap):
    _calls: list[tuple[str, int]]
    _label: str

    def __init__(
        self,
        label: str,
        calls: list[tuple[str, int]],
        shape: Shape,
        codomain_size: int,
        is_injective: bool | None,
        values: tuple[int, ...],
    ) -> None:
        super().__init__(shape, codomain_size, is_injective, values)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_calls", calls)

    def _index_ordinal(self, index: int) -> int:
        self._calls.append((self._label, index))
        return super()._index_ordinal(index)


def test_index_map_reports_immutable_domain_and_codomain_metadata():
    index_map = _TestIndexMap(Shape(2), 10, True, (4, 3))

    assert index_map.shape == Shape(2)
    assert index_map.size == 2
    assert index_map.codomain_size == 10
    assert index_map.is_injective is True
    assert index_map.index(0) == 4
    assert index_map.index(1) == 3


@pytest.mark.parametrize("is_injective", [True, False, None])
def test_index_map_preserves_tri_state_injectivity(is_injective):
    index_map = _TestIndexMap(Shape(2), 2, is_injective)

    assert index_map.is_injective is is_injective


def test_index_map_index_and_call_share_shape_coordinate_normalization():
    index_map = _TestIndexMap(Shape(5, [20, 4]), 400, True)

    for key in (214, (4, 42), (4, (2, 2)), [4, [2, 2]]):
        assert index_map.index(key) == 214
        assert index_map(key) == 214


def test_index_map_evaluation_rejects_wrong_key_kinds_and_arities():
    index_map = _TestIndexMap(Shape(2), 2, True)
    index = cast(Any, index_map.index)
    call = cast(Any, index_map)

    for key in (None, 1.0, "key"):
        with pytest.raises(TypeError):
            index_map.index(key)
        with pytest.raises(TypeError):
            index_map(key)
    with pytest.raises(TypeError):
        index()
    with pytest.raises(TypeError):
        index(0, 1)
    with pytest.raises(TypeError):
        call()
    with pytest.raises(TypeError):
        call(0, 1)


def test_index_map_rejects_direct_abstract_construction():
    index_map_type = cast(Any, IndexMap)

    with pytest.raises(TypeError, match="abstract"):
        index_map_type(Shape(1), 1, True)


@pytest.mark.parametrize(
    ("shape", "codomain_size", "is_injective", "error"),
    [
        (object(), 1, True, TypeError),
        (Shape(1), 1.0, True, TypeError),
        (Shape(1), 0, True, ValueError),
        (Shape(1), 1, 1, TypeError),
    ],
)
def test_index_map_rejects_invalid_metadata(shape, codomain_size, is_injective, error):
    with pytest.raises(error):
        _TestIndexMap(shape, codomain_size, is_injective)


def test_index_map_rejects_public_semantic_field_assignment_and_deletion():
    index_map = _TestIndexMap(Shape(2), 2, True)

    for name, value in (
        ("shape", Shape(1)),
        ("size", 1),
        ("codomain_size", 1),
        ("is_injective", False),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(index_map, name, value)
    with pytest.raises(AttributeError, match="immutable"):
        delattr(index_map, "shape")

    assert index_map.shape == Shape(2)
    assert index_map.size == 2
    assert index_map.codomain_size == 2
    assert index_map.is_injective is True


def test_index_map_composition_validates_type_bound_and_arity_before_lowering():
    outer = _TestIndexMap(Shape(10), 12, True)
    inner = _TestIndexMap(Shape(2), 8, True, (4, 3))
    oversized = _TestIndexMap(Shape(2), 11, True, (4, 3))
    compose = cast(Any, outer.compose)

    result = outer.compose(inner)

    assert result.shape == inner.shape
    assert result.codomain_size == outer.codomain_size
    assert result.index(0) == outer(inner(0))
    assert result.index(1) == outer(inner(1))
    with pytest.raises(TypeError):
        outer.compose(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inner codomain exceeds"):
        outer.compose(oversized)
    with pytest.raises(TypeError):
        compose()
    with pytest.raises(TypeError):
        compose(inner, inner)


def test_generic_composition_flattens_an_immutable_inner_to_outer_chain():
    inner = _GenericTestIndexMap(Shape(2), 2, False, (0, 0))
    middle = _GenericTestIndexMap(Shape(2), 3, True, (1, 0))
    outer = _GenericTestIndexMap(Shape(3), 5, None, (4, 2, 1))

    result = outer.compose(middle.compose(inner))
    maps = cast(Any, result)._maps

    assert maps == (inner, middle, outer)
    assert result.shape == inner.shape
    assert result.codomain_size == outer.codomain_size
    assert result.is_injective is False
    assert [result(index) for index in range(result.size)] == [2, 2]
    with pytest.raises(AttributeError, match="immutable"):
        result._maps = ()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match="immutable"):
        del result._maps  # type: ignore[attr-defined]


def test_generic_composition_infers_true_and_unknown_injectivity_conservatively():
    inner = _GenericTestIndexMap(Shape(2), 2, True, (1, 0))
    injective_outer = _GenericTestIndexMap(Shape(2), 3, True, (2, 1))
    unknown_outer = _GenericTestIndexMap(Shape(3), 4, None, (3, 2, 1))

    injective_result = injective_outer.compose(inner)
    unknown_result = unknown_outer.compose(injective_result)

    assert injective_result.is_injective is True
    assert unknown_result.is_injective is None
    assert cast(Any, unknown_result)._maps == (inner, injective_outer, unknown_outer)


def test_generic_composition_does_not_evaluate_children_during_construction():
    calls: list[tuple[str, int]] = []
    inner = _CountingGenericTestIndexMap(
        "inner",
        calls,
        Shape(2),
        2,
        True,
        (1, 0),
    )
    outer = _CountingGenericTestIndexMap(
        "outer",
        calls,
        Shape(2),
        3,
        True,
        (2, 1),
    )

    result = outer.compose(inner)

    assert calls == []
    assert result(0) == 1
    assert calls == [("inner", 0), ("outer", 1)]
