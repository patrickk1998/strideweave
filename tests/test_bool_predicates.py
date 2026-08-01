"""Bool storage and Generic predicate semantics."""

from __future__ import annotations

import math

import pytest

from strideweave import CPU, DType, Evictable, Generic, Layout, Shape, Stride, Tensor
from strideweave.carriers.generic.ops import GenericAddOperation
from strideweave.carriers.generic.predicate_ops import (
    GenericEqOperation,
    GenericLeOperation,
    GenericLogicalNotOperation,
    GenericLtOperation,
    GenericNeOperation,
)

ONE_MODE = Layout(Shape(4), Stride(1))


def generic_tensor(values, *, dtype=DType.Float32, layout=ONE_MODE):
    return Tensor(Generic(values, dtype=dtype), 0, layout)


def values_of(tensor):
    return [tensor[i] for i in range(tensor.size())]


def test_bool_storage_is_strict_and_initialized_false():
    carrier = Generic([True, False], dtype=DType.Bool)

    assert [carrier[i] for i in range(2)] == [True, False]
    assert carrier.allocate_like(2)[0] is False

    with pytest.raises(TypeError, match="must be a bool"):
        Generic([1], dtype=DType.Bool)
    with pytest.raises(TypeError, match="must be a bool"):
        carrier[0] = 1


def test_bool_tensor_indexing_and_mutation_preserve_logical_values():
    layout = Layout(Shape(2), Stride(1))
    tensor = generic_tensor([True, False], dtype=DType.Bool, layout=layout)

    assert tensor[0] is True
    tensor[1] = True
    assert values_of(tensor) == [True, True]
    with pytest.raises(TypeError, match="must be a bool"):
        tensor[0] = 0


def test_cpu_bool_storage_uses_normalized_bytes_and_strict_values():
    carrier = CPU(2, dtype=DType.Bool)

    carrier[0] = True
    carrier[1] = False

    assert carrier[0] is True
    assert carrier[1] is False
    assert carrier.allocate_like(1, dtype=DType.Bool)[0] is False
    with pytest.raises(TypeError, match="must be a bool"):
        carrier[0] = 1


def test_cpu_bool_scatter_copies_bytes_and_rejects_numeric_mixing():
    layout = Layout(Shape(2), Stride(1))
    destination_layout = Layout(Shape(4), Stride(1))
    source_carrier = CPU(2, dtype=DType.Bool)
    source_carrier[0] = True
    source_carrier[1] = False
    source = Tensor(source_carrier, 0, layout)
    destination_carrier = CPU(4, dtype=DType.Bool)
    destination = Tensor(destination_carrier, 0, destination_layout)

    destination_carrier.scatter(source, destination, layout, 1)

    assert [destination_carrier[i] for i in range(4)] == [False, True, False, False]

    numeric_source = Tensor(CPU(2, dtype=DType.Float32), 0, layout)
    with pytest.raises(TypeError, match="non-Bool"):
        destination_carrier.scatter(numeric_source, destination, layout, 0)


def test_evictable_composition_requires_and_preserves_bool_storage_support():
    carrier = Evictable(
        Generic([True], dtype=DType.Bool),
        Generic([False], dtype=DType.Bool),
    )

    assert carrier.supports_storage_dtype(DType.Bool)
    assert carrier[0] is True
    carrier.evict()
    carrier.promote()
    assert carrier[0] is True


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (GenericEqOperation, [True, False, False, False]),
        (GenericNeOperation, [False, True, True, True]),
        (GenericLtOperation, [False, True, False, False]),
        (GenericLeOperation, [True, True, False, False]),
    ],
)
def test_float32_predicates_return_bool_and_no_autograd_graph(operation, expected):
    lhs = generic_tensor([0.0, -0.0, 1.0, float("nan")])
    rhs = generic_tensor([0.0, 1.0, 0.0, float("nan")])

    result = operation().forward(lhs, rhs)

    assert result.dtype() is DType.Bool
    assert values_of(result) == expected
    assert result.autograd_ctx is None
    assert not result.is_differentiable()
    with pytest.raises(RuntimeError, match="non-differentiable"):
        result.backward()


def test_predicate_structural_broadcasting_expands_singleton_mode():
    lhs = generic_tensor([0.0, 2.0], layout=Layout(Shape(2), Stride(1)))
    singleton = generic_tensor([1.0], layout=Layout(Shape(1), Stride(1)))

    result = GenericLtOperation().forward(lhs, singleton)

    assert values_of(result) == [True, False]
    assert result.layout.shape == Shape(2)


def test_predicates_are_float32_only():
    layout = Layout(Shape(2), Stride(1))
    lhs = generic_tensor([1, 2], dtype=DType.Int32, layout=layout)
    rhs = generic_tensor([1.0, 2.0], layout=layout)

    with pytest.raises(TypeError, match=r"DType\.Float32"):
        GenericEqOperation().forward(lhs, rhs)


def test_logical_not_uses_zero_semantics_and_nan_is_true_input():
    tensor = generic_tensor(
        [0.0, -0.0, 1.0, -2.0, float("nan"), float("inf")],
        layout=Layout(Shape(6), Stride(1)),
    )

    result = GenericLogicalNotOperation().forward(tensor)

    assert values_of(result) == [True, True, False, False, False, False]
    assert math.isnan(tensor[4])
    assert result.dtype() is DType.Bool
    assert result.autograd_ctx is None


def test_logical_not_rejects_non_float32_storage():
    tensor = generic_tensor(
        [True], dtype=DType.Bool, layout=Layout(Shape(1), Stride(1))
    )

    with pytest.raises(TypeError, match=r"DType\.Float32"):
        GenericLogicalNotOperation().forward(tensor)


def test_logical_not_materializes_a_canonical_injective_layout():
    source_layout = Layout(Shape(1), Stride(1))
    source = generic_tensor([0.0], layout=source_layout)
    broadcast = source.carrier.dispatch_op("broadcast_to").forward(source, Shape(3))

    result = GenericLogicalNotOperation().forward(broadcast)

    assert values_of(result) == [True, True, True]
    assert result.layout.is_injective
    assert result.layout.stride == Stride(1)


def test_bool_does_not_enter_generic_numeric_arithmetic():
    layout = Layout(Shape(1), Stride(1))
    tensor = generic_tensor([True], dtype=DType.Bool, layout=layout)

    with pytest.raises(NotImplementedError, match=r"DType\.Bool"):
        GenericAddOperation().forward(tensor, tensor)
