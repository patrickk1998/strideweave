"""Focused tests for the isolated Generic ``conv_general`` operation."""

from __future__ import annotations

import math

import pytest

from strideweave import DType, Generic, Layout, Shape, Stride, Tensor
from strideweave.carriers.generic.convolution_ops import GenericConvGeneralOperation
from strideweave.carriers.operation_helpers import _canonical_layout_for_shape


def _tensor(values: list[float], shape: object) -> Tensor:
    normalized_shape = Shape(shape)
    return Tensor(
        Generic(values, dtype=DType.Float32),
        0,
        _canonical_layout_for_shape(normalized_shape),
    )


def _values(tensor: Tensor) -> list[object]:
    return [tensor[index] for index in range(tensor.size())]


def _tensor_with_layout(values: list[float], layout: Layout) -> Tensor:
    """Build a Generic Float32 tensor by placing values through ``layout``."""
    physical = [0.0] * layout.cosize
    for logical_index, value in enumerate(values):
        physical[layout.index(logical_index)] = value
    return Tensor(Generic(physical, dtype=DType.Float32), 0, layout)


def test_conv_general_matches_grouped_one_dimensional_cross_correlation() -> None:
    # First-mode-fast values: four input channels, and two output channels in
    # each of two feature groups.  Every output has three valid positions.
    lhs = _tensor(
        [
            1,
            2,
            3,
            4,
            2,
            3,
            4,
            5,
            3,
            4,
            5,
            6,
            4,
            5,
            6,
            7,
            5,
            6,
            7,
            8,
        ],
        [1, 4, 5],
    )
    kernel = _tensor(
        [
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            1,
            1,
            2,
            1,
            2,
            1,
            1,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            2,
        ],
        [4, 2, 3],
    )

    result = GenericConvGeneralOperation().forward(
        lhs, kernel, [1], [(0, 0)], None, None, 2
    )

    assert result.layout.shape == Shape([1, 4, 3])
    assert _values(result) == [
        17.0,
        19.0,
        32.0,
        37.0,
        24.0,
        27.0,
        39.0,
        45.0,
        31.0,
        35.0,
        46.0,
        53.0,
    ]


def test_conv_general_normalizes_roles_and_supports_dilation_and_padding() -> None:
    # Source modes are spatial, batch, feature rather than canonical.  The
    # input dilation inserts a zero between input samples and the one-cell
    # padding produces the expected six output positions.
    lhs = _tensor([1, 2, 3], [3, 1, 1])
    kernel = _tensor([2, 1], [2, 1, 1])
    result = GenericConvGeneralOperation().forward(
        lhs,
        kernel,
        [1],
        [(1, 1)],
        [2],
        [1],
        1,
        (1, 2, 0),
        (1, 2, 0),
        (0, 1, 2),
    )

    assert result.layout.shape == Shape([1, 1, 6])
    assert _values(result) == [1.0, 2.0, 2.0, 4.0, 3.0, 6.0]


def test_conv_general_uses_sequential_binary32_and_explicit_padding_zero() -> None:
    half_ulp = 2.0**-24
    lhs = _tensor([1.0, half_ulp, half_ulp], [1, 1, 3])
    kernel = _tensor([1.0, 1.0, 1.0], [1, 1, 3])
    result = GenericConvGeneralOperation().forward(lhs, kernel, [1], [(0, 0)])

    # The first + half-ulp rounds back to one, then the second half-ulp also
    # rounds back to one.  A binary64 accumulator would produce 1 + 2**-23.
    assert result[0] == 1.0

    inf_kernel = _tensor([math.inf], [1, 1, 1])
    padded = GenericConvGeneralOperation().forward(
        _tensor([2.0], [1, 1, 1]), inf_kernel, [1], [(1, 1)]
    )
    assert math.isnan(padded[0])
    assert padded[1] == math.inf
    assert math.isnan(padded[2])


def test_conv_general_vjp_accumulates_in_forward_coordinate_order() -> None:
    lhs = _tensor([1, 2, 3], [1, 1, 3])
    kernel = _tensor([2, 1], [1, 1, 2])
    operation = GenericConvGeneralOperation()
    result = operation.forward(lhs, kernel, [1], [(0, 0)])
    result.backward(_tensor([1, 1], [1, 1, 2]))

    lhs_gradient = lhs.grad
    kernel_gradient = kernel.grad
    assert lhs_gradient is not None
    assert kernel_gradient is not None
    assert _values(lhs_gradient) == [2.0, 3.0, 1.0]
    assert _values(kernel_gradient) == [3.0, 5.0]
    assert lhs_gradient.layout == _canonical_layout_for_shape(lhs.layout.shape)
    assert kernel_gradient.layout == _canonical_layout_for_shape(kernel.layout.shape)


@pytest.mark.parametrize(
    ("lhs_layout", "kernel_layout", "expected_lhs_layout", "expected_kernel_layout"),
    [
        (
            Layout(Shape([1, 1, 3]), Stride([1, 1, 2])),
            Layout(Shape([1, 1, 2]), Stride([1, 1, 2])),
            Layout(Shape([1, 1, 3]), Stride([1, 1, 2])),
            Layout(Shape([1, 1, 2]), Stride([1, 1, 2])),
        ),
        (
            Layout(Shape([1, 1, 3]), Stride([1, 1, 0])),
            Layout(Shape([1, 1, 2]), Stride([1, 1, 0])),
            _canonical_layout_for_shape(Shape([1, 1, 3])),
            _canonical_layout_for_shape(Shape([1, 1, 2])),
        ),
    ],
    ids=("injective-gapped", "noninjective-broadcast"),
)
def test_conv_general_vjp_preserves_or_canonicalizes_operand_layouts(
    lhs_layout: Layout,
    kernel_layout: Layout,
    expected_lhs_layout: Layout,
    expected_kernel_layout: Layout,
) -> None:
    lhs = _tensor_with_layout([1.0, 2.0, 3.0], lhs_layout)
    kernel = _tensor_with_layout([2.0, 3.0], kernel_layout)
    operation = GenericConvGeneralOperation()
    result = operation.forward(lhs, kernel, [1], [(0, 0)])

    result.backward(_tensor([1.0, 1.0], Shape([1, 1, 2])))

    lhs_gradient = lhs.grad
    kernel_gradient = kernel.grad
    assert lhs_gradient is not None
    assert kernel_gradient is not None
    assert lhs_gradient.layout == expected_lhs_layout
    assert kernel_gradient.layout == expected_kernel_layout


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (([0], [(0, 0)]), "strides must contain positive"),
        (([1, 1], [(0, 0)]), "strides must have one entry"),
        (([1], [(-1, 0)]), "padding values must be non-negative"),
    ],
)
def test_conv_general_rejects_invalid_spatial_configuration(
    arguments: tuple[object, object], message: str
) -> None:
    lhs = _tensor([1, 2, 3], [1, 1, 3])
    kernel = _tensor([1, 1], [1, 1, 2])
    with pytest.raises((TypeError, ValueError), match=message):
        GenericConvGeneralOperation().forward(lhs, kernel, *arguments)


def test_conv_general_rejects_bad_roles_groups_and_dtype() -> None:
    lhs = _tensor([1, 2, 3], [1, 1, 3])
    kernel = _tensor([1, 1], [1, 1, 2])
    integer = Tensor(
        Generic([1, 2, 3], dtype=DType.Int32),
        0,
        _canonical_layout_for_shape(Shape([1, 1, 3])),
    )

    with pytest.raises(TypeError, match="lhs must have dtype"):
        GenericConvGeneralOperation().forward(integer, kernel, [1], [(0, 0)])
    with pytest.raises(ValueError, match="permutation"):
        GenericConvGeneralOperation().forward(
            lhs, kernel, [1], [(0, 0)], None, None, 1, (0, 0, 0)
        )
    with pytest.raises(ValueError, match="feature_groups must be positive"):
        GenericConvGeneralOperation().forward(lhs, kernel, [1], [(0, 0)], None, None, 0)
    with pytest.raises(ValueError, match="divisible"):
        GenericConvGeneralOperation().forward(
            _tensor([1, 2, 3, 4], [1, 2, 2]),
            _tensor([1, 1], [1, 1, 2]),
            [1],
            [(0, 0)],
            None,
            None,
            3,
        )


def test_convolution_addition_does_not_regress_matmul_or_einsum() -> None:
    import strideweave as sw

    lhs = _tensor([1, 2, 3, 4], [2, 2])
    rhs = _tensor([2, 1, 1, 2], [2, 2])
    expected = [5.0, 8.0, 7.0, 10.0]

    assert _values(sw.matmul(lhs, rhs)) == expected
    assert _values(sw.einsum(lhs, rhs, "a b, c b -> a c")) == expected
