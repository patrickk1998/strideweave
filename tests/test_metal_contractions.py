from __future__ import annotations

import math
from typing import Any

import pytest

import strideweave as sw
from strideweave.carriers.base import Carrier
from strideweave.carriers.dtype import SimpleDType
from strideweave.carriers.metal import _contraction_executor as executor
from strideweave.carriers.metal import contraction_ops
from strideweave.carriers.metal.contraction_ops import MetalContractionOperation
from strideweave.carriers.operation_capability import UnsupportedOperationPlan
from strideweave.carriers.operation_helpers import _canonical_layout_for_shape

pytestmark = pytest.mark.metal


def _tensor_from_storage(
    storage: list[Any],
    layout: sw.Layout,
    *,
    metal: bool,
    offset: int = 0,
    dtype: sw.DType = sw.DType.Float32,
) -> sw.Tensor:
    if metal:
        carrier = sw.Metal(len(storage), dtype=dtype)
        for index, value in enumerate(storage):
            carrier[index] = value
    else:
        carrier = sw.Generic(storage, dtype=dtype)
    return sw.Tensor(carrier, offset, layout)


def _tensor_from_logical(
    values: list[Any],
    layout: sw.Layout,
    *,
    metal: bool,
    offset: int = 0,
    dtype: sw.DType = sw.DType.Float32,
) -> sw.Tensor:
    storage = [-91 if dtype is sw.DType.Int32 else -91.0] * (offset + layout.cosize)
    for logical_index, value in enumerate(values):
        storage[offset + layout.index(logical_index)] = value
    return _tensor_from_storage(
        storage,
        layout,
        metal=metal,
        offset=offset,
        dtype=dtype,
    )


def _canonical(shape: object) -> sw.Layout:
    return _canonical_layout_for_shape(sw.Shape(shape))


def _values(tensor: sw.Tensor) -> list[Any]:
    return [tensor[index] for index in range(tensor.size())]


def _assert_matches(actual: sw.Tensor, expected: sw.Tensor) -> None:
    assert actual.dtype() is expected.dtype()
    assert actual.layout == expected.layout
    assert _values(actual) == pytest.approx(
        _values(expected), rel=2e-6, abs=2e-6, nan_ok=True
    )


def _pair(
    values: list[Any],
    layout: sw.Layout,
    *,
    offset: int = 0,
    dtype: sw.DType = sw.DType.Float32,
) -> tuple[sw.Tensor, sw.Tensor]:
    return (
        _tensor_from_logical(values, layout, metal=True, offset=offset, dtype=dtype),
        _tensor_from_logical(values, layout, metal=False, offset=offset, dtype=dtype),
    )


def test_each_contraction_dispatch_is_fresh_and_metal_owned() -> None:
    carrier = sw.Metal.__new__(sw.Metal)
    Carrier.__init__(carrier)

    for name in ("matmul", "conv_general"):
        first = carrier.dispatch_op(name)
        second = carrier.dispatch_op(name)
        assert isinstance(first, MetalContractionOperation)
        assert isinstance(second, MetalContractionOperation)
        assert first is not second
        assert first._dispatch_carrier_class is sw.Metal
        assert first._operation_name == name


def test_matmul_public_paths_match_generic_and_preserve_hierarchical_rows() -> None:
    lhs_layout = sw.Layout(sw.Shape([[2, 2], 3]), sw.Stride([[1, 2], 4]))
    rhs_layout = sw.Layout(sw.Shape([[2, 1], 3]), sw.Stride([[1, 2], 2]))
    lhs_m, lhs_g = _pair([float(i) for i in range(1, 13)], lhs_layout)
    rhs_m, rhs_g = _pair([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], rhs_layout)

    _assert_matches(sw.matmul(lhs_m, rhs_m), sw.matmul(lhs_g, rhs_g))
    _assert_matches(lhs_m @ rhs_m, lhs_g @ rhs_g)
    result = sw.matmul(lhs_m, rhs_m)
    assert result.layout.shape == sw.Shape([[2, 2], [2, 1]])
    assert result.layout.is_injective
    assert type(result.carrier) is sw.Metal


@pytest.mark.parametrize(
    ("lhs_layout", "rhs_layout", "lhs_values", "rhs_values"),
    (
        (
            sw.Layout(sw.Shape([2, 3]), sw.Stride([2, 5])),
            sw.Layout(sw.Shape([3, 3]), sw.Stride([3, 7])),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 0.0, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ),
        (
            sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 0])),
            sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 0])),
            [2.0, 3.0, 2.0, 3.0, 2.0, 3.0],
            [5.0, 7.0, 5.0, 7.0, 5.0, 7.0],
        ),
    ),
    ids=("holes-offsets", "stride-zero-contraction"),
)
def test_matmul_address_plans_match_generic(
    lhs_layout: sw.Layout,
    rhs_layout: sw.Layout,
    lhs_values: list[float],
    rhs_values: list[float],
) -> None:
    lhs_m, lhs_g = _pair(lhs_values, lhs_layout, offset=2)
    rhs_m, rhs_g = _pair(rhs_values, rhs_layout, offset=1)

    _assert_matches(sw.matmul(lhs_m, rhs_m), sw.matmul(lhs_g, rhs_g))


def test_matmul_vjps_match_generic_and_preserve_gradient_layouts() -> None:
    lhs_layout = sw.Layout(sw.Shape([2, 3]), sw.Stride([2, 5]))
    rhs_layout = sw.Layout(sw.Shape([3, 3]), sw.Stride([3, 7]))
    lhs_m, lhs_g = _pair([1.0, -2.0, 3.0, 4.0, -5.0, 6.0], lhs_layout, offset=2)
    rhs_m, rhs_g = _pair(
        [2.0, 1.0, -1.0, 3.0, 0.5, 4.0, -2.0, 5.0, 6.0],
        rhs_layout,
        offset=1,
    )
    actual = sw.matmul(lhs_m, rhs_m)
    expected = sw.matmul(lhs_g, rhs_g)
    gradient_values = [1.0, -2.0, 3.0, 0.5, -1.0, 4.0]
    actual.backward(_tensor_from_logical(gradient_values, actual.layout, metal=True))
    expected.backward(
        _tensor_from_logical(gradient_values, expected.layout, metal=False)
    )

    assert lhs_m.grad is not None and lhs_g.grad is not None
    assert rhs_m.grad is not None and rhs_g.grad is not None
    _assert_matches(lhs_m.grad, lhs_g.grad)
    _assert_matches(rhs_m.grad, rhs_g.grad)
    assert lhs_m.grad.layout == lhs_layout
    assert rhs_m.grad.layout == rhs_layout
    assert type(lhs_m.grad.carrier) is sw.Metal
    assert type(rhs_m.grad.carrier) is sw.Metal


def test_matmul_stride_zero_vjp_uses_injective_metal_storage() -> None:
    layout = sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 0]))
    lhs_m, lhs_g = _pair([2.0, 3.0, 2.0, 3.0, 2.0, 3.0], layout)
    rhs_m, rhs_g = _pair([5.0, 7.0, 5.0, 7.0, 5.0, 7.0], layout)
    actual = sw.matmul(lhs_m, rhs_m)
    expected = sw.matmul(lhs_g, rhs_g)
    actual.backward(_tensor_from_logical([1.0] * 4, actual.layout, metal=True))
    expected.backward(_tensor_from_logical([1.0] * 4, expected.layout, metal=False))

    assert lhs_m.grad is not None and lhs_g.grad is not None
    assert rhs_m.grad is not None and rhs_g.grad is not None
    _assert_matches(lhs_m.grad, lhs_g.grad)
    _assert_matches(rhs_m.grad, rhs_g.grad)
    assert lhs_m.grad.layout.is_injective
    assert rhs_m.grad.layout.is_injective


def test_contractions_use_tilelang_without_host_element_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[Any] = []
    original_compile = executor._compile

    def capture(*args: Any, **kwargs: Any) -> Any:
        result = original_compile(*args, **kwargs)
        compiled.append(result)
        return result

    def unexpected_read(_self: sw.Metal, _index: int) -> Any:
        raise AssertionError("Metal contraction performed a host element read")

    executor._JIT_CACHE.clear()
    monkeypatch.setattr(executor, "_compile", capture)
    monkeypatch.setattr(sw.Metal, "get_value", unexpected_read)
    lhs = _tensor_from_logical([1.0, 2.0, 3.0, 4.0], _canonical([2, 2]), metal=True)
    rhs = _tensor_from_logical([2.0, 1.0, 0.0, 3.0], _canonical([2, 2]), metal=True)
    matmul = sw.matmul(lhs, rhs)
    conv_lhs = _tensor_from_logical([1.0, 2.0, 3.0], _canonical([1, 1, 3]), metal=True)
    kernel = _tensor_from_logical([2.0, 1.0], _canonical([1, 1, 2]), metal=True)
    conv = sw.conv_general(conv_lhs, kernel, (1,), ((0, 0),))

    assert matmul.carrier._require_storage().device.type == "mps"  # pyright: ignore[reportAttributeAccessIssue]
    assert conv.carrier._require_storage().device.type == "mps"  # pyright: ignore[reportAttributeAccessIssue]
    assert len(compiled) == 2
    assert all(item.facts.device_source and item.facts.host_source for item in compiled)


def test_matmul_cache_specializes_static_storage_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[Any] = []
    original_compile = executor._compile

    def capture(*args: Any, **kwargs: Any) -> Any:
        result = original_compile(*args, **kwargs)
        compiled.append(result)
        return result

    executor._JIT_CACHE.clear()
    monkeypatch.setattr(executor, "_compile", capture)
    layout = _canonical([1, 2])
    compact_lhs = _tensor_from_storage([1.0, 2.0], layout, metal=True)
    compact_rhs = _tensor_from_storage([3.0, 4.0], layout, metal=True)
    oversized_lhs = _tensor_from_storage(
        [1.0, 2.0, 91.0, 92.0, 93.0], layout, metal=True
    )
    oversized_rhs = _tensor_from_storage([3.0, 4.0, 81.0, 82.0], layout, metal=True)

    assert _values(sw.matmul(compact_lhs, compact_rhs)) == [11.0]
    assert _values(sw.matmul(oversized_lhs, oversized_rhs)) == [11.0]
    assert _values(sw.matmul(compact_lhs, compact_rhs)) == [11.0]
    assert len(compiled) == 2
    assert compiled[0].facts.device_source != compiled[1].facts.device_source


def test_grouped_one_dimensional_convolution_matches_generic() -> None:
    lhs_values = [float(i) for i in range(1, 21)]
    kernel_values = [
        1.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        1.0,
        1.0,
        2.0,
        1.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
    ]
    lhs_m, lhs_g = _pair(lhs_values, _canonical([1, 4, 5]))
    kernel_m, kernel_g = _pair(kernel_values, _canonical([4, 2, 3]))

    actual = sw.conv_general(lhs_m, kernel_m, (1,), ((0, 0),), feature_groups=2)
    expected = sw.conv_general(lhs_g, kernel_g, (1,), ((0, 0),), feature_groups=2)
    _assert_matches(actual, expected)
    assert type(actual.carrier) is sw.Metal


def test_convolution_public_role_permutations_groups_and_dilations_match_generic() -> (
    None
):
    lhs_layout = _canonical([3, 1, 3, 2])
    kernel_layout = _canonical([2, 2, 2, 1])
    lhs_values = [float((i % 7) - 3) for i in range(lhs_layout.size)]
    kernel_values = [float((i % 5) - 2) for i in range(kernel_layout.size)]
    lhs_m, lhs_g = _pair(lhs_values, lhs_layout)
    kernel_m, kernel_g = _pair(kernel_values, kernel_layout)
    kwargs = {
        "lhs_dilation": (2, 1),
        "kernel_dilation": (1, 2),
        "feature_groups": 2,
        "lhs_dims": (1, 3, 0, 2),
        "kernel_dims": (1, 3, 2, 0),
        "output_dims": (2, 0, 3, 1),
    }
    actual = sw.conv_general(
        lhs_m,
        kernel_m,
        (1, 1),
        ((1, 0), (0, 1)),
        **kwargs,
    )
    expected = sw.conv_general(
        lhs_g,
        kernel_g,
        (1, 1),
        ((1, 0), (0, 1)),
        **kwargs,
    )

    _assert_matches(actual, expected)
    assert actual.layout.shape == sw.Shape([5, 1, 2, 2])


def test_direct_convolution_role_permutations_and_vjps_match_generic() -> None:
    lhs_layout = _canonical([3, 1, 2])
    kernel_layout = _canonical([2, 2, 1])
    lhs_m, lhs_g = _pair([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_layout)
    kernel_m, kernel_g = _pair([2.0, 1.0, -1.0, 3.0], kernel_layout)
    arguments = (
        (1,),
        ((1, 1),),
        (2,),
        (1,),
        2,
        (1, 2, 0),
        (1, 2, 0),
        (0, 1, 2),
    )
    actual = lhs_m.carrier.dispatch_op("conv_general").forward(
        lhs_m, kernel_m, *arguments
    )
    expected = lhs_g.carrier.dispatch_op("conv_general").forward(
        lhs_g, kernel_g, *arguments
    )
    _assert_matches(actual, expected)
    gradient_values = [float(i + 1) for i in range(actual.size())]
    actual.backward(_tensor_from_logical(gradient_values, actual.layout, metal=True))
    expected.backward(
        _tensor_from_logical(gradient_values, expected.layout, metal=False)
    )

    assert lhs_m.grad is not None and lhs_g.grad is not None
    assert kernel_m.grad is not None and kernel_g.grad is not None
    _assert_matches(lhs_m.grad, lhs_g.grad)
    _assert_matches(kernel_m.grad, kernel_g.grad)


def test_three_dimensional_convolution_matches_generic() -> None:
    lhs_values = [float(i) for i in range(1, 9)]
    lhs_m, lhs_g = _pair(lhs_values, _canonical([1, 1, 2, 2, 2]))
    kernel_m, kernel_g = _pair([2.0], _canonical([1, 1, 1, 1, 1]))

    _assert_matches(
        sw.conv_general(lhs_m, kernel_m, (1, 1, 1), ((0, 0),) * 3),
        sw.conv_general(lhs_g, kernel_g, (1, 1, 1), ((0, 0),) * 3),
    )


@pytest.mark.parametrize(
    ("lhs_layout", "kernel_layout", "lhs_values", "kernel_values"),
    (
        (
            sw.Layout(sw.Shape([1, 1, 3]), sw.Stride([1, 1, 2])),
            sw.Layout(sw.Shape([1, 1, 2]), sw.Stride([1, 1, 3])),
            [1.0, 2.0, 3.0],
            [2.0, 1.0],
        ),
        (
            sw.Layout(sw.Shape([1, 1, 3]), sw.Stride([1, 1, 0])),
            sw.Layout(sw.Shape([1, 1, 2]), sw.Stride([1, 1, 0])),
            [2.0, 2.0, 2.0],
            [3.0, 3.0],
        ),
    ),
    ids=("holes-offsets", "stride-zero"),
)
def test_convolution_address_plans_match_generic(
    lhs_layout: sw.Layout,
    kernel_layout: sw.Layout,
    lhs_values: list[float],
    kernel_values: list[float],
) -> None:
    lhs_m, lhs_g = _pair(lhs_values, lhs_layout, offset=2)
    kernel_m, kernel_g = _pair(kernel_values, kernel_layout, offset=1)

    _assert_matches(
        sw.conv_general(lhs_m, kernel_m, (1,), ((0, 0),)),
        sw.conv_general(lhs_g, kernel_g, (1,), ((0, 0),)),
    )


def test_convolution_preserves_sequential_binary32_and_padded_ieee_terms() -> None:
    half_ulp = 2.0**-24
    lhs = _tensor_from_logical(
        [1.0, half_ulp, half_ulp], _canonical([1, 1, 3]), metal=True
    )
    kernel = _tensor_from_logical([1.0, 1.0, 1.0], _canonical([1, 1, 3]), metal=True)
    result = sw.conv_general(lhs, kernel, (1,), ((0, 0),))
    assert result[0] == 1.0

    padded = sw.conv_general(
        _tensor_from_logical([2.0], _canonical([1, 1, 1]), metal=True),
        _tensor_from_logical([math.inf], _canonical([1, 1, 1]), metal=True),
        (1,),
        ((1, 1),),
    )
    values = _values(padded)
    assert math.isnan(values[0])
    assert values[1] == math.inf
    assert math.isnan(values[2])


def test_grouped_dilated_convolution_vjps_match_generic() -> None:
    lhs_layout = sw.Layout(sw.Shape([1, 2, 3]), sw.Stride([1, 1, 3]))
    kernel_layout = sw.Layout(sw.Shape([2, 1, 2]), sw.Stride([1, 2, 5]))
    lhs_m, lhs_g = _pair([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_layout, offset=2)
    kernel_m, kernel_g = _pair([2.0, -1.0, 0.5, 3.0], kernel_layout, offset=1)
    arguments = ((1,), ((1, 0),))
    kwargs = {"lhs_dilation": (2,), "kernel_dilation": (1,), "feature_groups": 2}
    actual = sw.conv_general(lhs_m, kernel_m, *arguments, **kwargs)
    expected = sw.conv_general(lhs_g, kernel_g, *arguments, **kwargs)
    gradient_values = [float((i % 4) - 1) for i in range(actual.size())]
    actual.backward(_tensor_from_logical(gradient_values, actual.layout, metal=True))
    expected.backward(
        _tensor_from_logical(gradient_values, expected.layout, metal=False)
    )

    assert lhs_m.grad is not None and lhs_g.grad is not None
    assert kernel_m.grad is not None and kernel_g.grad is not None
    _assert_matches(lhs_m.grad, lhs_g.grad)
    _assert_matches(kernel_m.grad, kernel_g.grad)
    assert lhs_m.grad.layout == lhs_layout
    assert kernel_m.grad.layout == kernel_layout
    assert type(lhs_m.grad.carrier) is sw.Metal
    assert type(kernel_m.grad.carrier) is sw.Metal


@pytest.mark.parametrize("gradient_target", ("lhs", "kernel"))
def test_convolution_vjp_rounds_products_before_serial_addition(
    gradient_target: str,
) -> None:
    small = 2.856967235148791e-27
    large = 3.319672604740257e25
    negative = -19.603397369384766
    if gradient_target == "lhs":
        lhs_values = [1.0, 2.0]
        kernel_values = [large, 1.0]
        gradient_values = [negative, small, 0.0]
        selected = 0
    else:
        lhs_values = [1.0, large]
        kernel_values = [1.0, 2.0]
        gradient_values = [0.0, negative, small]
        selected = 0
    lhs_m, lhs_g = _pair(lhs_values, _canonical([1, 1, 2]))
    kernel_m, kernel_g = _pair(kernel_values, _canonical([1, 1, 2]))
    actual = sw.conv_general(lhs_m, kernel_m, (1,), ((1, 1),))
    expected = sw.conv_general(lhs_g, kernel_g, (1,), ((1, 1),))
    actual.backward(_tensor_from_logical(gradient_values, actual.layout, metal=True))
    expected.backward(
        _tensor_from_logical(gradient_values, expected.layout, metal=False)
    )
    actual_gradient = lhs_m.grad if gradient_target == "lhs" else kernel_m.grad
    expected_gradient = lhs_g.grad if gradient_target == "lhs" else kernel_g.grad
    assert actual_gradient is not None and expected_gradient is not None
    assert actual_gradient[selected] == expected_gradient[selected]


@pytest.mark.parametrize("operation", ("matmul", "conv_general"))
def test_saved_operand_mutation_rejects_backward_before_a_vjp_launch(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if operation == "matmul":
        lhs = _tensor_from_logical([1.0, 2.0], _canonical([1, 2]), metal=True)
        rhs = _tensor_from_logical([3.0, 4.0], _canonical([1, 2]), metal=True)
        result = sw.matmul(lhs, rhs)
    else:
        lhs = _tensor_from_logical([1.0, 2.0], _canonical([1, 1, 2]), metal=True)
        rhs = _tensor_from_logical([3.0], _canonical([1, 1, 1]), metal=True)
        result = sw.conv_general(lhs, rhs, (1,), ((0, 0),))
    lhs.carrier[0] = 9.0

    monkeypatch.setattr(
        contraction_ops,
        "execute_matmul" if operation == "matmul" else "execute_conv",
        lambda *_args, **_kwargs: pytest.fail("VJP launched before version rejection"),
    )
    with pytest.raises(RuntimeError, match="modified in-place"):
        result.backward(
            _tensor_from_logical([1.0] * result.size(), result.layout, metal=True)
        )


def test_matmul_default_backward_releases_operation_state() -> None:
    lhs = _tensor_from_logical([1.0, 2.0], _canonical([1, 2]), metal=True)
    rhs = _tensor_from_logical([3.0, 4.0], _canonical([1, 2]), metal=True)
    result = sw.matmul(lhs, rhs)
    operation = result.autograd_ctx
    assert isinstance(operation, MetalContractionOperation)
    result.backward(_tensor_from_logical([1.0], result.layout, metal=True))

    assert operation.inputs() == ()
    assert operation.input_versions() == ()
    assert operation.ctx == {}
    with pytest.raises(RuntimeError, match="retain_graph=True"):
        result.backward(_tensor_from_logical([1.0], result.layout, metal=True))


@pytest.mark.parametrize(
    ("dtype", "accumulator"),
    (
        (sw.DType.Int32, None),
        (sw.DType.Float32, sw.DType.Float64),
    ),
)
def test_unsupported_matmul_plans_refuse_before_allocation_or_jit(
    dtype: sw.DType,
    accumulator: SimpleDType | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lhs = _tensor_from_logical([1, 2], _canonical([1, 2]), metal=True, dtype=dtype)
    rhs = _tensor_from_logical([3, 4], _canonical([1, 2]), metal=True, dtype=dtype)
    monkeypatch.setattr(
        sw.Metal,
        "allocate_like",
        lambda *_args, **_kwargs: pytest.fail("unsupported plan allocated storage"),
    )
    monkeypatch.setattr(
        contraction_ops,
        "execute_matmul",
        lambda *_args, **_kwargs: pytest.fail("unsupported plan reached JIT"),
    )

    with pytest.raises(UnsupportedOperationPlan):
        sw.matmul(lhs, rhs, accumulator_dtype=accumulator)


@pytest.mark.parametrize(
    ("strides", "padding", "kwargs", "message"),
    (
        ((0,), ((0, 0),), {}, "strides must contain positive"),
        ((1,), ((-1, 0),), {}, "padding values must be non-negative"),
        ((1,), ((0, 0),), {"feature_groups": 0}, "feature_groups must be positive"),
        ((1,), ((0, 0),), {"lhs_dilation": (0,)}, "lhs_dilation must contain positive"),
        (
            (1,),
            ((0, 0),),
            {"kernel_dilation": (0,)},
            "kernel_dilation must contain positive",
        ),
    ),
)
def test_invalid_convolution_geometry_refuses_before_allocation_or_jit(
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    kwargs: dict[str, Any],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lhs = _tensor_from_logical([1.0, 2.0, 3.0], _canonical([1, 1, 3]), metal=True)
    kernel = _tensor_from_logical([1.0, 2.0], _canonical([1, 1, 2]), metal=True)
    snapshots = (
        (_values(lhs), lhs.carrier.version),
        (_values(kernel), kernel.carrier.version),
    )
    monkeypatch.setattr(
        sw.Metal,
        "allocate_like",
        lambda *_args, **_kwargs: pytest.fail("invalid convolution allocated storage"),
    )
    monkeypatch.setattr(
        contraction_ops,
        "execute_conv",
        lambda *_args, **_kwargs: pytest.fail("invalid convolution reached JIT"),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        sw.conv_general(lhs, kernel, strides, padding, **kwargs)
    assert (_values(lhs), lhs.carrier.version) == snapshots[0]
    assert (_values(kernel), kernel.carrier.version) == snapshots[1]


def test_convolution_rejects_bad_channels_output_and_dtype_before_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contraction_ops,
        "execute_conv",
        lambda *_args, **_kwargs: pytest.fail("invalid convolution reached JIT"),
    )
    lhs = _tensor_from_logical([1.0, 2.0], _canonical([1, 2, 1]), metal=True)
    bad_channels = _tensor_from_logical(
        [1.0, 2.0, 3.0, 4.0], _canonical([2, 2, 1]), metal=True
    )
    with pytest.raises(ValueError, match="kernel input feature extent"):
        sw.conv_general(lhs, bad_channels, (1,), ((0, 0),), feature_groups=2)

    kernel = _tensor_from_logical([1.0, 2.0, 3.0], _canonical([1, 1, 3]), metal=True)
    with pytest.raises(ValueError, match="output spatial extents must be positive"):
        sw.conv_general(
            _tensor_from_logical([1.0], _canonical([1, 1, 1]), metal=True),
            kernel,
            (1,),
            ((0, 0),),
        )

    int_lhs = _tensor_from_logical(
        [1, 2], _canonical([1, 1, 2]), metal=True, dtype=sw.DType.Int32
    )
    int_kernel = _tensor_from_logical(
        [1], _canonical([1, 1, 1]), metal=True, dtype=sw.DType.Int32
    )
    with pytest.raises((TypeError, UnsupportedOperationPlan)):
        sw.conv_general(int_lhs, int_kernel, (1,), ((0, 0),))


def test_shape_and_contraction_mismatches_refuse_before_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contraction_ops,
        "execute_matmul",
        lambda *_args, **_kwargs: pytest.fail("bad matmul reached executor"),
    )
    lhs = _tensor_from_logical([1.0, 2.0], _canonical([1, 2]), metal=True)
    rhs = _tensor_from_logical([1.0, 2.0, 3.0], _canonical([1, 3]), metal=True)
    with pytest.raises(ValueError, match="inner dimensions"):
        sw.matmul(lhs, rhs)
    one_mode = _tensor_from_logical([1.0, 2.0], _canonical(2), metal=True)
    with pytest.raises(ValueError, match="two-mode"):
        sw.matmul(one_mode, lhs)


def test_singleton_matmul_and_convolution_execute() -> None:
    lhs = _tensor_from_logical([2.0], _canonical([1, 1]), metal=True)
    rhs = _tensor_from_logical([3.0], _canonical([1, 1]), metal=True)
    assert _values(sw.matmul(lhs, rhs)) == [6.0]
    conv_lhs = _tensor_from_logical([2.0], _canonical([1, 1, 1]), metal=True)
    kernel = _tensor_from_logical([3.0], _canonical([1, 1, 1]), metal=True)
    assert _values(sw.conv_general(conv_lhs, kernel, (1,), ((0, 0),))) == [6.0]
