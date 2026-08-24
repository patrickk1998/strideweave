from __future__ import annotations

import math
from typing import Any

import pytest

import strideweave as sw
from strideweave.carriers.base import Carrier
from strideweave.carriers.metal import _reduction_executor as reduction_executor
from strideweave.carriers.metal.reduction_ops import MetalReductionOperation

pytestmark = pytest.mark.metal

_REDUCTION_NAMES = (
    "argmax",
    "argmin",
    "cumsum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "reduce_sum",
)


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
) -> sw.Tensor:
    storage = [-91.0] * (offset + layout.cosize)
    for logical_index, value in enumerate(values):
        storage[offset + layout.index(logical_index)] = value
    return _tensor_from_storage(storage, layout, metal=metal, offset=offset)


def _values(tensor: sw.Tensor) -> list[Any]:
    return [tensor[index] for index in range(tensor.size())]


def _direct(name: str, tensor: sw.Tensor, *extra: Any) -> sw.Tensor:
    return tensor.carrier.dispatch_op(name).forward(tensor, *extra)


def _assert_matches(actual: sw.Tensor, expected: sw.Tensor) -> None:
    assert actual.dtype() is expected.dtype()
    assert actual.layout == expected.layout
    assert _values(actual) == pytest.approx(
        _values(expected), rel=2e-6, abs=2e-6, nan_ok=True
    )


def _gradient(tensor: sw.Tensor, values: list[float]) -> sw.Tensor:
    return _tensor_from_logical(
        values, tensor.layout, metal=type(tensor.carrier) is sw.Metal
    )


def test_every_reduction_dispatch_returns_a_fresh_metal_owned_operation() -> None:
    carrier = sw.Metal.__new__(sw.Metal)
    Carrier.__init__(carrier)

    for name in _REDUCTION_NAMES:
        first = carrier.dispatch_op(name)
        second = carrier.dispatch_op(name)
        assert isinstance(first, MetalReductionOperation)
        assert isinstance(second, MetalReductionOperation)
        assert first is not second
        assert first._dispatch_carrier_class is sw.Metal
        assert first._operation_name == name


@pytest.mark.parametrize(
    "name",
    ("reduce_sum", "reduce_prod", "reduce_max", "reduce_min", "argmax", "argmin"),
)
def test_every_two_mode_reduction_matches_generic(name: str) -> None:
    layout = sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 2]))
    values = [1.0, -2.0, 3.0, 4.0, -5.0, 6.0]
    actual = _direct(name, _tensor_from_logical(values, layout, metal=True))
    expected = _direct(name, _tensor_from_logical(values, layout, metal=False))

    _assert_matches(actual, expected)
    assert type(actual.carrier) is sw.Metal


def test_reductions_use_tilelang_without_host_element_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[Any] = []
    original_compile = reduction_executor._compile_kernel

    def capture(*args: Any, **kwargs: Any) -> Any:
        result = original_compile(*args, **kwargs)
        compiled.append(result)
        return result

    def unexpected_read(_self: sw.Metal, _index: int) -> Any:
        raise AssertionError("Metal reduction performed a host element read")

    reduction_executor._JIT_CACHE.clear()
    monkeypatch.setattr(reduction_executor, "_compile_kernel", capture)
    monkeypatch.setattr(sw.Metal, "get_value", unexpected_read)
    layout = sw.Layout(sw.Shape([2, 2]), sw.Stride([1, 2]))
    tensor = _tensor_from_logical([1.0, 2.0, 3.0, 4.0], layout, metal=True)

    for name in ("reduce_sum", "reduce_prod", "reduce_max", "argmax"):
        result = _direct(name, tensor)
        assert isinstance(result.carrier, sw.Metal)
        assert result.carrier._require_storage().device.type == "mps"  # pyright: ignore[reportAttributeAccessIssue]
    result = _direct("cumsum", tensor, 1)
    assert isinstance(result.carrier, sw.Metal)
    assert result.carrier._require_storage().device.type == "mps"  # pyright: ignore[reportAttributeAccessIssue]
    assert len(compiled) == 5
    assert all(item.facts.device_source and item.facts.host_source for item in compiled)


@pytest.mark.parametrize(
    ("layout", "offset", "storage"),
    (
        (
            sw.Layout(sw.Shape([2, 3]), sw.Stride([2, 5])),
            2,
            [
                99.0,
                99.0,
                1.0,
                99.0,
                2.0,
                99.0,
                99.0,
                3.0,
                99.0,
                4.0,
                99.0,
                99.0,
                5.0,
                99.0,
                6.0,
            ],
        ),
        (
            sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 0])),
            0,
            [2.0, 4.0],
        ),
        (
            sw.Layout(sw.Shape([2, 3]), sw.Stride([0, 1])),
            0,
            [2.0, 3.0, 4.0],
        ),
    ),
)
def test_holes_offsets_and_stride_zero_reads_use_exact_address_plans(
    layout: sw.Layout,
    offset: int,
    storage: list[float],
) -> None:
    metal = _tensor_from_storage(storage, layout, metal=True, offset=offset)
    generic = _tensor_from_storage(storage, layout, metal=False, offset=offset)

    for name in (
        "argmax",
        "argmin",
        "reduce_sum",
        "reduce_prod",
        "reduce_max",
        "reduce_min",
    ):
        _assert_matches(_direct(name, metal), _direct(name, generic))
    _assert_matches(_direct("cumsum", metal, 1), _direct("cumsum", generic, 1))


def test_nested_permuted_and_described_reductions_preserve_hierarchy() -> None:
    layout = sw.Layout(
        sw.Shape([[2, 2], [3, 2]]),
        sw.Stride([[1, 2], [4, 12]]),
    )
    values = [float(index - 7) for index in range(layout.size)]
    metal = _tensor_from_logical(values, layout, metal=True)
    generic = _tensor_from_logical(values, layout, metal=False)

    for name in (
        "argmax",
        "argmin",
        "reduce_sum",
        "reduce_prod",
        "reduce_max",
        "reduce_min",
    ):
        _assert_matches(_direct(name, metal), _direct(name, generic))
    _assert_matches(_direct("cumsum", metal, 1), _direct("cumsum", generic, 1))
    for name in (
        "argmax",
        "argmin",
        "reduce_sum",
        "reduce_prod",
        "reduce_max",
        "reduce_min",
    ):
        public_operation = getattr(sw, name)
        _assert_matches(
            public_operation(metal, "(a b) (c d) -> a b"),
            public_operation(generic, "(a b) (c d) -> a b"),
        )

    metal_permuted = sw.permute(metal, [1, 0])
    generic_permuted = sw.permute(generic, [1, 0])
    for name in (
        "argmax",
        "argmin",
        "reduce_sum",
        "reduce_prod",
        "reduce_max",
        "reduce_min",
    ):
        _assert_matches(_direct(name, metal_permuted), _direct(name, generic_permuted))
    _assert_matches(
        _direct("cumsum", metal_permuted, 1),
        _direct("cumsum", generic_permuted, 1),
    )


def test_cumsum_axes_negative_axis_and_hierarchy_match_generic() -> None:
    layout = sw.Layout(
        sw.Shape([[2, 2], 3]),
        sw.Stride([[1, 2], 4]),
    )
    values = [float(index + 1) for index in range(layout.size)]
    metal = _tensor_from_logical(values, layout, metal=True)
    generic = _tensor_from_logical(values, layout, metal=False)

    for axis in (0, 1, -1):
        _assert_matches(sw.cumsum(metal, axis), sw.cumsum(generic, axis))
    with pytest.raises(TypeError, match="integer top-level mode"):
        sw.cumsum(metal, True)
    with pytest.raises(ValueError, match="out of range"):
        sw.cumsum(metal, 2)


def test_ieee_extrema_arg_ties_and_ordered_combining_are_faithful() -> None:
    pair = sw.Layout(sw.Shape([1, 2]), sw.Stride([1, 1]))
    maximum = _direct(
        "reduce_max",
        _tensor_from_logical([-0.0, 0.0], pair, metal=True),
    )
    minimum = _direct(
        "reduce_min",
        _tensor_from_logical([0.0, -0.0], pair, metal=True),
    )
    assert math.copysign(1.0, maximum[0]) > 0
    assert math.copysign(1.0, minimum[0]) < 0

    triples = sw.Layout(sw.Shape([1, 3]), sw.Stride([1, 1]))
    arg_values = _tensor_from_logical([2.0, math.nan, math.nan], triples, metal=True)
    assert _values(_direct("argmax", arg_values)) == [1]
    assert _values(_direct("argmin", arg_values)) == [1]
    tied = _tensor_from_logical([3.0, 1.0, 3.0], triples, metal=True)
    assert _values(_direct("argmax", tied)) == [0]
    minimum_tie = _tensor_from_logical([3.0, 1.0, 1.0], triples, metal=True)
    assert _values(_direct("argmin", minimum_tie)) == [1]
    zero_tie = _tensor_from_logical([-0.0, 0.0], pair, metal=True)
    assert _values(_direct("argmax", zero_tie)) == [0]
    assert _values(_direct("argmin", zero_tie)) == [0]

    infinities = _tensor_from_logical([math.inf, -math.inf, 5.0], triples, metal=True)
    assert _values(_direct("reduce_max", infinities)) == [math.inf]
    assert _values(_direct("reduce_min", infinities)) == [-math.inf]
    assert _values(_direct("argmax", infinities)) == [0]
    assert _values(_direct("argmin", infinities)) == [1]

    infinite_sum = _direct(
        "reduce_sum",
        _tensor_from_logical([math.inf, -math.inf], pair, metal=True),
    )
    assert math.isnan(infinite_sum[0])

    ordered = sw.Layout(sw.Shape([1, 4]), sw.Stride([1, 1]))
    product = _direct(
        "reduce_prod",
        _tensor_from_logical([1e20, 1e20, 1e-20, 1e-20], ordered, metal=True),
    )
    assert math.isinf(product[0])
    scan = _direct(
        "cumsum",
        _tensor_from_logical([2**24, 1.0, 1.0, -(2**24)], ordered, metal=True),
        1,
    )
    assert _values(scan) == [2**24, 2**24, 2**24, 0.0]


def test_empty_fibers_and_modes_are_refused_by_the_layout_boundary() -> None:
    # Shape's public contract makes zero extents unrepresentable, so an empty
    # reduction fiber or cumsum mode is rejected before carrier dispatch.
    with pytest.raises(ValueError, match="must not be less than 1"):
        sw.Shape([2, 0])
    with pytest.raises(ValueError, match="must not be less than 1"):
        sw.Shape([0, 3])


@pytest.mark.parametrize("unsupported", ("float64", "int32"))
def test_unsupported_sum_plans_refuse_before_allocation_or_jit(
    monkeypatch: pytest.MonkeyPatch,
    unsupported: str,
) -> None:
    dtype = sw.DType.Int32 if unsupported == "int32" else sw.DType.Float32
    values = [1, 2] if dtype is sw.DType.Int32 else [1.0, 2.0]
    layout = sw.Layout(sw.Shape([1, 2]), sw.Stride([1, 1]))
    tensor = _tensor_from_storage(values, layout, metal=True, dtype=dtype)

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unsupported reduction reached allocation or JIT")

    monkeypatch.setattr(sw.Metal, "allocate_like", unexpected)
    monkeypatch.setattr(
        "strideweave.carriers.metal.reduction_ops.execute_reduction_forward",
        unexpected,
    )
    operation = tensor.carrier.dispatch_op("reduce_sum")
    if unsupported == "float64":
        with pytest.raises(sw.UnsupportedOperationPlan, match="Metal declares no"):
            sw.reduce_sum(
                tensor,
                "a b -> a",
                accumulator_dtype=sw.DType.Float64,
            )
    else:
        with pytest.raises(sw.UnsupportedOperationPlan, match="Metal declares no"):
            operation.forward(tensor)


@pytest.mark.parametrize(
    "name",
    ("reduce_sum", "reduce_prod", "reduce_max", "reduce_min"),
)
def test_reduction_vjps_match_generic_and_remain_on_metal(name: str) -> None:
    layout = sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 2]))
    values = [0.0, 2.0, 3.0, 4.0, 3.0, 6.0]
    metal = _tensor_from_logical(values, layout, metal=True)
    generic = _tensor_from_logical(values, layout, metal=False)
    actual = _direct(name, metal)
    expected = _direct(name, generic)

    actual.backward(_gradient(actual, [2.0, 4.0]))
    expected.backward(_gradient(expected, [2.0, 4.0]))

    assert isinstance(metal.grad, sw.Tensor)
    assert isinstance(generic.grad, sw.Tensor)
    _assert_matches(metal.grad, generic.grad)
    assert type(metal.grad.carrier) is sw.Metal


def test_extrema_nan_vjp_and_cumsum_reverse_vjp_match_generic() -> None:
    reduction_layout = sw.Layout(sw.Shape([1, 3]), sw.Stride([1, 1]))
    metal_nan = _tensor_from_logical([1.0, math.nan, 2.0], reduction_layout, metal=True)
    result = _direct("reduce_max", metal_nan)
    result.backward(_gradient(result, [3.0]))
    assert isinstance(metal_nan.grad, sw.Tensor)
    assert all(math.isnan(value) for value in _values(metal_nan.grad))

    scan_layout = sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 2]))
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    metal = _tensor_from_logical(values, scan_layout, metal=True)
    generic = _tensor_from_logical(values, scan_layout, metal=False)
    actual = sw.cumsum(metal, 1)
    expected = sw.cumsum(generic, 1)
    actual.backward(_gradient(actual, [1.0] * 6))
    expected.backward(_gradient(expected, [1.0] * 6))
    assert isinstance(metal.grad, sw.Tensor)
    assert isinstance(generic.grad, sw.Tensor)
    _assert_matches(metal.grad, generic.grad)
    assert type(metal.grad.carrier) is sw.Metal

    ordered_layout = sw.Layout(sw.Shape([1, 3]), sw.Stride([1, 1]))
    metal_ordered = _tensor_from_logical([1.0, 2.0, 3.0], ordered_layout, metal=True)
    generic_ordered = _tensor_from_logical([1.0, 2.0, 3.0], ordered_layout, metal=False)
    metal_scan = sw.cumsum(metal_ordered, 1)
    generic_scan = sw.cumsum(generic_ordered, 1)
    upstream = [1e20, -1e20, 1.0]
    metal_scan.backward(_gradient(metal_scan, upstream))
    generic_scan.backward(_gradient(generic_scan, upstream))
    assert isinstance(metal_ordered.grad, sw.Tensor)
    assert isinstance(generic_ordered.grad, sw.Tensor)
    _assert_matches(metal_ordered.grad, generic_ordered.grad)
    ordered_gradient = _values(metal_ordered.grad)
    assert ordered_gradient[0] == 0.0
    assert ordered_gradient[1:] == pytest.approx([-1e20, 1.0])


@pytest.mark.parametrize(
    "name",
    ("reduce_sum", "reduce_prod", "reduce_max", "reduce_min"),
)
@pytest.mark.parametrize("layout_kind", ("holey_offset", "stride_zero"))
def test_reduction_vjps_cover_holey_offset_and_stride_zero_layouts(
    name: str,
    layout_kind: str,
) -> None:
    if layout_kind == "holey_offset":
        layout = sw.Layout(sw.Shape([2, 2]), sw.Stride([2, 5]))
        values = [1.0, 2.0, 3.0, 4.0]
        metal = _tensor_from_logical(values, layout, metal=True, offset=1)
        generic = _tensor_from_logical(values, layout, metal=False, offset=1)
    else:
        layout = sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 0]))
        metal = _tensor_from_storage([2.0, 4.0], layout, metal=True)
        generic = _tensor_from_storage([2.0, 4.0], layout, metal=False)

    actual = _direct(name, metal)
    expected = _direct(name, generic)
    actual.backward(_gradient(actual, [1.0, 2.0]))
    expected.backward(_gradient(expected, [1.0, 2.0]))
    assert isinstance(metal.grad, sw.Tensor)
    assert isinstance(generic.grad, sw.Tensor)
    _assert_matches(metal.grad, generic.grad)
    assert type(metal.grad.carrier) is sw.Metal


@pytest.mark.parametrize("layout_kind", ("holey_offset", "stride_zero"))
def test_cumsum_vjp_covers_holey_offset_and_stride_zero_layouts(
    layout_kind: str,
) -> None:
    if layout_kind == "holey_offset":
        layout = sw.Layout(sw.Shape([2, 2]), sw.Stride([2, 5]))
        values = [1.0, 2.0, 3.0, 4.0]
        metal = _tensor_from_logical(values, layout, metal=True, offset=1)
        generic = _tensor_from_logical(values, layout, metal=False, offset=1)
    else:
        layout = sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 0]))
        metal = _tensor_from_storage([2.0, 4.0], layout, metal=True)
        generic = _tensor_from_storage([2.0, 4.0], layout, metal=False)

    actual = _direct("cumsum", metal, 1)
    expected = _direct("cumsum", generic, 1)
    upstream = [float(index + 1) for index in range(actual.size())]
    actual.backward(_gradient(actual, upstream))
    expected.backward(_gradient(expected, upstream))
    assert isinstance(metal.grad, sw.Tensor)
    assert isinstance(generic.grad, sw.Tensor)
    _assert_matches(metal.grad, generic.grad)
    assert type(metal.grad.carrier) is sw.Metal


def test_holey_gradient_layout_and_stride_zero_leaf_reduction_are_preserved() -> None:
    holey = sw.Layout(sw.Shape([2, 2]), sw.Stride([2, 5]))
    tensor = _tensor_from_logical([1.0, 2.0, 3.0, 4.0], holey, metal=True)
    result = _direct("reduce_sum", tensor)
    result.backward(_gradient(result, [1.0, 2.0]))
    assert isinstance(tensor.grad, sw.Tensor)
    assert tensor.grad.layout == holey
    assert _values(tensor.grad) == [1.0, 2.0, 1.0, 2.0]

    scan_tensor = _tensor_from_logical([1.0, 2.0, 3.0, 4.0], holey, metal=True)
    scan = sw.cumsum(scan_tensor, 1)
    scan.backward(_gradient(scan, [1.0] * 4))
    assert isinstance(scan_tensor.grad, sw.Tensor)
    assert scan_tensor.grad.layout == holey
    assert _values(scan_tensor.grad) == [2.0, 2.0, 1.0, 1.0]

    broadcast = sw.Layout(sw.Shape([2, 3]), sw.Stride([1, 0]))
    repeated = _tensor_from_storage([2.0, 4.0], broadcast, metal=True)
    reduced = _direct("reduce_sum", repeated)
    reduced.backward(_gradient(reduced, [1.0, 2.0]))
    assert isinstance(repeated.grad, sw.Tensor)
    assert repeated.grad.layout.is_injective
    assert _values(repeated.grad) == [3.0, 6.0] * 3


def test_saved_version_and_default_graph_release_match_core_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = sw.Layout(sw.Shape([1, 2]), sw.Stride([1, 1]))
    mutated = _tensor_from_logical([1.0, 2.0], layout, metal=True)
    result = _direct("reduce_prod", mutated)
    calls = 0
    original = MetalReductionOperation.backward

    def spy(self: MetalReductionOperation, incoming: Any) -> tuple[Any, ...]:
        nonlocal calls
        calls += 1
        return original(self, incoming)

    monkeypatch.setattr(MetalReductionOperation, "backward", spy)
    mutated.carrier[0] = 9.0
    with pytest.raises(RuntimeError, match="modified in-place"):
        result.backward(_gradient(result, [1.0]))
    assert calls == 0

    tensor = _tensor_from_logical([1.0, 2.0], layout, metal=True)
    released = _direct("reduce_sum", tensor)
    operation = released.autograd_ctx
    assert isinstance(operation, sw.Operation)
    released.backward(_gradient(released, [1.0]))
    assert operation.inputs() == ()
    assert operation.input_versions() == ()
    assert operation.ctx == {}
    with pytest.raises(RuntimeError, match="retain_graph=True"):
        released.backward(_gradient(released, [1.0]))


def test_arg_reductions_are_int32_and_non_differentiable() -> None:
    layout = sw.Layout(sw.Shape([1, 2]), sw.Stride([1, 1]))
    tensor = _tensor_from_logical([1.0, 2.0], layout, metal=True)
    for name in ("argmax", "argmin"):
        result = _direct(name, tensor)
        assert result.dtype() is sw.DType.Int32
        assert result.autograd_ctx is None
        with pytest.raises(RuntimeError, match="non-differentiable"):
            result.backward()
