from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest

import strideweave as sw
from strideweave.carriers.base import Carrier
from strideweave.carriers.metal import _executor as metal_executor
from strideweave.carriers.metal.pointwise_ops import MetalPointwiseOperation
from strideweave.carriers.operation_policy import (
    Accumulation,
    resolve_operation_plan,
)

pytestmark = pytest.mark.metal

_POINTWISE_NAMES = (
    "abs",
    "add",
    "ceil",
    "clamp",
    "cos",
    "div",
    "elementwise_mul",
    "elu",
    "eq",
    "erf",
    "exp",
    "exp2",
    "floor",
    "gelu",
    "le",
    "leaky_relu",
    "log",
    "log2",
    "logical_not",
    "lt",
    "maximum",
    "minimum",
    "mul",
    "ne",
    "neg",
    "pow",
    "recip",
    "relu",
    "rem",
    "round",
    "rsqrt",
    "select",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "softplus",
    "sqrt",
    "sub",
    "tanh",
)


def _layout(size: int) -> sw.Layout:
    return sw.Layout(sw.Shape(size), sw.Stride(1))


def _tensor(
    values: list[Any],
    *,
    metal: bool,
    dtype: sw.DType = sw.DType.Float32,
    layout: sw.Layout | None = None,
    offset: int = 0,
) -> sw.Tensor:
    effective_layout = layout or _layout(len(values) - offset)
    if metal:
        carrier = sw.Metal(len(values), dtype=dtype)
        for index, value in enumerate(values):
            carrier[index] = value
    else:
        carrier = sw.Generic(values, dtype=dtype)
    return sw.Tensor(carrier, offset, effective_layout)


def _values(tensor: sw.Tensor) -> list[Any]:
    return [tensor[index] for index in range(tensor.size())]


def _require_tensor(value: sw.Tensor | None) -> sw.Tensor:
    assert isinstance(value, sw.Tensor)
    return value


def _require_operation(value: sw.Operation | None) -> sw.Operation:
    assert isinstance(value, sw.Operation)
    return value


def _invoke(name: str, *operands: Any) -> sw.Tensor:
    owner = next(operand for operand in operands if isinstance(operand, sw.Tensor))
    return owner.carrier.dispatch_op(name).forward(*operands)


def _assert_values(
    actual: sw.Tensor, expected: sw.Tensor, *, atol: float = 2e-3
) -> None:
    assert actual.dtype() is expected.dtype()
    assert actual.layout == expected.layout
    actual_values = _values(actual)
    expected_values = _values(expected)
    if actual.dtype() is sw.DType.Bool:
        assert actual_values == expected_values
        return
    assert actual_values == pytest.approx(
        expected_values, rel=atol, abs=atol, nan_ok=True
    )


def test_every_pointwise_dispatch_returns_a_fresh_metal_owned_operation() -> None:
    carrier = sw.Metal.__new__(sw.Metal)
    Carrier.__init__(carrier)

    for name in _POINTWISE_NAMES:
        first = carrier.dispatch_op(name)
        second = carrier.dispatch_op(name)
        assert isinstance(first, MetalPointwiseOperation)
        assert isinstance(second, MetalPointwiseOperation)
        assert first is not second
        assert first._dispatch_carrier_class is sw.Metal
        assert first._operation_name == name


def test_forward_uses_generated_tilelang_msl_without_host_element_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[Any] = []
    original_compile = metal_executor._compile_kernel

    def capture(*args: Any, **kwargs: Any) -> Any:
        result = original_compile(*args, **kwargs)
        compiled.append(result)
        return result

    def unexpected_read(_self: sw.Metal, _index: int) -> Any:
        raise AssertionError("Metal forward performed a host element read")

    metal_executor._JIT_CACHE.clear()
    monkeypatch.setattr(metal_executor, "_compile_kernel", capture)
    monkeypatch.setattr(sw.Metal, "get_value", unexpected_read)
    lhs = _tensor([1.0, 2.0, 3.0], metal=True)
    rhs = _tensor([4.0, 5.0, 6.0], metal=True)

    result = sw.add(lhs, rhs)
    repeated = sw.add(
        _tensor([7.0, 8.0, 9.0], metal=True),
        _tensor([10.0, 11.0, 12.0], metal=True),
    )

    assert result.carrier._require_storage().device.type == "mps"  # pyright: ignore[reportAttributeAccessIssue]
    assert repeated.carrier._require_storage().device.type == "mps"  # pyright: ignore[reportAttributeAccessIssue]
    assert len(compiled) == 1
    facts = compiled[0].facts
    assert "kernel void" in facts.device_source
    assert facts.host_source
    assert facts.runtime_artifact_digest is None


@pytest.mark.parametrize(
    "name",
    [
        "abs",
        "ceil",
        "cos",
        "elu",
        "erf",
        "exp",
        "exp2",
        "floor",
        "gelu",
        "leaky_relu",
        "log",
        "log2",
        "neg",
        "recip",
        "relu",
        "round",
        "rsqrt",
        "sigmoid",
        "sign",
        "silu",
        "sin",
        "softplus",
        "sqrt",
        "tanh",
    ],
)
def test_metal_unary_plans_execute_tilelang_and_match_generic(name: str) -> None:
    values = [0.125, 0.5, 1.25, 3.0]
    metal = _tensor(values, metal=True)
    generic = _tensor(values, metal=False)

    actual = _invoke(name, metal)
    expected = _invoke(name, generic)

    _assert_values(actual, expected)
    assert type(actual.carrier) is sw.Metal
    assert actual.layout.is_injective


@pytest.mark.parametrize(
    "name",
    [
        "add",
        "div",
        "elementwise_mul",
        "eq",
        "le",
        "lt",
        "maximum",
        "minimum",
        "mul",
        "ne",
        "pow",
        "rem",
        "sub",
    ],
)
def test_metal_binary_and_predicate_plans_match_generic(name: str) -> None:
    lhs_values = [0.25, 0.5, 1.5, 4.0]
    rhs_values = [2.0, 0.5, 3.0, 0.25]
    metal_lhs = _tensor(lhs_values, metal=True)
    metal_rhs = _tensor(rhs_values, metal=True)
    generic_lhs = _tensor(lhs_values, metal=False)
    generic_rhs = _tensor(rhs_values, metal=False)

    actual = _invoke(name, metal_lhs, metal_rhs)
    expected = _invoke(name, generic_lhs, generic_rhs)

    _assert_values(actual, expected)
    if name in {"eq", "le", "lt", "ne"}:
        assert actual.dtype() is sw.DType.Bool
        assert actual.autograd_ctx is None


def test_logical_not_select_clamp_and_weak_scalars_match_generic() -> None:
    condition_values = [True, False, False, True]
    true_values = [1.0, 2.0, 3.0, 4.0]
    false_values = [-1.0, -2.0, -3.0, -4.0]
    expected: tuple[sw.Tensor, ...] | None = None
    for metal in (False, True):
        condition = _tensor(condition_values, metal=metal, dtype=sw.DType.Bool)
        on_true = _tensor(true_values, metal=metal)
        on_false = _tensor(false_values, metal=metal)
        lower = _tensor([1.5] * 4, metal=metal)
        upper = _tensor([3.5] * 4, metal=metal)
        selected = _invoke("select", condition, on_true, on_false)
        clamped = _invoke("clamp", on_true, 1.5, 3.5)
        tensor_lower_clamped = _invoke("clamp", on_true, lower, 3.5)
        tensor_upper_clamped = _invoke("clamp", on_true, 1.5, upper)
        scaled = _invoke("mul", on_true, 2.5)
        powered = _invoke("pow", on_true, 2.5)
        reverse_powered = _invoke("pow", 2.5, on_true)
        logical = _invoke(
            "logical_not", _tensor([0.0, -0.0, 2.0, math.nan], metal=metal)
        )
        results = (
            selected,
            clamped,
            tensor_lower_clamped,
            tensor_upper_clamped,
            scaled,
            powered,
            reverse_powered,
            logical,
        )
        if not metal:
            expected = results
        else:
            assert expected is not None
            for actual, reference in zip(results, expected, strict=True):
                _assert_values(actual, reference)

    assert logical.dtype() is sw.DType.Bool
    assert logical.autograd_ctx is None
    with pytest.raises(RuntimeError, match="non-differentiable"):
        logical.backward()


def test_cancellation_sensitive_elu_and_softplus_match_binary32_reference() -> None:
    elu_values = [-1e-8, -1e-7, -1e-6, -1e-5, -1e-4]
    softplus_values = [-30.0, -20.0, -17.0, -16.0, -10.0]
    for name, values in (("elu", elu_values), ("softplus", softplus_values)):
        actual = _values(_invoke(name, _tensor(values, metal=True)))
        expected = _values(_invoke(name, _tensor(values, metal=False)))
        assert actual == pytest.approx(
            expected,
            rel=3e-6,
            abs=1e-14,
        )


def test_int32_to_float32_plans_convert_before_computation() -> None:
    values = [-3, -1, 1, 4]
    for name in ("exp", "sigmoid"):
        actual = _invoke(
            name,
            _tensor(values, metal=True, dtype=sw.DType.Int32),
        )
        expected = _invoke(
            name,
            _tensor(values, metal=False, dtype=sw.DType.Int32),
        )
        _assert_values(actual, expected)
        assert actual.dtype() is sw.DType.Float32

    actual_div = sw.div(
        _tensor([2, 3, -4, 5], metal=True, dtype=sw.DType.Int32),
        _tensor([4, -2, 2, 10], metal=True, dtype=sw.DType.Int32),
    )
    expected_div = sw.div(
        _tensor([2, 3, -4, 5], metal=False, dtype=sw.DType.Int32),
        _tensor([4, -2, 2, 10], metal=False, dtype=sw.DType.Int32),
    )
    _assert_values(actual_div, expected_div)
    assert actual_div.dtype() is sw.DType.Float32


def test_holey_offset_and_same_shape_different_strides_use_exact_address_maps() -> None:
    lhs_layout = sw.Layout(sw.Shape([2, 2]), sw.Stride([2, 5]))
    rhs_layout = sw.Layout(sw.Shape([2, 2]), sw.Stride([5, 2]))
    lhs_physical = [91.0, 1.0, 93.0, 2.0, 95.0, 96.0, 3.0, 98.0, 4.0]
    rhs_physical = [81.0, 10.0, 83.0, 30.0, 85.0, 86.0, 20.0, 88.0, 40.0]
    metal_lhs = _tensor(lhs_physical, metal=True, layout=lhs_layout, offset=1)
    metal_rhs = _tensor(rhs_physical, metal=True, layout=rhs_layout, offset=1)
    generic_lhs = _tensor(lhs_physical, metal=False, layout=lhs_layout, offset=1)
    generic_rhs = _tensor(rhs_physical, metal=False, layout=rhs_layout, offset=1)

    actual = sw.add(metal_lhs, metal_rhs)
    expected = sw.add(generic_lhs, generic_rhs)

    _assert_values(actual, expected)
    assert actual.layout == sw.Layout(sw.Shape([2, 2]), sw.Stride([1, 2]))


def test_nested_hierarchy_broadcast_uses_each_operand_address_plan() -> None:
    lhs_layout = sw.Layout(
        sw.Shape([2, [1, 2]]),
        sw.Stride([1, [2, 2]]),
    )
    rhs_layout = sw.Layout(
        sw.Shape([1, [3, 2]]),
        sw.Stride([1, [1, 3]]),
    )
    metal_lhs = _tensor([1.0, 2.0, 3.0, 4.0], metal=True, layout=lhs_layout)
    metal_rhs = _tensor(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        metal=True,
        layout=rhs_layout,
    )
    generic_lhs = _tensor([1.0, 2.0, 3.0, 4.0], metal=False, layout=lhs_layout)
    generic_rhs = _tensor(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        metal=False,
        layout=rhs_layout,
    )

    actual = sw.add(metal_lhs, metal_rhs)
    expected = sw.add(generic_lhs, generic_rhs)

    _assert_values(actual, expected)
    assert actual.layout == sw.Layout(
        sw.Shape([2, [3, 2]]),
        sw.Stride([1, [2, 6]]),
    )


def test_broadcast_backward_weak_scalar_exclusion_and_graph_release() -> None:
    singleton_layout = sw.Layout(sw.Shape([1, 2]), sw.Stride([1, 1]))
    full_layout = sw.Layout(sw.Shape([3, 2]), sw.Stride([1, 3]))
    lhs = _tensor([2.0, 3.0], metal=True, layout=singleton_layout)
    rhs = _tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], metal=True, layout=full_layout)
    result = sw.mul(sw.add(lhs, rhs), 2.5)
    scalar_operation = _require_operation(result.autograd_ctx)
    add_result = scalar_operation.inputs()[0]
    add_operation = _require_operation(add_result.autograd_ctx)
    assert len(scalar_operation.inputs()) == 1
    gradient = _tensor([1.0] * 6, metal=True, layout=full_layout)

    result.backward(gradient)

    lhs_gradient = _require_tensor(lhs.grad)
    rhs_gradient = _require_tensor(rhs.grad)
    assert _values(lhs_gradient) == pytest.approx([7.5, 7.5])
    assert _values(rhs_gradient) == pytest.approx([2.5] * 6)
    assert type(lhs_gradient.carrier) is sw.Metal
    assert type(rhs_gradient.carrier) is sw.Metal
    assert scalar_operation.inputs() == ()
    assert scalar_operation.ctx == {}
    assert add_operation.inputs() == ()


def test_select_and_clamp_backward_route_and_split_ties_on_metal() -> None:
    condition = _tensor([True, False, True, False], metal=True, dtype=sw.DType.Bool)
    on_true = _tensor([1.0, 2.0, 3.0, 4.0], metal=True)
    on_false = _tensor([5.0, 6.0, 7.0, 8.0], metal=True)
    gradient = _tensor([1.0, 2.0, 3.0, 4.0], metal=True)
    selected = sw.select(condition, on_true, on_false)
    selected.backward(gradient)
    assert _values(_require_tensor(on_true.grad)) == pytest.approx([1.0, 0.0, 3.0, 0.0])
    assert _values(_require_tensor(on_false.grad)) == pytest.approx(
        [0.0, 2.0, 0.0, 4.0]
    )
    with pytest.raises(RuntimeError, match="non-differentiable"):
        _ = condition.grad

    tensor = _tensor([0.0, 1.0, 2.0, 3.0], metal=True)
    lower = _tensor([0.0, 0.0, 0.0, 0.0], metal=True)
    upper = _tensor([2.0, 2.0, 2.0, 2.0], metal=True)
    clamped = sw.clamp(tensor, lower, upper)
    clamped.backward(_tensor([1.0] * 4, metal=True))
    assert _values(_require_tensor(tensor.grad)) == pytest.approx([0.5, 1.0, 0.5, 0.0])
    assert _values(_require_tensor(lower.grad)) == pytest.approx([0.5, 0.0, 0.0, 0.0])
    assert _values(_require_tensor(upper.grad)) == pytest.approx([0.0, 0.0, 0.5, 1.0])


@pytest.mark.parametrize("name", ["div", "maximum", "minimum", "pow", "rem"])
def test_binary_vjps_match_generic_and_remain_on_metal(name: str) -> None:
    lhs_values = [1.0, 2.0, 4.0, 4.0]
    rhs_values = [2.0, 2.0, 5.0, 3.0]
    gradient_values = [1.0, 2.0, 3.0, 4.0]

    def gradients(metal: bool) -> tuple[sw.Tensor, sw.Tensor]:
        lhs = _tensor(lhs_values, metal=metal)
        rhs = _tensor(rhs_values, metal=metal)
        result = _invoke(name, lhs, rhs)
        result.backward(_tensor(gradient_values, metal=metal))
        return _require_tensor(lhs.grad), _require_tensor(rhs.grad)

    expected_lhs, expected_rhs = gradients(False)
    actual_lhs, actual_rhs = gradients(True)

    assert _values(actual_lhs) == pytest.approx(
        _values(expected_lhs), rel=2e-3, abs=2e-3, nan_ok=True
    )
    assert _values(actual_rhs) == pytest.approx(
        _values(expected_rhs), rel=2e-3, abs=2e-3, nan_ok=True
    )
    assert type(actual_lhs.carrier) is sw.Metal
    assert type(actual_rhs.carrier) is sw.Metal


@pytest.mark.parametrize("name", ["elu", "exp", "gelu", "softplus", "sqrt", "tanh"])
def test_unary_activation_vjps_match_generic(name: str) -> None:
    values = [0.125, 0.5, 1.25, 3.0]
    gradient_values = [1.0, 2.0, 3.0, 4.0]

    def gradient(metal: bool) -> sw.Tensor:
        tensor = _tensor(values, metal=metal)
        result = _invoke(name, tensor)
        result.backward(_tensor(gradient_values, metal=metal))
        return _require_tensor(tensor.grad)

    expected = gradient(False)
    actual = gradient(True)

    assert _values(actual) == pytest.approx(
        _values(expected), rel=2e-3, abs=2e-3, nan_ok=True
    )
    assert type(actual.carrier) is sw.Metal


@pytest.mark.parametrize("scalar_first", [False, True])
def test_weak_scalar_pow_vjp_matches_generic(scalar_first: bool) -> None:
    values = [0.5, 1.0, 2.0, 4.0]
    gradient_values = [1.0, 2.0, 3.0, 4.0]

    def gradient(metal: bool) -> sw.Tensor:
        tensor = _tensor(values, metal=metal)
        result = sw.pow(2.5, tensor) if scalar_first else sw.pow(tensor, 2.5)
        operation = _require_operation(result.autograd_ctx)
        assert len(operation.inputs()) == 1
        result.backward(_tensor(gradient_values, metal=metal))
        return _require_tensor(tensor.grad)

    expected = gradient(False)
    actual = gradient(True)

    assert _values(actual) == pytest.approx(
        _values(expected), rel=2e-3, abs=2e-3, nan_ok=True
    )
    assert type(actual.carrier) is sw.Metal


def test_signed_zero_nan_extrema_and_masked_select_are_faithful() -> None:
    lhs = _tensor([-0.0, 0.0, math.nan, 2.0], metal=True)
    rhs = _tensor([0.0, -0.0, 2.0, math.nan], metal=True)
    maximum = _values(sw.maximum(lhs, rhs))
    lhs = _tensor([-0.0, 0.0, math.nan, 2.0], metal=True)
    rhs = _tensor([0.0, -0.0, 2.0, math.nan], metal=True)
    minimum = _values(sw.minimum(lhs, rhs))

    assert math.copysign(1.0, maximum[0]) == 1.0
    assert math.copysign(1.0, maximum[1]) == 1.0
    assert math.copysign(1.0, minimum[0]) == -1.0
    assert math.copysign(1.0, minimum[1]) == -1.0
    assert math.isnan(maximum[2]) and math.isnan(maximum[3])
    assert math.isnan(minimum[2]) and math.isnan(minimum[3])

    condition = _tensor([True, False], metal=True, dtype=sw.DType.Bool)
    chosen = _tensor([3.0, 4.0], metal=True)
    unchosen_nan = _tensor([math.nan, math.nan], metal=True)
    selected = sw.select(condition, chosen, unchosen_nan)
    assert _values(selected)[0] == 3.0


def test_ieee_boundaries_and_ordered_clamp_match_generic() -> None:
    boundary_values = [-0.0, 0.0, -math.inf, math.inf]
    for name in ("abs", "ceil", "exp", "floor", "recip", "round", "sign", "tanh"):
        actual = _invoke(name, _tensor(boundary_values, metal=True))
        expected = _invoke(name, _tensor(boundary_values, metal=False))
        _assert_values(actual, expected)
        for actual_value, expected_value in zip(
            _values(actual), _values(expected), strict=True
        ):
            if actual_value == expected_value == 0.0:
                assert math.copysign(1.0, actual_value) == math.copysign(
                    1.0, expected_value
                )

    lhs_values = [0.0, 1.0, math.inf, math.nan]
    rhs_values = [0.0, 0.0, -math.inf, 2.0]
    for name in ("add", "div", "rem"):
        actual = _invoke(
            name,
            _tensor(lhs_values, metal=True),
            _tensor(rhs_values, metal=True),
        )
        expected = _invoke(
            name,
            _tensor(lhs_values, metal=False),
            _tensor(rhs_values, metal=False),
        )
        _assert_values(actual, expected)

    clamp_values = [math.nan, -0.0, 1.0, math.inf]
    actual_clamp = sw.clamp(_tensor(clamp_values, metal=True), 2.0, -1.0)
    expected_clamp = sw.clamp(_tensor(clamp_values, metal=False), 2.0, -1.0)
    _assert_values(actual_clamp, expected_clamp)


@pytest.mark.parametrize("unsupported", ("dtype", "accumulator"))
def test_unsupported_dtype_or_accumulator_plan_refuses_before_executor(
    monkeypatch: pytest.MonkeyPatch,
    unsupported: str,
) -> None:
    if unsupported == "dtype":
        lhs = _tensor([1, 2], metal=True, dtype=sw.DType.Int32)
        rhs = _tensor([3, 4], metal=True, dtype=sw.DType.Int32)
    else:
        lhs = _tensor([1.0, 2.0], metal=True)
        rhs = _tensor([3.0, 4.0], metal=True)
        pointwise_plan = resolve_operation_plan(
            "add", sw.DType.Float32, sw.DType.Float32
        )
        accumulating_plan = replace(
            pointwise_plan,
            accumulation=Accumulation.SEQUENTIAL_BINARY32,
            accumulator_dtype=sw.DType.Float32,
        )
        monkeypatch.setattr(
            "strideweave.carriers.metal.pointwise_ops.resolve_operation_plan",
            lambda *_args: accumulating_plan,
        )

    def unexpected_execution(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("executor reached for an unsupported exact plan")

    monkeypatch.setattr(
        "strideweave.carriers.metal.pointwise_ops.execute_expression",
        unexpected_execution,
    )
    with pytest.raises(sw.UnsupportedOperationPlan, match="Metal declares no"):
        sw.add(lhs, rhs)


def test_saved_version_mutation_rejects_before_backward_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = _tensor([1.0, 2.0], metal=True)
    result = sw.relu(tensor)
    gradient = _tensor([1.0, 1.0], metal=True)
    calls = 0

    original = MetalPointwiseOperation.backward

    def spy(self: MetalPointwiseOperation, incoming: Any) -> tuple[Any, ...]:
        nonlocal calls
        calls += 1
        return original(self, incoming)

    monkeypatch.setattr(MetalPointwiseOperation, "backward", spy)
    tensor.carrier[0] = 9.0
    with pytest.raises(RuntimeError, match="modified in-place"):
        result.backward(gradient)
    assert calls == 0
