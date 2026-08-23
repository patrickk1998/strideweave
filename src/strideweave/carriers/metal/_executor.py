"""Lazy TileLang compilation and launch for Metal pointwise expressions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, cast

from ...layout import Layout
from ...tensor import Tensor
from ..dtype import DType
from ..generic.numerics import normalize_storage_value
from ..operation_policy import OperandPlan, OperandRole, OperationPlan
from ._address_plan import address_plan
from ._jit import (
    CompiledMetalKernel,
    GeneratedMetalFacts,
    LogicalKernel,
    MetalJITCache,
    MetalSpecializationKey,
)
from ._recipe import PointwiseRecipe, pointwise_recipe

_THREADS = 64
_TEMPLATE_REVISION = "strideweave.metal.pointwise.v1"
_SCALAR_ADDRESS_KEY = ("strideweave.metal.scalar-address.v1",)

_FORWARD_EXPRESSIONS = (
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

_BACKWARD_EXPRESSIONS = (
    "grad_abs",
    "grad_add",
    "grad_ceil",
    "grad_clamp_lower",
    "grad_clamp_tensor",
    "grad_clamp_upper",
    "grad_cos",
    "grad_div_lhs",
    "grad_div_rhs",
    "grad_elu",
    "grad_erf",
    "grad_exp",
    "grad_exp2",
    "grad_floor",
    "grad_gelu",
    "grad_leaky_relu",
    "grad_log",
    "grad_log2",
    "grad_maximum_lhs",
    "grad_maximum_rhs",
    "grad_minimum_lhs",
    "grad_minimum_rhs",
    "grad_mul_lhs",
    "grad_mul_rhs",
    "grad_neg",
    "grad_pow_base",
    "grad_pow_exponent",
    "grad_pow_scalar_base",
    "grad_pow_scalar_exponent",
    "grad_recip",
    "grad_relu",
    "grad_rem_lhs",
    "grad_rem_rhs",
    "grad_round",
    "grad_rsqrt",
    "grad_select_false",
    "grad_select_true",
    "grad_sigmoid",
    "grad_sign",
    "grad_silu",
    "grad_sin",
    "grad_softplus",
    "grad_sqrt",
    "grad_sub_lhs",
    "grad_sub_rhs",
    "grad_tanh",
)

_LOGICAL_KERNELS = tuple(
    LogicalKernel("metal.pointwise", expression)
    for expression in (*_FORWARD_EXPRESSIONS, *_BACKWARD_EXPRESSIONS)
)
_LOGICAL_BY_EXPRESSION = {logical.variant: logical for logical in _LOGICAL_KERNELS}
_JIT_CACHE = MetalJITCache(_LOGICAL_KERNELS, max_specializations=512)

_BUFFER_PATTERN = re.compile(
    r"\b(?P<name>output|input\d+|addresses)\b\s*"
    r"\[\[\s*buffer\((?P<index>\d+)\)\s*\]\]"
)


@dataclass(frozen=True, slots=True)
class _PreparedOperand:
    source: Any
    storage_dtype: DType
    convert_dtype: DType
    storage_size: int
    addresses: tuple[int, ...]
    address_key: object


@dataclass(frozen=True, slots=True)
class _OperandRecipe:
    storage_dtype: DType
    convert_dtype: DType
    storage_size: int


def _dtype_name(dtype: DType) -> str:
    if dtype is DType.Float32:
        return "float32"
    if dtype is DType.Int32:
        return "int32"
    if dtype is DType.Bool:
        return "bool"
    raise TypeError(f"Metal pointwise kernels do not support DType.{dtype.name}")


def _torch_dtype(runtime: Any, dtype: DType) -> Any:
    if dtype is DType.Float32:
        return runtime.torch.float32
    if dtype is DType.Int32:
        return runtime.torch.int32
    if dtype is DType.Bool:
        return runtime.torch.bool
    raise TypeError(f"Metal pointwise kernels do not support DType.{dtype.name}")


def _prepare_operand(
    runtime: Any,
    operand: Any,
    operand_plan: OperandPlan,
    logical_size: int,
) -> _PreparedOperand:
    if operand_plan.role is OperandRole.TENSOR:
        if not isinstance(operand, Tensor):
            raise TypeError("planned tensor operand must be a Tensor")
        carrier = cast(Any, operand.carrier)
        plan = address_plan(operand.layout, offset=operand.offset)
        if any(
            address < 0 or address >= carrier.size() or address > 2**31 - 1
            for address in plan.addresses
        ):
            raise RuntimeError(
                "Metal pointwise address plan contains an address outside the "
                "source storage or Int32 map range"
            )
        return _PreparedOperand(
            source=carrier._require_storage(),
            storage_dtype=operand.dtype(),
            convert_dtype=operand_plan.convert_to,
            storage_size=carrier.size(),
            addresses=plan.addresses,
            address_key=plan.key,
        )

    value = normalize_storage_value(
        operand_plan.convert_to, operand, name="Metal weak scalar"
    )
    source = runtime.torch.tensor(
        [value],
        dtype=_torch_dtype(runtime, operand_plan.convert_to),
        device="mps",
    )
    return _PreparedOperand(
        source=source,
        storage_dtype=operand_plan.convert_to,
        convert_dtype=operand_plan.convert_to,
        storage_size=1,
        addresses=(0,) * logical_size,
        address_key=_SCALAR_ADDRESS_KEY,
    )


def _synthetic_operand_plan(operand: Any, convert_dtype: DType) -> OperandPlan:
    from ..dtype import SimpleDType

    converted = cast(SimpleDType, convert_dtype)
    if isinstance(operand, Tensor):
        return OperandPlan(OperandRole.TENSOR, operand.dtype(), converted)
    return OperandPlan(OperandRole.WEAK_SCALAR, None, converted)


def _maximum(T: Any, lhs: Any, rhs: Any) -> Any:
    zero_tie = lhs + rhs
    return T.if_then_else(
        T.isnan(lhs),
        lhs,
        T.if_then_else(
            T.isnan(rhs),
            rhs,
            T.if_then_else(
                lhs == rhs,
                T.if_then_else(lhs == 0.0, zero_tie, lhs),
                T.if_then_else(lhs > rhs, lhs, rhs),
            ),
        ),
    )


def _minimum(T: Any, lhs: Any, rhs: Any) -> Any:
    zero_tie = -((-lhs) + (-rhs))
    return T.if_then_else(
        T.isnan(lhs),
        lhs,
        T.if_then_else(
            T.isnan(rhs),
            rhs,
            T.if_then_else(
                lhs == rhs,
                T.if_then_else(lhs == 0.0, zero_tie, lhs),
                T.if_then_else(lhs < rhs, lhs, rhs),
            ),
        ),
    )


def _sigmoid(T: Any, value: Any) -> Any:
    positive = 1.0 / (1.0 + T.exp(-value))
    exponential = T.exp(value)
    negative = exponential / (1.0 + exponential)
    return T.if_then_else(value >= 0.0, positive, negative)


def _expm1(T: Any, value: Any) -> Any:
    # MSL in TileLang 0.1.12 has no usable ``expm1`` symbol. Evaluate the
    # cancellation-sensitive neighborhood with a ninth-order Horner series;
    # outside it, subtraction from exp has enough Float32 separation.
    series = 1.0 / math.factorial(9)
    for order in range(8, 0, -1):
        series = 1.0 / math.factorial(order) + value * series
    return T.if_then_else(
        T.abs(value) < 0.5,
        value * series,
        T.exp(value) - 1.0,
    )


def _log1p(T: Any, value: Any) -> Any:
    # TileLang currently lowers ``T.log1p`` to an unavailable ``log1pf`` MSL
    # symbol. The alternating series retains tiny positive results that
    # ``log(1 + value)`` rounds to zero; direct log is stable above 0.25.
    series = -1.0 / 12.0
    for order in range(11, 0, -1):
        coefficient = (1.0 if order % 2 else -1.0) / order
        series = coefficient + value * series
    return T.if_then_else(value < 0.25, value * series, T.log(1.0 + value))


def _softplus(T: Any, value: Any) -> Any:
    return _log1p(T, T.exp(-T.abs(value))) + _maximum(T, value, 0.0)


def _extrema_multiplier(
    T: Any, lhs: Any, rhs: Any, *, maximum: bool, lhs_side: bool
) -> Any:
    selected_lhs = lhs > rhs if maximum else lhs < rhs
    strict = T.if_then_else(selected_lhs == lhs_side, 1.0, 0.0)
    finite_tie = (lhs == rhs) & T.isfinite(lhs)
    return T.if_then_else(
        T.isnan(lhs) | T.isnan(rhs),
        float("nan"),
        T.if_then_else(finite_tie, 0.5, strict),
    )


def _expression(T: Any, name: str, values: tuple[Any, ...]) -> Any:
    x = values[0]
    if name in {"grad_add", "grad_sub_lhs", "grad_rem_lhs"}:
        return x
    if name in {"grad_neg", "grad_sub_rhs"}:
        return -x
    if name in {"grad_sign", "grad_floor", "grad_ceil", "grad_round"}:
        return x * 0.0
    if name == "neg":
        return -x
    if name == "abs":
        return T.abs(x)
    if name == "sign":
        return T.if_then_else(
            T.isnan(x),
            x,
            T.if_then_else(x > 0.0, 1.0, T.if_then_else(x < 0.0, -1.0, 0.0)),
        )
    if name == "recip":
        return 1.0 / x
    if name == "sqrt":
        return T.sqrt(x)
    if name == "rsqrt":
        return T.rsqrt(x)
    if name == "exp2":
        return T.exp2(x)
    if name == "log":
        return T.log(x)
    if name == "log2":
        return T.log2(x)
    if name == "sin":
        return T.sin(x)
    if name == "cos":
        return T.cos(x)
    if name == "erf":
        return T.erf(x)
    if name == "floor":
        return T.if_then_else(x == 0.0, x, T.floor(x))
    if name == "ceil":
        return T.if_then_else(x == 0.0, x, T.ceil(x))
    if name == "round":
        return T.if_then_else(x == 0.0, x, T.round(x))
    if name == "exp":
        return T.exp(x)
    if name == "relu":
        return T.if_then_else(x > 0.0, x, 0.0)
    if name == "sigmoid":
        return _sigmoid(T, x)
    if name == "tanh":
        return T.if_then_else(x == 0.0, x, T.tanh(x))
    if name == "gelu":
        return 0.5 * x * (1.0 + T.erf(x * 0.7071067811865476))
    if name == "silu":
        return x * _sigmoid(T, x)
    if name == "softplus":
        return _softplus(T, x)
    if name == "elu":
        return T.if_then_else(x > 0.0, x, _expm1(T, x))
    if name == "leaky_relu":
        return T.if_then_else(x >= 0.0, x, 0.01 * x)
    if name == "logical_not":
        return x == 0.0

    y = values[1]
    if name == "add":
        return x + y
    if name == "sub":
        return x - y
    if name in {"elementwise_mul", "mul"}:
        return x * y
    if name == "div":
        return x / y
    if name == "pow":
        return T.pow(x, y)
    if name == "maximum":
        return _maximum(T, x, y)
    if name == "minimum":
        return _minimum(T, x, y)
    if name == "rem":
        return T.fmod(x, y)
    if name == "eq":
        return x == y
    if name == "ne":
        return x != y
    if name == "lt":
        return x < y
    if name == "le":
        return x <= y

    incoming = x
    value = y
    if name == "grad_abs":
        multiplier = T.if_then_else(
            T.isnan(value),
            value,
            T.if_then_else(
                value == 0.0,
                0.0,
                T.if_then_else(value > 0.0, 1.0, -1.0),
            ),
        )
        return incoming * multiplier
    if name == "grad_recip":
        return -incoming / (value * value)
    if name == "grad_sqrt":
        return incoming / (2.0 * T.sqrt(value))
    if name == "grad_rsqrt":
        result = T.rsqrt(value)
        return -incoming * result * result * result / 2.0
    if name == "grad_exp2":
        return incoming * 0.6931471805599453 * T.exp2(value)
    if name == "grad_log":
        return incoming / value
    if name == "grad_log2":
        return incoming / (value * 0.6931471805599453)
    if name == "grad_sin":
        return incoming * T.cos(value)
    if name == "grad_cos":
        return -incoming * T.sin(value)
    if name == "grad_erf":
        return incoming * 1.1283791670955126 * T.exp(-(value * value))
    if name == "grad_exp":
        return incoming * T.exp(value)
    if name == "grad_relu":
        return incoming * T.if_then_else(value > 0.0, 1.0, 0.0)
    if name == "grad_sigmoid":
        sigmoid = _sigmoid(T, value)
        return incoming * sigmoid * (1.0 - sigmoid)
    if name == "grad_tanh":
        result = T.tanh(value)
        return incoming * (1.0 - result * result)
    if name == "grad_gelu":
        derivative = (
            0.5 * (1.0 + T.erf(value * 0.7071067811865476))
            + value * T.exp(-0.5 * value * value) * 0.3989422804014327
        )
        return incoming * derivative
    if name == "grad_silu":
        sigmoid = _sigmoid(T, value)
        return incoming * (sigmoid + value * sigmoid * (1.0 - sigmoid))
    if name == "grad_softplus":
        return incoming * _sigmoid(T, value)
    if name == "grad_elu":
        return incoming * T.if_then_else(value > 0.0, 1.0, T.exp(value))
    if name == "grad_leaky_relu":
        return incoming * T.if_then_else(value >= 0.0, 1.0, 0.01)
    if name in {"grad_mul_lhs", "grad_mul_rhs", "grad_div_lhs"}:
        other = values[1]
        if name.startswith("grad_mul"):
            return incoming * other
        return incoming / other

    if name == "grad_pow_scalar_exponent":
        other = values[2]
        return incoming * other * T.pow(value, other - 1.0)

    if name in {
        "grad_div_rhs",
        "grad_rem_rhs",
        "grad_pow_base",
        "grad_pow_exponent",
        "grad_pow_scalar_base",
        "grad_maximum_lhs",
        "grad_maximum_rhs",
        "grad_minimum_lhs",
        "grad_minimum_rhs",
    }:
        lhs = values[1]
        rhs = values[2]
        if name == "grad_div_rhs":
            return -incoming * lhs / (rhs * rhs)
        if name == "grad_rem_rhs":
            return incoming * -T.trunc(lhs / rhs)
        if name == "grad_pow_base":
            return incoming * rhs * T.pow(lhs, rhs - 1.0)
        if name == "grad_pow_exponent":
            return incoming * T.pow(lhs, rhs) * T.log(lhs)
        if name == "grad_pow_scalar_base":
            return incoming * T.pow(lhs, rhs) * T.log(lhs)
        maximum = "maximum" in name
        lhs_side = name.endswith("lhs")
        return incoming * _extrema_multiplier(
            T, lhs, rhs, maximum=maximum, lhs_side=lhs_side
        )

    if name in {"select", "grad_select_true", "grad_select_false"}:
        condition = x
        if name == "select":
            return T.if_then_else(condition, values[1], values[2])
        selected = condition if name.endswith("true") else condition == False  # noqa: E712
        return T.if_then_else(selected, values[1], 0.0)

    if name == "clamp":
        return _minimum(T, _maximum(T, x, values[1]), values[2])

    if name.startswith("grad_clamp_"):
        incoming, data, low, high = values
        middle = _maximum(T, data, low)
        middle_gradient = incoming * _extrema_multiplier(
            T, middle, high, maximum=False, lhs_side=True
        )
        if name == "grad_clamp_upper":
            return incoming * _extrema_multiplier(
                T, middle, high, maximum=False, lhs_side=False
            )
        return middle_gradient * _extrema_multiplier(
            T,
            data,
            low,
            maximum=True,
            lhs_side=name == "grad_clamp_tensor",
        )

    raise AssertionError(f"unknown Metal pointwise expression {name!r}")


def _converted(T: Any, value: Any, storage: DType, convert: DType) -> Any:
    if storage is convert:
        return value
    return T.cast(value, _dtype_name(convert))


def _build_prim_func(
    runtime: Any,
    expression: str,
    operands: tuple[_OperandRecipe, ...],
    output_dtype: DType,
    logical_size: int,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])
    arity = len(operands)
    if not 1 <= arity <= 4:
        raise ValueError("Metal pointwise expressions require one to four operands")
    storage_dtypes = tuple(_dtype_name(item.storage_dtype) for item in operands)
    output_name = _dtype_name(output_dtype)
    storage_sizes = tuple(item.storage_size for item in operands)

    @T.macro
    def compute(*loaded: Any) -> Any:
        converted = tuple(
            _converted(T, value, item.storage_dtype, item.convert_dtype)
            for value, item in zip(loaded, operands, strict=True)
        )
        return _expression(T, expression, converted)

    if arity == 1:

        @tilelang.jit(out_idx=[2], target="metal", execution_backend="torch")
        def template(size0: int, dtype0: str, length: int, result_dtype: str) -> Any:
            @T.prim_func
            def main(
                input0: T.Tensor((size0,), dtype0),  # pyright: ignore[reportInvalidTypeForm]
                addresses: T.Tensor((1, length), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output: T.Tensor((length,), result_dtype),  # pyright: ignore[reportInvalidTypeForm]
            ):
                assert size0 >= 0 and dtype0 and length >= 0 and result_dtype
                with T.Kernel(T.ceildiv(length, _THREADS), threads=_THREADS) as block:
                    for thread in T.Parallel(_THREADS):
                        index = block * _THREADS + thread
                        if index < length:
                            output[index] = compute(input0[addresses[0, index]])

            return main

        return template.get_tir(
            storage_sizes[0], storage_dtypes[0], logical_size, output_name
        ), 2

    if arity == 2:

        @tilelang.jit(out_idx=[3], target="metal", execution_backend="torch")
        def template(
            size0: int,
            dtype0: str,
            size1: int,
            dtype1: str,
            length: int,
            result_dtype: str,
        ) -> Any:
            @T.prim_func
            def main(
                input0: T.Tensor((size0,), dtype0),  # pyright: ignore[reportInvalidTypeForm]
                input1: T.Tensor((size1,), dtype1),  # pyright: ignore[reportInvalidTypeForm]
                addresses: T.Tensor((2, length), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output: T.Tensor((length,), result_dtype),  # pyright: ignore[reportInvalidTypeForm]
            ):
                assert (
                    size0 >= 0
                    and dtype0
                    and size1 >= 0
                    and dtype1
                    and length >= 0
                    and result_dtype
                )
                with T.Kernel(T.ceildiv(length, _THREADS), threads=_THREADS) as block:
                    for thread in T.Parallel(_THREADS):
                        index = block * _THREADS + thread
                        if index < length:
                            output[index] = compute(
                                input0[addresses[0, index]],
                                input1[addresses[1, index]],
                            )

            return main

        return template.get_tir(
            storage_sizes[0],
            storage_dtypes[0],
            storage_sizes[1],
            storage_dtypes[1],
            logical_size,
            output_name,
        ), 3

    if arity == 3:

        @tilelang.jit(out_idx=[4], target="metal", execution_backend="torch")
        def template(
            size0: int,
            dtype0: str,
            size1: int,
            dtype1: str,
            size2: int,
            dtype2: str,
            length: int,
            result_dtype: str,
        ) -> Any:
            @T.prim_func
            def main(
                input0: T.Tensor((size0,), dtype0),  # pyright: ignore[reportInvalidTypeForm]
                input1: T.Tensor((size1,), dtype1),  # pyright: ignore[reportInvalidTypeForm]
                input2: T.Tensor((size2,), dtype2),  # pyright: ignore[reportInvalidTypeForm]
                addresses: T.Tensor((3, length), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output: T.Tensor((length,), result_dtype),  # pyright: ignore[reportInvalidTypeForm]
            ):
                assert (
                    size0 >= 0
                    and dtype0
                    and size1 >= 0
                    and dtype1
                    and size2 >= 0
                    and dtype2
                    and length >= 0
                    and result_dtype
                )
                with T.Kernel(T.ceildiv(length, _THREADS), threads=_THREADS) as block:
                    for thread in T.Parallel(_THREADS):
                        index = block * _THREADS + thread
                        if index < length:
                            output[index] = compute(
                                input0[addresses[0, index]],
                                input1[addresses[1, index]],
                                input2[addresses[2, index]],
                            )

            return main

        return template.get_tir(
            storage_sizes[0],
            storage_dtypes[0],
            storage_sizes[1],
            storage_dtypes[1],
            storage_sizes[2],
            storage_dtypes[2],
            logical_size,
            output_name,
        ), 4

    @tilelang.jit(out_idx=[5], target="metal", execution_backend="torch")
    def template(
        size0: int,
        dtype0: str,
        size1: int,
        dtype1: str,
        size2: int,
        dtype2: str,
        size3: int,
        dtype3: str,
        length: int,
        result_dtype: str,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((size0,), dtype0),  # pyright: ignore[reportInvalidTypeForm]
            input1: T.Tensor((size1,), dtype1),  # pyright: ignore[reportInvalidTypeForm]
            input2: T.Tensor((size2,), dtype2),  # pyright: ignore[reportInvalidTypeForm]
            input3: T.Tensor((size3,), dtype3),  # pyright: ignore[reportInvalidTypeForm]
            addresses: T.Tensor((4, length), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((length,), result_dtype),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert (
                size0 >= 0
                and dtype0
                and size1 >= 0
                and dtype1
                and size2 >= 0
                and dtype2
                and size3 >= 0
                and dtype3
                and length >= 0
                and result_dtype
            )
            with T.Kernel(T.ceildiv(length, _THREADS), threads=_THREADS) as block:
                for thread in T.Parallel(_THREADS):
                    index = block * _THREADS + thread
                    if index < length:
                        output[index] = compute(
                            input0[addresses[0, index]],
                            input1[addresses[1, index]],
                            input2[addresses[2, index]],
                            input3[addresses[3, index]],
                        )

        return main

    return template.get_tir(
        storage_sizes[0],
        storage_dtypes[0],
        storage_sizes[1],
        storage_dtypes[1],
        storage_sizes[2],
        storage_dtypes[2],
        storage_sizes[3],
        storage_dtypes[3],
        logical_size,
        output_name,
    ), 5


def _buffer_order(device_source: str, arity: int) -> tuple[str, ...]:
    expected = {"output", "addresses", *(f"input{index}" for index in range(arity))}
    bindings: dict[str, int] = {}
    for match in _BUFFER_PATTERN.finditer(device_source):
        name = match.group("name")
        index = int(match.group("index"))
        if name in bindings and bindings[name] != index:
            raise RuntimeError(f"TileLang emitted ambiguous Metal buffer {name!r}")
        bindings[name] = index
    if set(bindings) != expected or set(bindings.values()) != set(range(len(expected))):
        raise RuntimeError(
            "TileLang emitted an unexpected Metal buffer ABI; "
            f"expected={sorted(expected)}, bindings={bindings}"
        )
    return tuple(name for name, _ in sorted(bindings.items(), key=lambda item: item[1]))


def _compile_kernel(
    runtime: Any,
    key: MetalSpecializationKey,
    expression: str,
    operands: tuple[_OperandRecipe, ...],
    output_dtype: DType,
    logical_size: int,
) -> CompiledMetalKernel:
    prim_func, output_index = _build_prim_func(
        runtime, expression, operands, output_dtype, logical_size
    )
    kernel = runtime.tilelang.JITKernel(
        prim_func,
        out_idx=[output_index],
        target="metal",
        execution_backend="torch",
    )
    device_source = kernel.kernel_source
    host_source = kernel.host_source
    order = _buffer_order(device_source, len(operands))

    def executable(output: Any, *canonical: Any) -> object:
        sources = canonical[:-1]
        addresses = canonical[-1]
        named = {"output": output, "addresses": addresses}
        named.update({f"input{index}": source for index, source in enumerate(sources)})
        return kernel(*(named[name] for name in order))

    facts = GeneratedMetalFacts(
        provider="tilelang.JITKernel",
        target="metal",
        execution_backend="torch",
        tilelang_version=str(runtime.tilelang.__version__),
        torch_version=str(runtime.torch.__version__),
        device_source=device_source,
        host_source=host_source,
        runtime_artifact_digest=None,
    )
    return CompiledMetalKernel(executable, facts)


def _recompile_from_key(
    runtime: Any, key: MetalSpecializationKey
) -> CompiledMetalKernel:
    """Compile pointwise facts from immutable specialization axes only."""
    if key.logical_kernel not in _LOGICAL_KERNELS:
        raise RuntimeError("unknown Metal pointwise reconstruction recipe")
    recipe = _recipe_from_key(key)
    return _recompile_recipe(runtime, key, recipe)


def _recipe_from_key(key: MetalSpecializationKey) -> PointwiseRecipe:
    if key.logical_kernel not in _LOGICAL_KERNELS:
        raise RuntimeError("unknown Metal pointwise reconstruction recipe")
    return pointwise_recipe(key, template_revision=_TEMPLATE_REVISION)


def _recompile_recipe(
    runtime: Any,
    key: MetalSpecializationKey,
    recipe: PointwiseRecipe,
) -> CompiledMetalKernel:
    prepared = tuple(
        _OperandRecipe(item.storage_dtype, item.convert_dtype, item.storage_size)
        for item in recipe.operands
    )
    return _compile_kernel(
        runtime,
        key,
        recipe.expression,
        prepared,
        recipe.output_dtype,
        recipe.logical_size,
    )


def execute_expression(
    expression: str,
    operands: tuple[Any, ...],
    *,
    output_layout: Layout,
    output_dtype: DType = DType.Float32,
    plan: OperationPlan | None = None,
    convert_dtypes: tuple[DType, ...] | None = None,
) -> Tensor:
    """Compile/cache/launch one exact expression into fresh Metal storage."""
    try:
        logical_kernel = _LOGICAL_BY_EXPRESSION[expression]
    except KeyError as exc:
        raise ValueError(f"unknown Metal pointwise expression {expression!r}") from exc
    if not operands:
        raise ValueError("Metal pointwise execution requires operands")
    tensor_operands = tuple(item for item in operands if isinstance(item, Tensor))
    if not tensor_operands:
        raise TypeError("Metal pointwise execution requires a Tensor operand")
    target = tensor_operands[0]
    target_carrier = cast(Any, target.carrier)
    runtime = target_carrier._runtime
    logical_size = output_layout.size

    if plan is not None:
        operand_plans = plan.operands
    else:
        if convert_dtypes is None or len(convert_dtypes) != len(operands):
            raise ValueError("internal Metal expression conversion arity mismatch")
        operand_plans = tuple(
            _synthetic_operand_plan(operand, convert_dtype)
            for operand, convert_dtype in zip(operands, convert_dtypes, strict=True)
        )
    if len(operand_plans) != len(operands):
        raise ValueError("Metal operation-plan operand arity mismatch")

    prepared = tuple(
        _prepare_operand(runtime, operand, operand_plan, logical_size)
        for operand, operand_plan in zip(operands, operand_plans, strict=True)
    )
    result_carrier = target_carrier.allocate_like(
        output_layout.cosize, dtype=output_dtype, empty=True
    )
    result = Tensor(result_carrier, 0, output_layout)
    if logical_size == 0:
        return result

    axes = {
        "address_plans": tuple(item.address_key for item in prepared),
        "convert_dtypes": tuple(item.convert_dtype.name for item in prepared),
        "expression": expression,
        "logical_size": logical_size,
        "output_dtype": output_dtype.name,
        "physical_sizes": tuple(item.storage_size for item in prepared),
        "plan": (
            (
                plan.operation,
                tuple(
                    (
                        operand.role.value,
                        operand.dtype.name if operand.dtype is not None else None,
                        operand.convert_to.name,
                    )
                    for operand in plan.operands
                ),
                plan.compute.value,
                plan.accumulation.value if plan.accumulation is not None else None,
                (
                    plan.accumulator_dtype.name
                    if plan.accumulator_dtype is not None
                    else None
                ),
                plan.output.name,
            )
            if plan is not None
            else ("internal-gradient", expression)
        ),
        "storage_dtypes": tuple(item.storage_dtype.name for item in prepared),
        "template_revision": _TEMPLATE_REVISION,
    }
    key = MetalSpecializationKey.from_axes(logical_kernel, axes)
    recipe_operands = tuple(
        _OperandRecipe(
            storage_dtype=item.storage_dtype,
            convert_dtype=item.convert_dtype,
            storage_size=item.storage_size,
        )
        for item in prepared
    )
    compiled = _JIT_CACHE.get_or_compile(
        key,
        lambda specialization: _compile_kernel(
            runtime,
            specialization,
            expression,
            recipe_operands,
            output_dtype,
            logical_size,
        ),
    )
    addresses = runtime.torch.tensor(
        [item.addresses for item in prepared],
        dtype=runtime.torch.int32,
        device="mps",
    )
    compiled.executable(
        result_carrier._require_storage(),
        *(item.source for item in prepared),
        addresses,
    )
    return result


__all__: list[str] = []
