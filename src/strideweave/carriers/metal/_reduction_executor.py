"""Lazy serial TileLang execution for Metal reductions and inclusive scans."""

from __future__ import annotations

from typing import Any, cast

from ...layout import Layout
from ...tensor import Tensor
from ..dtype import DType
from ..operation_policy import OperationPlan
from ._jit import (
    CompiledMetalKernel,
    LogicalKernel,
    MetalJITCache,
    MetalSpecializationKey,
)
from ._kernel_support import (
    PreparedTensor as _PreparedTensor,
)
from ._kernel_support import (
    address_tensor as _address_tensor,
)
from ._kernel_support import (
    compile_kernel as _shared_compile_kernel,
)
from ._kernel_support import (
    dtype_name as _dtype_name,
)
from ._kernel_support import (
    layout_address_tensor as _shared_layout_address_tensor,
)
from ._kernel_support import (
    plan_axis as _plan_axis,
)
from ._kernel_support import (
    prepare_tensor as _shared_prepare_tensor,
)

_THREADS = 64
_TEMPLATE_REVISION = "strideweave.metal.reduction.v1"
_PASS_CONFIGS = (("tl.disable_data_race_check", True),)
_FORWARD_REDUCTIONS = (
    "argmax",
    "argmin",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "reduce_sum",
)
_BACKWARD_REDUCTIONS = (
    "grad_reduce_max",
    "grad_reduce_min",
    "grad_reduce_prod",
    "grad_reduce_sum",
)
_SCAN_VARIANTS = ("cumsum", "grad_cumsum")
_LOGICAL_KERNELS = tuple(
    LogicalKernel(
        "metal.scan" if variant in _SCAN_VARIANTS else "metal.reduction",
        variant,
    )
    for variant in (*_FORWARD_REDUCTIONS, *_BACKWARD_REDUCTIONS, *_SCAN_VARIANTS)
)
_LOGICAL_BY_VARIANT = {kernel.variant: kernel for kernel in _LOGICAL_KERNELS}
_JIT_CACHE = MetalJITCache(_LOGICAL_KERNELS, max_specializations=256)


def _prepare_tensor(tensor: Tensor) -> _PreparedTensor:
    return _shared_prepare_tensor(tensor, family="reduction")


def _maximum(T: Any, lhs: Any, rhs: Any) -> Any:
    return T.if_then_else(
        T.isnan(lhs),
        lhs,
        T.if_then_else(
            T.isnan(rhs),
            rhs,
            T.if_then_else(
                lhs == rhs,
                T.if_then_else(lhs == 0.0, lhs + rhs, lhs),
                T.if_then_else(lhs > rhs, lhs, rhs),
            ),
        ),
    )


def _minimum(T: Any, lhs: Any, rhs: Any) -> Any:
    return T.if_then_else(
        T.isnan(lhs),
        lhs,
        T.if_then_else(
            T.isnan(rhs),
            rhs,
            T.if_then_else(
                lhs == rhs,
                T.if_then_else(lhs == 0.0, -((-lhs) + (-rhs)), lhs),
                T.if_then_else(lhs < rhs, lhs, rhs),
            ),
        ),
    )


def _build_reduction_forward(
    runtime: Any,
    operation: str,
    prepared: _PreparedTensor,
    output_dtype: DType,
    row_count: int,
    fiber_size: int,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])
    input_size = prepared.storage_size
    input_dtype = _dtype_name(prepared.storage_dtype)
    result_dtype = _dtype_name(output_dtype)

    if operation in {"reduce_sum", "reduce_prod"}:

        @T.macro
        def combine(lhs: Any, rhs: Any) -> Any:
            if operation == "reduce_sum":
                return lhs + rhs
            return lhs * rhs

        @tilelang.jit(out_idx=[2], target="metal", execution_backend="torch")
        def template(
            size: int,
            dtype: str,
            rows: int,
            fiber: int,
            output_name: str,
        ) -> Any:
            @T.prim_func
            def main(
                input0: T.Tensor((size,), dtype),  # pyright: ignore[reportInvalidTypeForm]
                addresses0: T.Tensor((rows * fiber,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output: T.Tensor((rows,), output_name),  # pyright: ignore[reportInvalidTypeForm]
            ):
                assert size >= 0 and dtype and rows >= 0 and fiber > 0 and output_name
                with T.Kernel(T.ceildiv(rows, _THREADS), threads=_THREADS) as block:
                    for thread in T.Parallel(_THREADS):
                        row = block * _THREADS + thread
                        if row < rows:
                            accumulator = T.alloc_fragment((1,), "float32")
                            accumulator[0] = 0.0 if operation == "reduce_sum" else 1.0
                            for ordinal in T.Serial(fiber):
                                accumulator[0] = combine(
                                    accumulator[0],
                                    input0[addresses0[row + rows * ordinal]],
                                )
                            output[row] = accumulator[0]

            return main

        return template.get_tir(
            input_size,
            input_dtype,
            row_count,
            fiber_size,
            result_dtype,
        ), 2

    if operation in {"reduce_max", "reduce_min"}:

        @T.macro
        def combine(lhs: Any, rhs: Any) -> Any:
            if operation == "reduce_max":
                return _maximum(T, lhs, rhs)
            return _minimum(T, lhs, rhs)

        @tilelang.jit(out_idx=[2], target="metal", execution_backend="torch")
        def template(
            size: int,
            dtype: str,
            rows: int,
            fiber: int,
            output_name: str,
        ) -> Any:
            @T.prim_func
            def main(
                input0: T.Tensor((size,), dtype),  # pyright: ignore[reportInvalidTypeForm]
                addresses0: T.Tensor((rows * fiber,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output: T.Tensor((rows,), output_name),  # pyright: ignore[reportInvalidTypeForm]
            ):
                assert size >= 0 and dtype and rows >= 0 and fiber > 0 and output_name
                with T.Kernel(T.ceildiv(rows, _THREADS), threads=_THREADS) as block:
                    for thread in T.Parallel(_THREADS):
                        row = block * _THREADS + thread
                        if row < rows:
                            accumulator = T.alloc_fragment((1,), "float32")
                            accumulator[0] = input0[addresses0[row]]
                            for ordinal in T.Serial(1, fiber):
                                accumulator[0] = combine(
                                    accumulator[0],
                                    input0[addresses0[row + rows * ordinal]],
                                )
                            output[row] = accumulator[0]

            return main

        return template.get_tir(
            input_size,
            input_dtype,
            row_count,
            fiber_size,
            result_dtype,
        ), 2

    maximum = operation == "argmax"

    @tilelang.jit(out_idx=[2], target="metal", execution_backend="torch")
    def template(
        size: int,
        dtype: str,
        rows: int,
        fiber: int,
        output_name: str,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((size,), dtype),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor((rows * fiber,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((rows,), output_name),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert size >= 0 and dtype and rows >= 0 and fiber > 0 and output_name
            with T.Kernel(T.ceildiv(rows, _THREADS), threads=_THREADS) as block:
                for thread in T.Parallel(_THREADS):
                    row = block * _THREADS + thread
                    if row < rows:
                        winner_value = T.alloc_fragment((1,), "float32")
                        winner_index = T.alloc_fragment((1,), "int32")
                        winner_value[0] = input0[addresses0[row]]
                        winner_index[0] = 0
                        for ordinal in T.Serial(1, fiber):
                            candidate = input0[addresses0[row + rows * ordinal]]
                            candidate_nan = T.isnan(candidate)
                            winner_nan = T.isnan(winner_value[0])
                            better = (
                                candidate > winner_value[0]
                                if maximum
                                else candidate < winner_value[0]
                            )
                            if (candidate_nan & (winner_nan == False)) | (  # noqa: E712
                                (candidate_nan == winner_nan) & better
                            ):
                                winner_value[0] = candidate
                                winner_index[0] = ordinal
                        output[row] = winner_index[0]

        return main

    return template.get_tir(
        input_size,
        input_dtype,
        row_count,
        fiber_size,
        result_dtype,
    ), 2


def _build_reduction_backward(
    runtime: Any,
    operation: str,
    source: _PreparedTensor,
    gradient: _PreparedTensor,
    row_count: int,
    fiber_size: int,
    output_storage_size: int,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])

    if operation == "grad_reduce_sum":

        @tilelang.jit(out_idx=[3], target="metal", execution_backend="torch")
        def template(size: int, rows: int, fiber: int, output_size: int) -> Any:
            @T.prim_func
            def main(
                input0: T.Tensor((size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
                addresses0: T.Tensor((rows,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output_addresses: T.Tensor((rows * fiber,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output: T.Tensor((output_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            ):
                assert size >= 0 and rows > 0 and fiber > 0 and output_size > 0
                with T.Kernel(
                    T.ceildiv(rows * fiber, _THREADS), threads=_THREADS
                ) as block:
                    for thread in T.Parallel(_THREADS):
                        index = block * _THREADS + thread
                        if index < rows * fiber:
                            output[output_addresses[index]] = input0[
                                addresses0[index % rows]
                            ]

            return main

        return template.get_tir(
            gradient.storage_size,
            row_count,
            fiber_size,
            output_storage_size,
        ), 3

    if operation == "grad_reduce_prod":

        @tilelang.jit(out_idx=[5], target="metal", execution_backend="torch")
        def template(
            source_size: int,
            gradient_size: int,
            rows: int,
            fiber: int,
            output_size: int,
        ) -> Any:
            @T.prim_func
            def main(
                input0: T.Tensor((source_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
                addresses0: T.Tensor((rows * fiber,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                input1: T.Tensor((gradient_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
                addresses1: T.Tensor((rows,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output_addresses: T.Tensor((rows * fiber,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output: T.Tensor((output_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            ):
                assert (
                    source_size >= 0
                    and gradient_size >= 0
                    and rows > 0
                    and fiber > 0
                    and output_size > 0
                )
                with T.Kernel(
                    T.ceildiv(rows * fiber, _THREADS), threads=_THREADS
                ) as block:
                    for thread in T.Parallel(_THREADS):
                        index = block * _THREADS + thread
                        if index < rows * fiber:
                            row = index % rows
                            reduced_ordinal = index // rows
                            product = T.alloc_fragment((1,), "float32")
                            product[0] = 1.0
                            for ordinal in T.Serial(fiber):
                                if ordinal != reduced_ordinal:
                                    product[0] = (
                                        product[0]
                                        * input0[addresses0[row + rows * ordinal]]
                                    )
                            output[output_addresses[index]] = (
                                input1[addresses1[row]] * product[0]
                            )

            return main

        return template.get_tir(
            source.storage_size,
            gradient.storage_size,
            row_count,
            fiber_size,
            output_storage_size,
        ), 5

    maximum = operation == "grad_reduce_max"

    @T.macro
    def combine(lhs: Any, rhs: Any) -> Any:
        if maximum:
            return _maximum(T, lhs, rhs)
        return _minimum(T, lhs, rhs)

    @tilelang.jit(out_idx=[5], target="metal", execution_backend="torch")
    def template(
        source_size: int,
        gradient_size: int,
        rows: int,
        fiber: int,
        output_size: int,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((source_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor((rows * fiber,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            input1: T.Tensor((gradient_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses1: T.Tensor((rows,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output_addresses: T.Tensor((rows * fiber,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((output_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert (
                source_size >= 0
                and gradient_size >= 0
                and rows > 0
                and fiber > 0
                and output_size > 0
            )
            with T.Kernel(T.ceildiv(rows * fiber, _THREADS), threads=_THREADS) as block:
                for thread in T.Parallel(_THREADS):
                    index = block * _THREADS + thread
                    if index < rows * fiber:
                        row = index % rows
                        reduced = T.alloc_fragment((1,), "float32")
                        winners = T.alloc_fragment((1,), "int32")
                        reduced[0] = input0[addresses0[row]]
                        for ordinal in T.Serial(1, fiber):
                            reduced[0] = combine(
                                reduced[0],
                                input0[addresses0[row + rows * ordinal]],
                            )
                        winners[0] = 0
                        for ordinal in T.Serial(fiber):
                            if input0[addresses0[row + rows * ordinal]] == reduced[0]:
                                winners[0] = winners[0] + 1
                        if T.isnan(reduced[0]):
                            output[output_addresses[index]] = float("nan")
                        elif input0[addresses0[index]] == reduced[0]:
                            output[output_addresses[index]] = input1[
                                addresses1[row]
                            ] / T.cast(winners[0], "float32")
                        else:
                            output[output_addresses[index]] = 0.0

        return main

    return template.get_tir(
        source.storage_size,
        gradient.storage_size,
        row_count,
        fiber_size,
        output_storage_size,
    ), 5


def _build_cumsum(
    runtime: Any,
    prepared: _PreparedTensor,
    logical_size: int,
    axis_stride: int,
    axis_extent: int,
    output_storage_size: int,
    *,
    reverse: bool,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])

    @tilelang.jit(out_idx=[3], target="metal", execution_backend="torch")
    def template(
        size: int,
        length: int,
        stride: int,
        extent: int,
        output_size: int,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor((length,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output_addresses: T.Tensor((length,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((output_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert (
                size >= 0
                and length > 0
                and stride > 0
                and extent > 0
                and output_size > 0
            )
            with T.Kernel(T.ceildiv(length, _THREADS), threads=_THREADS) as block:
                for thread in T.Parallel(_THREADS):
                    index = block * _THREADS + thread
                    if index < length:
                        coordinate = (index // stride) % extent
                        base = index - coordinate * stride
                        accumulator = T.alloc_fragment((1,), "float32")
                        accumulator[0] = 0.0
                        for serial_ordinal in T.Serial(extent):
                            position = (
                                extent - 1 - serial_ordinal
                                if reverse
                                else serial_ordinal
                            )
                            selected = (
                                position >= coordinate
                                if reverse
                                else position <= coordinate
                            )
                            if selected:
                                source_ordinal = base + position * stride
                                accumulator[0] = (
                                    accumulator[0] + input0[addresses0[source_ordinal]]
                                )
                        output[output_addresses[index]] = accumulator[0]

        return main

    return template.get_tir(
        prepared.storage_size,
        logical_size,
        axis_stride,
        axis_extent,
        output_storage_size,
    ), 3


def _compile_kernel(
    runtime: Any,
    prim_func: Any,
    output_index: int,
    expected_buffers: tuple[str, ...],
) -> CompiledMetalKernel:
    return _shared_compile_kernel(
        runtime,
        prim_func,
        output_index,
        expected_buffers,
        family="reduction",
        pass_configs=_PASS_CONFIGS,
    )


def _layout_address_tensor(runtime: Any, layout: Layout) -> tuple[object, Any]:
    return _shared_layout_address_tensor(runtime, layout, family="reduction")


def execute_reduction_forward(
    operation: str,
    tensor: Tensor,
    *,
    output_layout: Layout,
    output_dtype: DType,
    row_count: int,
    fiber_size: int,
    plan: OperationPlan,
) -> Tensor:
    """Launch one two-mode reduction into fresh canonical Metal storage."""
    logical_kernel = _LOGICAL_BY_VARIANT[operation]
    target_carrier = cast(Any, tensor.carrier)
    runtime = target_carrier._runtime
    prepared = _prepare_tensor(tensor)
    result_carrier = target_carrier.allocate_like(
        output_layout.cosize, dtype=output_dtype, empty=True
    )
    result = Tensor(result_carrier, 0, output_layout)
    if row_count == 0:
        return result

    axes = {
        "address_plan": prepared.address_key,
        "fiber_size": fiber_size,
        "operation": operation,
        "output_dtype": output_dtype.name,
        "pass_configs": _PASS_CONFIGS,
        "plan": _plan_axis(plan, operation),
        "row_count": row_count,
        "storage_dtype": prepared.storage_dtype.name,
        "storage_size": prepared.storage_size,
        "template_revision": _TEMPLATE_REVISION,
    }
    key = MetalSpecializationKey.from_axes(logical_kernel, axes)

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_reduction_forward(
            runtime,
            operation,
            prepared,
            output_dtype,
            row_count,
            fiber_size,
        )
        return _compile_kernel(
            runtime,
            prim_func,
            output_index,
            ("input0", "addresses0", "output"),
        )

    compiled = _JIT_CACHE.get_or_compile(key, compiler)
    compiled.executable(
        input0=prepared.source,
        addresses0=_address_tensor(runtime, prepared),
        output=result_carrier._require_storage(),
    )
    return result


def execute_reduction_backward(
    operation: str,
    tensor: Tensor,
    gradient: Tensor,
    *,
    output_layout: Layout,
    row_count: int,
    fiber_size: int,
) -> Tensor:
    """Launch one reduction VJP into fresh injective Metal storage."""
    logical_kernel = _LOGICAL_BY_VARIANT[operation]
    target_carrier = cast(Any, tensor.carrier)
    runtime = target_carrier._runtime
    source = _prepare_tensor(tensor)
    prepared_gradient = _prepare_tensor(gradient)
    output_address_key, output_addresses = _layout_address_tensor(
        runtime, output_layout
    )
    result_carrier = target_carrier.allocate_like(
        output_layout.cosize,
        dtype=DType.Float32,
        empty=output_layout.cosize == output_layout.size,
    )
    result = Tensor(result_carrier, 0, output_layout)
    if output_layout.size == 0:
        return result

    axes = {
        "fiber_size": fiber_size,
        "gradient_address_plan": prepared_gradient.address_key,
        "gradient_storage_size": prepared_gradient.storage_size,
        "operation": operation,
        "output_address_plan": output_address_key,
        "output_storage_size": output_layout.cosize,
        "pass_configs": _PASS_CONFIGS,
        "plan": _plan_axis(None, operation),
        "row_count": row_count,
        "source_address_plan": source.address_key,
        "source_storage_size": source.storage_size,
        "template_revision": _TEMPLATE_REVISION,
    }
    key = MetalSpecializationKey.from_axes(logical_kernel, axes)

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_reduction_backward(
            runtime,
            operation,
            source,
            prepared_gradient,
            row_count,
            fiber_size,
            output_layout.cosize,
        )
        expected = (
            ("input0", "addresses0", "output_addresses", "output")
            if operation == "grad_reduce_sum"
            else (
                "input0",
                "addresses0",
                "input1",
                "addresses1",
                "output_addresses",
                "output",
            )
        )
        return _compile_kernel(runtime, prim_func, output_index, expected)

    compiled = _JIT_CACHE.get_or_compile(key, compiler)
    buffers = {
        "output": result_carrier._require_storage(),
        "output_addresses": output_addresses,
    }
    if operation == "grad_reduce_sum":
        buffers.update(
            input0=prepared_gradient.source,
            addresses0=_address_tensor(runtime, prepared_gradient),
        )
    else:
        buffers.update(
            input0=source.source,
            addresses0=_address_tensor(runtime, source),
            input1=prepared_gradient.source,
            addresses1=_address_tensor(runtime, prepared_gradient),
        )
    compiled.executable(**buffers)
    return result


def execute_cumsum(
    tensor: Tensor,
    *,
    output_layout: Layout,
    axis_stride: int,
    axis_extent: int,
    reverse: bool = False,
    plan: OperationPlan | None = None,
) -> Tensor:
    """Launch one forward or reverse inclusive serial scan."""
    variant = "grad_cumsum" if reverse else "cumsum"
    logical_kernel = _LOGICAL_BY_VARIANT[variant]
    target_carrier = cast(Any, tensor.carrier)
    runtime = target_carrier._runtime
    prepared = _prepare_tensor(tensor)
    output_address_key, output_addresses = _layout_address_tensor(
        runtime, output_layout
    )
    result_carrier = target_carrier.allocate_like(
        output_layout.cosize,
        dtype=DType.Float32,
        empty=output_layout.cosize == output_layout.size,
    )
    result = Tensor(result_carrier, 0, output_layout)
    logical_size = output_layout.size
    if logical_size == 0:
        return result

    axes = {
        "address_plan": prepared.address_key,
        "axis_extent": axis_extent,
        "axis_stride": axis_stride,
        "logical_size": logical_size,
        "operation": variant,
        "output_address_plan": output_address_key,
        "output_storage_size": output_layout.cosize,
        "pass_configs": _PASS_CONFIGS,
        "plan": _plan_axis(plan, variant),
        "storage_size": prepared.storage_size,
        "template_revision": _TEMPLATE_REVISION,
    }
    key = MetalSpecializationKey.from_axes(logical_kernel, axes)

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_cumsum(
            runtime,
            prepared,
            logical_size,
            axis_stride,
            axis_extent,
            output_layout.cosize,
            reverse=reverse,
        )
        return _compile_kernel(
            runtime,
            prim_func,
            output_index,
            ("input0", "addresses0", "output_addresses", "output"),
        )

    compiled = _JIT_CACHE.get_or_compile(key, compiler)
    compiled.executable(
        input0=prepared.source,
        addresses0=_address_tensor(runtime, prepared),
        output_addresses=output_addresses,
        output=result_carrier._require_storage(),
    )
    return result


__all__: list[str] = []
