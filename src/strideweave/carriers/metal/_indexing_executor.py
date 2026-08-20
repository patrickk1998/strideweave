"""Lazy TileLang execution for Metal indexing, scatter, sort, and top-k."""

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
_TEMPLATE_REVISION = "strideweave.metal.indexing.v1"
_PASS_CONFIGS = (("tl.disable_data_race_check", True),)
_VARIANTS = (
    "validate_indices",
    "gather",
    "scatter",
    "scatter_add",
    "grad_gather",
    "grad_scatter_base",
    "grad_scatter_add_base",
    "grad_scatter_updates",
    "_sort_values",
    "_sort_indices",
    "_topk_values",
    "_topk_indices",
    "grad_selection",
)
_LOGICAL_KERNELS = tuple(
    LogicalKernel(
        "metal.selection"
        if "sort" in variant or "topk" in variant or variant == "grad_selection"
        else "metal.indexing",
        variant,
    )
    for variant in _VARIANTS
)
_LOGICAL_BY_VARIANT = {kernel.variant: kernel for kernel in _LOGICAL_KERNELS}
_JIT_CACHE = MetalJITCache(_LOGICAL_KERNELS, max_specializations=256)


def _prepare_tensor(tensor: Tensor) -> _PreparedTensor:
    return _shared_prepare_tensor(tensor, family="indexing")


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
        family="indexing",
        pass_configs=_PASS_CONFIGS,
    )


def _layout_address_tensor(runtime: Any, layout: Layout) -> tuple[object, Any]:
    return _shared_layout_address_tensor(runtime, layout, family="indexing")


def _build_validate_indices(
    runtime: Any,
    prepared: _PreparedTensor,
    extent: int,
    unique: bool,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])
    logical_size = len(prepared.addresses)

    @tilelang.jit(out_idx=[2], target="metal", execution_backend="torch")
    def template(size: int, logical: int, selected_extent: int, distinct: bool) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((size,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor((logical,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            status: T.Tensor((3,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert size >= 0 and logical >= 0 and selected_extent > 0
            with T.Kernel(1, threads=1):
                status[0] = 0
                status[1] = 0
                status[2] = 0
                for ordinal in T.Serial(logical):
                    value = input0[addresses0[ordinal]]
                    if status[0] == 0:
                        if value < 0:
                            status[0] = 1
                            status[1] = value
                            status[2] = ordinal
                        elif value >= selected_extent:
                            status[0] = 1
                            status[1] = value
                            status[2] = ordinal
                        elif distinct:
                            for previous in T.Serial(logical):
                                if previous < ordinal:
                                    if status[0] == 0:
                                        if input0[addresses0[previous]] == value:
                                            status[0] = 2
                                            status[1] = value
                                            status[2] = ordinal

        return main

    return template.get_tir(
        prepared.storage_size,
        logical_size,
        extent,
        unique,
    ), 2


def _build_gather(
    runtime: Any,
    data: _PreparedTensor,
    indices: _PreparedTensor,
    output_size: int,
    axis_stride: int,
    axis_extent: int,
    *,
    output_addresses: bool,
    output_storage_size: int,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])
    index_size = len(indices.addresses)

    if output_addresses:

        @tilelang.jit(out_idx=[5], target="metal", execution_backend="torch")
        def template(
            data_size: int,
            logical_data: int,
            index_storage_size: int,
            logical_indices: int,
            logical_output: int,
            stride: int,
            extent: int,
            result_storage_size: int,
        ) -> Any:
            @T.prim_func
            def main(
                input0: T.Tensor((data_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
                addresses0: T.Tensor((logical_data,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                input1: T.Tensor((index_storage_size,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                addresses1: T.Tensor((logical_indices,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output_addresses: T.Tensor((logical_output,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output: T.Tensor((result_storage_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            ):
                assert (
                    data_size >= 0
                    and logical_data > 0
                    and index_storage_size >= 0
                    and logical_indices > 0
                    and logical_output > 0
                    and stride > 0
                    and extent > 0
                    and result_storage_size > 0
                )
                with T.Kernel(
                    T.ceildiv(logical_output, _THREADS), threads=_THREADS
                ) as block:
                    for thread in T.Parallel(_THREADS):
                        ordinal = block * _THREADS + thread
                        if ordinal < logical_output:
                            prefix = ordinal % stride
                            middle = (ordinal // stride) % logical_indices
                            suffix = ordinal // (stride * logical_indices)
                            selected = input1[addresses1[middle]]
                            source = (
                                prefix + stride * selected + stride * extent * suffix
                            )
                            output[output_addresses[ordinal]] = input0[
                                addresses0[source]
                            ]

            return main

        return template.get_tir(
            data.storage_size,
            len(data.addresses),
            indices.storage_size,
            index_size,
            output_size,
            axis_stride,
            axis_extent,
            output_storage_size,
        ), 5

    @tilelang.jit(out_idx=[4], target="metal", execution_backend="torch")
    def template(
        data_size: int,
        logical_data: int,
        index_storage_size: int,
        logical_indices: int,
        logical_output: int,
        stride: int,
        extent: int,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((data_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor((logical_data,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            input1: T.Tensor((index_storage_size,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses1: T.Tensor((logical_indices,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((logical_output,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert (
                data_size >= 0
                and logical_data > 0
                and index_storage_size >= 0
                and logical_indices > 0
                and logical_output > 0
                and stride > 0
                and extent > 0
            )
            with T.Kernel(
                T.ceildiv(logical_output, _THREADS), threads=_THREADS
            ) as block:
                for thread in T.Parallel(_THREADS):
                    ordinal = block * _THREADS + thread
                    if ordinal < logical_output:
                        prefix = ordinal % stride
                        middle = (ordinal // stride) % logical_indices
                        suffix = ordinal // (stride * logical_indices)
                        selected = input1[addresses1[middle]]
                        source = prefix + stride * selected + stride * extent * suffix
                        output[ordinal] = input0[addresses0[source]]

        return main

    return template.get_tir(
        data.storage_size,
        len(data.addresses),
        indices.storage_size,
        index_size,
        output_size,
        axis_stride,
        axis_extent,
    ), 4


def _build_scatter(
    runtime: Any,
    base: _PreparedTensor,
    indices: _PreparedTensor,
    updates: _PreparedTensor,
    axis_stride: int,
    axis_extent: int,
    *,
    add: bool,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])
    base_size = len(base.addresses)
    index_size = len(indices.addresses)
    update_size = len(updates.addresses)

    @tilelang.jit(out_idx=[6], target="metal", execution_backend="torch")
    def template(
        base_storage_size: int,
        logical_base: int,
        index_storage_size: int,
        logical_indices: int,
        update_storage_size: int,
        logical_updates: int,
        stride: int,
        extent: int,
        accumulate: bool,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((base_storage_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor((logical_base,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            input1: T.Tensor((index_storage_size,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses1: T.Tensor((logical_indices,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            input2: T.Tensor((update_storage_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses2: T.Tensor((logical_updates,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((logical_base,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert (
                base_storage_size >= 0
                and logical_base > 0
                and index_storage_size >= 0
                and logical_indices > 0
                and update_storage_size >= 0
                and logical_updates > 0
                and stride > 0
                and extent > 0
            )
            with T.Kernel(1, threads=1):
                for ordinal in T.Serial(logical_base):
                    output[ordinal] = input0[addresses0[ordinal]]
                for ordinal in T.Serial(logical_updates):
                    prefix = ordinal % stride
                    middle = (ordinal // stride) % logical_indices
                    suffix = ordinal // (stride * logical_indices)
                    selected = input1[addresses1[middle]]
                    destination = prefix + stride * selected + stride * extent * suffix
                    if accumulate:
                        output[destination] = (
                            output[destination] + input2[addresses2[ordinal]]
                        )
                    else:
                        output[destination] = input2[addresses2[ordinal]]

        return main

    return template.get_tir(
        base.storage_size,
        base_size,
        indices.storage_size,
        index_size,
        updates.storage_size,
        update_size,
        axis_stride,
        axis_extent,
        add,
    ), 6


def _build_gather_backward(
    runtime: Any,
    indices: _PreparedTensor,
    gradient: _PreparedTensor,
    source_size: int,
    axis_stride: int,
    axis_extent: int,
    output_storage_size: int,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])
    index_size = len(indices.addresses)
    gradient_size = len(gradient.addresses)

    @tilelang.jit(out_idx=[5], target="metal", execution_backend="torch")
    def template(
        index_storage_size: int,
        logical_indices: int,
        gradient_storage_size: int,
        logical_gradient: int,
        logical_source: int,
        stride: int,
        extent: int,
        result_storage_size: int,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((index_storage_size,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor((logical_indices,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            input1: T.Tensor((gradient_storage_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses1: T.Tensor((logical_gradient,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output_addresses: T.Tensor((logical_source,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((result_storage_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert (
                index_storage_size >= 0
                and logical_indices > 0
                and gradient_storage_size >= 0
                and logical_gradient > 0
                and logical_source > 0
                and stride > 0
                and extent > 0
                and result_storage_size > 0
            )
            with T.Kernel(1, threads=1):
                for ordinal in T.Serial(logical_source):
                    output[output_addresses[ordinal]] = 0.0
                for ordinal in T.Serial(logical_gradient):
                    prefix = ordinal % stride
                    middle = (ordinal // stride) % logical_indices
                    suffix = ordinal // (stride * logical_indices)
                    selected = input0[addresses0[middle]]
                    destination = prefix + stride * selected + stride * extent * suffix
                    physical = output_addresses[destination]
                    output[physical] = output[physical] + input1[addresses1[ordinal]]

        return main

    return template.get_tir(
        indices.storage_size,
        index_size,
        gradient.storage_size,
        gradient_size,
        source_size,
        axis_stride,
        axis_extent,
        output_storage_size,
    ), 5


def _build_scatter_base_backward(
    runtime: Any,
    indices: _PreparedTensor,
    gradient: _PreparedTensor,
    axis_stride: int,
    axis_extent: int,
    output_storage_size: int,
    *,
    overwrite: bool,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])
    index_size = len(indices.addresses)
    base_size = len(gradient.addresses)
    update_size = base_size // axis_extent * index_size

    @tilelang.jit(out_idx=[5], target="metal", execution_backend="torch")
    def template(
        index_storage_size: int,
        logical_indices: int,
        gradient_storage_size: int,
        logical_base: int,
        logical_updates: int,
        stride: int,
        extent: int,
        result_storage_size: int,
        zero_written: bool,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((index_storage_size,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor((logical_indices,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            input1: T.Tensor((gradient_storage_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses1: T.Tensor((logical_base,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output_addresses: T.Tensor((logical_base,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((result_storage_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert (
                index_storage_size >= 0
                and logical_indices > 0
                and gradient_storage_size >= 0
                and logical_base > 0
                and logical_updates > 0
                and stride > 0
                and extent > 0
                and result_storage_size > 0
            )
            with T.Kernel(1, threads=1):
                for ordinal in T.Serial(logical_base):
                    output[output_addresses[ordinal]] = input1[addresses1[ordinal]]
                if zero_written:
                    for ordinal in T.Serial(logical_updates):
                        prefix = ordinal % stride
                        middle = (ordinal // stride) % logical_indices
                        suffix = ordinal // (stride * logical_indices)
                        selected = input0[addresses0[middle]]
                        destination = (
                            prefix + stride * selected + stride * extent * suffix
                        )
                        output[output_addresses[destination]] = 0.0

        return main

    return template.get_tir(
        indices.storage_size,
        index_size,
        gradient.storage_size,
        base_size,
        update_size,
        axis_stride,
        axis_extent,
        output_storage_size,
        overwrite,
    ), 5


def _before_macro(T: Any, descending: bool) -> Any:
    if descending:

        @T.macro
        def before(lhs: Any, lhs_ordinal: Any, rhs: Any, rhs_ordinal: Any) -> Any:
            return T.if_then_else(
                T.isnan(lhs),
                T.if_then_else(T.isnan(rhs), lhs_ordinal < rhs_ordinal, True),
                T.if_then_else(
                    T.isnan(rhs),
                    False,
                    T.if_then_else(
                        lhs == rhs,
                        lhs_ordinal < rhs_ordinal,
                        lhs > rhs,
                    ),
                ),
            )

        return before

    @T.macro
    def before(lhs: Any, lhs_ordinal: Any, rhs: Any, rhs_ordinal: Any) -> Any:
        return T.if_then_else(
            T.isnan(lhs),
            T.if_then_else(T.isnan(rhs), lhs_ordinal < rhs_ordinal, False),
            T.if_then_else(
                T.isnan(rhs),
                True,
                T.if_then_else(
                    lhs == rhs,
                    lhs_ordinal < rhs_ordinal,
                    lhs < rhs,
                ),
            ),
        )

    return before


def _build_selection(
    runtime: Any,
    source: _PreparedTensor,
    output_dtype: DType,
    output_size: int,
    axis_stride: int,
    axis_extent: int,
    k: int,
    descending: bool,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])
    before = _before_macro(T, descending)
    result_dtype = _dtype_name(output_dtype)

    @tilelang.jit(out_idx=[2], target="metal", execution_backend="torch")
    def template(
        storage_size: int,
        logical_source: int,
        logical_output: int,
        stride: int,
        extent: int,
        retained: int,
        output_name: str,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((storage_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor((logical_source,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((logical_output,), output_name),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert (
                storage_size >= 0
                and logical_source > 0
                and logical_output > 0
                and stride > 0
                and extent > 0
                and retained > 0
                and output_name
            )
            with T.Kernel(
                T.ceildiv(logical_output, _THREADS), threads=_THREADS
            ) as block:
                for thread in T.Parallel(_THREADS):
                    ordinal = block * _THREADS + thread
                    if ordinal < logical_output:
                        prefix = ordinal % stride
                        position = (ordinal // stride) % retained
                        suffix = ordinal // (stride * retained)
                        if output_name == "float32":
                            output[ordinal] = 0.0
                        else:
                            output[ordinal] = 0
                        for candidate in T.Serial(extent):
                            source_ordinal = (
                                prefix + stride * candidate + stride * extent * suffix
                            )
                            candidate_value = input0[addresses0[source_ordinal]]
                            rank = T.alloc_fragment((1,), "int32")
                            rank[0] = 0
                            for other in T.Serial(extent):
                                other_ordinal = (
                                    prefix + stride * other + stride * extent * suffix
                                )
                                other_value = input0[addresses0[other_ordinal]]
                                rank[0] = rank[0] + T.cast(
                                    before(
                                        other_value, other, candidate_value, candidate
                                    ),
                                    "int32",
                                )
                            if rank[0] == position:
                                if output_name == "float32":
                                    output[ordinal] = candidate_value
                                else:
                                    output[ordinal] = candidate

        return main

    return template.get_tir(
        source.storage_size,
        len(source.addresses),
        output_size,
        axis_stride,
        axis_extent,
        k,
        result_dtype,
    ), 2


def _build_selection_backward(
    runtime: Any,
    source: _PreparedTensor,
    gradient: _PreparedTensor,
    source_size: int,
    axis_stride: int,
    axis_extent: int,
    k: int,
    descending: bool,
    output_storage_size: int,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])
    before = _before_macro(T, descending)

    @tilelang.jit(out_idx=[5], target="metal", execution_backend="torch")
    def template(
        source_storage_size: int,
        logical_source: int,
        gradient_storage_size: int,
        logical_gradient: int,
        stride: int,
        extent: int,
        retained: int,
        result_storage_size: int,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((source_storage_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor((logical_source,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            input1: T.Tensor((gradient_storage_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses1: T.Tensor((logical_gradient,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output_addresses: T.Tensor((logical_source,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((result_storage_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert (
                source_storage_size >= 0
                and logical_source > 0
                and gradient_storage_size >= 0
                and logical_gradient > 0
                and stride > 0
                and extent > 0
                and retained > 0
                and result_storage_size > 0
            )
            with T.Kernel(
                T.ceildiv(logical_source, _THREADS), threads=_THREADS
            ) as block:
                for thread in T.Parallel(_THREADS):
                    ordinal = block * _THREADS + thread
                    if ordinal < logical_source:
                        prefix = ordinal % stride
                        candidate = (ordinal // stride) % extent
                        suffix = ordinal // (stride * extent)
                        candidate_value = input0[addresses0[ordinal]]
                        rank = T.alloc_fragment((1,), "int32")
                        rank[0] = 0
                        for other in T.Serial(extent):
                            other_ordinal = (
                                prefix + stride * other + stride * extent * suffix
                            )
                            other_value = input0[addresses0[other_ordinal]]
                            rank[0] = rank[0] + T.cast(
                                before(other_value, other, candidate_value, candidate),
                                "int32",
                            )
                        if rank[0] < retained:
                            gradient_ordinal = (
                                prefix + stride * rank[0] + stride * retained * suffix
                            )
                            output[output_addresses[ordinal]] = input1[
                                addresses1[gradient_ordinal]
                            ]
                        else:
                            output[output_addresses[ordinal]] = 0.0

        return main

    return template.get_tir(
        source.storage_size,
        source_size,
        gradient.storage_size,
        len(gradient.addresses),
        axis_stride,
        axis_extent,
        k,
        output_storage_size,
    ), 5


def _result_tensor(
    target: Tensor,
    layout: Layout,
    dtype: DType,
) -> tuple[Any, Tensor]:
    carrier = cast(Any, target.carrier).allocate_like(
        layout.cosize,
        dtype=dtype,
        empty=layout.cosize == layout.size,
    )
    return carrier, Tensor(carrier, 0, layout)


def validate_indices(indices: Tensor, *, extent: int, unique: bool) -> None:
    """Validate device-resident Int32 indices before result allocation."""
    logical_kernel = _LOGICAL_BY_VARIANT["validate_indices"]
    target_carrier = cast(Any, indices.carrier)
    runtime = target_carrier._runtime
    prepared = _prepare_tensor(indices)
    key = MetalSpecializationKey.from_axes(
        logical_kernel,
        {
            "address_plan": prepared.address_key,
            "extent": extent,
            "logical_size": len(prepared.addresses),
            "pass_configs": _PASS_CONFIGS,
            "storage_size": prepared.storage_size,
            "template_revision": _TEMPLATE_REVISION,
            "unique": unique,
        },
    )

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_validate_indices(
            runtime, prepared, extent, unique
        )
        return _compile_kernel(
            runtime,
            prim_func,
            output_index,
            ("input0", "addresses0", "status"),
        )

    compiled = _JIT_CACHE.get_or_compile(key, compiler)
    status = runtime.torch.zeros(3, dtype=runtime.torch.int32, device="mps")
    compiled.executable(
        input0=prepared.source,
        addresses0=_address_tensor(runtime, prepared),
        status=status,
    )
    runtime.synchronize()
    code, value, _ = status.to("cpu").tolist()
    if code == 1:
        raise IndexError(
            f"index {value} is out of range for selected axis extent {extent}"
        )
    if code == 2:
        raise ValueError("scatter indices must contain distinct logical values")
    if code != 0:
        raise RuntimeError(f"Metal index validation returned unknown status {code}")


def execute_gather(
    tensor: Tensor,
    indices: Tensor,
    *,
    output_layout: Layout,
    axis_stride: int,
    axis_extent: int,
    plan: OperationPlan,
) -> Tensor:
    logical_kernel = _LOGICAL_BY_VARIANT["gather"]
    target_carrier = cast(Any, tensor.carrier)
    runtime = target_carrier._runtime
    source = _prepare_tensor(tensor)
    prepared_indices = _prepare_tensor(indices)
    result_carrier, result = _result_tensor(tensor, output_layout, DType.Float32)
    axes = {
        "axis_extent": axis_extent,
        "axis_stride": axis_stride,
        "index_address_plan": prepared_indices.address_key,
        "index_size": len(prepared_indices.addresses),
        "index_storage_size": prepared_indices.storage_size,
        "operation": "gather",
        "output_size": output_layout.size,
        "pass_configs": _PASS_CONFIGS,
        "plan": _plan_axis(plan, "gather"),
        "source_address_plan": source.address_key,
        "source_storage_size": source.storage_size,
        "template_revision": _TEMPLATE_REVISION,
    }
    key = MetalSpecializationKey.from_axes(logical_kernel, axes)

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_gather(
            runtime,
            source,
            prepared_indices,
            output_layout.size,
            axis_stride,
            axis_extent,
            output_addresses=False,
            output_storage_size=output_layout.cosize,
        )
        return _compile_kernel(
            runtime,
            prim_func,
            output_index,
            ("input0", "addresses0", "input1", "addresses1", "output"),
        )

    compiled = _JIT_CACHE.get_or_compile(key, compiler)
    compiled.executable(
        input0=source.source,
        addresses0=_address_tensor(runtime, source),
        input1=prepared_indices.source,
        addresses1=_address_tensor(runtime, prepared_indices),
        output=result_carrier._require_storage(),
    )
    return result


def execute_scatter(
    operation: str,
    base: Tensor,
    indices: Tensor,
    updates: Tensor,
    *,
    output_layout: Layout,
    axis_stride: int,
    axis_extent: int,
    plan: OperationPlan,
) -> Tensor:
    logical_kernel = _LOGICAL_BY_VARIANT[operation]
    target_carrier = cast(Any, base.carrier)
    runtime = target_carrier._runtime
    prepared_base = _prepare_tensor(base)
    prepared_indices = _prepare_tensor(indices)
    prepared_updates = _prepare_tensor(updates)
    result_carrier, result = _result_tensor(base, output_layout, DType.Float32)
    key = MetalSpecializationKey.from_axes(
        logical_kernel,
        {
            "axis_extent": axis_extent,
            "axis_stride": axis_stride,
            "base_address_plan": prepared_base.address_key,
            "base_storage_size": prepared_base.storage_size,
            "index_address_plan": prepared_indices.address_key,
            "index_storage_size": prepared_indices.storage_size,
            "operation": operation,
            "pass_configs": _PASS_CONFIGS,
            "plan": _plan_axis(plan, operation),
            "template_revision": _TEMPLATE_REVISION,
            "update_address_plan": prepared_updates.address_key,
            "update_storage_size": prepared_updates.storage_size,
        },
    )

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_scatter(
            runtime,
            prepared_base,
            prepared_indices,
            prepared_updates,
            axis_stride,
            axis_extent,
            add=operation == "scatter_add",
        )
        return _compile_kernel(
            runtime,
            prim_func,
            output_index,
            (
                "input0",
                "addresses0",
                "input1",
                "addresses1",
                "input2",
                "addresses2",
                "output",
            ),
        )

    compiled = _JIT_CACHE.get_or_compile(key, compiler)
    compiled.executable(
        input0=prepared_base.source,
        addresses0=_address_tensor(runtime, prepared_base),
        input1=prepared_indices.source,
        addresses1=_address_tensor(runtime, prepared_indices),
        input2=prepared_updates.source,
        addresses2=_address_tensor(runtime, prepared_updates),
        output=result_carrier._require_storage(),
    )
    return result


def execute_gather_backward(
    indices: Tensor,
    gradient: Tensor,
    *,
    output_layout: Layout,
    axis_stride: int,
    axis_extent: int,
    source_size: int,
) -> Tensor:
    logical_kernel = _LOGICAL_BY_VARIANT["grad_gather"]
    target_carrier = cast(Any, gradient.carrier)
    runtime = target_carrier._runtime
    prepared_indices = _prepare_tensor(indices)
    prepared_gradient = _prepare_tensor(gradient)
    output_key, output_addresses = _layout_address_tensor(runtime, output_layout)
    result_carrier, result = _result_tensor(gradient, output_layout, DType.Float32)
    key = MetalSpecializationKey.from_axes(
        logical_kernel,
        {
            "axis_extent": axis_extent,
            "axis_stride": axis_stride,
            "gradient_address_plan": prepared_gradient.address_key,
            "gradient_storage_size": prepared_gradient.storage_size,
            "index_address_plan": prepared_indices.address_key,
            "index_storage_size": prepared_indices.storage_size,
            "operation": "grad_gather",
            "output_address_plan": output_key,
            "output_storage_size": output_layout.cosize,
            "pass_configs": _PASS_CONFIGS,
            "source_size": source_size,
            "template_revision": _TEMPLATE_REVISION,
        },
    )

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_gather_backward(
            runtime,
            prepared_indices,
            prepared_gradient,
            source_size,
            axis_stride,
            axis_extent,
            output_layout.cosize,
        )
        return _compile_kernel(
            runtime,
            prim_func,
            output_index,
            (
                "input0",
                "addresses0",
                "input1",
                "addresses1",
                "output_addresses",
                "output",
            ),
        )

    compiled = _JIT_CACHE.get_or_compile(key, compiler)
    compiled.executable(
        input0=prepared_indices.source,
        addresses0=_address_tensor(runtime, prepared_indices),
        input1=prepared_gradient.source,
        addresses1=_address_tensor(runtime, prepared_gradient),
        output_addresses=output_addresses,
        output=result_carrier._require_storage(),
    )
    return result


def execute_scatter_base_backward(
    operation: str,
    indices: Tensor,
    gradient: Tensor,
    *,
    output_layout: Layout,
    axis_stride: int,
    axis_extent: int,
) -> Tensor:
    variant = "grad_scatter_base" if operation == "scatter" else "grad_scatter_add_base"
    logical_kernel = _LOGICAL_BY_VARIANT[variant]
    target_carrier = cast(Any, gradient.carrier)
    runtime = target_carrier._runtime
    prepared_indices = _prepare_tensor(indices)
    prepared_gradient = _prepare_tensor(gradient)
    output_key, output_addresses = _layout_address_tensor(runtime, output_layout)
    result_carrier, result = _result_tensor(gradient, output_layout, DType.Float32)
    key = MetalSpecializationKey.from_axes(
        logical_kernel,
        {
            "axis_extent": axis_extent,
            "axis_stride": axis_stride,
            "gradient_address_plan": prepared_gradient.address_key,
            "gradient_storage_size": prepared_gradient.storage_size,
            "index_address_plan": prepared_indices.address_key,
            "index_storage_size": prepared_indices.storage_size,
            "operation": variant,
            "output_address_plan": output_key,
            "output_storage_size": output_layout.cosize,
            "pass_configs": _PASS_CONFIGS,
            "template_revision": _TEMPLATE_REVISION,
        },
    )

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_scatter_base_backward(
            runtime,
            prepared_indices,
            prepared_gradient,
            axis_stride,
            axis_extent,
            output_layout.cosize,
            overwrite=operation == "scatter",
        )
        expected = (
            (
                "input0",
                "addresses0",
                "input1",
                "addresses1",
                "output_addresses",
                "output",
            )
            if operation == "scatter"
            else ("input1", "addresses1", "output_addresses", "output")
        )
        return _compile_kernel(runtime, prim_func, output_index, expected)

    compiled = _JIT_CACHE.get_or_compile(key, compiler)
    compiled.executable(
        input0=prepared_indices.source,
        addresses0=_address_tensor(runtime, prepared_indices),
        input1=prepared_gradient.source,
        addresses1=_address_tensor(runtime, prepared_gradient),
        output_addresses=output_addresses,
        output=result_carrier._require_storage(),
    )
    return result


def execute_scatter_updates_backward(
    indices: Tensor,
    gradient: Tensor,
    *,
    output_layout: Layout,
    axis_stride: int,
    axis_extent: int,
) -> Tensor:
    logical_kernel = _LOGICAL_BY_VARIANT["grad_scatter_updates"]
    target_carrier = cast(Any, gradient.carrier)
    runtime = target_carrier._runtime
    prepared_indices = _prepare_tensor(indices)
    prepared_gradient = _prepare_tensor(gradient)
    output_key, output_addresses = _layout_address_tensor(runtime, output_layout)
    result_carrier, result = _result_tensor(gradient, output_layout, DType.Float32)
    key = MetalSpecializationKey.from_axes(
        logical_kernel,
        {
            "axis_extent": axis_extent,
            "axis_stride": axis_stride,
            "gradient_address_plan": prepared_gradient.address_key,
            "gradient_storage_size": prepared_gradient.storage_size,
            "index_address_plan": prepared_indices.address_key,
            "index_storage_size": prepared_indices.storage_size,
            "operation": "grad_scatter_updates",
            "output_address_plan": output_key,
            "output_storage_size": output_layout.cosize,
            "pass_configs": _PASS_CONFIGS,
            "template_revision": _TEMPLATE_REVISION,
        },
    )

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_gather(
            runtime,
            prepared_gradient,
            prepared_indices,
            output_layout.size,
            axis_stride,
            axis_extent,
            output_addresses=True,
            output_storage_size=output_layout.cosize,
        )
        return _compile_kernel(
            runtime,
            prim_func,
            output_index,
            (
                "input0",
                "addresses0",
                "input1",
                "addresses1",
                "output_addresses",
                "output",
            ),
        )

    compiled = _JIT_CACHE.get_or_compile(key, compiler)
    compiled.executable(
        input0=prepared_gradient.source,
        addresses0=_address_tensor(runtime, prepared_gradient),
        input1=prepared_indices.source,
        addresses1=_address_tensor(runtime, prepared_indices),
        output_addresses=output_addresses,
        output=result_carrier._require_storage(),
    )
    return result


def execute_selection(
    operation: str,
    tensor: Tensor,
    *,
    output_layout: Layout,
    axis_stride: int,
    axis_extent: int,
    k: int,
    descending: bool,
    output_dtype: DType,
    plan: OperationPlan,
) -> Tensor:
    logical_kernel = _LOGICAL_BY_VARIANT[operation]
    target_carrier = cast(Any, tensor.carrier)
    runtime = target_carrier._runtime
    prepared = _prepare_tensor(tensor)
    result_carrier, result = _result_tensor(tensor, output_layout, output_dtype)
    key = MetalSpecializationKey.from_axes(
        logical_kernel,
        {
            "address_plan": prepared.address_key,
            "axis_extent": axis_extent,
            "axis_stride": axis_stride,
            "descending": descending,
            "k": k,
            "operation": operation,
            "output_dtype": output_dtype.name,
            "output_size": output_layout.size,
            "pass_configs": _PASS_CONFIGS,
            "plan": _plan_axis(plan, operation),
            "storage_size": prepared.storage_size,
            "template_revision": _TEMPLATE_REVISION,
        },
    )

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_selection(
            runtime,
            prepared,
            output_dtype,
            output_layout.size,
            axis_stride,
            axis_extent,
            k,
            descending,
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


def execute_selection_backward(
    tensor: Tensor,
    gradient: Tensor,
    *,
    output_layout: Layout,
    axis_stride: int,
    axis_extent: int,
    k: int,
    descending: bool,
) -> Tensor:
    logical_kernel = _LOGICAL_BY_VARIANT["grad_selection"]
    target_carrier = cast(Any, tensor.carrier)
    runtime = target_carrier._runtime
    prepared = _prepare_tensor(tensor)
    prepared_gradient = _prepare_tensor(gradient)
    output_key, output_addresses = _layout_address_tensor(runtime, output_layout)
    result_carrier, result = _result_tensor(tensor, output_layout, DType.Float32)
    key = MetalSpecializationKey.from_axes(
        logical_kernel,
        {
            "address_plan": prepared.address_key,
            "axis_extent": axis_extent,
            "axis_stride": axis_stride,
            "descending": descending,
            "gradient_address_plan": prepared_gradient.address_key,
            "gradient_storage_size": prepared_gradient.storage_size,
            "k": k,
            "operation": "grad_selection",
            "output_address_plan": output_key,
            "output_storage_size": output_layout.cosize,
            "pass_configs": _PASS_CONFIGS,
            "source_storage_size": prepared.storage_size,
            "template_revision": _TEMPLATE_REVISION,
        },
    )

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_selection_backward(
            runtime,
            prepared,
            prepared_gradient,
            tensor.size(),
            axis_stride,
            axis_extent,
            k,
            descending,
            output_layout.cosize,
        )
        return _compile_kernel(
            runtime,
            prim_func,
            output_index,
            (
                "input0",
                "addresses0",
                "input1",
                "addresses1",
                "output_addresses",
                "output",
            ),
        )

    compiled = _JIT_CACHE.get_or_compile(key, compiler)
    compiled.executable(
        input0=prepared.source,
        addresses0=_address_tensor(runtime, prepared),
        input1=prepared_gradient.source,
        addresses1=_address_tensor(runtime, prepared_gradient),
        output_addresses=output_addresses,
        output=result_carrier._require_storage(),
    )
    return result


__all__: list[str] = []
