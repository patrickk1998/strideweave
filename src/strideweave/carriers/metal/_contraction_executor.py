"""Lazy serial TileLang execution for Metal matmul and convolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from ...layout import Layout
from ...tensor import Tensor
from ..dtype import DType
from ..operation_policy import OperationPlan
from ._address_plan import address_plan
from ._jit import (
    CompiledMetalKernel,
    LogicalKernel,
    MetalJITCache,
    MetalSpecializationKey,
)
from ._kernel_support import (
    PreparedTensor,
    address_tensor,
    compile_kernel,
    plan_axis,
    prepare_tensor,
)
from ._recipe import (
    ContractionRecipe,
    ConvolutionRecipe,
    MatmulRecipe,
    contraction_recipe,
)

_TEMPLATE_REVISION = "strideweave.metal.contraction.v1"
_PASS_CONFIGS = (("tl.disable_data_race_check", True),)
_VARIANTS = (
    "matmul_forward",
    "matmul_grad_lhs",
    "matmul_grad_rhs",
    "conv_forward",
    "conv_grad_lhs",
    "conv_grad_kernel",
)
_LOGICAL_KERNELS = tuple(
    LogicalKernel(
        "metal.matmul" if variant.startswith("matmul") else "metal.conv_general",
        variant,
    )
    for variant in _VARIANTS
)
_LOGICAL_BY_VARIANT = {kernel.variant: kernel for kernel in _LOGICAL_KERNELS}
_JIT_CACHE = MetalJITCache(_LOGICAL_KERNELS, max_specializations=128)


@dataclass(frozen=True, slots=True)
class ConvGeometry:
    """Normalized canonical-role convolution geometry."""

    batch_size: int
    input_features: int
    output_features: int
    channels_per_group: int
    input_spatial: tuple[int, ...]
    kernel_spatial: tuple[int, ...]
    output_spatial: tuple[int, ...]
    strides: tuple[int, ...]
    padding: tuple[tuple[int, int], ...]
    lhs_dilation: tuple[int, ...]
    kernel_dilation: tuple[int, ...]
    feature_groups: int

    @property
    def output_count(self) -> int:
        return self.batch_size * self.output_features * _product(self.output_spatial)

    @property
    def contraction_count(self) -> int:
        return self.channels_per_group * _product(self.kernel_spatial)

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.batch_size,
            self.input_features,
            self.output_features,
            self.channels_per_group,
            self.input_spatial,
            self.kernel_spatial,
            self.output_spatial,
            self.strides,
            self.padding,
            self.lhs_dilation,
            self.kernel_dilation,
            self.feature_groups,
        )


@dataclass(frozen=True, slots=True)
class _ConvTables:
    lhs_source: tuple[int, ...]
    kernel_source: tuple[int, ...]
    lhs_logical: tuple[int, ...]
    kernel_logical: tuple[int, ...]
    valid: tuple[int, ...]


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _coordinates(ordinal: int, extents: tuple[int, ...]) -> tuple[int, ...]:
    coordinates: list[int] = []
    for extent in extents:
        coordinates.append(ordinal % extent)
        ordinal //= extent
    return tuple(coordinates)


def _ordinal(coordinates: tuple[int, ...], extents: tuple[int, ...]) -> int:
    result = 0
    stride = 1
    for coordinate, extent in zip(coordinates, extents, strict=True):
        result += coordinate * stride
        stride *= extent
    return result


def _result_plan(layout: Layout) -> tuple[object, tuple[int, ...]]:
    plan = address_plan(layout)
    if not plan.is_injective:
        raise RuntimeError("Metal contraction results require an injective layout")
    if any(address < 0 or address > 2**31 - 1 for address in plan.addresses):
        raise RuntimeError(
            "Metal contraction result address plan exceeds the Int32 map range"
        )
    return plan.key, plan.addresses


def _int32_tensor(runtime: Any, values: tuple[int, ...]) -> Any:
    return runtime.torch.tensor(values, dtype=runtime.torch.int32, device="mps")


def _conv_tables(
    lhs: PreparedTensor, kernel: PreparedTensor, geometry: ConvGeometry
) -> _ConvTables:
    output_extents = (
        geometry.batch_size,
        geometry.output_features,
        *geometry.output_spatial,
    )
    contraction_extents = (geometry.channels_per_group, *geometry.kernel_spatial)
    lhs_extents = (
        geometry.batch_size,
        geometry.input_features,
        *geometry.input_spatial,
    )
    kernel_extents = (
        geometry.output_features,
        geometry.channels_per_group,
        *geometry.kernel_spatial,
    )
    lhs_source: list[int] = []
    kernel_source: list[int] = []
    lhs_logical: list[int] = []
    kernel_logical: list[int] = []
    valid_values: list[int] = []
    outputs = tuple(
        _coordinates(i, output_extents) for i in range(geometry.output_count)
    )
    contractions = tuple(
        _coordinates(i, contraction_extents) for i in range(geometry.contraction_count)
    )
    for contraction in contractions:
        channel, kernel_position = contraction[0], contraction[1:]
        for output in outputs:
            batch, output_feature = output[0], output[1]
            output_position = output[2:]
            group = output_feature // (
                geometry.output_features // geometry.feature_groups
            )
            input_position: list[int] = []
            valid = True
            for dimension, kernel_coordinate in enumerate(kernel_position):
                dilated = (
                    output_position[dimension] * geometry.strides[dimension]
                    + kernel_coordinate * geometry.kernel_dilation[dimension]
                    - geometry.padding[dimension][0]
                )
                if dilated < 0 or dilated % geometry.lhs_dilation[dimension] != 0:
                    valid = False
                    break
                coordinate = dilated // geometry.lhs_dilation[dimension]
                if coordinate >= geometry.input_spatial[dimension]:
                    valid = False
                    break
                input_position.append(coordinate)
            kernel_index = _ordinal(
                (output_feature, channel, *kernel_position), kernel_extents
            )
            kernel_logical.append(kernel_index)
            kernel_source.append(kernel.addresses[kernel_index])
            if valid:
                lhs_index = _ordinal(
                    (
                        batch,
                        group * geometry.channels_per_group + channel,
                        *input_position,
                    ),
                    lhs_extents,
                )
                lhs_logical.append(lhs_index)
                lhs_source.append(lhs.addresses[lhs_index])
                valid_values.append(1)
            else:
                lhs_logical.append(0)
                lhs_source.append(0)
                valid_values.append(0)
    return _ConvTables(
        tuple(lhs_source),
        tuple(kernel_source),
        tuple(lhs_logical),
        tuple(kernel_logical),
        tuple(valid_values),
    )


def _build_matmul(
    runtime: Any,
    variant: str,
    first: PreparedTensor,
    second: PreparedTensor,
    output_storage_size: int,
    n_size: int,
    m_size: int,
    k_size: int,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])

    @tilelang.jit(out_idx=[5], target="metal", execution_backend="torch")
    def template(
        first_size: int,
        second_size: int,
        output_size: int,
        n: int,
        m: int,
        k: int,
        selected_variant: str,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((first_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor(
                (n * m if selected_variant != "matmul_forward" else n * k,), "int32"
            ),  # pyright: ignore[reportInvalidTypeForm]
            input1: T.Tensor((second_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses1: T.Tensor(
                (m * k if selected_variant != "matmul_grad_rhs" else n * k,), "int32"
            ),  # pyright: ignore[reportInvalidTypeForm]
            output_addresses: T.Tensor(
                (
                    n * m
                    if selected_variant == "matmul_forward"
                    else (n * k if selected_variant == "matmul_grad_lhs" else m * k),
                ),
                "int32",
            ),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((output_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert first_size > 0 and second_size > 0 and output_size > 0
            assert n > 0 and m > 0 and k > 0 and selected_variant
            with T.Kernel(1, threads=1):
                if selected_variant == "matmul_forward":
                    for j in T.Serial(m):
                        for i in T.Serial(n):
                            accumulator = T.alloc_fragment((1,), "float32")
                            product = T.alloc_fragment((1,), "float32")
                            accumulator[0] = 0.0
                            for contracted in T.Serial(k):
                                product[0] = (
                                    input0[addresses0[i + n * contracted]]
                                    * input1[addresses1[j + m * contracted]]
                                )
                                accumulator[0] = accumulator[0] + product[0]
                            ordinal = i + n * j
                            output[output_addresses[ordinal]] = accumulator[0]
                elif selected_variant == "matmul_grad_lhs":
                    for contracted in T.Serial(k):
                        for i in T.Serial(n):
                            accumulator = T.alloc_fragment((1,), "float32")
                            product = T.alloc_fragment((1,), "float32")
                            accumulator[0] = 0.0
                            for j in T.Serial(m):
                                product[0] = (
                                    input0[addresses0[i + n * j]]
                                    * input1[addresses1[j + m * contracted]]
                                )
                                accumulator[0] = accumulator[0] + product[0]
                            ordinal = i + n * contracted
                            output[output_addresses[ordinal]] = accumulator[0]
                else:
                    for contracted in T.Serial(k):
                        for j in T.Serial(m):
                            accumulator = T.alloc_fragment((1,), "float32")
                            product = T.alloc_fragment((1,), "float32")
                            accumulator[0] = 0.0
                            for i in T.Serial(n):
                                product[0] = (
                                    input0[addresses0[i + n * j]]
                                    * input1[addresses1[i + n * contracted]]
                                )
                                accumulator[0] = accumulator[0] + product[0]
                            ordinal = j + m * contracted
                            output[output_addresses[ordinal]] = accumulator[0]

        return main

    return template.get_tir(
        first.storage_size,
        second.storage_size,
        output_storage_size,
        n_size,
        m_size,
        k_size,
        variant,
    ), 5


def _build_conv(
    runtime: Any,
    variant: str,
    first_storage_size: int,
    second_storage_size: int,
    output_storage_size: int,
    output_count: int,
    contraction_count: int,
) -> tuple[Any, int]:
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])
    if variant == "conv_forward":

        @tilelang.jit(out_idx=[6], target="metal", execution_backend="torch")
        def template(
            first_size: int,
            second_size: int,
            output_size: int,
            outputs: int,
            contractions: int,
        ) -> Any:
            @T.prim_func
            def main(
                input0: T.Tensor((first_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
                input1: T.Tensor((second_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
                addresses1: T.Tensor((outputs * contractions,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                addresses2: T.Tensor((outputs * contractions,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                addresses3: T.Tensor((outputs * contractions,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output_addresses: T.Tensor((outputs,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
                output: T.Tensor((output_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            ):
                assert first_size > 0 and second_size > 0 and output_size > 0
                assert outputs > 0 and contractions > 0
                with T.Kernel(1, threads=1):
                    for output_ordinal in T.Serial(outputs):
                        accumulator = T.alloc_fragment((1,), "float32")
                        product = T.alloc_fragment((1,), "float32")
                        accumulator[0] = 0.0
                        for contracted in T.Serial(contractions):
                            event = output_ordinal + outputs * contracted
                            lhs_value = T.if_then_else(
                                addresses3[event] != 0,
                                input0[addresses1[event]],
                                0.0,
                            )
                            product[0] = lhs_value * input1[addresses2[event]]
                            accumulator[0] = accumulator[0] + product[0]
                        output[output_addresses[output_ordinal]] = accumulator[0]

            return main

        return template.get_tir(
            first_storage_size,
            second_storage_size,
            output_storage_size,
            output_count,
            contraction_count,
        ), 6

    lhs_gradient = variant == "conv_grad_lhs"

    @tilelang.jit(out_idx=[6], target="metal", execution_backend="torch")
    def template(
        first_size: int,
        second_size: int,
        output_size: int,
        outputs: int,
        contractions: int,
        gradient_lhs: bool,
    ) -> Any:
        @T.prim_func
        def main(
            input0: T.Tensor((first_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses0: T.Tensor((outputs,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            input1: T.Tensor((second_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses1: T.Tensor((outputs * contractions,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            addresses3: T.Tensor((outputs * contractions,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output_addresses: T.Tensor((outputs * contractions,), "int32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((output_size,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            assert first_size > 0 and second_size > 0 and output_size > 0
            assert outputs > 0 and contractions > 0
            with T.Kernel(1, threads=1):
                product = T.alloc_fragment((1,), "float32")
                for output_ordinal in T.Serial(outputs):
                    incoming = input0[addresses0[output_ordinal]]
                    for contracted in T.Serial(contractions):
                        event = output_ordinal + outputs * contracted
                        if gradient_lhs:
                            if addresses3[event] != 0:
                                destination = output_addresses[event]
                                product[0] = incoming * input1[addresses1[event]]
                                output[destination] = output[destination] + product[0]
                        else:
                            lhs_value = T.if_then_else(
                                addresses3[event] != 0,
                                input1[addresses1[event]],
                                0.0,
                            )
                            destination = output_addresses[event]
                            product[0] = incoming * lhs_value
                            output[destination] = output[destination] + product[0]

        return main

    return template.get_tir(
        first_storage_size,
        second_storage_size,
        output_storage_size,
        output_count,
        contraction_count,
        lhs_gradient,
    ), 6


def _compile(
    runtime: Any,
    prim_func: Any,
    output_index: int,
    expected_buffers: tuple[str, ...],
) -> CompiledMetalKernel:
    return compile_kernel(
        runtime,
        prim_func,
        output_index,
        expected_buffers,
        family="contraction",
        pass_configs=_PASS_CONFIGS,
    )


def _recompile_from_key(
    runtime: Any, key: MetalSpecializationKey
) -> CompiledMetalKernel:
    """Compile contraction facts from immutable specialization axes only."""
    if key.logical_kernel not in _LOGICAL_KERNELS:
        raise RuntimeError("unknown Metal contraction reconstruction recipe")
    recipe = _recipe_from_key(key)
    return _recompile_recipe(runtime, key, recipe)


def _recipe_from_key(key: MetalSpecializationKey) -> ContractionRecipe:
    if key.logical_kernel not in _LOGICAL_KERNELS:
        raise RuntimeError("unknown Metal contraction reconstruction recipe")
    return contraction_recipe(
        key,
        template_revision=_TEMPLATE_REVISION,
        pass_configs=_PASS_CONFIGS,
    )


def _recompile_recipe(
    runtime: Any, _key: MetalSpecializationKey, recipe: ContractionRecipe
) -> CompiledMetalKernel:
    if isinstance(recipe, MatmulRecipe):
        prim_func, output_index = _build_matmul(
            runtime,
            recipe.operation,
            recipe.first,
            recipe.second,
            recipe.output.storage_size,
            recipe.n_size,
            recipe.m_size,
            recipe.k_size,
        )
        return _compile(
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

    assert isinstance(recipe, ConvolutionRecipe)
    prim_func, output_index = _build_conv(
        runtime,
        recipe.operation,
        recipe.first.storage_size,
        recipe.second.storage_size,
        recipe.output.storage_size,
        recipe.geometry.output_count,
        recipe.geometry.contraction_count,
    )
    expected = (
        (
            "input0",
            "input1",
            "addresses1",
            "addresses2",
            "addresses3",
            "output_addresses",
            "output",
        )
        if recipe.operation == "conv_forward"
        else (
            "input0",
            "addresses0",
            "input1",
            "addresses1",
            "addresses3",
            "output_addresses",
            "output",
        )
    )
    return _compile(runtime, prim_func, output_index, expected)


def execute_matmul(
    variant: str,
    lhs: Tensor,
    rhs: Tensor,
    *,
    output_layout: Layout,
    n_size: int,
    m_size: int,
    k_size: int,
    plan: OperationPlan | None = None,
) -> Tensor:
    """Launch a serial matmul forward or VJP specialization."""
    logical_kernel = _LOGICAL_BY_VARIANT[variant]
    target_carrier = cast(Any, lhs.carrier)
    runtime = target_carrier._runtime
    prepared_lhs = prepare_tensor(lhs, family="contraction")
    prepared_rhs = prepare_tensor(rhs, family="contraction")
    output_key, output_addresses = _result_plan(output_layout)
    axes = {
        "first_address_plan": prepared_lhs.address_key,
        "first_storage_size": prepared_lhs.storage_size,
        "k_size": k_size,
        "m_size": m_size,
        "n_size": n_size,
        "operation": variant,
        "output_address_plan": output_key,
        "output_storage_size": output_layout.cosize,
        "pass_configs": _PASS_CONFIGS,
        "plan": plan_axis(plan, variant),
        "second_address_plan": prepared_rhs.address_key,
        "second_storage_size": prepared_rhs.storage_size,
        "template_revision": _TEMPLATE_REVISION,
    }
    key = MetalSpecializationKey.from_axes(logical_kernel, axes)
    result_carrier = target_carrier.allocate_like(
        output_layout.cosize,
        dtype=DType.Float32,
        empty=output_layout.cosize == output_layout.size,
    )
    result = Tensor(result_carrier, 0, output_layout)

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_matmul(
            runtime,
            variant,
            prepared_lhs,
            prepared_rhs,
            output_layout.cosize,
            n_size,
            m_size,
            k_size,
        )
        return _compile(
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
        input0=prepared_lhs.source,
        addresses0=address_tensor(runtime, prepared_lhs),
        input1=prepared_rhs.source,
        addresses1=address_tensor(runtime, prepared_rhs),
        output_addresses=_int32_tensor(runtime, output_addresses),
        output=result_carrier._require_storage(),
    )
    return result


def execute_conv(
    variant: str,
    lhs: Tensor,
    kernel: Tensor,
    *,
    output_layout: Layout,
    geometry: ConvGeometry,
    gradient: Tensor | None = None,
    plan: OperationPlan | None = None,
) -> Tensor:
    """Launch serial grouped cross-correlation forward or one VJP."""
    logical_kernel = _LOGICAL_BY_VARIANT[variant]
    target_carrier = cast(Any, lhs.carrier)
    runtime = target_carrier._runtime
    prepared_lhs = prepare_tensor(lhs, family="contraction")
    prepared_kernel = prepare_tensor(kernel, family="contraction")
    tables = _conv_tables(prepared_lhs, prepared_kernel, geometry)
    output_key, output_addresses = _result_plan(output_layout)
    prepared_gradient = (
        None if gradient is None else prepare_tensor(gradient, family="contraction")
    )
    first = prepared_lhs if prepared_gradient is None else prepared_gradient
    second = prepared_kernel if variant != "conv_grad_kernel" else prepared_lhs
    axes = {
        "convolution_signature": geometry.signature,
        "first_address_plan": first.address_key,
        "first_storage_size": first.storage_size,
        "operation": variant,
        "output_address_plan": output_key,
        "output_storage_size": output_layout.cosize,
        "pass_configs": _PASS_CONFIGS,
        "plan": plan_axis(plan, variant),
        "second_address_plan": second.address_key,
        "second_storage_size": second.storage_size,
        "template_revision": _TEMPLATE_REVISION,
    }
    key = MetalSpecializationKey.from_axes(logical_kernel, axes)
    result_carrier = target_carrier.allocate_like(
        output_layout.cosize,
        dtype=DType.Float32,
        empty=variant == "conv_forward" and output_layout.cosize == output_layout.size,
    )
    result = Tensor(result_carrier, 0, output_layout)

    def compiler(_key: MetalSpecializationKey) -> CompiledMetalKernel:
        prim_func, output_index = _build_conv(
            runtime,
            variant,
            first.storage_size,
            second.storage_size,
            output_layout.cosize,
            geometry.output_count,
            geometry.contraction_count,
        )
        expected = (
            (
                "input0",
                "input1",
                "addresses1",
                "addresses2",
                "addresses3",
                "output_addresses",
                "output",
            )
            if variant == "conv_forward"
            else (
                "input0",
                "addresses0",
                "input1",
                "addresses1",
                "addresses3",
                "output_addresses",
                "output",
            )
        )
        return _compile(runtime, prim_func, output_index, expected)

    compiled = _JIT_CACHE.get_or_compile(key, compiler)
    output_storage = result_carrier._require_storage()
    validity = _int32_tensor(runtime, tables.valid)
    if variant == "conv_forward":
        buffers = {
            "input0": prepared_lhs.source,
            "input1": prepared_kernel.source,
            "addresses1": _int32_tensor(runtime, tables.lhs_source),
            "addresses2": _int32_tensor(runtime, tables.kernel_source),
            "addresses3": validity,
            "output_addresses": _int32_tensor(runtime, output_addresses),
            "output": output_storage,
        }
    elif variant == "conv_grad_lhs":
        assert prepared_gradient is not None
        destination = tuple(output_addresses[index] for index in tables.lhs_logical)
        buffers = {
            "input0": prepared_gradient.source,
            "addresses0": _int32_tensor(runtime, prepared_gradient.addresses),
            "input1": prepared_kernel.source,
            "addresses1": _int32_tensor(runtime, tables.kernel_source),
            "addresses3": validity,
            "output_addresses": _int32_tensor(runtime, destination),
            "output": output_storage,
        }
    else:
        assert prepared_gradient is not None
        destination = tuple(output_addresses[index] for index in tables.kernel_logical)
        buffers = {
            "input0": prepared_gradient.source,
            "addresses0": _int32_tensor(runtime, prepared_gradient.addresses),
            "input1": prepared_lhs.source,
            "addresses1": _int32_tensor(runtime, tables.lhs_source),
            "addresses3": validity,
            "output_addresses": _int32_tensor(runtime, destination),
            "output": output_storage,
        }
    compiled.executable(**buffers)
    return result


__all__: list[str] = []
