"""End-to-end exercise of the public carrier extension authoring surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

import strideweave as sw
from strideweave.carriers.extension import (
    PreparedComposite,
    PreparedKernel,
    PreparedTransfer,
    ProviderResult,
    TransferProvider,
    TransferRequest,
    Unsupported,
)
from strideweave.carriers.operation_capability import OperationCapability
from strideweave.carriers.operation_policy import (
    Arithmetic,
    OperandPlan,
    OperandRole,
    OperationPlan,
)


def _storage() -> sw.StorageProvider:
    def allocate(carrier: Any, slots: int) -> None:
        carrier.values = [0.0] * slots

    def release(carrier: Any) -> None:
        carrier.values = []

    def size(carrier: Any) -> int:
        return len(carrier.values)

    def read(carrier: Any, index: int) -> object:
        return carrier.values[index]

    def write(carrier: Any, index: int, value: object) -> None:
        carrier.values[index] = value

    return sw.StorageProvider(
        lambda _carrier, dtype: dtype is sw.DType.Float32,
        size,
        allocate,
        release,
        read,
        write,
    )


def _tensor(carrier_class: type[sw.Carrier], values: list[float]) -> sw.Tensor:
    carrier = carrier_class(len(values), dtype=sw.DType.Float32)
    for index, value in enumerate(values):
        carrier[index] = value
    return sw.Tensor(
        carrier,
        0,
        sw.Layout(sw.Shape(len(values)), sw.Stride(1)),
    )


@dataclass
class _World:
    compute: type[sw.Carrier]
    storage: type[sw.Carrier]
    dependent: type[sw.DependentCarrier]
    operation: sw.OperationDefinition
    plan: OperationPlan
    attempts: list[str]
    pattern: sw.KernelPattern
    use_kernel: bool = True


@pytest.fixture(scope="module")
def world() -> _World:
    compute = type("E2EComputeCarrier", (sw.Carrier,), {})
    storage = type("E2EStorageCarrier", (sw.Carrier,), {})
    attempts: list[str] = []

    schema = sw.OperationSchema(
        (sw.OperandSpec("value", sw.OperandKind.TENSOR, differentiable=True),),
        (sw.OptionSpec("tag", 0),),
    )

    def make_plan(value: sw.Tensor) -> OperationPlan:
        return OperationPlan(
            "e2e.square",
            (OperandPlan(OperandRole.TENSOR, value.dtype(), value.dtype()),),
            Arithmetic.BINARY32,
            None,
            None,
            value.dtype(),
        )

    def resolve(
        call: sw.operation.BoundOperationCall,
    ) -> sw.operation.ResolvedInvocation:
        value = cast(sw.Tensor, call.operands[0])
        return sw.operation.ResolvedInvocation(
            call,
            make_plan(value),
            (sw.ResultSpec(value.dtype(), value.layout.shape, value.layout),),
            saved_operands=(0,),
        )

    def result(
        invocation: sw.operation.ResolvedInvocation,
        values: list[float],
        carrier: type[sw.Carrier] | None = None,
    ) -> ProviderResult:
        source = cast(sw.Tensor, invocation.operands[0])
        output_carrier = source.carrier.new_like(
            values, dtype=invocation.results[0].dtype
        )
        if carrier is not None:
            output_carrier = carrier(len(values), dtype=invocation.results[0].dtype)
            for index, value in enumerate(values):
                output_carrier[index] = value
        return ProviderResult(
            (sw.Tensor(output_carrier, 0, invocation.results[0].layout),)
        )

    def reference(invocation: sw.operation.ResolvedInvocation) -> sw.Tensor:
        attempts.append("reference")
        value = cast(sw.Tensor, invocation.operands[0])
        output = result(
            invocation,
            [float(value[index]) ** 2 for index in range(value.size())],
        )
        return output.outputs[0]

    def vjp(
        context: sw.VJPContext,
        cotangents: tuple[sw.Tensor | None, ...],
    ) -> tuple[sw.Tensor | None, ...]:
        value = context.saved_operands[0]
        cotangent = cast(sw.Tensor, cotangents[0])
        return (
            sw.Tensor(
                value.carrier.new_like(
                    [
                        2.0 * float(value[index]) * float(cotangent[index])
                        for index in range(value.size())
                    ],
                    dtype=value.dtype(),
                ),
                0,
                value.layout,
            ),
        )

    operation = sw.define_operation("e2e.square", schema, resolve, reference, vjp)
    plan = OperationPlan(
        "e2e.square",
        (OperandPlan(OperandRole.TENSOR, sw.DType.Float32, sw.DType.Float32),),
        Arithmetic.BINARY32,
        None,
        None,
        sw.DType.Float32,
    )
    capability = OperationCapability.from_plan(plan)
    interface = sw.KernelExecutionInterface("e2e-v1")

    # The pack is deliberately switchable: the optimized candidate can be
    # rejected at preparation time, in which case a lower-preference pattern
    # executes the operation's public reference callback.
    state: _World

    def prepare(
        invocation: sw.operation.ResolvedInvocation,
        selected: sw.KernelExecutionInterface,
    ) -> PreparedKernel | Unsupported:
        attempts.append("kernel")
        if not state.use_kernel:
            return Unsupported("test requested reference fallback")
        return PreparedKernel(
            selected,
            lambda: result(
                invocation,
                [
                    float(cast(sw.Tensor, invocation.operands[0])[index]) ** 2
                    for index in range(cast(sw.Tensor, invocation.operands[0]).size())
                ],
            ),
        )

    class Dependent(sw.DependentCarrier):
        def __init__(
            self,
            slots: int,
            *,
            dtype: sw.SimpleDType,
            mutable: bool = True,
        ) -> None:
            super().__init__(slots, dtype=dtype, mutable=mutable)
            self._finalize_dependent_capabilities()

    pattern = sw.KernelPattern(
        operation,
        (capability,),
        "e2e.optimized.square",
        prepare,
        preference=10,
    )

    def prepare_reference(
        invocation: sw.operation.ResolvedInvocation,
        selected: sw.KernelExecutionInterface,
    ) -> PreparedKernel:
        reference_callback = invocation.definition.reference
        assert reference_callback is not None

        def submit() -> ProviderResult:
            returned = reference_callback(invocation)
            outputs = (returned,) if isinstance(returned, sw.Tensor) else returned
            return ProviderResult(outputs)

        return PreparedKernel(selected, submit)

    reference_pattern = sw.KernelPattern(
        operation,
        (capability,),
        "e2e.reference.square",
        prepare_reference,
    )
    state = _World(compute, storage, Dependent, operation, plan, attempts, pattern)

    def copy_prepare(request: TransferRequest) -> PreparedTransfer:
        def submit() -> None:
            source = cast(sw.Carrier, request.tensor.carrier)
            destination = cast(sw.Carrier, request.destination)
            for index in range(request.physical_span):
                destination[index] = source[index + request.tensor.offset]

        return PreparedTransfer(submit)

    forward = sw.TransferRoute(
        compute,
        storage,
        "e2e.forward",
        TransferProvider(copy_prepare),
    )
    reverse = sw.TransferRoute(
        storage,
        compute,
        "e2e.reverse",
        TransferProvider(copy_prepare),
    )
    sw.register_carrier_definition(
        compute,
        sw.CarrierDefinition(
            _storage(),
            sw.KernelProvider(interface),
            transfers=(forward,),
        ),
    )
    sw.register_carrier_definition(
        storage,
        sw.CarrierDefinition(_storage(), transfers=(reverse,)),
    )
    sw.register_kernel_pack(
        sw.KernelPack("e2e", "1", compute, interface, (pattern, reference_pattern))
    )

    dependent = state.dependent

    def composite_capabilities(carrier: object) -> tuple[OperationCapability, ...]:
        return (capability,)

    def composite_results(carrier: object, invocation: object) -> tuple[type, ...]:
        return (compute,)

    def composite_prepare(carrier: object, invocation: object) -> PreparedComposite:
        resolved = cast(sw.operation.ResolvedInvocation, invocation)
        value = cast(sw.Tensor, resolved.operands[0])
        return PreparedComposite(
            lambda: result(
                resolved,
                [float(value[index]) ** 2 for index in range(value.size())],
                compute,
            )
        )

    sw.register_carrier_definition(
        dependent,
        sw.CarrierDefinition(
            _storage(),
            composite=sw.CompositeProvider(
                composite_capabilities,
                composite_results,
                composite_prepare,
            ),
        ),
    )
    return state


def test_kernel_pack_and_reference_fallback_share_capability_authority(
    world: _World,
) -> None:
    world.attempts.clear()
    world.use_kernel = True
    value = _tensor(world.compute, [-2.0, 3.0])
    result = cast(sw.Tensor, world.operation(value))
    assert [result[index] for index in range(result.size())] == [4.0, 9.0]
    assert world.attempts == ["kernel"]
    assert world.compute(1, dtype=sw.DType.Float32).supports_operation_plan(world.plan)

    world.use_kernel = False
    fallback = cast(sw.Tensor, world.operation(_tensor(world.compute, [4.0])))
    assert fallback[0] == 16.0
    assert world.attempts[-2:] == ["kernel", "reference"]
    world.use_kernel = True


def test_custom_semantic_options_keep_dispatch_metadata_and_autograd(
    world: _World,
) -> None:
    value = _tensor(world.compute, [3.0])
    result = cast(sw.Tensor, world.operation(value, tag=7))

    assert result.autograd_ctx is not None
    assert result.autograd_ctx._dispatch_carrier_class is world.compute
    result.backward(_tensor(world.compute, [1.0]))

    assert value.grad is not None
    assert value.grad[0] == 6.0


def test_exact_class_attachment_and_dispatch_metadata_are_authoritative(
    world: _World,
) -> None:
    subclass = type("E2EComputeSubclass", (world.compute,), {})
    with pytest.raises(TypeError, match="exact CarrierDefinition"):
        sw.register_kernel_pack(
            sw.KernelPack(
                "e2e-subclass",
                "1",
                subclass,
                sw.KernelExecutionInterface("e2e-v1"),
                (world.pattern,),
            )
        )
    carrier = world.compute(1, dtype=sw.DType.Float32)
    first = carrier.dispatch_op("e2e.square")
    second = carrier.dispatch_op("e2e.square")
    assert first is not second
    assert first._dispatch_carrier_class is world.compute

    for target in (
        sw.Generic,
        sw.CPU,
        sw.Metal,
        sw.FileBacked,
        sw.BlockDeviceCarrier,
        sw.Evictable,
        sw.TiledEvictable,
        world.dependent,
    ):
        with pytest.raises(
            TypeError, match=r"exact CarrierDefinition|DependentCarrier"
        ):
            sw.register_kernel_pack(
                sw.KernelPack(
                    f"e2e-refused-{target.__name__.lower()}",
                    "1",
                    target,
                    sw.KernelExecutionInterface("e2e-v1"),
                    (world.pattern,),
                )
            )


def test_definition_backed_dependent_composition_executes_public_operation(
    world: _World,
) -> None:
    dependent = world.dependent(2, dtype=sw.DType.Float32)
    dependent[0], dependent[1] = -3.0, 2.0
    value = sw.Tensor(
        dependent,
        0,
        sw.Layout(sw.Shape(2), sw.Stride(1)),
    )
    result = cast(sw.Tensor, world.operation(value))
    assert type(result.carrier) is world.compute
    assert [result[index] for index in range(result.size())] == [9.0, 4.0]


def test_provider_failure_preserves_both_custom_carriers() -> None:
    source_class = type("E2EFailureSource", (sw.Carrier,), {})
    destination_class = type("E2EFailureDestination", (sw.Carrier,), {})

    def fail_prepare(request: TransferRequest) -> PreparedTransfer:
        del request

        def submit() -> None:
            raise RuntimeError("provider boom")

        return PreparedTransfer(submit)

    route = sw.TransferRoute(
        source_class,
        destination_class,
        "e2e.failure",
        TransferProvider(fail_prepare),
    )
    reverse = sw.TransferRoute(
        destination_class,
        source_class,
        "e2e.failure.reverse",
        TransferProvider(lambda request: PreparedTransfer(lambda: None)),
    )
    sw.register_carrier_definition(
        source_class, sw.CarrierDefinition(_storage(), transfers=(route,))
    )
    sw.register_carrier_definition(
        destination_class, sw.CarrierDefinition(_storage(), transfers=(reverse,))
    )
    source = _tensor(source_class, [1.0, 2.0])
    destination = destination_class(2, dtype=sw.DType.Float32)
    handle = sw.move_async(source, destination)
    with pytest.raises(RuntimeError, match="provider boom"):
        handle.wait()
    assert not source.carrier.is_released()
    assert not source.carrier.is_owned()
    assert not destination.is_owned()


def test_move_autograd_crosses_custom_carriers_and_tiled_projection_accepts_them(
    world: _World,
) -> None:
    # A complete pair of public TransferRoutes makes the move an autograd
    # boundary in both directions.
    source_class = type("E2EMoveSource", (sw.Carrier,), {})
    destination_class = type("E2EMoveDestination", (sw.Carrier,), {})

    def copying(request: TransferRequest) -> PreparedTransfer:
        def submit() -> None:
            source = cast(sw.Carrier, request.tensor.carrier)
            destination = cast(sw.Carrier, request.destination)
            for index in range(request.physical_span):
                destination[index] = source[index + request.tensor.offset]

        return PreparedTransfer(submit)

    sw.register_carrier_definition(
        source_class,
        sw.CarrierDefinition(
            _storage(),
            transfers=(
                sw.TransferRoute(
                    source_class,
                    destination_class,
                    "forward",
                    TransferProvider(copying),
                ),
            ),
        ),
    )
    sw.register_carrier_definition(
        destination_class,
        sw.CarrierDefinition(
            _storage(),
            transfers=(
                sw.TransferRoute(
                    destination_class,
                    source_class,
                    "reverse",
                    TransferProvider(copying),
                ),
            ),
        ),
    )
    source = _tensor(source_class, [2.0, 4.0])
    moved = sw.move(source, destination_class(2, dtype=sw.DType.Float32))
    moved.backward(_tensor(destination_class, [1.0, 1.0]))
    assert source.grad is not None
    assert [source.grad[index] for index in range(2)] == [1.0, 1.0]

    primary = world.compute(2, dtype=sw.DType.Float32)
    primary[0], primary[1] = 1.0, 2.0
    tiled = sw.TiledEvictable(
        primary,
        world.storage(0, dtype=sw.DType.Float32),
        (2,),
        (1,),
    )
    projected = tiled.project(
        sw.Tensor(tiled, 0, sw.Layout(sw.Shape(2), sw.Stride(1))),
        sw.TileSelection(([1, 0],)),
    ).wait()
    assert [projected[index] for index in range(2)] == [2.0, 1.0]


def test_tiled_composes_multiple_legacy_dependency_pairs_identically() -> None:
    values = [1.0, -2.0, 3.0, -4.0]

    cpu_primary = sw.CPU(len(values), dtype=sw.DType.Float32)
    for index, value in enumerate(values):
        cpu_primary[index] = value
    cases = (
        (
            sw.Generic(values, dtype=sw.DType.Float32),
            sw.CPU(0, dtype=sw.DType.Float32),
        ),
        (cpu_primary, sw.FileBacked(dtype=sw.DType.Float32)),
    )

    for primary, secondary in cases:
        tiled = sw.TiledEvictable(primary, secondary, (2,), (2,))
        tensor = sw.Tensor(
            tiled,
            0,
            sw.Layout(sw.Shape(len(values)), sw.Stride(1)),
        )
        selected = sw.TileSet(((0,),))
        try:
            tiled.evict(selected)
            assert [tensor[index] for index in range(tensor.size())] == values

            result = sw.relu(tensor)
            assert [result[index] for index in range(result.size())] == [
                1.0,
                0.0,
                3.0,
                0.0,
            ]

            tiled.promote(selected)
            assert [tensor[index] for index in range(tensor.size())] == values
        finally:
            tiled.release()


def test_move_api_is_single_tensor_and_blocking_only() -> None:
    assert not hasattr(sw.AwaitMove, "__await__")
    assert not hasattr(sw, "move_batch")
    assert not hasattr(sw, "move_many")
    tensor = sw.Tensor(
        sw.Generic([1.0]),
        0,
        sw.Layout(sw.Shape(1), sw.Stride(1)),
    )
    with pytest.raises(TypeError):
        sw.move_async([tensor], sw.Generic([0.0]))  # type: ignore[arg-type]
