from __future__ import annotations

import gc
import weakref
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError
from threading import Barrier, Thread
from typing import Any, cast

import pytest

import strideweave as sw
from strideweave.carriers.extension import (
    PreparedComposite,
    PreparedKernel,
    ProviderResult,
    Unsupported,
)
from strideweave.carriers.operation_capability import (
    OperationCapability,
    capabilities_for_carrier_class,
)
from strideweave.carriers.operation_policy import (
    Arithmetic,
    OperandPlan,
    OperandRole,
    OperationPlan,
    resolve_operation_plan,
)
from strideweave.operation_definition import _operation_definition


class FakeInterface(sw.KernelExecutionInterface):
    __slots__ = ("label",)

    def __init__(self, version: str, label: str = "fake") -> None:
        super().__init__(version)
        object.__setattr__(self, "label", label)


class Completion:
    def __init__(self, result: ProviderResult | Exception) -> None:
        self._result = result
        self._done = False
        self.wait_count = 0

    @property
    def done(self) -> bool:
        return self._done

    def wait(self) -> ProviderResult:
        self.wait_count += 1
        self._done = True
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def fresh_carrier(name: str) -> type[sw.Carrier]:
    return type(name, (sw.Carrier,), {})


def storage_provider() -> sw.StorageProvider:
    def allocate(carrier: Any, slots: int) -> None:
        carrier.values = [None] * slots

    def release(carrier: Any) -> None:
        carrier.values = []

    def physical_size(carrier: Any) -> int:
        return len(carrier.values)

    def read(carrier: Any, index: int) -> object:
        return carrier.values[index]

    def write(carrier: Any, index: int, value: object) -> None:
        carrier.values[index] = value

    return sw.StorageProvider(
        lambda _carrier, dtype: dtype in (sw.DType.Float32, sw.DType.Int32),
        physical_size,
        allocate,
        release,
        read,
        write,
    )


def make_tensor(
    carrier_class: type[sw.Carrier],
    values: Sequence[object],
    *,
    dtype: sw.SimpleDType = sw.DType.Float32,
    layout: sw.Layout | None = None,
) -> sw.Tensor:
    if layout is None:
        layout = sw.Layout(sw.Shape(len(values)), sw.Stride(1))
    carrier = carrier_class(len(values), dtype=dtype)
    for index, value in enumerate(values):
        carrier.set_value(index, value)
    return sw.Tensor(carrier, 0, layout)


def result_for(
    invocation: sw.operation.ResolvedInvocation,
    values: list[object],
    *,
    carrier_class: type[sw.Carrier] | None = None,
) -> ProviderResult:
    source = next(
        operand for operand in invocation.operands if isinstance(operand, sw.Tensor)
    )
    spec = invocation.results[0]
    if carrier_class is None:
        carrier = source.carrier.new_like(values, dtype=spec.dtype)
    else:
        carrier = carrier_class(len(values), dtype=spec.dtype)
        for index, value in enumerate(values):
            carrier.set_value(index, value)
    return ProviderResult((sw.Tensor(carrier, 0, spec.layout),))


def relu_capability(dtype: sw.SimpleDType = sw.DType.Float32) -> OperationCapability:
    return OperationCapability.from_plan(resolve_operation_plan("relu", dtype))


def relu_pattern(
    kernel_id: str,
    prepare: Callable[[Any, Any], PreparedKernel | Unsupported],
    *,
    dtype: sw.SimpleDType = sw.DType.Float32,
    preference: int = 0,
) -> sw.KernelPattern:
    return sw.KernelPattern(
        _operation_definition("relu"),
        (relu_capability(dtype),),
        kernel_id,
        prepare,
        preference,
    )


def relu_prepared(
    invocation: sw.operation.ResolvedInvocation,
    interface: sw.KernelExecutionInterface,
    *,
    asynchronous: bool = False,
) -> PreparedKernel:
    tensor = cast(sw.Tensor, invocation.operands[0])

    def result() -> ProviderResult:
        return result_for(
            invocation,
            [max(0, tensor[index]) for index in range(tensor.size())],
        )

    if asynchronous:
        return PreparedKernel(interface, lambda: Completion(result()))
    return PreparedKernel(interface, result)


def register_fake(
    carrier_class: type[sw.Carrier],
    interface: sw.KernelExecutionInterface,
    patterns: tuple[sw.KernelPattern, ...] = (),
) -> None:
    sw.register_carrier_definition(
        carrier_class,
        sw.CarrierDefinition(
            storage_provider(), sw.KernelProvider(interface, patterns)
        ),
    )


def multi_result_operation(
    name: str,
    vjp: object,
) -> tuple[sw.OperationDefinition, OperationCapability]:
    schema = sw.OperationSchema(
        (sw.OperandSpec("value", sw.OperandKind.TENSOR, differentiable=True),)
    )

    def resolve(
        call: sw.operation.BoundOperationCall,
    ) -> sw.operation.ResolvedInvocation:
        value = cast(sw.Tensor, call.operands[0])
        plan = OperationPlan(
            name,
            (OperandPlan(OperandRole.TENSOR, value.dtype(), value.dtype()),),
            Arithmetic.BINARY32,
            None,
            None,
            value.dtype(),
        )
        result = sw.ResultSpec(value.dtype(), value.layout.shape, value.layout)
        return sw.operation.ResolvedInvocation(
            call,
            plan,
            (result, result),
            saved_operands=(0,),
        )

    operation = sw.define_operation(name, schema, resolve, vjp=cast(Any, vjp))
    plan = OperationPlan(
        name,
        (
            OperandPlan(
                OperandRole.TENSOR,
                sw.DType.Float32,
                sw.DType.Float32,
            ),
        ),
        Arithmetic.BINARY32,
        None,
        None,
        sw.DType.Float32,
    )
    return operation, OperationCapability.from_plan(plan)


def multi_provider_result(
    invocation: sw.operation.ResolvedInvocation,
    carrier_classes: tuple[type[sw.Carrier], type[sw.Carrier]],
    values: tuple[Sequence[object], Sequence[object]],
) -> ProviderResult:
    outputs = []
    for spec, carrier_class, output_values in zip(
        invocation.results, carrier_classes, values, strict=True
    ):
        carrier = carrier_class(spec.layout.cosize, dtype=spec.dtype)
        for index, value in enumerate(output_values):
            carrier.set_value(index, value)
        outputs.append(sw.Tensor(carrier, 0, spec.layout))
    return ProviderResult(outputs)


def mutating_operation(
    name: str,
    *,
    mutated_operands: tuple[int, ...],
    result_count: int = 2,
    differentiable: bool = False,
) -> tuple[sw.OperationDefinition, OperationCapability]:
    schema = sw.OperationSchema(
        (
            sw.OperandSpec("lhs", sw.OperandKind.TENSOR, differentiable=differentiable),
            sw.OperandSpec("rhs", sw.OperandKind.TENSOR),
        )
    )

    def resolve(
        call: sw.operation.BoundOperationCall,
    ) -> sw.operation.ResolvedInvocation:
        lhs = cast(sw.Tensor, call.operands[0])
        rhs = cast(sw.Tensor, call.operands[1])
        plan = OperationPlan(
            name,
            (
                OperandPlan(OperandRole.TENSOR, lhs.dtype(), lhs.dtype()),
                OperandPlan(OperandRole.TENSOR, rhs.dtype(), rhs.dtype()),
            ),
            Arithmetic.BINARY32,
            None,
            None,
            lhs.dtype(),
        )
        result = sw.ResultSpec(lhs.dtype(), lhs.layout.shape, lhs.layout)
        return sw.operation.ResolvedInvocation(
            call,
            plan,
            (result,) * result_count,
            mutated_operands=mutated_operands,
            saved_operands=(0,) if differentiable else (),
        )

    vjp: object = (
        (lambda context, cotangents: (cotangents[0],))
        if differentiable
        else sw.NON_DIFFERENTIABLE
    )
    operation = sw.define_operation(name, schema, resolve, vjp=cast(Any, vjp))
    plan = OperationPlan(
        name,
        (
            OperandPlan(
                OperandRole.TENSOR,
                sw.DType.Float32,
                sw.DType.Float32,
            ),
            OperandPlan(
                OperandRole.TENSOR,
                sw.DType.Float32,
                sw.DType.Float32,
            ),
        ),
        Arithmetic.BINARY32,
        None,
        None,
        sw.DType.Float32,
    )
    return operation, OperationCapability.from_plan(plan)


def unary_custom_operation(
    name: str,
    *,
    differentiable: bool = False,
) -> tuple[sw.OperationDefinition, OperationCapability]:
    schema = sw.OperationSchema(
        (sw.OperandSpec("value", sw.OperandKind.TENSOR, differentiable=differentiable),)
    )

    def resolve(
        call: sw.operation.BoundOperationCall,
    ) -> sw.operation.ResolvedInvocation:
        value = cast(sw.Tensor, call.operands[0])
        plan = OperationPlan(
            name,
            (OperandPlan(OperandRole.TENSOR, value.dtype(), value.dtype()),),
            Arithmetic.BINARY32,
            None,
            None,
            value.dtype(),
        )
        return sw.operation.ResolvedInvocation(
            call,
            plan,
            (sw.ResultSpec(value.dtype(), value.layout.shape, value.layout),),
            saved_operands=(0,) if differentiable else (),
        )

    vjp: object = (
        (lambda context, cotangents: (cotangents[0],))
        if differentiable
        else sw.NON_DIFFERENTIABLE
    )
    operation = sw.define_operation(name, schema, resolve, vjp=cast(Any, vjp))
    plan = OperationPlan(
        name,
        (
            OperandPlan(
                OperandRole.TENSOR,
                sw.DType.Float32,
                sw.DType.Float32,
            ),
        ),
        Arithmetic.BINARY32,
        None,
        None,
        sw.DType.Float32,
    )
    return operation, OperationCapability.from_plan(plan)


def schema_autograd_operation(
    name: str,
    differentiable: tuple[bool, bool],
    result_count: int,
    vjp: Callable[
        [sw.VJPContext, tuple[sw.Tensor | None, ...]],
        tuple[sw.Tensor | None, ...],
    ],
) -> tuple[sw.OperationDefinition, OperationCapability]:
    schema = sw.OperationSchema(
        (
            sw.OperandSpec(
                "lhs", sw.OperandKind.TENSOR, differentiable=differentiable[0]
            ),
            sw.OperandSpec(
                "rhs", sw.OperandKind.TENSOR, differentiable=differentiable[1]
            ),
        )
    )

    def resolve(
        call: sw.operation.BoundOperationCall,
    ) -> sw.operation.ResolvedInvocation:
        lhs = cast(sw.Tensor, call.operands[0])
        rhs = cast(sw.Tensor, call.operands[1])
        plan = OperationPlan(
            name,
            (
                OperandPlan(OperandRole.TENSOR, lhs.dtype(), lhs.dtype()),
                OperandPlan(OperandRole.TENSOR, rhs.dtype(), rhs.dtype()),
            ),
            Arithmetic.BINARY32,
            None,
            None,
            lhs.dtype(),
        )
        result = sw.ResultSpec(lhs.dtype(), lhs.layout.shape, lhs.layout)
        return sw.operation.ResolvedInvocation(
            call,
            plan,
            (result,) * result_count,
            saved_operands=(0, 1),
        )

    operation = sw.define_operation(name, schema, resolve, vjp=vjp)
    plan = OperationPlan(
        name,
        (
            OperandPlan(
                OperandRole.TENSOR,
                sw.DType.Float32,
                sw.DType.Float32,
            ),
            OperandPlan(
                OperandRole.TENSOR,
                sw.DType.Float32,
                sw.DType.Float32,
            ),
        ),
        Arithmetic.BINARY32,
        None,
        None,
        sw.DType.Float32,
    )
    return operation, OperationCapability.from_plan(plan)


def autograd_provider_result(
    carrier_class: type[sw.Carrier],
    invocation: sw.operation.ResolvedInvocation,
) -> ProviderResult:
    tensors = tuple(
        operand for operand in invocation.operands if isinstance(operand, sw.Tensor)
    )
    if invocation.name == "relu":
        values = tuple(
            max(0.0, float(tensors[0][index])) for index in range(tensors[0].size())
        )
    else:
        values = tuple(
            sum(float(tensor[index]) for tensor in tensors)
            for index in range(tensors[0].size())
        )
    outputs: list[sw.Tensor] = []
    for result_index, spec in enumerate(invocation.results):
        carrier = carrier_class(spec.layout.cosize, dtype=spec.dtype)
        for index, value in enumerate(values):
            carrier.set_value(index, value + result_index)
        outputs.append(sw.Tensor(carrier, 0, spec.layout))
    return ProviderResult(outputs)


def register_autograd_test_carrier(
    provider_kind: str,
    label: str,
    operation: sw.OperationDefinition,
    capability: OperationCapability,
) -> type[sw.Carrier]:
    if provider_kind == "kernel":
        carrier_class = fresh_carrier(f"{label}KernelCarrier")
        interface = FakeInterface("v1")
        pattern = sw.KernelPattern(
            operation,
            (capability,),
            f"{operation.name}.f32",
            lambda invocation, selected: PreparedKernel(
                selected,
                lambda: autograd_provider_result(carrier_class, invocation),
            ),
        )
        register_fake(
            carrier_class,
            interface,
            (
                pattern,
                relu_pattern(
                    f"{operation.name}.relu",
                    lambda invocation, selected: relu_prepared(invocation, selected),
                ),
            ),
        )
        return carrier_class

    class CompositeCarrier(sw.DependentCarrier):
        def __init__(
            self,
            slots: int,
            *,
            dtype: sw.SimpleDType,
            mutable: bool = True,
        ) -> None:
            super().__init__(slots, dtype=dtype, mutable=mutable)
            self._finalize_dependent_capabilities()

    CompositeCarrier.__name__ = f"{label}CompositeCarrier"

    def result_carriers(
        carrier: object, invocation: object
    ) -> tuple[type[sw.Carrier], ...]:
        del carrier
        resolved = cast(sw.operation.ResolvedInvocation, invocation)
        return (CompositeCarrier,) * len(resolved.results)

    def prepare(carrier: object, invocation: object) -> PreparedComposite:
        del carrier
        resolved = cast(sw.operation.ResolvedInvocation, invocation)
        return PreparedComposite(
            lambda: autograd_provider_result(CompositeCarrier, resolved)
        )

    sw.register_carrier_definition(
        CompositeCarrier,
        sw.CarrierDefinition(
            storage_provider(),
            composite=sw.CompositeProvider(
                lambda carrier: (capability, relu_capability()),
                result_carriers,
                prepare,
            ),
        ),
    )
    return CompositeCarrier


def alias_test_operation(
    name: str,
    *,
    vjp: object,
    result_count: int = 1,
    alias_result: int | None = 0,
) -> tuple[sw.OperationDefinition, OperationCapability]:
    schema = sw.OperationSchema(
        (sw.OperandSpec("value", sw.OperandKind.TENSOR, differentiable=True),)
    )

    def resolve(
        call: sw.operation.BoundOperationCall,
    ) -> sw.operation.ResolvedInvocation:
        value = cast(sw.Tensor, call.operands[0])
        plan = OperationPlan(
            name,
            (OperandPlan(OperandRole.TENSOR, value.dtype(), value.dtype()),),
            Arithmetic.BINARY32,
            None,
            None,
            value.dtype(),
        )
        results = tuple(
            sw.ResultSpec(
                value.dtype(),
                value.layout.shape,
                value.layout,
                alias_of=0 if index == alias_result else None,
            )
            for index in range(result_count)
        )
        return sw.operation.ResolvedInvocation(
            call,
            plan,
            results,
            saved_operands=(0,),
        )

    operation = sw.define_operation(name, schema, resolve, vjp=cast(Any, vjp))
    plan = OperationPlan(
        name,
        (
            OperandPlan(
                OperandRole.TENSOR,
                sw.DType.Float32,
                sw.DType.Float32,
            ),
        ),
        Arithmetic.BINARY32,
        None,
        None,
        sw.DType.Float32,
    )
    return operation, OperationCapability.from_plan(plan)


def alias_provider_result(
    carrier_class: type[sw.Carrier],
    invocation: sw.operation.ResolvedInvocation,
    *,
    storage_sharing_view: bool = False,
) -> ProviderResult:
    source = cast(sw.Tensor, invocation.operands[0])
    outputs: list[sw.Tensor] = []
    for index, spec in enumerate(invocation.results):
        if spec.alias_of is not None:
            outputs.append(source)
        elif storage_sharing_view:
            outputs.append(sw.Tensor(source.carrier, source.offset, spec.layout))
        else:
            carrier = carrier_class(spec.layout.cosize, dtype=spec.dtype)
            for value_index in range(spec.layout.cosize):
                carrier.set_value(value_index, float(index + value_index + 1))
            outputs.append(sw.Tensor(carrier, 0, spec.layout))
    return ProviderResult(outputs)


def register_counted_alias_carrier(
    provider_kind: str,
    label: str,
    operation: sw.OperationDefinition,
    capability: OperationCapability,
    counters: dict[str, int],
    *,
    storage_sharing_view: bool = False,
) -> type[sw.Carrier]:
    def submit(
        carrier_class: type[sw.Carrier],
        invocation: sw.operation.ResolvedInvocation,
    ) -> ProviderResult:
        counters["submit"] += 1
        return alias_provider_result(
            carrier_class,
            invocation,
            storage_sharing_view=storage_sharing_view,
        )

    if provider_kind == "kernel":
        carrier_class = fresh_carrier(f"{label}KernelCarrier")
        interface = FakeInterface("alias")

        def prepare(
            invocation: sw.operation.ResolvedInvocation,
            selected: sw.KernelExecutionInterface,
        ) -> PreparedKernel:
            counters["prepare"] += 1
            return PreparedKernel(
                selected,
                lambda: submit(carrier_class, invocation),
            )

        register_fake(
            carrier_class,
            interface,
            (
                sw.KernelPattern(
                    operation,
                    (capability,),
                    f"{operation.name}.alias",
                    prepare,
                ),
                relu_pattern(
                    f"{operation.name}.relu",
                    lambda invocation, selected: relu_prepared(invocation, selected),
                ),
            ),
        )
        return carrier_class

    class CompositeCarrier(sw.DependentCarrier):
        def __init__(
            self,
            slots: int,
            *,
            dtype: sw.SimpleDType,
            mutable: bool = True,
        ) -> None:
            super().__init__(slots, dtype=dtype, mutable=mutable)
            self._finalize_dependent_capabilities()

    CompositeCarrier.__name__ = f"{label}CompositeCarrier"

    def capabilities(carrier: object) -> tuple[OperationCapability, ...]:
        del carrier
        counters["capabilities"] += 1
        return (capability, relu_capability())

    def result_carriers(
        carrier: object, invocation: object
    ) -> tuple[type[sw.Carrier], ...]:
        del carrier
        resolved = cast(sw.operation.ResolvedInvocation, invocation)
        if resolved.name == operation.name:
            counters["result_carriers"] += 1
        return (CompositeCarrier,) * len(resolved.results)

    def prepare_composite(carrier: object, invocation: object) -> PreparedComposite:
        del carrier
        resolved = cast(sw.operation.ResolvedInvocation, invocation)
        if resolved.name == operation.name:
            counters["prepare"] += 1
            return PreparedComposite(lambda: submit(CompositeCarrier, resolved))
        return PreparedComposite(
            lambda: autograd_provider_result(CompositeCarrier, resolved)
        )

    sw.register_carrier_definition(
        CompositeCarrier,
        sw.CarrierDefinition(
            storage_provider(),
            composite=sw.CompositeProvider(
                capabilities,
                result_carriers,
                prepare_composite,
            ),
        ),
    )
    return CompositeCarrier


def test_kernel_pack_exports_are_identity_preserving() -> None:
    import strideweave.carriers.extension as extension

    for name in ("KernelPack", "KernelPattern", "register_kernel_pack"):
        assert getattr(sw, name) is getattr(extension, name)
        assert name in sw.__all__


def test_pattern_materializes_plans_and_validates_all_fields() -> None:
    operation = _operation_definition("relu")
    capability = relu_capability()
    pattern = sw.KernelPattern(
        operation,
        iter((capability,)),
        "relu.valid",
        lambda invocation, interface: Unsupported("not used"),
        preference=-3,
    )
    assert pattern.plans == (capability,)
    assert pattern.preference == -3
    with pytest.raises(FrozenInstanceError):
        pattern.preference = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="must not be empty"):
        sw.KernelPattern(
            operation,
            (),
            "empty",
            lambda invocation, interface: Unsupported("unused"),
        )
    with pytest.raises(ValueError, match="duplicate"):
        sw.KernelPattern(
            operation,
            (capability, capability),
            "duplicate",
            lambda invocation, interface: Unsupported("unused"),
        )
    add = OperationCapability.from_plan(
        resolve_operation_plan("add", sw.DType.Float32, sw.DType.Float32)
    )
    with pytest.raises(ValueError, match=r"operation\.name"):
        sw.KernelPattern(
            operation,
            (add,),
            "wrong.operation",
            lambda invocation, interface: Unsupported("unused"),
        )
    with pytest.raises(TypeError, match="signed integer"):
        sw.KernelPattern(
            operation,
            (capability,),
            "bool.preference",
            lambda invocation, interface: Unsupported("unused"),
            True,
        )


def test_pack_materializes_patterns_and_rejects_local_duplicates() -> None:
    carrier_class = fresh_carrier("PackValueCarrier")
    interface = FakeInterface("v1")
    pattern = relu_pattern(
        "pack.value", lambda invocation, selected: Unsupported("unused")
    )
    pack = sw.KernelPack("acme", "1", carrier_class, interface, iter((pattern,)))
    assert pack.patterns == (pattern,)
    with pytest.raises(ValueError, match="must not be empty"):
        sw.KernelPack("acme", "1", carrier_class, interface, ())
    with pytest.raises(ValueError, match="duplicate kernel_id"):
        sw.KernelPack("acme", "1", carrier_class, interface, (pattern, pattern))


def test_capability_union_and_execution_use_the_same_frozen_patterns() -> None:
    carrier_class = fresh_carrier("UnionCarrier")
    interface = FakeInterface("v1")
    float_pattern = relu_pattern(
        "relu.float",
        lambda invocation, selected: relu_prepared(invocation, selected),
    )
    int_pattern = relu_pattern(
        "relu.int",
        lambda invocation, selected: relu_prepared(invocation, selected),
        dtype=sw.DType.Int32,
    )
    register_fake(carrier_class, interface, (float_pattern,))
    sw.register_kernel_pack(
        sw.KernelPack("extra", "1", carrier_class, interface, (int_pattern,))
    )

    capabilities = capabilities_for_carrier_class(carrier_class)
    assert set(capabilities) == {relu_capability(), relu_capability(sw.DType.Int32)}
    assert capabilities == capabilities_for_carrier_class(carrier_class)

    for dtype, values in (
        (sw.DType.Float32, [-2.0, 3.0]),
        (sw.DType.Int32, [-2, 3]),
    ):
        value = make_tensor(carrier_class, values, dtype=dtype)
        result = sw.relu(value)
        assert [result[index] for index in range(2)] == [0, 3]
        assert type(result.carrier) is carrier_class


def test_preference_then_builtin_and_sorted_pack_order_control_fallback() -> None:
    carrier_class = fresh_carrier("FallbackCarrier")
    interface = FakeInterface("v1")
    prepared: list[str] = []

    def rejecting(name: str) -> Callable[[Any, Any], Unsupported]:
        def prepare(invocation: Any, selected: Any) -> Unsupported:
            del invocation, selected
            prepared.append(name)
            return Unsupported(f"{name} rejected")

        return prepare

    builtin = relu_pattern("builtin", rejecting("builtin"))
    pack_z = relu_pattern(
        "pack-z",
        lambda invocation, selected: (
            prepared.append("pack-z")
            or relu_prepared(invocation, selected, asynchronous=True)
        ),
    )
    pack_a = relu_pattern("pack-a", rejecting("pack-a"))
    register_fake(carrier_class, interface, (builtin,))
    sw.register_kernel_pack(
        sw.KernelPack("zeta", "1", carrier_class, interface, (pack_z,))
    )
    sw.register_kernel_pack(
        sw.KernelPack("alpha", "1", carrier_class, interface, (pack_a,))
    )
    result = sw.relu(make_tensor(carrier_class, [-1.0, 2.0]))
    assert prepared == ["builtin", "pack-a", "pack-z"]
    assert [result[index] for index in range(2)] == [0, 2.0]


def test_higher_preference_prepares_before_builtin_order() -> None:
    carrier_class = fresh_carrier("PreferenceCarrier")
    interface = FakeInterface("v1")
    prepared: list[str] = []
    builtin = relu_pattern(
        "builtin",
        lambda invocation, selected: (
            prepared.append("builtin") or relu_prepared(invocation, selected)
        ),
    )
    optimized = relu_pattern(
        "optimized",
        lambda invocation, selected: (
            prepared.append("optimized") or relu_prepared(invocation, selected)
        ),
        preference=10,
    )
    register_fake(carrier_class, interface, (builtin,))
    sw.register_kernel_pack(
        sw.KernelPack("optimized", "1", carrier_class, interface, (optimized,))
    )
    sw.relu(make_tensor(carrier_class, [1.0]))
    assert prepared == ["optimized"]


def test_preparation_exception_is_terminal_and_never_falls_back() -> None:
    carrier_class = fresh_carrier("TerminalPrepareCarrier")
    interface = FakeInterface("v1")
    prepared: list[str] = []

    def fail(invocation: Any, selected: Any) -> object:
        del invocation, selected
        prepared.append("fail")
        raise RuntimeError("compile failed")

    fallback = relu_pattern(
        "fallback",
        lambda invocation, selected: (
            prepared.append("fallback") or relu_prepared(invocation, selected)
        ),
    )
    register_fake(
        carrier_class,
        interface,
        (
            relu_pattern(
                "terminal",
                cast(
                    Callable[[Any, Any], PreparedKernel | Unsupported],
                    fail,
                ),
                preference=1,
            ),
            fallback,
        ),
    )
    with pytest.raises(RuntimeError, match="compile failed"):
        sw.relu(make_tensor(carrier_class, [1.0]))
    assert prepared == ["fail"]


def test_all_unsupported_reasons_are_combined_in_stable_order() -> None:
    carrier_class = fresh_carrier("AllUnsupportedCarrier")
    interface = FakeInterface("v1")
    register_fake(
        carrier_class,
        interface,
        (
            relu_pattern(
                "first",
                lambda invocation, selected: Unsupported("alignment"),
                preference=2,
            ),
            relu_pattern("second", lambda invocation, selected: Unsupported("shape")),
        ),
    )
    with pytest.raises(
        sw.UnsupportedOperationPlan,
        match=r"first: alignment; second: shape",
    ):
        sw.relu(make_tensor(carrier_class, [1.0]))


def test_live_definition_dispatch_gaps_use_unsupported_operation_plan() -> None:
    operation, _ = unary_custom_operation("acme.unsupported_dispatch")
    interface = FakeInterface("v1")

    storage_only = fresh_carrier("UnsupportedStorageOnly")
    sw.register_carrier_definition(
        storage_only,
        sw.CarrierDefinition(storage_provider()),
    )

    independent = fresh_carrier("UnsupportedIndependent")
    register_fake(independent, interface)

    callbacks: list[str] = []

    class UnsupportedDependent(sw.DependentCarrier):
        def __init__(self, slots: int, *, dtype: sw.SimpleDType) -> None:
            super().__init__(slots, dtype=dtype)
            self._finalize_dependent_capabilities()

    sw.register_carrier_definition(
        UnsupportedDependent,
        sw.CarrierDefinition(
            storage_provider(),
            composite=sw.CompositeProvider(
                lambda carrier: (),
                lambda carrier, invocation: callbacks.append("result_carriers") or (),
                lambda carrier, invocation: (
                    callbacks.append("prepare") or Unsupported("unreachable")
                ),
            ),
        ),
    )

    cases = (
        (storage_only, "storage-only"),
        (independent, "no kernel"),
        (UnsupportedDependent, "does not support"),
    )
    for carrier_class, message in cases:
        value = make_tensor(carrier_class, [1.0])
        with pytest.raises(sw.UnsupportedOperationPlan, match=message):
            value.carrier.dispatch_op(operation.name)
        with pytest.raises(sw.UnsupportedOperationPlan, match=message):
            operation(value)

    assert callbacks == []


def test_released_definition_dispatch_precedes_unknown_and_unsupported_names() -> None:
    operation, _ = unary_custom_operation("acme.released_dispatch")
    carrier_class = fresh_carrier("ReleasedUnsupportedDefinition")
    sw.register_carrier_definition(
        carrier_class,
        sw.CarrierDefinition(storage_provider()),
    )
    carrier = carrier_class(1, dtype=sw.DType.Float32)
    carrier.release()

    for name in (operation.name, "acme.unknown_released_dispatch"):
        with pytest.raises(
            RuntimeError, match=r"^ReleasedUnsupportedDefinition is released$"
        ):
            carrier.dispatch_op(name)


def test_fake_jit_completion_and_precompiled_kernel_share_one_interface() -> None:
    carrier_class = fresh_carrier("JitAndPrecompiledCarrier")
    interface = FakeInterface("v7", "device-0")
    compiled_shapes: list[sw.Shape] = []
    completions: list[Completion] = []

    def prepare(invocation: Any, selected: Any) -> PreparedKernel:
        assert selected is interface
        compiled_shapes.append(invocation.results[0].shape)
        prepared = relu_prepared(invocation, selected, asynchronous=True)

        def submit() -> Completion:
            completion = cast(Completion, prepared.submit())
            completions.append(completion)
            return completion

        return PreparedKernel(selected, submit)

    register_fake(carrier_class, interface, (relu_pattern("jit", prepare),))
    result = sw.relu(make_tensor(carrier_class, [-1.0, 4.0]))
    assert compiled_shapes == [sw.Shape(2)]
    assert len(completions) == 1
    assert completions[0].done
    assert completions[0].wait_count == 1
    assert result[1] == 4.0


def test_namespaced_custom_operation_executes_through_a_pack() -> None:
    carrier_class = fresh_carrier("CustomOperationCarrier")
    interface = FakeInterface("v1")
    schema = sw.OperationSchema(
        (sw.OperandSpec("value", sw.OperandKind.TENSOR, differentiable=True),)
    )

    def resolve(
        call: sw.operation.BoundOperationCall,
    ) -> sw.operation.ResolvedInvocation:
        value = cast(sw.Tensor, call.operands[0])
        plan = OperationPlan(
            "acme.square",
            (OperandPlan(OperandRole.TENSOR, value.dtype(), value.dtype()),),
            Arithmetic.BINARY32,
            None,
            None,
            value.dtype(),
        )
        return sw.operation.ResolvedInvocation(
            call,
            plan,
            (sw.ResultSpec(value.dtype(), value.layout.shape, value.layout),),
            saved_operands=(0,),
        )

    square = sw.define_operation(
        "acme.square",
        schema,
        resolve,
        vjp=lambda context, cotangents: (cotangents[0],),
    )
    capability = OperationCapability.from_plan(
        OperationPlan(
            "acme.square",
            (OperandPlan(OperandRole.TENSOR, sw.DType.Float32, sw.DType.Float32),),
            Arithmetic.BINARY32,
            None,
            None,
            sw.DType.Float32,
        )
    )

    def prepare(invocation: Any, selected: Any) -> PreparedKernel:
        value = cast(sw.Tensor, invocation.operands[0])
        return PreparedKernel(
            selected,
            lambda: result_for(
                invocation,
                [value[index] * value[index] for index in range(value.size())],
            ),
        )

    pattern = sw.KernelPattern(square, (capability,), "square.f32", prepare)
    register_fake(carrier_class, interface)
    sw.register_kernel_pack(
        sw.KernelPack("acme", "1", carrier_class, interface, (pattern,))
    )
    value = make_tensor(carrier_class, [2.0, 3.0])
    result = square(value)
    assert isinstance(result, sw.Tensor)
    assert [result[index] for index in range(2)] == [4.0, 9.0]
    assert result.autograd_ctx is not None


@pytest.mark.parametrize("provider_kind", ["kernel", "composite"])
def test_schema_nondifferentiable_tensors_do_not_admit_autograd(
    provider_kind: str,
) -> None:
    vjp_calls: list[tuple[sw.Tensor | None, ...]] = []

    def vjp(
        context: sw.VJPContext,
        cotangents: tuple[sw.Tensor | None, ...],
    ) -> tuple[sw.Tensor | None, ...]:
        del context
        vjp_calls.append(cotangents)
        return ()

    operation, capability = schema_autograd_operation(
        f"review.nondifferentiable_{provider_kind}",
        (False, False),
        1,
        vjp,
    )
    carrier_class = register_autograd_test_carrier(
        provider_kind,
        f"Nondifferentiable{provider_kind.title()}",
        operation,
        capability,
    )

    source = make_tensor(carrier_class, [1.0, 2.0])
    nonleaf = sw.relu(source)
    rhs = make_tensor(carrier_class, [3.0, 4.0])
    rhs_reference = weakref.ref(rhs)
    dispatched_operation = cast(Any, nonleaf.carrier.dispatch_op(operation.name))
    result = cast(sw.Tensor, dispatched_operation.forward(nonleaf, rhs))

    assert nonleaf.autograd_ctx is not None
    assert result.autograd_ctx is None
    assert dispatched_operation.inputs() == ()
    assert dispatched_operation._resolved_invocation is None
    assert dispatched_operation._saved_operands == ()
    del rhs
    gc.collect()
    assert rhs_reference() is None

    result.backward(make_tensor(carrier_class, [5.0, 6.0]))
    assert vjp_calls == []
    assert source.grad is None

    leaf_result = cast(
        sw.Tensor,
        operation(
            make_tensor(carrier_class, [7.0, 8.0]),
            make_tensor(carrier_class, [9.0, 10.0]),
        ),
    )
    assert leaf_result.autograd_ctx is None


@pytest.mark.parametrize("provider_kind", ["kernel", "composite"])
def test_mixed_schema_admits_only_differentiable_operand_and_clears_state(
    provider_kind: str,
) -> None:
    vjp_calls: list[tuple[sw.Tensor | None, ...]] = []

    def vjp(
        context: sw.VJPContext,
        cotangents: tuple[sw.Tensor | None, ...],
    ) -> tuple[sw.Tensor | None, ...]:
        assert len(context.outputs) == 2
        vjp_calls.append(cotangents)
        return (next(entry for entry in cotangents if entry is not None),)

    operation, capability = schema_autograd_operation(
        f"review.mixed_schema_{provider_kind}",
        (False, True),
        2,
        vjp,
    )
    carrier_class = register_autograd_test_carrier(
        provider_kind,
        f"MixedSchema{provider_kind.title()}",
        operation,
        capability,
    )

    source = make_tensor(carrier_class, [1.0, 2.0])
    lhs = sw.relu(source)
    rhs = make_tensor(carrier_class, [3.0, 4.0])
    outputs = cast(tuple[sw.Tensor, sw.Tensor], operation(lhs, rhs))
    assert outputs[0].autograd_ctx is not None
    assert outputs[1].autograd_ctx is not None
    owner = cast(Any, outputs[1].autograd_ctx)._owner
    assert owner.inputs() == (lhs, rhs)

    gradient = make_tensor(carrier_class, [5.0, 6.0])
    outputs[1].backward(gradient)

    assert vjp_calls == [(None, gradient)]
    assert lhs.grad is None
    assert source.grad is None
    assert rhs.grad is not None
    assert [rhs.grad[index] for index in range(rhs.grad.size())] == [5.0, 6.0]
    assert owner.inputs() == ()
    assert owner._resolved_invocation is None
    assert owner._saved_operands == ()

    disabled_rhs = make_tensor(carrier_class, [11.0, 12.0])
    disabled_reference = weakref.ref(disabled_rhs)
    disabled_operation = cast(Any, lhs.carrier.dispatch_op(operation.name))
    with sw.no_grad():
        disabled = cast(
            tuple[sw.Tensor, sw.Tensor],
            disabled_operation.forward(lhs, disabled_rhs),
        )
    assert all(output.autograd_ctx is None for output in disabled)
    assert disabled_operation.inputs() == ()
    assert disabled_operation._resolved_invocation is None
    assert disabled_operation._saved_operands == ()
    del disabled_rhs
    gc.collect()
    assert disabled_reference() is None


@pytest.mark.parametrize("provider_kind", ["kernel", "composite"])
@pytest.mark.parametrize("source_kind", ["leaf", "nonleaf"])
@pytest.mark.parametrize("result_count", [1, 2])
def test_graph_building_exact_alias_rejects_before_provider_work(
    provider_kind: str,
    source_kind: str,
    result_count: int,
) -> None:
    vjp_calls: list[tuple[sw.Tensor | None, ...]] = []

    def vjp(
        context: sw.VJPContext,
        cotangents: tuple[sw.Tensor | None, ...],
    ) -> tuple[sw.Tensor | None, ...]:
        del context
        vjp_calls.append(cotangents)
        return (next(entry for entry in cotangents if entry is not None),)

    operation, capability = alias_test_operation(
        f"review.alias_reject_{provider_kind}_{source_kind}_{result_count}",
        vjp=vjp,
        result_count=result_count,
        alias_result=result_count - 1,
    )
    counters = {
        "capabilities": 0,
        "result_carriers": 0,
        "prepare": 0,
        "submit": 0,
    }
    carrier_class = register_counted_alias_carrier(
        provider_kind,
        f"AliasReject{provider_kind.title()}{source_kind.title()}{result_count}",
        operation,
        capability,
        counters,
    )
    leaf = make_tensor(carrier_class, [1.0, 2.0])
    value = leaf if source_kind == "leaf" else sw.relu(leaf)
    original_context = value.autograd_ctx
    original_carrier = value.carrier
    original_layout = value.layout
    original_version = value._version_token()
    original_values = tuple(value[index] for index in range(value.size()))
    value_reference = weakref.ref(value)
    callback_baseline = counters.copy()
    dispatched_operation = cast(Any, value.carrier.dispatch_op(operation.name))

    with pytest.raises(ValueError, match="exact-object aliased"):
        dispatched_operation.forward(value)

    assert counters == callback_baseline
    assert vjp_calls == []
    assert dispatched_operation.inputs() == ()
    assert dispatched_operation.input_versions() == ()
    assert value.autograd_ctx is original_context
    assert value.carrier is original_carrier
    assert value.layout is original_layout
    assert value._version_token() == original_version
    assert tuple(value[index] for index in range(value.size())) == original_values
    if source_kind == "nonleaf":
        value.backward(make_tensor(carrier_class, [1.0, 1.0]))
        assert leaf.grad is not None
        assert [leaf.grad[index] for index in range(leaf.grad.size())] == [1.0, 1.0]
    else:
        assert leaf.grad is None
    del value
    if source_kind == "leaf":
        del leaf
    gc.collect()
    assert value_reference() is None


@pytest.mark.parametrize("provider_kind", ["kernel", "composite"])
@pytest.mark.parametrize("execution_mode", ["no_grad", "non_differentiable"])
def test_exact_alias_remains_valid_without_graph_publication(
    provider_kind: str,
    execution_mode: str,
) -> None:
    vjp_calls: list[tuple[sw.Tensor | None, ...]] = []

    def vjp(
        context: sw.VJPContext,
        cotangents: tuple[sw.Tensor | None, ...],
    ) -> tuple[sw.Tensor | None, ...]:
        del context
        vjp_calls.append(cotangents)
        return (cotangents[0],)

    vjp_policy: object = (
        sw.NON_DIFFERENTIABLE if execution_mode == "non_differentiable" else vjp
    )
    operation, capability = alias_test_operation(
        f"review.alias_permitted_{provider_kind}_{execution_mode}",
        vjp=vjp_policy,
    )
    counters = {
        "capabilities": 0,
        "result_carriers": 0,
        "prepare": 0,
        "submit": 0,
    }
    carrier_class = register_counted_alias_carrier(
        provider_kind,
        f"AliasPermitted{provider_kind.title()}{execution_mode.title()}",
        operation,
        capability,
        counters,
    )
    leaf = make_tensor(carrier_class, [3.0, 4.0])
    value = sw.relu(leaf)
    original_context = value.autograd_ctx
    callback_baseline = counters.copy()

    if execution_mode == "no_grad":
        with sw.no_grad():
            result = cast(sw.Tensor, operation(value))
    else:
        result = cast(sw.Tensor, operation(value))

    assert result is value
    assert value.autograd_ctx is original_context
    assert original_context is not None
    assert vjp_calls == []
    assert counters["capabilities"] == callback_baseline["capabilities"]
    assert counters["prepare"] == callback_baseline["prepare"] + 1
    assert counters["submit"] == callback_baseline["submit"] + 1
    assert counters["result_carriers"] == callback_baseline["result_carriers"] + (
        1 if provider_kind == "composite" else 0
    )
    result.backward(make_tensor(carrier_class, [1.0, 1.0]))
    assert leaf.grad is not None
    assert [leaf.grad[index] for index in range(leaf.grad.size())] == [1.0, 1.0]
    assert vjp_calls == []


@pytest.mark.parametrize("provider_kind", ["kernel", "composite"])
def test_distinct_storage_sharing_view_keeps_definition_autograd(
    provider_kind: str,
) -> None:
    vjp_calls: list[tuple[sw.Tensor | None, ...]] = []

    def vjp(
        context: sw.VJPContext,
        cotangents: tuple[sw.Tensor | None, ...],
    ) -> tuple[sw.Tensor | None, ...]:
        del context
        vjp_calls.append(cotangents)
        return (cotangents[0],)

    operation, capability = alias_test_operation(
        f"review.storage_sharing_view_{provider_kind}",
        vjp=vjp,
        alias_result=None,
    )
    counters = {
        "capabilities": 0,
        "result_carriers": 0,
        "prepare": 0,
        "submit": 0,
    }
    carrier_class = register_counted_alias_carrier(
        provider_kind,
        f"StorageSharingView{provider_kind.title()}",
        operation,
        capability,
        counters,
        storage_sharing_view=True,
    )
    value = make_tensor(carrier_class, [5.0, 6.0])

    result = cast(sw.Tensor, operation(value))

    assert result is not value
    assert result.carrier is value.carrier
    assert result.autograd_ctx is not None
    gradient = make_tensor(carrier_class, [7.0, 8.0])
    result.backward(gradient)
    assert vjp_calls == [(gradient,)]
    assert value.grad is not None
    assert [value.grad[index] for index in range(value.grad.size())] == [7.0, 8.0]


@pytest.mark.parametrize("provider_kind", ["kernel", "composite"])
def test_released_state_precedes_exact_alias_rejection(provider_kind: str) -> None:
    operation, capability = alias_test_operation(
        f"review.alias_released_{provider_kind}",
        vjp=lambda context, cotangents: (cotangents[0],),
    )
    counters = {
        "capabilities": 0,
        "result_carriers": 0,
        "prepare": 0,
        "submit": 0,
    }
    carrier_class = register_counted_alias_carrier(
        provider_kind,
        f"AliasReleased{provider_kind.title()}",
        operation,
        capability,
        counters,
    )
    value = make_tensor(carrier_class, [9.0, 10.0])
    callback_baseline = counters.copy()
    value.carrier.release()

    with pytest.raises(RuntimeError, match="released"):
        operation(value)

    assert counters == callback_baseline


def test_definition_backed_kernel_publishes_ordered_multi_results_and_vjps() -> None:
    carrier_class = fresh_carrier("MultiResultKernelCarrier")
    interface = FakeInterface("v1")
    seen_cotangents: list[tuple[sw.Tensor | None, ...]] = []

    def vjp(
        context: sw.VJPContext,
        cotangents: tuple[sw.Tensor | None, ...],
    ) -> tuple[sw.Tensor | None, ...]:
        assert len(context.outputs) == 2
        seen_cotangents.append(cotangents)
        return (next(entry for entry in cotangents if entry is not None),)

    operation, capability = multi_result_operation("review.multi_kernel", vjp)

    def prepare(
        invocation: sw.operation.ResolvedInvocation,
        selected: sw.KernelExecutionInterface,
    ) -> PreparedKernel:
        value = cast(sw.Tensor, invocation.operands[0])
        return PreparedKernel(
            selected,
            lambda: multi_provider_result(
                invocation,
                (carrier_class, carrier_class),
                (
                    tuple(float(value[index]) + 1.0 for index in range(value.size())),
                    tuple(float(value[index]) * 10.0 for index in range(value.size())),
                ),
            ),
        )

    pattern = sw.KernelPattern(operation, (capability,), "review.multi", prepare)
    register_fake(carrier_class, interface, (pattern,))

    first_input = make_tensor(carrier_class, [1.0, 2.0])
    with sw.profile(record_shapes=True) as profiler:
        first = operation(first_input)
    assert isinstance(first, tuple)
    assert len(first) == 2
    assert [first[0][index] for index in range(2)] == [2.0, 3.0]
    assert [first[1][index] for index in range(2)] == [10.0, 20.0]
    assert first[0].autograd_ctx is not None
    assert first[1].autograd_ctx is not None
    assert first[0].autograd_ctx is not first[1].autograd_ctx
    assert first[0].autograd_ctx._dispatch_carrier_class is carrier_class
    (event,) = profiler.events()
    assert event.name == "review.multi_kernel"
    assert event.carrier_type is carrier_class
    assert event.input_shapes == ((2,),)

    first_gradient = make_tensor(carrier_class, [3.0, 4.0])
    first[0].backward(first_gradient)
    assert seen_cotangents[-1] == (first_gradient, None)
    assert first_input.grad is not None
    assert [first_input.grad[index] for index in range(2)] == [3.0, 4.0]

    second_input = make_tensor(carrier_class, [5.0, 6.0])
    second = cast(tuple[sw.Tensor, sw.Tensor], operation(second_input))
    second_gradient = make_tensor(carrier_class, [7.0, 8.0])
    second[1].backward(second_gradient)
    assert seen_cotangents[-1] == (None, second_gradient)
    assert second_input.grad is not None
    assert [second_input.grad[index] for index in range(2)] == [7.0, 8.0]


def test_definition_backed_composite_publishes_ordered_multi_results() -> None:
    result_class = fresh_carrier("MultiResultCompositeOutput")
    register_fake(result_class, FakeInterface("result"))
    operation, capability = multi_result_operation(
        "review.multi_composite", sw.NON_DIFFERENTIABLE
    )

    class CompositeCarrier(sw.DependentCarrier):
        def __init__(
            self,
            slots: int,
            *,
            dtype: sw.SimpleDType,
            mutable: bool = True,
        ) -> None:
            super().__init__(slots, dtype=dtype, mutable=mutable)
            self._finalize_dependent_capabilities()

    def prepare(carrier: object, invocation: object) -> PreparedComposite:
        del carrier
        resolved = cast(sw.operation.ResolvedInvocation, invocation)
        value = cast(sw.Tensor, resolved.operands[0])
        return PreparedComposite(
            lambda: multi_provider_result(
                resolved,
                (result_class, result_class),
                (
                    tuple(float(value[index]) - 1.0 for index in range(value.size())),
                    tuple(float(value[index]) + 2.0 for index in range(value.size())),
                ),
            )
        )

    sw.register_carrier_definition(
        CompositeCarrier,
        sw.CarrierDefinition(
            storage_provider(),
            composite=sw.CompositeProvider(
                lambda carrier: (capability,),
                lambda carrier, invocation: (result_class, result_class),
                prepare,
            ),
        ),
    )
    value = make_tensor(CompositeCarrier, [4.0, 8.0])
    outputs = operation(value)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    assert tuple(type(output.carrier) for output in outputs) == (
        result_class,
        result_class,
    )
    assert [outputs[0][index] for index in range(2)] == [3.0, 7.0]
    assert [outputs[1][index] for index in range(2)] == [6.0, 10.0]
    assert all(output.autograd_ctx is None for output in outputs)


@pytest.mark.parametrize(
    ("conflict", "error", "message"),
    [
        ("count", sw.ProviderContractError, "count"),
        ("type", TypeError, "completion"),
        ("layout", sw.ProviderContractError, "layout"),
        ("dtype", sw.ProviderContractError, "dtype"),
        ("carrier", sw.ProviderContractError, "carrier role"),
        ("alias", sw.ProviderContractError, "aliases"),
    ],
)
def test_multi_result_provider_conflicts_fail_before_publication(
    conflict: str, error: type[Exception], message: str
) -> None:
    carrier_class = fresh_carrier(f"MultiContract{conflict.title()}Carrier")
    other_class = fresh_carrier(f"MultiContract{conflict.title()}Other")
    interface = FakeInterface("v1")
    register_fake(other_class, FakeInterface("other"))
    operation, capability = multi_result_operation(
        f"review.multi_contract_{conflict}",
        lambda context, cotangents: (cotangents[0],),
    )
    produced: list[sw.Tensor] = []

    def prepare(
        invocation: sw.operation.ResolvedInvocation,
        selected: sw.KernelExecutionInterface,
    ) -> PreparedKernel:
        source = cast(sw.Tensor, invocation.operands[0])

        def submit() -> object:
            if conflict == "type":
                return object()
            first = multi_provider_result(
                invocation,
                (carrier_class, carrier_class),
                ((1.0, 2.0), (3.0, 4.0)),
            ).outputs
            outputs = [cast(sw.Tensor, first[0]), cast(sw.Tensor, first[1])]
            if conflict == "count":
                outputs.pop()
            elif conflict == "layout":
                outputs[1] = sw.Tensor(
                    carrier_class(1, dtype=sw.DType.Float32),
                    0,
                    sw.Layout(invocation.results[1].shape, sw.Stride(0)),
                )
            elif conflict == "dtype":
                outputs[1] = sw.Tensor(
                    carrier_class(2, dtype=sw.DType.Int32),
                    0,
                    invocation.results[1].layout,
                )
            elif conflict == "carrier":
                outputs[1] = sw.Tensor(
                    other_class(2, dtype=sw.DType.Float32),
                    0,
                    invocation.results[1].layout,
                )
            elif conflict == "alias":
                outputs[1] = source
            produced.extend(outputs)
            return ProviderResult(outputs)

        return PreparedKernel(selected, cast(Any, submit))

    pattern = sw.KernelPattern(
        operation, (capability,), f"review.invalid.{conflict}", prepare
    )
    register_fake(carrier_class, interface, (pattern,))
    source = make_tensor(carrier_class, [5.0, 6.0])

    with pytest.raises(error, match=message):
        operation(source)
    assert source.autograd_ctx is None
    assert all(output.autograd_ctx is None for output in produced)


@pytest.mark.parametrize("provider_kind", ["kernel", "composite"])
@pytest.mark.parametrize(
    ("conflict", "expected_error", "message"),
    [
        ("count", sw.ProviderContractError, "count"),
        ("dtype", sw.ProviderContractError, "dtype"),
        ("layout", sw.ProviderContractError, "layout"),
        ("carrier", sw.ProviderContractError, "carrier role"),
        ("alias", sw.ProviderContractError, "aliases"),
        ("completion", RuntimeError, "terminal provider failure"),
        ("unauthorized", sw.ProviderContractError, "unauthorized operand"),
    ],
)
def test_failed_provider_work_restores_all_operand_state(
    provider_kind: str,
    conflict: str,
    expected_error: type[Exception],
    message: str,
) -> None:
    if provider_kind == "kernel":
        carrier_class = fresh_carrier(f"AtomicKernel{conflict.title()}")
    else:

        class AtomicComposite(sw.DependentCarrier):
            def __init__(
                self,
                slots: int,
                *,
                dtype: sw.SimpleDType,
                mutable: bool = True,
            ) -> None:
                super().__init__(slots, dtype=dtype, mutable=mutable)
                self._finalize_dependent_capabilities()

        carrier_class = AtomicComposite

    other_class = fresh_carrier(f"Atomic{provider_kind.title()}{conflict.title()}Other")
    register_fake(other_class, FakeInterface("other"))
    mutated_operands = () if conflict == "unauthorized" else (0, 1)
    operation, capability = mutating_operation(
        f"review.atomic_{provider_kind}_{conflict}",
        mutated_operands=mutated_operands,
    )

    def submit(
        invocation: sw.operation.ResolvedInvocation,
    ) -> ProviderResult | Completion:
        lhs = cast(sw.Tensor, invocation.operands[0])
        rhs = cast(sw.Tensor, invocation.operands[1])
        lhs.carrier.set_value(0, 91.0)
        lhs.carrier.set_value(1, 92.0)
        rhs.carrier.set_value(0, 81.0)
        if conflict == "completion":
            return Completion(RuntimeError("terminal provider failure"))

        outputs = list(
            multi_provider_result(
                invocation,
                (carrier_class, carrier_class),
                ((10.0, 20.0), (30.0, 40.0)),
            ).outputs
        )
        if conflict == "count":
            outputs.pop()
        elif conflict == "dtype":
            outputs[1] = make_tensor(carrier_class, [30, 40], dtype=sw.DType.Int32)
        elif conflict == "layout":
            outputs[1] = sw.Tensor(
                carrier_class(1, dtype=sw.DType.Float32),
                0,
                sw.Layout(invocation.results[1].shape, sw.Stride(0)),
            )
        elif conflict == "carrier":
            outputs[1] = make_tensor(other_class, [30.0, 40.0])
        elif conflict == "alias":
            outputs[1] = lhs
        return ProviderResult(outputs)

    if provider_kind == "kernel":
        interface = FakeInterface("atomic")
        pattern = sw.KernelPattern(
            operation,
            (capability,),
            f"atomic.{conflict}",
            lambda invocation, selected: PreparedKernel(
                selected, lambda: submit(invocation)
            ),
        )
        register_fake(carrier_class, interface, (pattern,))
    else:
        sw.register_carrier_definition(
            carrier_class,
            sw.CarrierDefinition(
                storage_provider(),
                composite=sw.CompositeProvider(
                    lambda carrier: (capability,),
                    lambda carrier, invocation: (carrier_class, carrier_class),
                    lambda carrier, invocation: PreparedComposite(
                        lambda: submit(
                            cast(sw.operation.ResolvedInvocation, invocation)
                        )
                    ),
                ),
            ),
        )

    lhs = make_tensor(carrier_class, [1.0, 2.0])
    rhs = make_tensor(carrier_class, [3.0, 4.0])
    lhs_carrier = lhs.carrier
    rhs_carrier = rhs.carrier
    lhs_layout = lhs.layout
    rhs_layout = rhs.layout
    versions = (lhs.version, rhs.version)
    ownership = (lhs.carrier.is_owned(), rhs.carrier.is_owned())

    with pytest.raises(expected_error, match=message):
        operation(lhs, rhs)

    assert [lhs[index] for index in range(lhs.size())] == [1.0, 2.0]
    assert [rhs[index] for index in range(rhs.size())] == [3.0, 4.0]
    assert (lhs.version, rhs.version) == versions
    assert lhs.carrier is lhs_carrier
    assert rhs.carrier is rhs_carrier
    assert lhs.layout is lhs_layout
    assert rhs.layout is rhs_layout
    assert (lhs.carrier.is_owned(), rhs.carrier.is_owned()) == ownership
    assert lhs.autograd_ctx is None
    assert rhs.autograd_ctx is None


@pytest.mark.parametrize("provider_kind", ["kernel", "composite"])
def test_successful_declared_provider_mutation_commits_one_version(
    provider_kind: str,
) -> None:
    if provider_kind == "kernel":
        carrier_class = fresh_carrier("AtomicKernelSuccess")
    else:

        class AtomicCompositeSuccess(sw.DependentCarrier):
            def __init__(
                self,
                slots: int,
                *,
                dtype: sw.SimpleDType,
                mutable: bool = True,
            ) -> None:
                super().__init__(slots, dtype=dtype, mutable=mutable)
                self._finalize_dependent_capabilities()

        carrier_class = AtomicCompositeSuccess

    operation, capability = mutating_operation(
        f"review.atomic_{provider_kind}_success",
        mutated_operands=(0,),
        result_count=1,
        differentiable=True,
    )

    def submit(invocation: sw.operation.ResolvedInvocation) -> ProviderResult:
        lhs = cast(sw.Tensor, invocation.operands[0])
        lhs.carrier.set_value(0, 11.0)
        lhs.carrier.set_value(1, 12.0)
        return result_for(invocation, [21.0, 22.0])

    if provider_kind == "kernel":
        interface = FakeInterface("atomic-success")
        register_fake(
            carrier_class,
            interface,
            (
                sw.KernelPattern(
                    operation,
                    (capability,),
                    "atomic.success",
                    lambda invocation, selected: PreparedKernel(
                        selected, lambda: submit(invocation)
                    ),
                ),
            ),
        )
    else:
        sw.register_carrier_definition(
            carrier_class,
            sw.CarrierDefinition(
                storage_provider(),
                composite=sw.CompositeProvider(
                    lambda carrier: (capability,),
                    lambda carrier, invocation: (carrier_class,),
                    lambda carrier, invocation: PreparedComposite(
                        lambda: submit(
                            cast(sw.operation.ResolvedInvocation, invocation)
                        )
                    ),
                ),
            ),
        )

    lhs = make_tensor(carrier_class, [1.0, 2.0])
    rhs = make_tensor(carrier_class, [3.0, 4.0])
    version = lhs.version
    result = cast(sw.Tensor, operation(lhs, rhs))

    assert [lhs[index] for index in range(lhs.size())] == [11.0, 12.0]
    assert lhs.version == version + 1
    assert [rhs[index] for index in range(rhs.size())] == [3.0, 4.0]
    assert result.autograd_ctx is not None
    result.backward(make_tensor(carrier_class, [5.0, 6.0]))
    assert lhs.grad is not None
    assert [lhs.grad[index] for index in range(lhs.grad.size())] == [5.0, 6.0]


def test_late_duplicate_and_interface_mismatched_packs_are_atomic() -> None:
    carrier_class = fresh_carrier("PackRegistrationCarrier")
    interface = FakeInterface("v1")
    first = relu_pattern(
        "first", lambda invocation, selected: relu_prepared(invocation, selected)
    )
    second = relu_pattern(
        "second", lambda invocation, selected: relu_prepared(invocation, selected)
    )
    register_fake(carrier_class, interface)
    sw.register_kernel_pack(
        sw.KernelPack("acme", "1", carrier_class, interface, (first,))
    )
    with pytest.raises(ValueError, match="namespace and version"):
        sw.register_kernel_pack(
            sw.KernelPack("acme", "1", carrier_class, interface, (second,))
        )
    with pytest.raises(ValueError, match="kernel_id"):
        sw.register_kernel_pack(
            sw.KernelPack("other", "1", carrier_class, interface, (first,))
        )
    with pytest.raises(TypeError, match="does not match"):
        sw.register_kernel_pack(
            sw.KernelPack(
                "other",
                "2",
                carrier_class,
                FakeInterface("v2"),
                (second,),
            )
        )
    assert set(capabilities_for_carrier_class(carrier_class)) == {relu_capability()}
    late = relu_pattern(
        "late", lambda invocation, selected: relu_prepared(invocation, selected)
    )
    with pytest.raises(TypeError, match="already observed"):
        sw.register_kernel_pack(
            sw.KernelPack("late", "1", carrier_class, interface, (late,))
        )


def test_pack_attachment_is_exact_and_rejects_invalid_targets() -> None:
    interface = FakeInterface("v1")
    target = fresh_carrier("ExactPackTarget")
    register_fake(target, interface)
    pattern = relu_pattern(
        "exact", lambda invocation, selected: relu_prepared(invocation, selected)
    )
    subclass = type("ExactPackSubclass", (target,), {})
    with pytest.raises(TypeError, match="exact CarrierDefinition"):
        sw.register_kernel_pack(
            sw.KernelPack("subclass", "1", subclass, interface, (pattern,))
        )
    with pytest.raises(TypeError, match="exact CarrierDefinition"):
        sw.register_kernel_pack(
            sw.KernelPack("metal", "1", sw.Metal, interface, (pattern,))
        )

    dependent = type("PackDependent", (sw.DependentCarrier,), {})
    composite = sw.CompositeProvider(
        lambda carrier: (),
        lambda carrier, invocation: (),
        lambda carrier, invocation: Unsupported("none"),
    )
    sw.register_carrier_definition(
        dependent, sw.CarrierDefinition(storage_provider(), composite=composite)
    )
    with pytest.raises(TypeError, match="DependentCarrier"):
        sw.register_kernel_pack(
            sw.KernelPack("dependent", "1", dependent, interface, (pattern,))
        )


@pytest.mark.parametrize("conflict", ["dtype", "layout", "carrier", "size"])
def test_provider_result_conflicts_fail_before_publication(conflict: str) -> None:
    carrier_class = fresh_carrier(f"InvalidResult{conflict.title()}")
    other_class = fresh_carrier(f"OtherResult{conflict.title()}")
    interface = FakeInterface("v1")

    def prepare(invocation: Any, selected: Any) -> PreparedKernel:
        spec = invocation.results[0]
        if conflict == "dtype":
            carrier = carrier_class(spec.layout.cosize, dtype=sw.DType.Int32)
            output = sw.Tensor(carrier, 0, spec.layout)
        elif conflict == "layout":
            carrier = carrier_class(1, dtype=spec.dtype)
            output = sw.Tensor(carrier, 0, sw.Layout(spec.shape, sw.Stride(0)))
        elif conflict == "carrier":
            carrier = other_class(spec.layout.cosize, dtype=spec.dtype)
            output = sw.Tensor(carrier, 0, spec.layout)
        else:
            carrier = carrier_class(spec.layout.cosize + 1, dtype=spec.dtype)
            output = sw.Tensor(carrier, 0, spec.layout)
        return PreparedKernel(selected, lambda: ProviderResult((output,)))

    register_fake(carrier_class, interface, (relu_pattern("invalid", prepare),))
    register_fake(other_class, FakeInterface("other"))
    source = make_tensor(carrier_class, [-1.0, 2.0])
    with pytest.raises(sw.ProviderContractError, match=conflict):
        sw.relu(source)
    assert [source[index] for index in range(2)] == [-1.0, 2.0]


@pytest.mark.parametrize("wrong_carrier", [False, True])
def test_composite_provider_validates_declared_result_carrier(
    wrong_carrier: bool,
) -> None:
    result_class = fresh_carrier(f"CompositeResult{wrong_carrier}")
    other_class = fresh_carrier(f"CompositeOther{wrong_carrier}")
    register_fake(result_class, FakeInterface("result"))
    register_fake(other_class, FakeInterface("other"))

    class CompositeCarrier(sw.DependentCarrier):
        def __init__(
            self,
            slots: int,
            *,
            dtype: sw.SimpleDType,
            mutable: bool = True,
        ) -> None:
            super().__init__(slots, dtype=dtype, mutable=mutable)
            self._finalize_dependent_capabilities()

    def prepare(invocation: Any) -> PreparedComposite:
        source = cast(sw.Tensor, invocation.operands[0])
        output_class = other_class if wrong_carrier else result_class
        return PreparedComposite(
            lambda: result_for(
                invocation,
                [max(0, source[index]) for index in range(source.size())],
                carrier_class=output_class,
            )
        )

    composite = sw.CompositeProvider(
        lambda carrier: (relu_capability(),),
        lambda carrier, invocation: (result_class,),
        lambda carrier, invocation: prepare(invocation),
    )
    sw.register_carrier_definition(
        CompositeCarrier,
        sw.CarrierDefinition(storage_provider(), composite=composite),
    )
    source = make_tensor(CompositeCarrier, [-1.0, 2.0])

    if wrong_carrier:
        with pytest.raises(sw.ProviderContractError, match="carrier role"):
            sw.relu(source)
    else:
        result = sw.relu(source)
        assert type(result.carrier) is result_class
        assert [result[index] for index in range(result.size())] == [0.0, 2.0]


def test_wrong_prepared_or_submission_types_and_interfaces_fail_closed() -> None:
    cases: tuple[tuple[str, Callable[[Any, Any], object], str], ...] = (
        ("prepare", lambda invocation, selected: object(), "prepare"),
        (
            "interface",
            lambda invocation, selected: PreparedKernel(
                FakeInterface("wrong"), lambda: result_for(invocation, [0.0])
            ),
            "interface",
        ),
        (
            "submit",
            lambda invocation, selected: PreparedKernel(
                selected, lambda: cast(Any, invocation.operands[0])
            ),
            "completion",
        ),
    )
    for name, prepare, message in cases:
        carrier_class = fresh_carrier(f"Wrong{name.title()}Carrier")
        interface = FakeInterface("v1")
        register_fake(
            carrier_class,
            interface,
            (
                relu_pattern(
                    f"wrong.{name}",
                    cast(
                        Callable[[Any, Any], PreparedKernel | Unsupported],
                        prepare,
                    ),
                ),
            ),
        )
        with pytest.raises(TypeError, match=message):
            sw.relu(make_tensor(carrier_class, [1.0]))


def test_definition_backed_dispatch_is_fresh_and_central_vjp_owns_autograd() -> None:
    carrier_class = fresh_carrier("AutogradKernelCarrier")
    interface = FakeInterface("v1")
    register_fake(
        carrier_class,
        interface,
        (
            relu_pattern(
                "relu.autograd",
                lambda invocation, selected: relu_prepared(invocation, selected),
            ),
        ),
    )
    value = make_tensor(carrier_class, [-1.0, 2.0])
    first = value.carrier.dispatch_op("relu")
    second = value.carrier.dispatch_op("relu")
    assert first is not second
    assert first._operation_name == second._operation_name == "relu"
    assert first._dispatch_carrier_class is carrier_class

    result = sw.relu(value)
    gradient = make_tensor(carrier_class, [3.0, 4.0])
    result.backward(gradient)
    assert value.grad is not None
    assert [value.grad[index] for index in range(2)] == [0.0, 4.0]


def test_registration_and_first_observation_are_atomic() -> None:
    for attempt in range(10):
        carrier_class = fresh_carrier(f"PackObservationRace{attempt}")
        interface = FakeInterface("v1")
        register_fake(carrier_class, interface)
        pattern = relu_pattern(
            f"race.{attempt}",
            lambda invocation, selected: relu_prepared(invocation, selected),
        )
        pack = sw.KernelPack("race", str(attempt), carrier_class, interface, (pattern,))
        barrier = Barrier(2)
        outcomes: list[object] = []

        def register() -> None:
            barrier.wait()
            try:
                sw.register_kernel_pack(pack)
                outcomes.append("registered")
            except Exception as error:
                outcomes.append(error)

        def observe() -> None:
            barrier.wait()
            outcomes.append(capabilities_for_carrier_class(carrier_class))

        threads = (Thread(target=register), Thread(target=observe))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        capabilities = capabilities_for_carrier_class(carrier_class)
        if capabilities:
            assert "registered" in outcomes
            assert capabilities == (relu_capability(),)
        else:
            assert any(isinstance(outcome, TypeError) for outcome in outcomes)
