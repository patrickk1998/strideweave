from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

import strideweave as sw
import strideweave.operation as operation
from strideweave.carriers.operation_policy import (
    Arithmetic,
    OperandPlan,
    OperandRole,
    OperationPlan,
)
from strideweave.operation_definition import (
    _call_vjp,
    _execute_reference,
    _operation_definition,
)


class OneShot:
    def __init__(self, values: tuple[object, ...]) -> None:
        self.values = values
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("iterable was consumed more than once")
        yield from self.values


def tensor(
    values: list[object],
    *,
    dtype: sw.SimpleDType = sw.DType.Float32,
    layout: sw.Layout | None = None,
) -> sw.Tensor:
    if layout is None:
        layout = sw.Layout(sw.Shape(len(values)), sw.Stride(1))
    return sw.Tensor(sw.Generic(values, dtype=dtype), 0, layout)


def unary_plan(name: str, dtype: sw.SimpleDType = sw.DType.Float32) -> OperationPlan:
    return OperationPlan(
        name,
        (OperandPlan(OperandRole.TENSOR, dtype, dtype),),
        Arithmetic.BINARY32,
        None,
        None,
        dtype,
    )


def unary_invocation(
    call: operation.BoundOperationCall,
    *,
    saved: tuple[int, ...] = (),
) -> operation.ResolvedInvocation:
    value = call.operands[0]
    assert isinstance(value, sw.Tensor)
    return operation.ResolvedInvocation(
        call,
        unary_plan(call.definition.name, value.dtype()),
        (operation.ResultSpec(value.dtype(), value.layout.shape, value.layout),),
        saved_operands=saved,
    )


def test_operation_extension_exports_are_identity_preserving() -> None:
    top_level = (
        "define_operation",
        "OperationDefinition",
        "OperationSchema",
        "OperandSpec",
        "OptionSpec",
        "OperandKind",
        "REQUIRED",
        "NON_DIFFERENTIABLE",
        "ResultSpec",
        "ProviderContractError",
    )
    module_only = (
        "OperationCall",
        "BoundOperationCall",
        "OperationPlan",
        "ResolvedInvocation",
        "VJPContext",
    )
    for name in top_level:
        assert getattr(sw, name) is getattr(operation, name)
        assert name in sw.__all__
    for name in module_only:
        assert name in operation.__all__
        assert name not in sw.__all__


def test_operand_kinds_have_explicit_string_values() -> None:
    assert tuple(kind.value for kind in sw.OperandKind) == (
        "tensor",
        "weak_scalar",
        "structural",
    )


@pytest.mark.parametrize("name", ["", "not valid", "1name"])
def test_operand_and_option_names_must_be_identifiers(name: str) -> None:
    with pytest.raises(ValueError, match="identifier"):
        sw.OperandSpec(name, sw.OperandKind.TENSOR)
    with pytest.raises(ValueError, match="identifier"):
        sw.OptionSpec(name)


def test_operand_spec_validates_kind_and_differentiability() -> None:
    with pytest.raises(TypeError, match="OperandKind"):
        sw.OperandSpec("value", "tensor")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        sw.OperandSpec("value", sw.OperandKind.TENSOR, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only a Tensor"):
        sw.OperandSpec("value", sw.OperandKind.WEAK_SCALAR, True)


def test_schema_materializes_each_iterable_once_and_is_immutable() -> None:
    operands = OneShot((sw.OperandSpec("value", sw.OperandKind.TENSOR),))
    options = OneShot((sw.OptionSpec("axis", 0),))
    schema = sw.OperationSchema(operands, options)  # type: ignore[arg-type]
    assert operands.iterations == options.iterations == 1
    assert schema.operands[0].name == "value"
    with pytest.raises(FrozenInstanceError):
        schema.operands = ()  # type: ignore[misc]


def test_schema_rejects_empty_wrong_and_duplicate_entries() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        sw.OperationSchema(())
    with pytest.raises(TypeError, match="OperandSpec"):
        sw.OperationSchema((object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="OptionSpec"):
        sw.OperationSchema(
            (sw.OperandSpec("value", sw.OperandKind.TENSOR),),
            (object(),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unique"):
        sw.OperationSchema(
            (sw.OperandSpec("value", sw.OperandKind.TENSOR),),
            (sw.OptionSpec("value"),),
        )


def test_binding_applies_defaults_before_resolve_and_keeps_options_read_only() -> None:
    seen: list[operation.BoundOperationCall] = []

    def resolve(call: operation.BoundOperationCall) -> operation.ResolvedInvocation:
        seen.append(call)
        return unary_invocation(call)

    definition = sw.define_operation(
        "binding.defaults",
        sw.OperationSchema(
            (sw.OperandSpec("value", sw.OperandKind.TENSOR),),
            (sw.OptionSpec("axis", 3),),
        ),
        resolve,
        vjp=sw.NON_DIFFERENTIABLE,
    )
    value = tensor([1.0])
    invocation = definition.resolve_call(value)
    assert invocation.call is seen[0]
    assert invocation.operands == (value,)
    assert invocation.options == {"axis": 3}
    with pytest.raises(TypeError):
        invocation.options["axis"] = 4  # type: ignore[index]


def test_binding_rejects_count_options_and_operand_kind_before_resolve() -> None:
    calls = 0

    def resolve(call: operation.BoundOperationCall) -> operation.ResolvedInvocation:
        nonlocal calls
        calls += 1
        return unary_invocation(call)

    definition = sw.define_operation(
        "binding.errors",
        sw.OperationSchema(
            (sw.OperandSpec("value", sw.OperandKind.TENSOR),),
            (sw.OptionSpec("required"),),
        ),
        resolve,
        vjp=sw.NON_DIFFERENTIABLE,
    )
    with pytest.raises(TypeError, match="expects 1 operands"):
        definition.resolve_call()
    with pytest.raises(TypeError, match="missing required"):
        definition.resolve_call(tensor([1.0]))
    with pytest.raises(TypeError, match="unknown option"):
        definition.resolve_call(tensor([1.0]), required=1, extra=2)
    with pytest.raises(TypeError, match="must be a Tensor"):
        definition.resolve_call(1, required=1)
    assert calls == 0


def test_weak_scalar_binding_rejects_tensors_and_non_real_values() -> None:
    definition = sw.define_operation(
        "binding.weak",
        sw.OperationSchema((sw.OperandSpec("scale", sw.OperandKind.WEAK_SCALAR),)),
        lambda call: (_ for _ in ()).throw(AssertionError(call)),
        vjp=sw.NON_DIFFERENTIABLE,
    )
    with pytest.raises(TypeError, match="weak scalar"):
        definition.bind(tensor([1.0]))
    with pytest.raises(TypeError, match="weak scalar"):
        definition.bind("one")
    assert definition.bind(True).operand_kinds == (sw.OperandKind.WEAK_SCALAR,)


@pytest.mark.parametrize(
    "name",
    [
        "plain",
        ".missing",
        "acme.",
        "acme.not-valid",
        "strideweave.custom",
        "add.custom",
    ],
)
def test_custom_names_must_be_valid_unreserved_namespaces(name: str) -> None:
    schema = sw.OperationSchema((sw.OperandSpec("value", sw.OperandKind.TENSOR),))
    with pytest.raises(ValueError, match=r"namespaced|reserved"):
        sw.define_operation(
            name,
            schema,
            lambda call: unary_invocation(call),
            vjp=sw.NON_DIFFERENTIABLE,
        )


def test_definition_requires_explicit_vjp_and_registers_once() -> None:
    schema = sw.OperationSchema((sw.OperandSpec("value", sw.OperandKind.TENSOR),))
    with pytest.raises(ValueError, match="supplied explicitly"):
        sw.define_operation("registration.missing_vjp", schema, unary_invocation)
    with pytest.raises(LookupError):
        _operation_definition("registration.missing_vjp")

    first = sw.define_operation(
        "registration.once",
        schema,
        unary_invocation,
        vjp=sw.NON_DIFFERENTIABLE,
    )
    assert _operation_definition("registration.once") is first
    with pytest.raises(ValueError, match="already defined"):
        sw.define_operation(
            "registration.once",
            schema,
            unary_invocation,
            vjp=sw.NON_DIFFERENTIABLE,
        )


def test_resolve_must_return_the_same_bound_call() -> None:
    schema = sw.OperationSchema((sw.OperandSpec("value", sw.OperandKind.TENSOR),))
    wrong = sw.define_operation(
        "resolution.wrong_call_source",
        schema,
        unary_invocation,
        vjp=sw.NON_DIFFERENTIABLE,
    )
    wrong_call = wrong.bind(tensor([2.0]))
    definition = sw.define_operation(
        "resolution.wrong_call",
        schema,
        lambda _call: operation.ResolvedInvocation(
            wrong_call,
            unary_plan("resolution.wrong_call_source"),
            (
                operation.ResultSpec(
                    sw.DType.Float32,
                    wrong_call.operands[0].layout.shape,  # type: ignore[union-attr]
                    wrong_call.operands[0].layout,  # type: ignore[union-attr]
                ),
            ),
        ),
        vjp=sw.NON_DIFFERENTIABLE,
    )
    with pytest.raises(ValueError, match="bound-call identity"):
        definition.resolve_call(tensor([1.0]))

    non_invocation = sw.define_operation(
        "resolution.wrong_type",
        schema,
        lambda _call: None,  # type: ignore[arg-type,return-value]
        vjp=sw.NON_DIFFERENTIABLE,
    )
    with pytest.raises(TypeError, match="ResolvedInvocation"):
        non_invocation.resolve_call(tensor([1.0]))


def test_result_spec_validates_dtype_shape_layout_alias_and_tolerances() -> None:
    shape = sw.Shape(2)
    layout = sw.Layout(shape, sw.Stride(1))
    with pytest.raises(TypeError, match="DType"):
        sw.ResultSpec("Float32", shape, layout)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Shape"):
        sw.ResultSpec(
            sw.DType.Float32,
            shape,
            sw.Layout(sw.Shape(3), sw.Stride(1)),
        )
    with pytest.raises(ValueError, match="non-negative"):
        sw.ResultSpec(sw.DType.Float32, shape, layout, -1)
    with pytest.raises(TypeError, match="finite"):
        sw.ResultSpec(sw.DType.Float32, shape, layout, atol=True)
    with pytest.raises(ValueError, match="finite"):
        sw.ResultSpec(sw.DType.Float32, shape, layout, rtol=float("inf"))


def test_resolved_invocation_materializes_effects_and_checks_roles() -> None:
    definition = sw.define_operation(
        "resolution.effects",
        sw.OperationSchema((sw.OperandSpec("value", sw.OperandKind.TENSOR),)),
        unary_invocation,
        vjp=sw.NON_DIFFERENTIABLE,
    )
    value = tensor([1.0])
    call = definition.bind(value)
    results = OneShot((sw.ResultSpec(value.dtype(), value.layout.shape, value.layout),))
    saved = OneShot((0,))
    invocation = operation.ResolvedInvocation(
        call,
        unary_plan(definition.name),
        results,  # type: ignore[arg-type]
        saved_operands=saved,  # type: ignore[arg-type]
    )
    assert results.iterations == saved.iterations == 1
    assert invocation.saved_operands == (0,)
    assert invocation.kernel_id == definition.name
    with pytest.raises(ValueError, match="roles"):
        operation.ResolvedInvocation(
            call,
            OperationPlan(
                definition.name,
                (OperandPlan(OperandRole.WEAK_SCALAR, None, sw.DType.Float32),),
                Arithmetic.BINARY32,
                None,
                None,
                sw.DType.Float32,
            ),
            invocation.results,
        )


def test_resolved_invocation_rejects_invalid_effect_and_alias_indices() -> None:
    definition = sw.define_operation(
        "resolution.indices",
        sw.OperationSchema(
            (
                sw.OperandSpec("value", sw.OperandKind.TENSOR),
                sw.OperandSpec("axis", sw.OperandKind.STRUCTURAL),
            )
        ),
        lambda call: (_ for _ in ()).throw(AssertionError(call)),
        vjp=sw.NON_DIFFERENTIABLE,
    )
    value = tensor([1.0])
    call = definition.bind(value, "x")
    spec = sw.ResultSpec(value.dtype(), value.layout.shape, value.layout)
    with pytest.raises(ValueError, match="Tensor operands"):
        operation.ResolvedInvocation(
            call,
            unary_plan(definition.name),
            (spec,),
            mutated_operands=(1,),
        )
    alias = sw.ResultSpec(value.dtype(), value.layout.shape, value.layout, alias_of=2)
    with pytest.raises(ValueError, match="out-of-range"):
        operation.ResolvedInvocation(call, unary_plan(definition.name), (alias,))


def test_reference_alias_contract_distinguishes_identity_from_shared_storage() -> None:
    schema = sw.OperationSchema(
        (sw.OperandSpec("value", sw.OperandKind.TENSOR, differentiable=True),)
    )

    def alias_invocation(
        call: operation.BoundOperationCall,
    ) -> operation.ResolvedInvocation:
        value = cast(sw.Tensor, call.operands[0])
        return operation.ResolvedInvocation(
            call,
            unary_plan(call.definition.name),
            (
                sw.ResultSpec(
                    value.dtype(),
                    value.layout.shape,
                    value.layout,
                    alias_of=0,
                ),
            ),
        )

    exact = sw.define_operation(
        "reference.exact_alias_identity",
        schema,
        alias_invocation,
        reference=lambda invocation: cast(sw.Tensor, invocation.operands[0]),
        vjp=sw.NON_DIFFERENTIABLE,
    )
    value = tensor([1.0, 2.0])
    assert _execute_reference(exact.resolve_call(value)) == (value,)

    wrong_identity = sw.define_operation(
        "reference.alias_is_not_shared_storage",
        schema,
        alias_invocation,
        reference=lambda invocation: sw.Tensor(
            cast(sw.Tensor, invocation.operands[0]).carrier,
            cast(sw.Tensor, invocation.operands[0]).offset,
            invocation.results[0].layout,
        ),
        vjp=sw.NON_DIFFERENTIABLE,
    )
    with pytest.raises(sw.ProviderContractError, match="aliasing"):
        _execute_reference(wrong_identity.resolve_call(value))

    def view_invocation(
        call: operation.BoundOperationCall,
    ) -> operation.ResolvedInvocation:
        resolved_value = cast(sw.Tensor, call.operands[0])
        return operation.ResolvedInvocation(
            call,
            unary_plan(call.definition.name),
            (
                sw.ResultSpec(
                    resolved_value.dtype(),
                    resolved_value.layout.shape,
                    resolved_value.layout,
                ),
            ),
        )

    distinct_view = sw.define_operation(
        "reference.distinct_storage_sharing_view",
        schema,
        view_invocation,
        reference=lambda invocation: sw.Tensor(
            cast(sw.Tensor, invocation.operands[0]).carrier,
            cast(sw.Tensor, invocation.operands[0]).offset,
            invocation.results[0].layout,
        ),
        vjp=sw.NON_DIFFERENTIABLE,
    )
    (view,) = _execute_reference(distinct_view.resolve_call(value))
    assert view is not value
    assert view.carrier is value.carrier


def test_all_six_builtins_resolve_to_central_definitions_and_references() -> None:
    vector = sw.Layout(sw.Shape(2), sw.Stride(1))
    matrix = sw.Layout(sw.Shape([2, 2]), sw.Stride([1, 2]))
    x = tensor([1.0, 2.0], layout=vector)
    y = tensor([3.0, 4.0], layout=vector)
    lhs = tensor([1.0, 2.0, 3.0, 4.0], layout=matrix)
    rhs = tensor([5.0, 6.0, 7.0, 8.0], layout=matrix)
    cases = {
        "add": (x, y),
        "elementwise_mul": (x, y),
        "mul": (x, 2.0),
        "relu": (x,),
        "reduce_sum": (lhs,),
        "matmul": (lhs, rhs),
    }
    for name, operands in cases.items():
        definition = _operation_definition(name)
        invocation = definition.resolve_call(*operands)
        assert invocation.definition is definition
        assert invocation.name == name
        assert invocation.plan.operation == name
        (result,) = _execute_reference(invocation)
        assert result.dtype() is invocation.results[0].dtype
        assert result.layout == invocation.results[0].layout


def test_mul_preserves_both_weak_scalar_orders_in_the_plan() -> None:
    value = tensor([1, 2], dtype=sw.DType.Int32)
    definition = _operation_definition("mul")
    forward = definition.resolve_call(value, 3)
    reverse = definition.resolve_call(3, value)
    assert tuple(entry.role for entry in forward.plan.operands) == (
        OperandRole.TENSOR,
        OperandRole.WEAK_SCALAR,
    )
    assert tuple(entry.role for entry in reverse.plan.operands) == (
        OperandRole.WEAK_SCALAR,
        OperandRole.TENSOR,
    )
    assert _execute_reference(forward)[0][1] == 6
    assert _execute_reference(reverse)[0][1] == 6


def test_builtin_resolution_validates_shape_and_options_before_execution() -> None:
    lhs = tensor([1.0, 2.0])
    rhs = tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="broadcast-compatible"):
        _operation_definition("add").resolve_call(lhs, rhs)
    with pytest.raises(TypeError, match="unknown option"):
        _operation_definition("relu").resolve_call(lhs, accumulator_dtype=None)
    with pytest.raises(TypeError, match="accumulator_dtype"):
        _operation_definition("reduce_sum").resolve_call(
            tensor(
                [1, 2, 3, 4],
                dtype=sw.DType.Int32,
                layout=sw.Layout(sw.Shape([2, 2]), sw.Stride([1, 2])),
            ),
            accumulator_dtype=sw.DType.Float64,
        )


def test_reference_results_are_normalized_and_validated() -> None:
    value = tensor([1.0])
    schema = sw.OperationSchema((sw.OperandSpec("value", sw.OperandKind.TENSOR),))

    empty = sw.define_operation(
        "reference.empty",
        schema,
        unary_invocation,
        reference=lambda _invocation: (),
        vjp=sw.NON_DIFFERENTIABLE,
    )
    with pytest.raises(sw.ProviderContractError, match="no results"):
        _execute_reference(empty.resolve_call(value))

    wrong = sw.define_operation(
        "reference.wrong_type",
        schema,
        unary_invocation,
        reference=lambda _invocation: (object(),),  # type: ignore[return-value]
        vjp=sw.NON_DIFFERENTIABLE,
    )
    with pytest.raises(TypeError, match="must be a Tensor"):
        _execute_reference(wrong.resolve_call(value))

    no_reference = sw.define_operation(
        "reference.absent",
        schema,
        unary_invocation,
        vjp=sw.NON_DIFFERENTIABLE,
    )
    with pytest.raises(sw.UnsupportedOperationPlan, match="no reference"):
        _execute_reference(no_reference.resolve_call(value))


def test_vjp_context_and_callback_validation_are_semantic_definition_owned() -> None:
    value = tensor([1.0, 2.0])
    cotangent = tensor([3.0, 4.0])
    definition = sw.define_operation(
        "gradient.explicit",
        sw.OperationSchema(
            (sw.OperandSpec("value", sw.OperandKind.TENSOR, differentiable=True),)
        ),
        lambda call: unary_invocation(call, saved=(0,)),
        vjp=lambda _context, cotangents: (cotangents[0],),
    )
    invocation = definition.resolve_call(value)
    context = operation.VJPContext(invocation, (value,), (value,))
    assert _call_vjp(context, (cotangent,)) == (cotangent,)
    with pytest.raises(ValueError, match="save recipe"):
        operation.VJPContext(invocation, (value,), ())
    with pytest.raises(ValueError, match="one entry per result"):
        _call_vjp(context, ())

    nondifferentiable = sw.define_operation(
        "gradient.none",
        sw.OperationSchema((sw.OperandSpec("value", sw.OperandKind.TENSOR),)),
        unary_invocation,
        vjp=sw.NON_DIFFERENTIABLE,
    )
    no_grad_invocation = nondifferentiable.resolve_call(value)
    no_grad_context = operation.VJPContext(no_grad_invocation, (value,), ())
    with pytest.raises(RuntimeError, match="no VJP"):
        _call_vjp(no_grad_context, (cotangent,))


def test_builtin_vjps_execute_the_central_gradient_rules() -> None:
    vector_layout = sw.Layout(sw.Shape(2), sw.Stride(1))
    x = tensor([-1.0, 2.0], layout=vector_layout)
    y = tensor([3.0, 4.0], layout=vector_layout)
    cotangent = tensor([5.0, 6.0], layout=vector_layout)

    for name, operands, expected in (
        ("add", (x, y), ([5.0, 6.0], [5.0, 6.0])),
        ("elementwise_mul", (x, y), ([15.0, 24.0], [-5.0, 12.0])),
        ("mul", (x, 3.0), ([15.0, 18.0],)),
        ("mul", (3.0, x), ([15.0, 18.0],)),
        ("relu", (x,), ([0.0, 6.0],)),
    ):
        invocation = _operation_definition(name).resolve_call(*operands)
        outputs = _execute_reference(invocation)
        saved = tuple(
            cast(sw.Tensor, invocation.operands[index])
            for index in invocation.saved_operands
        )
        context = operation.VJPContext(invocation, outputs, saved)
        gradients = _call_vjp(context, (cotangent,))
        assert (
            tuple(
                [gradient[index] for index in range(gradient.size())]
                for gradient in gradients
                if gradient is not None
            )
            == expected
        )

    matrix_layout = sw.Layout(sw.Shape([2, 2]), sw.Stride([1, 2]))
    matrix = tensor([1.0, 2.0, 3.0, 4.0], layout=matrix_layout)
    reduction = _operation_definition("reduce_sum").resolve_call(matrix)
    reduction_output = _execute_reference(reduction)
    reduction_context = operation.VJPContext(reduction, reduction_output, ())
    reduction_gradient = _call_vjp(reduction_context, (tensor([10.0, 20.0]),))[0]
    assert reduction_gradient is not None
    assert [reduction_gradient[index] for index in range(4)] == [
        10.0,
        20.0,
        10.0,
        20.0,
    ]

    rhs = tensor([5.0, 6.0, 7.0, 8.0], layout=matrix_layout)
    matmul = _operation_definition("matmul").resolve_call(matrix, rhs)
    matmul_output = _execute_reference(matmul)
    matmul_context = operation.VJPContext(matmul, matmul_output, (matrix, rhs))
    matrix_cotangent = tensor([1.0, 1.0, 1.0, 1.0], layout=matrix_layout)
    lhs_gradient, rhs_gradient = _call_vjp(matmul_context, (matrix_cotangent,))
    assert lhs_gradient is not None and rhs_gradient is not None
    assert [lhs_gradient[index] for index in range(4)] == [11.0, 11.0, 15.0, 15.0]
    assert [rhs_gradient[index] for index in range(4)] == [3.0, 3.0, 7.0, 7.0]


def test_add_vjp_reduces_structural_broadcast_axes() -> None:
    lhs_layout = sw.Layout(sw.Shape([1, 2]), sw.Stride([1, 1]))
    rhs_layout = sw.Layout(sw.Shape([3, 2]), sw.Stride([1, 3]))
    lhs = tensor([1.0, 2.0], layout=lhs_layout)
    rhs = tensor([1.0] * 6, layout=rhs_layout)
    invocation = _operation_definition("add").resolve_call(lhs, rhs)
    outputs = _execute_reference(invocation)
    cotangent = tensor([1.0] * 6, layout=rhs_layout)
    context = operation.VJPContext(invocation, outputs, ())
    lhs_gradient, rhs_gradient = _call_vjp(context, (cotangent,))
    assert lhs_gradient is not None and rhs_gradient is not None
    assert [lhs_gradient[index] for index in range(2)] == [3.0, 3.0]
    assert [rhs_gradient[index] for index in range(6)] == [1.0] * 6


def test_legacy_carriers_keep_their_existing_operation_classes() -> None:
    from strideweave.carriers.metal.carrier import Metal
    from strideweave.carriers.metal.pointwise_ops import MetalPointwiseOperation

    layout = sw.Layout(sw.Shape(1), sw.Stride(1))
    generic = sw.Generic([1.0], dtype=sw.DType.Float32)
    assert type(generic.dispatch_op("add")) is operation.GenericAddOperation
    assert type(generic.dispatch_op("relu")) is operation.GenericReLUOperation
    cpu = sw.CPU(1, dtype=sw.DType.Float32)
    assert type(cpu.dispatch_op("add")).__name__ == "_CPUAddOperation"
    assert type(cpu.dispatch_op("relu")).__name__ == "_CPUReLUOperation"
    assert (
        type(
            Metal._dispatch_op(  # strideweave-lint: ignore=RT011
                None,  # type: ignore[arg-type]
                "add",
            )
        )
        is MetalPointwiseOperation
    )
    assert (
        type(
            Metal._dispatch_op(  # strideweave-lint: ignore=RT011
                None,  # type: ignore[arg-type]
                "relu",
            )
        )
        is MetalPointwiseOperation
    )
    assert sw.add(sw.Tensor(generic, 0, layout), tensor([2.0]))[0] == 3.0
    for carrier_class in (
        sw.Generic,
        sw.CPU,
        sw.Metal,
        sw.FileBacked,
        sw.BlockDeviceCarrier,
        sw.Evictable,
    ):
        assert not sw.carriers.extension._has_carrier_definition(carrier_class)
