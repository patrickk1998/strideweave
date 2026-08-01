"""The backend capability registry for already-resolved operation plans.

These tests pin the registry as *operational* infrastructure: the entries a
backend enumerates are the entries execution matches against, matching is exact
rather than approximate, and a shape nobody declared is refused rather than
routed to a nearby implemented branch. Each test declares its capabilities on a
class of its own, so the registry's built-in entries are never involved and the
suite stays order-independent.
"""

from __future__ import annotations

import ast
import dataclasses
import gc
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import strideweave as sw
import strideweave.carriers as sw_carriers
import strideweave.carriers._built_in_capabilities as built_in_capability_module
import strideweave.carriers.cpu.capabilities as cpu_capability_module
import strideweave.carriers.generic.capabilities as generic_capability_module
import strideweave.carriers.operation_capability as operation_capability
from strideweave import DType
from strideweave.carriers._built_in_capabilities import (
    _initialize_built_in_capabilities,
)
from strideweave.carriers.cpu.capabilities import cpu_capabilities
from strideweave.carriers.generic.capabilities import generic_capabilities
from strideweave.carriers.operation_capability import (
    OperandCapability,
    OperationCapability,
    UnsupportedOperationPlan,
    capabilities_for_carrier_class,
    matching_capability,
    register_operation_capabilities,
    require_capability,
    supports_operation_plan,
    unsupported_plan_reason,
)
from strideweave.carriers.operation_policy import (
    Accumulation,
    Arithmetic,
    OperandPlan,
    OperandRole,
    OperationPlan,
    resolve_operation_plan,
)

F32 = DType.Float32
I32 = DType.Int32
BOOL = DType.Bool


def plan(
    operation="add",
    *,
    operands=None,
    compute=Arithmetic.BINARY32,
    accumulation=None,
    output=F32,
):
    """Build a plan directly, so a shape the resolver never emits can be tested."""
    if operands is None:
        operands = ((F32, F32), (F32, F32))
    return OperationPlan(
        operation=operation,
        operands=tuple(
            OperandPlan(role=OperandRole.TENSOR, dtype=dtype, convert_to=convert_to)
            for dtype, convert_to in operands
        ),
        compute=compute,
        accumulation=accumulation,
        output=output,
    )


def capability(**overrides):
    """Build the capability accepting exactly the plan ``plan(**overrides)``."""
    return OperationCapability.from_plan(plan(**overrides))


def carrier_class(name="ExampleCarrier", base=sw.Carrier):
    """Return a fresh Carrier implementation, so one test's declarations never
    reach another."""
    return type(name, (base,), {})


# --- Registration and enumeration -------------------------------------------


def test_a_registered_capability_is_enumerated_and_matched():
    backend = carrier_class()
    entry = capability()

    register_operation_capabilities(backend, [entry])

    assert capabilities_for_carrier_class(backend) == (entry,)
    assert matching_capability(backend, plan()) is entry
    assert supports_operation_plan(backend, plan())


def test_enumeration_is_deterministic_rather_than_registration_ordered():
    backend = carrier_class()
    relu = OperationCapability.from_plan(resolve_operation_plan("relu", F32))
    add = OperationCapability.from_plan(resolve_operation_plan("add", I32, I32))
    reduce = OperationCapability.from_plan(resolve_operation_plan("reduce_sum", F32))

    register_operation_capabilities(backend, [relu, reduce, add])

    assert [entry.operation for entry in capabilities_for_carrier_class(backend)] == [
        "add",
        "reduce_sum",
        "relu",
    ]


def test_enumeration_can_be_restricted_to_one_operation():
    backend = carrier_class()
    integer = OperationCapability.from_plan(resolve_operation_plan("add", I32, I32))
    floating = OperationCapability.from_plan(resolve_operation_plan("add", F32, F32))
    unrelated = OperationCapability.from_plan(resolve_operation_plan("relu", F32))

    register_operation_capabilities(backend, [integer, floating, unrelated])

    assert set(capabilities_for_carrier_class(backend, "add")) == {integer, floating}
    assert capabilities_for_carrier_class(backend, "matmul") == ()


def test_enumeration_returns_an_immutable_view_of_immutable_entries():
    backend = carrier_class()
    entry = capability()
    register_operation_capabilities(backend, [entry])

    (enumerated,) = capabilities_for_carrier_class(backend)

    assert isinstance(capabilities_for_carrier_class(backend), tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        enumerated.output = I32  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        enumerated.operands[0].convert_to = I32  # type: ignore[misc]
    # A capability is hashable, so a backend may key a lowered form by it.
    assert {enumerated} == {capability()}


def test_a_class_declaring_one_shape_twice_is_rejected():
    # A declaration is complete and happens once, so a repeated shape can only
    # appear within it.
    backend = carrier_class()

    with pytest.raises(ValueError, match="already declares a capability"):
        register_operation_capabilities(backend, [capability(), capability()])

    assert capabilities_for_carrier_class(backend) == ()


def test_a_second_declaration_is_refused_and_changes_nothing():
    backend = carrier_class()
    declared = capability()
    register_operation_capabilities(backend, [declared])
    extra = OperationCapability.from_plan(resolve_operation_plan("relu", I32))

    with pytest.raises(TypeError, match="has already declared"):
        register_operation_capabilities(backend, [extra])

    assert capabilities_for_carrier_class(backend) == (declared,)
    assert not supports_operation_plan(backend, resolve_operation_plan("relu", I32))


def test_an_empty_declaration_is_complete_and_closes_the_class():
    # Declaring nothing is a statement — this class executes no planned
    # operation — not a class still waiting for its entries.
    backend = carrier_class()

    register_operation_capabilities(backend, [])

    with pytest.raises(TypeError, match="has already declared"):
        register_operation_capabilities(backend, [capability()])
    assert capabilities_for_carrier_class(backend) == ()


def test_a_rejected_declaration_leaves_the_class_open():
    # Nothing was published, so nothing was decided: a caller may declare the
    # corrected set.
    backend = carrier_class()
    with pytest.raises(TypeError, match="must be OperationCapability entries"):
        register_operation_capabilities(backend, [plan()])  # type: ignore[list-item]

    register_operation_capabilities(backend, [capability()])

    assert capabilities_for_carrier_class(backend) == (capability(),)


@pytest.mark.parametrize(
    "observe",
    [
        capabilities_for_carrier_class,
        lambda backend: supports_operation_plan(backend, plan()),
        lambda backend: unsupported_plan_reason(backend, plan()),
        lambda backend: matching_capability(backend, plan()),
    ],
    ids=["enumerate", "supports", "reason", "match"],
)
def test_observing_an_undeclared_class_seals_its_empty_set(observe):
    # The confirmed hazard: a dependent carrier that snapshots this class is
    # told "nothing" and would be silently wrong once a later declaration
    # widened it. First observation is therefore the final answer.
    backend = carrier_class()

    observe(backend)

    with pytest.raises(TypeError, match="first observed without a declaration"):
        register_operation_capabilities(backend, [capability()])
    assert capabilities_for_carrier_class(backend) == ()


def test_requiring_a_plan_seals_an_undeclared_class():
    backend = carrier_class()

    with pytest.raises(UnsupportedOperationPlan):
        require_capability(backend, plan())

    with pytest.raises(TypeError, match="has already declared"):
        register_operation_capabilities(backend, [capability()])


def test_a_declaration_before_any_observation_is_the_answer():
    backend = carrier_class()

    register_operation_capabilities(backend, [capability()])

    assert supports_operation_plan(backend, plan())
    assert capabilities_for_carrier_class(backend) == (capability(),)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (sw.CPU, "is a shipped backend"),
        (carrier_class("PreviouslyDeclared"), "has already declared"),
    ],
    ids=["shipped", "custom"],
)
def test_a_closed_class_says_why_it_is_closed(target, message):
    # "The framework already declared this one" and "you already declared this
    # one" are different mistakes with different fixes.
    if target is not sw.CPU:
        register_operation_capabilities(target, [capability()])

    with pytest.raises(TypeError, match=message):
        register_operation_capabilities(target, [])


def test_a_declaration_racing_observers_is_all_or_nothing():
    # Whatever interleaving wins, an observer sees the complete declared set or
    # the sealed empty one, and the class never widens after an observation.
    entries = tuple(
        OperationCapability.from_plan(resolve_operation_plan(operation, I32))
        for operation in ("relu", "reduce_sum")
    )
    for _ in range(50):
        backend = carrier_class("RacedCarrier")
        start = threading.Barrier(4)
        observations: list[tuple[OperationCapability, ...]] = []
        declared: list[bool] = []

        def declare(backend=backend, declared=declared, start=start):
            start.wait()
            try:
                register_operation_capabilities(backend, entries)
            except TypeError:
                declared.append(False)
            else:
                declared.append(True)

        def observe(backend=backend, observations=observations, start=start):
            start.wait()
            observations.append(capabilities_for_carrier_class(backend))

        threads = [threading.Thread(target=declare)] + [
            threading.Thread(target=observe) for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        final = capabilities_for_carrier_class(backend)
        assert set(final) in ({*entries}, set())
        assert declared == [set(final) == {*entries}]
        # An observation that sealed the empty set is exactly what makes the
        # declaration fail, so the two outcomes never mix.
        assert all(observed == final for observed in observations)


def test_registration_takes_a_class_rather_than_a_carrier_instance():
    backend = carrier_class()

    with pytest.raises(TypeError, match="must be a carrier class"):
        register_operation_capabilities(backend(), [capability()])  # type: ignore[arg-type]


def test_only_capability_entries_can_be_registered():
    backend = carrier_class()

    with pytest.raises(TypeError, match="must be OperationCapability entries"):
        register_operation_capabilities(backend, [plan()])  # type: ignore[list-item]

    assert capabilities_for_carrier_class(backend) == ()


# --- Exact matching ---------------------------------------------------------
#
# The registry exists to stop a backend running a plan whose shape it never
# implemented, so every field must be part of the match. Each case below differs
# from the declared capability in exactly one field.


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"compute": Arithmetic.INT32_EXACT_CHECKED}, "compute"),
        ({"accumulation": Accumulation.SEQUENTIAL_BINARY32}, "accumulation"),
        ({"output": I32}, "output"),
        ({"operands": (((F32, F32), (I32, I32)))}, "conversion"),
        ({"operation": "sub"}, "operation"),
    ],
    ids=["compute", "accumulation", "output", "conversion", "operation"],
)
def test_a_plan_differing_in_one_field_does_not_match(overrides, reason):
    backend = carrier_class()
    register_operation_capabilities(backend, [capability()])

    assert not supports_operation_plan(backend, plan(**overrides))
    assert reason  # names the field this case varies


def test_a_declared_accumulation_is_not_satisfied_by_a_plan_without_one():
    # The confirmed CPU bug: an Int32 reduce-sum plan with no accumulation must
    # not reach the accumulating branch a reduce capability describes.
    backend = carrier_class()
    accumulating = resolve_operation_plan("reduce_sum", I32)
    register_operation_capabilities(
        backend, [OperationCapability.from_plan(accumulating)]
    )

    assert supports_operation_plan(backend, accumulating)
    assert not supports_operation_plan(
        backend, dataclasses.replace(accumulating, accumulation=None)
    )


def test_a_compute_and_output_mismatch_does_not_match_either_direction():
    # The confirmed Generic bug: a binary32 compute with an Int32 output, and
    # its mirror image, are shapes no backend declares.
    backend = carrier_class()
    register_operation_capabilities(
        backend,
        [
            OperationCapability.from_plan(resolve_operation_plan("add", F32, F32)),
            OperationCapability.from_plan(resolve_operation_plan("add", I32, I32)),
        ],
    )

    floating_compute_integer_output = plan(compute=Arithmetic.BINARY32, output=I32)
    integer_compute_floating_output = plan(
        operands=((I32, I32), (I32, I32)),
        compute=Arithmetic.INT32_EXACT_CHECKED,
        output=F32,
    )

    assert not supports_operation_plan(backend, floating_compute_integer_output)
    assert not supports_operation_plan(backend, integer_compute_floating_output)


def test_operand_order_is_part_of_the_match():
    backend = carrier_class()
    register_operation_capabilities(
        backend,
        [OperationCapability.from_plan(plan(operands=((F32, F32), (I32, F32))))],
    )

    assert not supports_operation_plan(backend, plan(operands=((I32, F32), (F32, F32))))


def test_a_query_takes_a_resolved_plan():
    backend = carrier_class()

    with pytest.raises(TypeError, match="must be an OperationPlan"):
        supports_operation_plan(backend, "add")  # type: ignore[arg-type]


# --- Refusal ----------------------------------------------------------------


def test_requiring_an_undeclared_operation_names_the_operation():
    backend = carrier_class("QuietCarrier")

    with pytest.raises(
        UnsupportedOperationPlan,
        match=r"QuietCarrier declares no operation-plan capability for 'add'",
    ):
        require_capability(backend, plan())


def test_requiring_an_undeclared_shape_describes_the_shape():
    backend = carrier_class("PartialCarrier")
    register_operation_capabilities(
        backend,
        [OperationCapability.from_plan(resolve_operation_plan("add", F32, F32))],
    )

    with pytest.raises(
        UnsupportedOperationPlan,
        match=(
            r"PartialCarrier declares no operation-plan capability for this 'add' "
            r"plan: add with operands \(tensor Int32->Int32, tensor Int32->Int32\), "
            r"int32_exact_checked compute, no accumulation, Int32 output"
        ),
    ):
        require_capability(backend, resolve_operation_plan("add", I32, I32))


def test_a_supported_plan_has_no_unsupported_reason():
    backend = carrier_class()
    entry = capability()
    register_operation_capabilities(backend, [entry])

    assert unsupported_plan_reason(backend, plan()) is None
    assert require_capability(backend, plan()) is entry


def test_an_unsupported_reason_distinguishes_operation_from_shape():
    backend = carrier_class("NarrowCarrier")
    register_operation_capabilities(backend, [capability()])

    unknown_operation = unsupported_plan_reason(backend, plan(operation="matmul"))
    unknown_shape = unsupported_plan_reason(backend, plan(output=I32))

    assert unknown_operation == (
        "NarrowCarrier declares no operation-plan capability for 'matmul'"
    )
    assert unknown_shape is not None
    assert "for this 'add' plan" in unknown_shape


# --- The registry decides nothing --------------------------------------------


def test_an_undeclared_backend_supports_nothing_the_policy_resolves():
    # The registry never infers support from central policy: a resolvable plan
    # is not an executable one until a backend says so.
    backend = carrier_class()

    for operation, operands in (
        ("add", (F32, F32)),
        ("relu", (I32,)),
        ("matmul", (I32, I32)),
    ):
        resolved = resolve_operation_plan(operation, *operands)
        assert matching_capability(backend, resolved) is None
        assert not supports_operation_plan(backend, resolved)


def test_an_unmatched_plan_is_never_answered_with_a_nearby_capability():
    backend = carrier_class()
    integer = OperationCapability.from_plan(resolve_operation_plan("add", I32, I32))
    register_operation_capabilities(backend, [integer])

    floating = resolve_operation_plan("add", F32, F32)

    assert matching_capability(backend, floating) is None
    with pytest.raises(UnsupportedOperationPlan):
        require_capability(backend, floating)


# --- Extension ---------------------------------------------------------------


def test_a_declaration_belongs_to_the_exact_class_that_made_it():
    # Support is a claim about an implementation, and an implementation is not
    # inherited: a class that declared nothing supports nothing, whatever a base
    # of it declared.
    base = carrier_class("BaseCarrier")
    register_operation_capabilities(base, [capability()])
    derived = carrier_class("DerivedCarrier", base)

    assert capabilities_for_carrier_class(base) == (capability(),)
    assert capabilities_for_carrier_class(derived) == ()
    assert not supports_operation_plan(derived, plan())


def test_a_declaration_never_widens_another_class():
    base = carrier_class("StableCarrier")
    register_operation_capabilities(base, [capability()])
    derived = carrier_class("ExtendedCarrier", base)
    extra = OperationCapability.from_plan(resolve_operation_plan("relu", I32))

    register_operation_capabilities(derived, [extra])

    assert capabilities_for_carrier_class(derived) == (extra,)
    assert extra not in capabilities_for_carrier_class(base)
    assert not supports_operation_plan(base, resolve_operation_plan("relu", I32))


def test_a_late_declaration_reaches_only_the_class_it_names():
    base = carrier_class("LateBase")
    derived = carrier_class("EarlyDerived", base)
    assert capabilities_for_carrier_class(derived) == ()

    register_operation_capabilities(base, [capability()])

    assert capabilities_for_carrier_class(base) == (capability(),)
    assert capabilities_for_carrier_class(derived) == ()


def test_sibling_classes_stay_isolated():
    base = carrier_class("SharedBase")
    register_operation_capabilities(base, [capability()])
    left = carrier_class("LeftCarrier", base)
    right = carrier_class("RightCarrier", base)
    extra = OperationCapability.from_plan(resolve_operation_plan("relu", F32))

    register_operation_capabilities(left, [extra])

    assert supports_operation_plan(left, resolve_operation_plan("relu", F32))
    assert not supports_operation_plan(right, resolve_operation_plan("relu", F32))
    assert not supports_operation_plan(base, resolve_operation_plan("relu", F32))


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (object, "not a Carrier implementation"),
        (type("Unrelated", (), {}), "not a Carrier implementation"),
        (sw.Carrier, "cannot be declared for the Carrier base class"),
    ],
    ids=["object", "unrelated", "carrier-root"],
)
def test_only_an_independent_carrier_implementation_may_declare(target, message):
    # The confirmed bug this rule exists for: declaring one capability on object
    # made every backend advertise a shape none of them implements.
    contamination = plan(operation="count", operands=((I32, I32),), output=I32)

    with pytest.raises(TypeError, match=message):
        register_operation_capabilities(
            target, [OperationCapability.from_plan(contamination)]
        )

    assert capabilities_for_carrier_class(object) == ()
    assert capabilities_for_carrier_class(sw.Carrier) == ()
    assert not supports_operation_plan(sw.CPU, contamination)
    assert not supports_operation_plan(sw.Generic, contamination)


# Evictable is shipped too, but it is dependent: it finalizes per instance and
# has no class declaration to seal, which
# test_evictable_has_no_class_declaration_to_seal covers.
SHIPPED_BACKENDS = [sw.Generic, sw.CPU, sw.FileBacked]
STORAGE_CARRIERS = [sw.FileBacked]


@pytest.mark.parametrize(
    "built_in", SHIPPED_BACKENDS, ids=lambda cls: cls.__name__.lower()
)
def test_a_shipped_backends_declarations_are_sealed(built_in):
    before = capabilities_for_carrier_class(built_in)
    extra = OperationCapability.from_plan(
        plan(operation="count", operands=((I32, I32),), output=I32)
    )

    with pytest.raises(TypeError, match="sealed"):
        register_operation_capabilities(built_in, [extra])
    with pytest.raises(RuntimeError, match="already declared"):
        operation_capability._declare_built_in_capabilities(built_in, [extra])

    assert capabilities_for_carrier_class(built_in) == before


@pytest.mark.parametrize(
    "storage_carrier", STORAGE_CARRIERS, ids=lambda cls: cls.__name__.lower()
)
def test_a_storage_carrier_declares_and_seals_an_empty_set(storage_carrier):
    # The confirmed bug: only the backends that declare kernels were sealed, so
    # a fake relu(Int32) capability registered against the closed FileBacked
    # class was reported as executable by a carrier that plans no operation of
    # its own.
    relu = resolve_operation_plan("relu", I32)

    assert capabilities_for_carrier_class(storage_carrier) == ()
    with pytest.raises(TypeError, match="sealed"):
        register_operation_capabilities(
            storage_carrier, [OperationCapability.from_plan(relu)]
        )

    assert capabilities_for_carrier_class(storage_carrier) == ()
    assert not supports_operation_plan(storage_carrier, relu)


def test_no_exported_declaration_path_reaches_a_shipped_backend():
    # Sealing is framework-internal: the only exported way into the registry is
    # public registration, which refuses every shipped class.
    assert not hasattr(operation_capability, "declare_built_in_capabilities")
    assert not hasattr(sw_carriers, "initialize_built_in_capabilities")
    assert not hasattr(built_in_capability_module, "initialize_built_in_capabilities")
    assert built_in_capability_module.__all__ == []
    assert [name for name in operation_capability.__all__ if "declare" in name] == []
    assert not any(
        hasattr(module, name)
        for module in (
            sw,
            sw_carriers,
            cpu_capability_module,
            generic_capability_module,
        )
        for name in (
            "declare_built_in_capabilities",
            "declare_cpu_capabilities",
            "declare_generic_capabilities",
        )
    )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (object, "not a Carrier implementation"),
        (type("Unrelated", (), {}), "not a Carrier implementation"),
        (sw.Carrier, "cannot be declared for the Carrier base class"),
    ],
    ids=["object", "unrelated", "carrier-root"],
)
def test_the_internal_declaration_path_is_not_a_wider_escape_hatch(target, message):
    # Being private is not the only guard: the bootstrap path validates its class
    # exactly as public registration does, so no framework-internal caller can
    # contaminate object, the Carrier root, or a non-carrier either.
    contamination = plan(operation="count", operands=((I32, I32),), output=I32)

    with pytest.raises(TypeError, match=message):
        operation_capability._declare_built_in_capabilities(
            target, [OperationCapability.from_plan(contamination)]
        )

    assert capabilities_for_carrier_class(object) == ()
    assert capabilities_for_carrier_class(sw.Carrier) == ()


def test_reinitializing_the_built_in_capabilities_changes_nothing():
    before = {
        built_in: capabilities_for_carrier_class(built_in)
        for built_in in SHIPPED_BACKENDS
    }

    _initialize_built_in_capabilities()

    assert {
        built_in: capabilities_for_carrier_class(built_in)
        for built_in in SHIPPED_BACKENDS
    } == before


@pytest.mark.parametrize(
    "first_import",
    [
        "strideweave",
        "strideweave.carriers.cpu",
        "strideweave.carriers.generic.carrier",
        "strideweave.carriers.file_backed",
        "strideweave.carriers.operation_capability",
    ],
)
def test_every_shipped_backend_is_sealed_whatever_is_imported_first(first_import):
    # Whichever module a user reaches for, the carrier package initializes first
    # and no shipped class is ever observable in an unsealed state.
    program = f"""
import importlib

importlib.import_module({first_import!r})

import strideweave as sw
from strideweave.carriers.operation_capability import (
    capabilities_for_carrier_class,
    register_operation_capabilities,
)

assert capabilities_for_carrier_class(sw.CPU)
assert capabilities_for_carrier_class(sw.Generic)
for built_in in (sw.Generic, sw.CPU, sw.FileBacked):
    try:
        register_operation_capabilities(built_in, [])
    except TypeError as error:
        assert "sealed" in str(error), error
    else:
        raise AssertionError(built_in)
try:
    register_operation_capabilities(sw.Evictable, [])
except TypeError as error:
    assert "DependentCarrier" in str(error), error
else:
    raise AssertionError(sw.Evictable)
"""

    subprocess.run([sys.executable, "-c", program], check=True)


def test_a_shipped_backend_declares_exactly_what_it_executes():
    # Introspection and execution read the same entries, and those entries are
    # the ones the bootstrap declared, not a resolution through a base.
    assert set(capabilities_for_carrier_class(sw.Generic)) == set(
        generic_capabilities()
    )
    assert set(capabilities_for_carrier_class(sw.CPU)) == set(cpu_capabilities())


@pytest.mark.parametrize("backend", [sw.Generic, sw.CPU], ids=["generic", "cpu"])
def test_shipped_backends_declare_select_and_every_clamp_overload(backend):
    select_plan = resolve_operation_plan("select", BOOL, F32, F32)
    clamp_plans = (
        resolve_operation_plan("clamp", F32, F32, F32),
        resolve_operation_plan("clamp", F32, F32, 1.0),
        resolve_operation_plan("clamp", F32, -1.0, F32),
        resolve_operation_plan("clamp", F32, -1.0, 1.0),
    )

    assert set(capabilities_for_carrier_class(backend, "select")) == {
        OperationCapability.from_plan(select_plan)
    }
    assert set(capabilities_for_carrier_class(backend, "clamp")) == {
        OperationCapability.from_plan(plan) for plan in clamp_plans
    }
    assert supports_operation_plan(backend, select_plan)
    assert all(supports_operation_plan(backend, plan) for plan in clamp_plans)

    clamp_roles = {
        tuple(operand.role for operand in entry.operands)
        for entry in capabilities_for_carrier_class(backend, "clamp")
    }
    assert clamp_roles == {
        (OperandRole.TENSOR, OperandRole.TENSOR, OperandRole.TENSOR),
        (OperandRole.TENSOR, OperandRole.TENSOR, OperandRole.WEAK_SCALAR),
        (OperandRole.TENSOR, OperandRole.WEAK_SCALAR, OperandRole.TENSOR),
        (OperandRole.TENSOR, OperandRole.WEAK_SCALAR, OperandRole.WEAK_SCALAR),
    }


# --- Capability construction --------------------------------------------------


def test_a_capability_round_trips_the_plan_it_was_built_from():
    for operation, operands in (
        ("add", (I32, I32)),
        ("div", (I32, F32)),
        ("mul", (I32, 3)),
        ("pow", (F32, 0.5)),
        ("reduce_sum", (I32,)),
        ("matmul", (I32, I32)),
        ("exp", (I32,)),
        ("select", (BOOL, F32, F32)),
        ("clamp", (F32, F32, F32)),
        ("clamp", (F32, F32, 1.0)),
        ("clamp", (F32, -1.0, F32)),
        ("clamp", (F32, -1.0, 1.0)),
    ):
        resolved = resolve_operation_plan(operation, *operands)
        entry = OperationCapability.from_plan(resolved)

        assert entry.matches(resolved)
        assert entry.operation == resolved.operation
        assert entry.compute is resolved.compute
        assert entry.accumulation is resolved.accumulation
        assert entry.output is resolved.output


def test_a_weak_scalar_capability_carries_no_storage_dtype():
    entry = OperationCapability.from_plan(resolve_operation_plan("mul", I32, 3))
    tensor, scalar = entry.operands

    assert tensor.role is OperandRole.TENSOR
    assert tensor.dtype is I32
    assert scalar.role is OperandRole.WEAK_SCALAR
    assert scalar.dtype is None
    assert scalar.convert_to is I32


def test_clamp_weak_scalar_capabilities_carry_no_storage_dtype():
    entry = OperationCapability.from_plan(
        resolve_operation_plan("clamp", F32, -1.0, 1.0)
    )
    tensor, lower, upper = entry.operands

    assert tensor.role is OperandRole.TENSOR
    assert tensor.dtype is F32
    for bound in (lower, upper):
        assert bound.role is OperandRole.WEAK_SCALAR
        assert bound.dtype is None
        assert bound.convert_to is F32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"role": OperandRole.TENSOR, "dtype": None, "convert_to": F32},
            "tensor operand capability needs a storage dtype",
        ),
        (
            {"role": OperandRole.WEAK_SCALAR, "dtype": I32, "convert_to": I32},
            "weak scalar operand capability has no storage dtype",
        ),
        (
            {"role": "tensor", "dtype": F32, "convert_to": F32},
            "role must be an OperandRole",
        ),
        (
            {"role": OperandRole.TENSOR, "dtype": F32, "convert_to": DType.Floating},
            "convert_to must be a SimpleDType",
        ),
    ],
    ids=["missing-dtype", "scalar-dtype", "role", "convert-to"],
)
def test_a_malformed_operand_capability_is_rejected(kwargs, message):
    with pytest.raises(TypeError, match=message):
        OperandCapability(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"operands": ()}, ValueError, "needs at least one operand"),
        ({"compute": "binary32"}, TypeError, "compute must be an Arithmetic"),
        (
            {"accumulation": "exact_integer"},
            TypeError,
            "accumulation must be an Accumulation",
        ),
        ({"output": DType.Any}, TypeError, "output must be a SimpleDType"),
        ({"operation": 7}, TypeError, "operation must be a str"),
    ],
    ids=["operands", "compute", "accumulation", "output", "operation"],
)
def test_a_malformed_capability_is_rejected(overrides, error, message):
    fields = {
        "operation": "add",
        "operands": (
            OperandCapability(role=OperandRole.TENSOR, dtype=F32, convert_to=F32),
        ),
        "compute": Arithmetic.BINARY32,
        "accumulation": None,
        "output": F32,
        **overrides,
    }

    with pytest.raises(error, match=message):
        OperationCapability(**fields)


# --- The public introspection surface ----------------------------------------
#
# A user asks a carrier what its backend executes and gets the same entries
# execution is accepted against, without running a kernel.


@pytest.mark.parametrize("carrier", [sw.CPU(1), sw.Generic([1.0])])
def test_a_carrier_reports_its_backend_capabilities(carrier):
    reported = carrier.operation_capabilities()

    assert reported == capabilities_for_carrier_class(type(carrier))
    assert isinstance(reported, tuple)
    assert all(isinstance(entry, sw.OperationCapability) for entry in reported)
    # Predicate plans and index-producing reductions coexist with the numeric
    # plans in one backend registry; introspection must not narrow the result to
    # a single output dtype.
    outputs = {entry.output for entry in reported}
    assert {DType.Bool, DType.Int32} <= outputs


def test_capability_queries_do_not_filter_by_the_querying_carriers_dtype():
    # The query describes the backend, not this carrier's storage: a Float32 CPU
    # carrier still reports the Int32 plans CPU executes.
    floating = sw.CPU(1, dtype=F32)
    integer = sw.CPU(1, dtype=I32)

    assert floating.operation_capabilities() == integer.operation_capabilities()
    assert floating.supports_operation_plan(resolve_operation_plan("relu", I32))


def test_a_carrier_query_is_restricted_to_one_operation_in_a_stable_order():
    carrier = sw.CPU(1)

    # ``mul`` has tensor/tensor and tensor/weak-scalar overloads. The query is
    # still restricted to the name and returns every exact shape in a stable
    # order, rather than assuming one role signature per operation spec.
    reported = carrier.operation_capabilities("mul")

    assert {entry.operation for entry in reported} == {"mul"}
    assert {entry.output for entry in reported} == {DType.Float32, DType.Int32}
    assert any(
        any(operand.role is OperandRole.WEAK_SCALAR for operand in entry.operands)
        for entry in reported
    )
    assert any(
        all(operand.role is OperandRole.TENSOR for operand in entry.operands)
        for entry in reported
    )
    assert reported == carrier.operation_capabilities("mul")


def test_an_unknown_operation_is_distinguished_from_an_unsupported_plan():
    carrier = sw.CPU(1)
    unsupported = dataclasses.replace(
        resolve_operation_plan("reduce_sum", I32), accumulation=None
    )

    assert carrier.operation_capabilities("no_such_operation") == ()
    assert carrier.unsupported_plan_reason(plan(operation="no_such_operation")) == (
        "CPU declares no operation-plan capability for 'no_such_operation'"
    )
    reason = carrier.unsupported_plan_reason(unsupported)
    assert reason is not None
    assert "for this 'reduce_sum' plan" in reason
    assert (
        carrier.unsupported_plan_reason(resolve_operation_plan("reduce_sum", I32))
        is None
    )


def test_a_carrier_requires_the_same_capability_execution_uses():
    carrier = sw.CPU(1)
    supported = resolve_operation_plan("matmul", I32, I32)

    required = carrier.require_operation_plan(supported)

    assert required in carrier.operation_capabilities("matmul")
    assert carrier.supports_operation_plan(supported)
    with pytest.raises(UnsupportedOperationPlan, match="CPU declares no"):
        carrier.require_operation_plan(
            dataclasses.replace(supported, accumulation=None)
        )


def test_a_reported_capability_is_immutable_and_exposes_its_fields():
    (capability,) = [
        entry
        for entry in sw.CPU(1).operation_capabilities("reduce_sum")
        if entry.output is I32
    ]

    assert capability.compute is Arithmetic.INT32_EXACT_CHECKED
    assert capability.accumulation is Accumulation.EXACT_INTEGER
    assert capability.operands[0].dtype is I32
    with pytest.raises(dataclasses.FrozenInstanceError):
        capability.output = F32  # type: ignore[misc]


class CountingCarrier(sw.Carrier):
    """An independent carrier composing CPU storage.

    The shipped concrete carriers are closed, so a custom backend is a sibling
    `Carrier` owning one rather than a subclass of it. Capability registration
    remains exact-class state even when this discovery-only fixture implements
    no operation factories.
    """

    def __init__(self, size, dtype=I32):
        super().__init__()
        self._inner = sw.CPU(size, dtype=dtype)

    def size(self):
        return self._inner.size()

    def dtype(self):
        return self._inner.dtype()

    def get_value(self, index):
        return self._inner[index]

    def _dispatch_op(self, operation_name):
        raise NotImplementedError(operation_name)


def test_a_custom_carrier_is_discoverable_without_altering_a_built_in():
    extra = OperationCapability.from_plan(
        plan(operation="count", operands=((I32, I32),), output=I32)
    )
    register_operation_capabilities(CountingCarrier, [extra])
    custom = CountingCarrier(1)

    assert custom.operation_capabilities("count") == (extra,)
    assert sw.CPU(1).operation_capabilities("count") == ()
    # A custom carrier declares its own reach and nothing else: the backend it
    # lowers onto keeps declaring exactly what its kernels execute.
    relu = resolve_operation_plan("relu", I32)
    assert not custom.supports_operation_plan(relu)
    assert sw.CPU(1, dtype=I32).supports_operation_plan(relu)


def test_capability_queries_expose_no_executor_or_mutable_registry():
    carrier = sw.CPU(1)
    reported = carrier.operation_capabilities()

    assert not hasattr(reported, "append")
    assert all(not hasattr(entry, "forward") for entry in reported)
    # Mutating what a query returned cannot change what the backend executes.
    with pytest.raises(TypeError):
        reported[0] = None  # type: ignore[index]


def test_the_public_capability_stub_names_the_real_contracts():
    # A typed user discovers descriptor fields and has an invalid plan argument
    # rejected statically, so the stub must name OperationPlan and
    # OperationCapability rather than Any on all four query methods.
    stub_path = Path(__file__).parents[1] / "src/strideweave/_carrier.pyi"
    tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
    (carrier,) = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Carrier"
    ]

    def unparse(annotation):
        assert annotation is not None
        return ast.unparse(annotation)

    annotations = {
        node.name: (
            [unparse(argument.annotation) for argument in node.args.args[1:]],
            unparse(node.returns),
        )
        for node in carrier.body
        if isinstance(node, ast.FunctionDef)
    }

    assert annotations["operation_capabilities"] == (
        ["str | None"],
        "tuple[OperationCapability, ...]",
    )
    assert annotations["supports_operation_plan"] == (["OperationPlan"], "bool")
    assert annotations["unsupported_plan_reason"] == (["OperationPlan"], "str | None")
    assert annotations["require_operation_plan"] == (
        ["OperationPlan"],
        "OperationCapability",
    )


# --- Carriers whose capabilities depend on the carriers they compose ---------
#
# A dependent carrier owns its capabilities per constructed instance: two
# hierarchies of one class reach different carriers and therefore execute
# different plans. The lifecycle is cooperative rather than automatic, so these
# tests pin when an instance starts answering and what a failed generation
# leaves behind.


RELU_I32 = OperationCapability.from_plan(resolve_operation_plan("relu", I32))
RELU_F32 = OperationCapability.from_plan(resolve_operation_plan("relu", F32))


def dependent_class(name="ExampleDependent", generate=None, base=None):
    """Return a dependent carrier finalizing ``generate()`` during construction."""

    def __init__(self, entries=()):
        sw.DependentCarrier.__init__(self)
        self.entries = tuple(entries)
        self._finalize_dependent_capabilities()

    return type(
        name,
        (base or sw.DependentCarrier,),
        {
            "__init__": __init__,
            "_generate_operation_capabilities": generate or (lambda self: self.entries),
        },
    )


def test_a_dependent_instance_answers_with_what_it_generated():
    dependent = dependent_class()

    carrier = dependent([RELU_I32])

    assert carrier._operation_capability_snapshot() == (RELU_I32,)
    assert carrier._operation_capability_snapshot("relu") == (RELU_I32,)
    assert carrier._operation_capability_snapshot("no_such_operation") == ()


def test_two_instances_of_one_dependent_class_freeze_different_capabilities():
    # The point of the model: one class, two dependency sets, two answers.
    dependent = dependent_class()

    integer = dependent([RELU_I32])
    floating = dependent([RELU_F32])

    assert integer._operation_capability_snapshot() == (RELU_I32,)
    assert floating._operation_capability_snapshot() == (RELU_F32,)
    assert capabilities_for_carrier_class(dependent) == ()


def test_a_dependent_instance_answers_nothing_before_it_finalizes():
    # An unfinished hierarchy is not a hierarchy that supports nothing: it is
    # not ready to be asked, and saying so is what keeps a partially built
    # carrier from advertising an empty set some caller then trusts.
    unfinalized = type("Unfinalized", (sw.DependentCarrier,), {})()

    with pytest.raises(RuntimeError, match="has not finalized"):
        unfinalized._operation_capability_snapshot()


def test_the_base_constructor_never_generates_capabilities():
    generated = []

    def generate(self):
        generated.append(self)
        return ()

    bare = type(
        "BareDependent",
        (sw.DependentCarrier,),
        {"_generate_operation_capabilities": generate},
    )()

    assert generated == []

    bare._finalize_dependent_capabilities()

    assert generated == [bare]


def test_finalizing_a_second_time_is_refused_and_changes_nothing():
    dependent = dependent_class()
    carrier = dependent([RELU_I32])
    carrier.entries = (RELU_F32,)

    with pytest.raises(RuntimeError, match="already finalized"):
        carrier._finalize_dependent_capabilities()

    assert carrier._operation_capability_snapshot() == (RELU_I32,)


def test_the_base_generator_is_unimplemented():
    # The algorithm is the concrete carrier's own; the base supplies none.
    incomplete = type("Incomplete", (sw.DependentCarrier,), {})()

    with pytest.raises(NotImplementedError, match="Incomplete must implement"):
        incomplete._finalize_dependent_capabilities()
    with pytest.raises(RuntimeError, match="has not finalized"):
        incomplete._operation_capability_snapshot()


def test_a_generator_that_raises_publishes_nothing():
    def generate(self):
        raise ValueError("dependency is not ready")

    failing = type(
        "FailingDependent",
        (sw.DependentCarrier,),
        {"_generate_operation_capabilities": generate},
    )()

    with pytest.raises(ValueError, match="dependency is not ready"):
        failing._finalize_dependent_capabilities()
    with pytest.raises(RuntimeError, match="has not finalized"):
        failing._operation_capability_snapshot()


@pytest.mark.parametrize(
    ("generated", "error", "message"),
    [
        ([RELU_I32, "relu"], TypeError, "must be OperationCapability entries"),
        ([RELU_I32, RELU_I32], ValueError, "generated two capabilities"),
    ],
    ids=["not-a-capability", "duplicate-shape"],
)
def test_an_invalid_generated_set_leaves_the_instance_unfinalized(
    generated, error, message
):
    invalid = type(
        "InvalidDependent",
        (sw.DependentCarrier,),
        {"_generate_operation_capabilities": lambda self: generated},
    )()

    with pytest.raises(error, match=message):
        invalid._finalize_dependent_capabilities()
    with pytest.raises(RuntimeError, match="has not finalized"):
        invalid._operation_capability_snapshot()


def test_a_one_shot_generator_is_materialized_once():
    calls = []

    def generate(self):
        calls.append(None)
        return iter([RELU_F32, RELU_I32])

    dependent = dependent_class(generate=generate)
    carrier = dependent()

    assert len(calls) == 1
    assert carrier._operation_capability_snapshot() == (RELU_F32, RELU_I32)
    assert carrier._operation_capability_snapshot() == (RELU_F32, RELU_I32)


def test_a_snapshot_is_ordered_by_name_rather_than_by_generation_order():
    forward = dependent_class()([RELU_F32, RELU_I32])
    reversed_generation = dependent_class()([RELU_I32, RELU_F32])

    assert (
        forward._operation_capability_snapshot()
        == reversed_generation._operation_capability_snapshot()
        == (RELU_F32, RELU_I32)
    )


def test_a_snapshot_is_an_immutable_view_of_immutable_entries():
    carrier = dependent_class()([RELU_I32])

    reported = carrier._operation_capability_snapshot()

    assert isinstance(reported, tuple)
    assert reported[0] is RELU_I32
    with pytest.raises(dataclasses.FrozenInstanceError):
        reported[0].output = F32  # type: ignore[misc]


def test_a_dependent_carrier_stays_open_to_further_subclassing():
    # DependentCarrier is an extension interface: a dependent carrier may be
    # specialized, and each level's instances still answer for themselves.
    dependent = dependent_class("OpenDependent")
    specialized = dependent_class("SpecializedDependent", base=dependent)

    assert issubclass(specialized, dependent)
    assert specialized([RELU_F32])._operation_capability_snapshot() == (RELU_F32,)
    assert dependent([RELU_I32])._operation_capability_snapshot() == (RELU_I32,)


def test_instances_comparing_equal_keep_their_own_snapshots():
    # Ownership is per instance, and an extension interface cannot assume how a
    # subclass defines equality: two carriers that compare equal must not share
    # or overwrite one another's frozen capabilities.
    dependent = dependent_class(
        "EqualDependent",
        base=type(
            "AlwaysEqual",
            (sw.DependentCarrier,),
            {"__eq__": lambda self, other: True, "__hash__": lambda self: 7},
        ),
    )

    integer = dependent([RELU_I32])
    floating = dependent([RELU_F32])

    assert integer == floating
    assert integer._operation_capability_snapshot() == (RELU_I32,)
    assert floating._operation_capability_snapshot() == (RELU_F32,)


def test_a_collected_dependent_carrier_leaves_no_snapshot_behind():
    dependent = dependent_class()
    # Carriers other tests built may still be reachable from a traceback or a
    # cycle, so the baseline is what survives a collection rather than whatever
    # happens to be recorded now.
    gc.collect()
    before = len(operation_capability._FROZEN)

    carrier = dependent([RELU_I32])
    assert len(operation_capability._FROZEN) == before + 1

    del carrier
    gc.collect()

    assert len(operation_capability._FROZEN) == before


@pytest.mark.parametrize(
    "declare",
    [
        register_operation_capabilities,
        operation_capability._declare_built_in_capabilities,
    ],
    ids=["public", "internal"],
)
def test_a_dependent_class_cannot_declare_class_capabilities(declare):
    # A dependent class knows nothing about the carriers its instances will
    # compose, so a class-level declaration would answer for hierarchies it has
    # never seen.
    dependent = dependent_class("UndeclarableDependent")

    with pytest.raises(TypeError, match="is a DependentCarrier implementation"):
        declare(dependent, [RELU_I32])

    assert capabilities_for_carrier_class(dependent) == ()
    assert dependent([RELU_F32])._operation_capability_snapshot() == (RELU_F32,)


def test_the_instance_snapshot_path_takes_a_dependent_carrier():
    with pytest.raises(TypeError, match="must be a DependentCarrier instance"):
        operation_capability._instance_capabilities(sw.CPU(1))  # type: ignore[arg-type]


# --- One question, whichever model owns the answer ---------------------------
#
# A caller asks the carrier. Whether the answer comes from an exact class's
# sealed declarations or from a dependent instance's frozen snapshot is the
# carrier's business, and the same resolution serves introspection and the
# execution gate.


CARRIER_QUERIES = {
    "enumerate": lambda carrier: carrier.operation_capabilities(),
    "supports": lambda carrier: carrier.supports_operation_plan(
        resolve_operation_plan("relu", I32)
    ),
    "reason": lambda carrier: carrier.unsupported_plan_reason(
        resolve_operation_plan("relu", I32)
    ),
    "require": lambda carrier: carrier.require_operation_plan(
        resolve_operation_plan("relu", I32)
    ),
}


def test_a_dependent_carrier_answers_for_its_instance_not_its_class():
    dependent = dependent_class("ReportingDependent")
    integer = dependent([RELU_I32])
    floating = dependent([RELU_F32])

    assert integer.operation_capabilities() == (RELU_I32,)
    assert floating.operation_capabilities() == (RELU_F32,)
    assert integer.operation_capabilities("relu") == (RELU_I32,)
    # The class itself declares nothing, and querying the instances did not
    # make it declare anything either.
    assert capabilities_for_carrier_class(dependent) == ()


def test_a_dependent_carriers_public_answers_agree_with_each_other():
    dependent = dependent_class("AgreeingDependent")
    relu = resolve_operation_plan("relu", I32)

    supported = dependent([RELU_I32])
    unsupported = dependent([RELU_F32])

    assert supported.supports_operation_plan(relu)
    assert supported.unsupported_plan_reason(relu) is None
    assert supported.require_operation_plan(relu) is RELU_I32
    assert supported.require_operation_plan(relu) in supported.operation_capabilities()

    assert not unsupported.supports_operation_plan(relu)
    # This instance froze the Float32 relu shape, so the refusal is about the
    # shape rather than about the operation.
    assert unsupported.unsupported_plan_reason(relu) == (
        "AgreeingDependent declares no operation-plan capability for this 'relu' "
        "plan: relu with operands (tensor Int32->Int32), int32_exact compute, "
        "no accumulation, Int32 output"
    )
    with pytest.raises(UnsupportedOperationPlan, match="AgreeingDependent declares no"):
        unsupported.require_operation_plan(relu)


def test_a_dependent_carrier_distinguishes_an_unknown_operation_from_a_shape():
    dependent = dependent_class("DiscriminatingDependent")
    carrier = dependent([RELU_I32])

    reason = carrier.unsupported_plan_reason(resolve_operation_plan("relu", F32))

    assert reason is not None
    assert "for this 'relu' plan" in reason
    assert carrier.unsupported_plan_reason(
        resolve_operation_plan("reduce_sum", I32)
    ) == (
        "DiscriminatingDependent declares no operation-plan capability for 'reduce_sum'"
    )
    assert carrier.operation_capabilities("reduce_sum") == ()


@pytest.mark.parametrize("query", CARRIER_QUERIES.values(), ids=CARRIER_QUERIES)
def test_an_unfinalized_dependent_carrier_answers_no_public_query(query):
    unfinalized = type("UnfinalizedDependent", (sw.DependentCarrier,), {})()

    with pytest.raises(RuntimeError, match="has not finalized"):
        query(unfinalized)


@pytest.mark.parametrize("carrier", [sw.CPU(1, dtype=I32), sw.Generic([1])])
def test_an_independent_carrier_still_answers_from_its_exact_class(carrier):
    assert carrier.operation_capabilities() == capabilities_for_carrier_class(
        type(carrier)
    )
    relu = resolve_operation_plan("relu", I32)
    assert carrier.supports_operation_plan(relu) is supports_operation_plan(
        type(carrier), relu
    )
    assert carrier.unsupported_plan_reason(relu) == unsupported_plan_reason(
        type(carrier), relu
    )


def test_a_custom_independent_carrier_answers_from_its_own_class():
    # No public method is overridden and no class name is inspected: a new
    # backend gets the right answer by declaring, and a new dependent carrier by
    # generating.
    backend = carrier_class("AnsweringCarrier")
    register_operation_capabilities(backend, [RELU_I32])

    carrier = backend()

    assert carrier.operation_capabilities() == (RELU_I32,)
    assert carrier.supports_operation_plan(resolve_operation_plan("relu", I32))
    assert not carrier.supports_operation_plan(resolve_operation_plan("relu", F32))


def test_a_carrier_query_is_asked_of_an_instance_rather_than_a_class():
    with pytest.raises(TypeError, match="not a class"):
        operation_capability.carrier_operation_capabilities(sw.CPU)
    with pytest.raises(TypeError, match="must be a Carrier instance"):
        operation_capability.carrier_operation_capabilities(object())


def test_evictable_has_no_class_declaration_to_seal():
    # Evictable is shipped, but it is dependent: a class-level declaration
    # could only describe hierarchies it has never seen, so the bootstrap does
    # not make one and no caller can.
    assert capabilities_for_carrier_class(sw.Evictable) == ()
    assert sw.Evictable not in dict(built_in_capability_module._BUILT_INS)
    with pytest.raises(TypeError, match="is a DependentCarrier implementation"):
        register_operation_capabilities(sw.Evictable, [RELU_I32])
    with pytest.raises(TypeError, match="is a DependentCarrier implementation"):
        operation_capability._declare_built_in_capabilities(sw.Evictable, [])
