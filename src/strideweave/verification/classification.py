"""Explicit verification classifications for the native CPU manifest."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module

from strideweave.carriers.cpu.capabilities import cpu_capabilities

from .model import (
    ClassificationDisposition,
    KernelDescriptor,
    KernelPlanDescriptor,
    PlanKey,
    VerificationClass,
)

_carrier = import_module("strideweave._carrier")

_CLASSIFICATIONS: dict[tuple[str, str], tuple[VerificationClass, ...]] = {
    ("cpu.add", "default"): (VerificationClass.EXACT_ARITHMETIC,),
    ("cpu.div", "default"): (VerificationClass.EXACT_ARITHMETIC,),
    ("cpu.elementwise_mul", "default"): (VerificationClass.EXACT_ARITHMETIC,),
    ("cpu.elu", "default"): (VerificationClass.DEFERRED,),
    ("cpu.exp", "default"): (VerificationClass.DEFERRED,),
    ("cpu.gelu", "default"): (VerificationClass.DEFERRED,),
    ("cpu.leaky_relu", "default"): (VerificationClass.EXACT_ARITHMETIC,),
    ("cpu.matmul", "default"): (
        VerificationClass.STRUCTURAL,
        VerificationClass.ANALYTIC,
        VerificationClass.NUMERICAL,
    ),
    ("cpu.scalar_mul", "default"): (VerificationClass.EXACT_ARITHMETIC,),
    ("cpu.pow", "default"): (
        VerificationClass.EXACT_ARITHMETIC,
        VerificationClass.DEFERRED,
    ),
    ("cpu.reduce_sum", "default"): (
        VerificationClass.STRUCTURAL,
        VerificationClass.ANALYTIC,
        VerificationClass.NUMERICAL,
    ),
    ("cpu.relu", "default"): (VerificationClass.EXACT_ARITHMETIC,),
    ("cpu.sigmoid", "default"): (VerificationClass.DEFERRED,),
    ("cpu.silu", "default"): (VerificationClass.DEFERRED,),
    ("cpu.softplus", "default"): (VerificationClass.DEFERRED,),
    ("cpu.sub", "default"): (VerificationClass.EXACT_ARITHMETIC,),
    ("cpu.tanh", "default"): (VerificationClass.DEFERRED,),
}


def native_cpu_kernel_manifest() -> tuple[KernelDescriptor, ...]:
    return tuple(
        KernelDescriptor(*entry) for entry in _carrier._cpu_native_kernel_metadata()
    )


def classifications_for_kernel(
    kernel: KernelDescriptor,
) -> tuple[VerificationClass, ...]:
    try:
        return _CLASSIFICATIONS[(kernel.kernel_id, kernel.variant)]
    except KeyError as error:
        raise ValueError(
            f"native kernel {kernel.kernel_id!r} has no verification classification"
        ) from error


def require_complete_classification(
    kernels: Iterable[KernelDescriptor],
) -> tuple[tuple[KernelDescriptor, tuple[VerificationClass, ...]], ...]:
    materialized = tuple(kernels)
    keys = tuple((kernel.kernel_id, kernel.variant) for kernel in materialized)
    if len(set(keys)) != len(keys):
        raise ValueError("native kernel manifest contains a duplicate kernel/variant")
    missing = set(keys) - _CLASSIFICATIONS.keys()
    stale = _CLASSIFICATIONS.keys() - set(keys)
    if missing or stale:
        raise ValueError(
            "verification classification does not exactly match native manifest: "
            f"missing={sorted(missing)!r}, stale={sorted(stale)!r}"
        )
    return tuple(
        (kernel, classifications_for_kernel(kernel)) for kernel in materialized
    )


MOVEMENT_CLASSIFICATIONS = {
    operation: (VerificationClass.BIT_EXACT,)
    for operation in ("move", "view", "permute", "rearrange", "broadcast_to")
}


def classify_cpu_kernel_plans() -> tuple[KernelPlanDescriptor, ...]:
    manifest = native_cpu_kernel_manifest()
    require_complete_classification(manifest)
    by_operation = {kernel.operation: kernel for kernel in manifest}
    descriptors: list[KernelPlanDescriptor] = []
    seen_operations: set[str] = set()
    for capability in cpu_capabilities():
        kernel = by_operation.get(capability.operation)
        if kernel is None:
            raise ValueError(
                f"CPU capability {capability.operation!r} has no native kernel metadata"
            )
        seen_operations.add(capability.operation)
        plan = PlanKey.from_plan_like(capability)
        classes = classifications_for_kernel(kernel)
        deferred_reason = None
        disposition = ClassificationDisposition.ACTIVE
        if classes == (VerificationClass.DEFERRED,):
            disposition = ClassificationDisposition.DEFERRED
            classes = ()
            deferred_reason = "vendor transcendental implementation"
        elif capability.operation == "pow" and capability.compute.name == "BINARY32":
            disposition = ClassificationDisposition.DEFERRED
            classes = ()
            deferred_reason = "floating pow depends on the vendor math library"
        elif capability.operation == "pow":
            classes = (VerificationClass.EXACT_ARITHMETIC,)
        descriptors.append(
            KernelPlanDescriptor(kernel, plan, classes, disposition, deferred_reason)
        )
    missing_operations = set(by_operation) - seen_operations
    if missing_operations:
        raise ValueError(
            f"native kernels have no executable CPU plan: {sorted(missing_operations)!r}"
        )
    return tuple(descriptors)
