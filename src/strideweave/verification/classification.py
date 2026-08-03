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

_EXACT = (VerificationClass.EXACT_ARITHMETIC,)
_DEFERRED = (VerificationClass.DEFERRED,)
_STRUCTURAL = (VerificationClass.STRUCTURAL,)
_CERTIFIED_SUM = (
    VerificationClass.STRUCTURAL,
    VerificationClass.ANALYTIC,
    VerificationClass.NUMERICAL,
)

# The reason attached to every plan deferred because its result is produced by
# the platform math library rather than by an arithmetic StrideWeave pins.
VENDOR_TRANSCENDENTAL_REASON = "vendor transcendental implementation"

# Kernels whose result is a correctly rounded composition of pinned binary32 or
# exact-integer operations, so `Generic` and native CPU must agree bit for bit
# on arbitrary finite encoded inputs.
_EXACT_KERNELS = (
    "abs",
    "add",
    "argmax",
    "argmin",
    "ceil",
    "clamp",
    "div",
    "elementwise_mul",
    "eq",
    "floor",
    "gather",
    "le",
    "leaky_relu",
    "logical_not",
    "lt",
    "maximum",
    "minimum",
    "ne",
    "neg",
    "recip",
    "reduce_max",
    "reduce_min",
    "relu",
    "rem",
    "round",
    "rsqrt",
    "scalar_mul",
    "scatter",
    "select",
    "sign",
    "sort_indices",
    "sort_values",
    "sqrt",
    "sub",
    "topk_indices",
    "topk_values",
)

# Kernels that combine many floating terms under a normative order. Their
# payloads are chosen so every partial result is exactly representable, which
# checks traversal and addressing without pinning a particular association.
_STRUCTURAL_KERNELS = ("conv_general", "cumsum", "reduce_prod", "scatter_add")

# Kernels whose value comes from the platform math library. StrideWeave pins no
# accuracy contract for them yet, so they are deferred with a concrete reason
# rather than compared against a second implementation of the same library.
_TRANSCENDENTAL_KERNELS = (
    "cos",
    "elu",
    "erf",
    "exp",
    "exp2",
    "gelu",
    "log",
    "log2",
    "sigmoid",
    "silu",
    "sin",
    "softplus",
    "tanh",
)

_CLASSIFICATIONS: dict[tuple[str, str], tuple[VerificationClass, ...]] = {
    **{(f"cpu.{name}", "default"): _EXACT for name in _EXACT_KERNELS},
    **{(f"cpu.{name}", "default"): _STRUCTURAL for name in _STRUCTURAL_KERNELS},
    **{(f"cpu.{name}", "default"): _DEFERRED for name in _TRANSCENDENTAL_KERNELS},
    ("cpu.matmul", "default"): _CERTIFIED_SUM,
    ("cpu.reduce_sum", "default"): _CERTIFIED_SUM,
    # `pow` is one kernel with two dispositions: the checked-integer plan is
    # exact, while the floating plan calls the vendor `pow`.
    ("cpu.pow", "default"): (
        VerificationClass.EXACT_ARITHMETIC,
        VerificationClass.DEFERRED,
    ),
}


def native_cpu_kernel_manifest() -> tuple[KernelDescriptor, ...]:
    """Return the native CPU kernel metadata exposed by the compiled carrier.

    Args:
        None.

    Returns:
        Tuple of operation descriptors with stable kernel IDs and variants.

    Examples:
        >>> bool(native_cpu_kernel_manifest())
        True
    """
    return tuple(
        KernelDescriptor(*entry) for entry in _carrier._cpu_native_kernel_metadata()
    )


def classifications_for_kernel(
    kernel: KernelDescriptor,
) -> tuple[VerificationClass, ...]:
    """Return the required verification classes for one native kernel.

    Args:
        kernel: Native kernel descriptor to classify.

    Returns:
        Ordered verification classes required for certification.

    Examples:
        >>> bool(classifications_for_kernel(native_cpu_kernel_manifest()[0]))
        True
    """
    try:
        return _CLASSIFICATIONS[(kernel.kernel_id, kernel.variant)]
    except KeyError as error:
        raise ValueError(
            f"native kernel {kernel.kernel_id!r} has no verification classification"
        ) from error


def require_complete_classification(
    kernels: Iterable[KernelDescriptor],
) -> tuple[tuple[KernelDescriptor, tuple[VerificationClass, ...]], ...]:
    """Validate and return a one-to-one manifest/classification pairing.

    Args:
        kernels: Native kernel descriptors to check.

    Returns:
        Descriptors paired with their declared verification classes.

    Examples:
        >>> bool(require_complete_classification(native_cpu_kernel_manifest()))
        True
    """
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
    """Resolve executable CPU capability plans and their dispositions.

    Args:
        None.

    Returns:
        Tuple of active or explicitly deferred kernel plan descriptors.

    Examples:
        >>> all(plan.kernel.operation for plan in classify_cpu_kernel_plans())
        True
    """
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
        if classes == _DEFERRED:
            disposition = ClassificationDisposition.DEFERRED
            classes = ()
            deferred_reason = VENDOR_TRANSCENDENTAL_REASON
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
