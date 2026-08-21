"""Provider-neutral verification profile classification.

This module describes the finite, logical verification obligations of a
provider. Compilation specializations deliberately do not appear here: they
belong to the provenance receipt that records an individual execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

from strideweave.carriers.cpu.capabilities import cpu_capabilities
from strideweave.carriers.operation_capability import OperationCapability

from .model import (
    ClassificationDisposition,
    PlanKey,
    VerificationClass,
    VerificationStage,
)

_NATIVE_KERNEL_METADATA_BINDING = "_cpu_native_kernel_metadata"
_NATIVE_EXTENSION_REBUILD_COMMAND = (
    "uv sync --reinstall-package strideweave --group dev"
)

_EXACT = (VerificationClass.EXACT_ARITHMETIC,)
_DEFERRED = (VerificationClass.DEFERRED,)
_STRUCTURAL = (VerificationClass.STRUCTURAL,)
_CERTIFIED_SUM = (
    VerificationClass.STRUCTURAL,
    VerificationClass.ANALYTIC,
    VerificationClass.NUMERICAL,
)

# The reason attached to every plan deferred because its result is produced by
# the platform math library rather than by arithmetic StrideWeave pins.
VENDOR_TRANSCENDENTAL_REASON = "vendor transcendental implementation"

# Kernels whose result is a correctly rounded composition of pinned binary32 or
# exact-integer operations, so Generic and a provider implementation must agree
# bit for bit on arbitrary finite encoded inputs.
_EXACT_OPERATIONS = frozenset(
    {
        "abs",
        "add",
        "argmax",
        "argmin",
        "ceil",
        "clamp",
        "div",
        "_sort_indices",
        "_sort_values",
        "_topk_indices",
        "_topk_values",
        "elementwise_mul",
        "eq",
        "floor",
        "gather",
        "le",
        "leaky_relu",
        "logical_not",
        "lt",
        "mul",
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
        "scatter",
        "select",
        "sign",
        "sqrt",
        "sub",
    }
)
_STRUCTURAL_OPERATIONS = frozenset(
    {"conv_general", "cumsum", "reduce_prod", "scatter_add"}
)
_TRANSCENDENTAL_OPERATIONS = frozenset(
    {
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
    }
)


@dataclass(frozen=True, slots=True)
class VerificationProfile:
    """One registered provider/carrier verification profile."""

    profile_id: str
    carrier: str
    provider: str
    stages: tuple[VerificationStage, ...]


@dataclass(frozen=True, slots=True)
class LogicalKernel:
    """Stable provider-neutral identity of one finite verification kernel."""

    profile_id: str
    operation: str
    kernel_id: str
    variant: str


@dataclass(frozen=True, slots=True)
class PlanClassification:
    """One logical kernel's exact plan and verification disposition."""

    kernel: LogicalKernel
    plan: PlanKey
    classes: tuple[VerificationClass, ...]
    disposition: ClassificationDisposition
    deferred_reason: str | None = None


@dataclass(frozen=True, slots=True)
class VerificationSubject:
    """One non-numerical provider obligation classified before execution."""

    profile_id: str
    subject_kind: str
    operation: str
    classes: tuple[VerificationClass, ...]


@dataclass(frozen=True, slots=True)
class _ProfileDefinition:
    profile: VerificationProfile
    kernel_prefix: str
    capabilities: Callable[[], tuple[OperationCapability, ...]]


_CPU_COMPILED = VerificationProfile(
    "cpu-compiled",
    "CPU",
    "compiled-executable",
    (VerificationStage.ORACLE, VerificationStage.TARGET),
)
_SYNTHETIC_JIT = VerificationProfile(
    "synthetic-jit",
    "SyntheticJIT",
    "jit-specialization",
    (VerificationStage.TARGET,),
)
_PROFILE_DEFINITIONS = tuple(
    sorted(
        (
            _ProfileDefinition(_CPU_COMPILED, "cpu", cpu_capabilities),
            _ProfileDefinition(_SYNTHETIC_JIT, "synthetic-jit", cpu_capabilities),
        ),
        key=lambda definition: definition.profile.profile_id,
    )
)
_PROFILES_BY_ID = {
    definition.profile.profile_id: definition for definition in _PROFILE_DEFINITIONS
}
_MOVEMENT_OPERATIONS = (
    "move",
    "view",
    "permute",
    "rearrange",
    "broadcast_to",
)


def verification_profiles() -> tuple[VerificationProfile, ...]:
    """Return registered verification profiles in canonical profile-ID order.

    Args:
        None.

    Returns:
        Immutable registered provider/carrier profile records.

    Examples:
        >>> [profile.profile_id for profile in verification_profiles()]
        ['cpu-compiled', 'synthetic-jit']
    """
    return tuple(definition.profile for definition in _PROFILE_DEFINITIONS)


def verification_profile(profile_id: str) -> VerificationProfile:
    """Resolve one registered profile without initializing a provider runtime.

    Args:
        profile_id: Exact registered profile identifier.

    Returns:
        Immutable registered profile record.

    Examples:
        >>> verification_profile('cpu-compiled').provider
        'compiled-executable'
    """
    if type(profile_id) is not str:
        raise TypeError("verification profile ID must be a string")
    try:
        return _PROFILES_BY_ID[profile_id].profile
    except KeyError as error:
        raise ValueError(f"unknown verification profile {profile_id!r}") from error


def _definition(profile: VerificationProfile) -> _ProfileDefinition:
    if not isinstance(profile, VerificationProfile):
        raise TypeError("profile must be a VerificationProfile")
    try:
        definition = _PROFILES_BY_ID[profile.profile_id]
    except KeyError as error:
        raise ValueError(
            f"unregistered verification profile {profile.profile_id!r}"
        ) from error
    if profile != definition.profile:
        raise ValueError("verification profile facts do not match its registration")
    return definition


def _native_metadata() -> tuple[tuple[str, str, str, str, str], ...]:
    carrier = import_module("strideweave._carrier")
    try:
        binding = getattr(carrier, _NATIVE_KERNEL_METADATA_BINDING)
    except AttributeError as error:
        raise RuntimeError(
            "StrideWeave's native extension is stale or incompatible with its "
            "Python verification sources: required binding "
            f"{_NATIVE_KERNEL_METADATA_BINDING!r} is missing. Rebuild the active "
            f"environment with '{_NATIVE_EXTENSION_REBUILD_COMMAND}'."
        ) from error
    return tuple(binding())


def require_native_verification_api() -> None:
    """Require the installed native metadata binding used by the CPU profile."""
    _native_metadata()


def kernel_manifest(profile: VerificationProfile) -> tuple[LogicalKernel, ...]:
    """Return one profile's exact finite logical-kernel manifest.

    Args:
        profile: Registered provider/carrier verification profile.

    Returns:
        Immutable logical kernels sorted by kernel identifier and variant.

    Examples:
        >>> bool(kernel_manifest(verification_profile('cpu-compiled')))
        True
    """
    definition = _definition(profile)
    kernels = tuple(
        LogicalKernel(
            profile.profile_id,
            operation,
            f"{definition.kernel_prefix}.{operation}",
            variant,
        )
        for operation, _, variant, _, _ in _native_metadata()
    )
    ordered = tuple(
        sorted(kernels, key=lambda kernel: (kernel.kernel_id, kernel.variant))
    )
    _validate_manifest(profile, ordered)
    return ordered


def profile_subjects(profile: VerificationProfile) -> tuple[VerificationSubject, ...]:
    """Return every classified movement/structural subject for one profile.

    Args:
        profile: Registered provider/carrier verification profile.

    Returns:
        Immutable non-numerical verification obligations.

    Examples:
        >>> bool(profile_subjects(verification_profile('cpu-compiled')))
        True
    """
    _definition(profile)
    subjects = tuple(
        VerificationSubject(
            profile.profile_id,
            "movement",
            operation,
            (VerificationClass.BIT_EXACT,),
        )
        for operation in _MOVEMENT_OPERATIONS
    )
    _validate_subjects(profile, subjects)
    return subjects


def _validate_manifest(
    profile: VerificationProfile, kernels: tuple[LogicalKernel, ...]
) -> None:
    keys = tuple((kernel.kernel_id, kernel.variant) for kernel in kernels)
    if len(set(keys)) != len(keys):
        raise ValueError("logical kernel manifest contains a duplicate kernel/variant")
    if any(kernel.profile_id != profile.profile_id for kernel in kernels):
        raise ValueError("logical kernel manifest contains a mismatched profile")
    if any(
        not kernel.operation or not kernel.kernel_id or not kernel.variant
        for kernel in kernels
    ):
        raise ValueError("logical kernel manifest contains an incomplete kernel")


def _validate_subjects(
    profile: VerificationProfile, subjects: tuple[VerificationSubject, ...]
) -> None:
    keys = tuple((subject.subject_kind, subject.operation) for subject in subjects)
    if len(set(keys)) != len(keys):
        raise ValueError("profile subjects contain a duplicate subject")
    expected = {("movement", operation) for operation in _MOVEMENT_OPERATIONS}
    if set(keys) != expected:
        raise ValueError(
            "profile subjects do not exactly match required movement subjects: "
            f"missing={sorted(expected - set(keys))!r}, "
            f"stale={sorted(set(keys) - expected)!r}"
        )
    if any(subject.profile_id != profile.profile_id for subject in subjects):
        raise ValueError("profile subjects contain a mismatched profile")
    if any(subject.classes != (VerificationClass.BIT_EXACT,) for subject in subjects):
        raise ValueError("profile movement subjects require bit-exact classification")


def _classes_for(
    kernel: LogicalKernel, plan: PlanKey
) -> tuple[tuple[VerificationClass, ...], ClassificationDisposition, str | None]:
    if kernel.operation != plan.operation:
        raise ValueError(
            "classification plan operation does not match its logical kernel"
        )
    if kernel.operation in _EXACT_OPERATIONS:
        return (_EXACT, ClassificationDisposition.ACTIVE, None)
    if kernel.operation in _STRUCTURAL_OPERATIONS:
        return (_STRUCTURAL, ClassificationDisposition.ACTIVE, None)
    if kernel.operation in _TRANSCENDENTAL_OPERATIONS:
        return (
            _DEFERRED,
            ClassificationDisposition.DEFERRED,
            VENDOR_TRANSCENDENTAL_REASON,
        )
    if kernel.operation in {"matmul", "reduce_sum"}:
        return (_CERTIFIED_SUM, ClassificationDisposition.ACTIVE, None)
    if kernel.operation == "pow":
        if plan.compute == "BINARY32":
            return (
                _DEFERRED,
                ClassificationDisposition.DEFERRED,
                "floating pow depends on the vendor math library",
            )
        return (_EXACT, ClassificationDisposition.ACTIVE, None)
    raise ValueError(
        f"logical kernel {kernel.kernel_id!r} has no verification classification"
    )


def _validate_classification_catalog(manifest: tuple[LogicalKernel, ...]) -> None:
    declared_operations = (
        _EXACT_OPERATIONS
        | _STRUCTURAL_OPERATIONS
        | _TRANSCENDENTAL_OPERATIONS
        | {"matmul", "pow", "reduce_sum"}
    )
    declared_keys = {(operation, "default") for operation in declared_operations}
    manifest_keys = {(kernel.operation, kernel.variant) for kernel in manifest}
    missing = manifest_keys - declared_keys
    stale = declared_keys - manifest_keys
    if missing or stale:
        raise ValueError(
            "verification classifications do not exactly match the logical "
            f"manifest: missing={sorted(missing)!r}, stale={sorted(stale)!r}"
        )


def classify_profile_plans(
    profile: VerificationProfile,
) -> tuple[PlanClassification, ...]:
    """Return every capability plan classified by one registered profile.

    The complete manifest, capability declaration, plan identities, and
    non-numerical subjects are validated before returning any executable fact.

    Args:
        profile: Registered provider/carrier verification profile.

    Returns:
        Immutable plan classifications in deterministic logical-plan order.

    Examples:
        >>> bool(classify_profile_plans(verification_profile('cpu-compiled')))
        True
    """
    definition = _definition(profile)
    manifest = kernel_manifest(profile)
    _validate_classification_catalog(manifest)
    by_operation = {kernel.operation: kernel for kernel in manifest}
    if len(by_operation) != len(manifest):
        raise ValueError("logical kernel manifest has duplicate operations")
    capabilities = tuple(definition.capabilities())
    classifications: list[PlanClassification] = []
    seen_plan_keys: set[tuple[LogicalKernel, PlanKey]] = set()
    seen_operations: set[str] = set()
    for capability in capabilities:
        operation = capability.operation
        try:
            kernel = by_operation[operation]
        except KeyError as error:
            raise ValueError(
                f"profile capability {operation!r} has no logical kernel"
            ) from error
        plan = PlanKey.from_plan_like(capability)
        classes, disposition, reason = _classes_for(kernel, plan)
        if disposition is ClassificationDisposition.DEFERRED:
            classes = ()
        key = (kernel, plan)
        if key in seen_plan_keys:
            raise ValueError("profile capability declaration contains a duplicate plan")
        seen_plan_keys.add(key)
        seen_operations.add(operation)
        classifications.append(
            PlanClassification(kernel, plan, classes, disposition, reason)
        )
    missing_operations = set(by_operation) - seen_operations
    if missing_operations:
        raise ValueError(
            "logical kernels have no executable profile plan: "
            f"{sorted(missing_operations)!r}"
        )
    _validate_subjects(profile, profile_subjects(profile))
    _validate_plan_classifications(profile, tuple(classifications))
    return tuple(
        sorted(
            classifications,
            key=lambda item: (
                item.kernel.kernel_id,
                item.kernel.variant,
                item.plan.operands,
                item.plan.compute,
                item.plan.accumulation or "",
                item.plan.accumulator_dtype or "",
                item.plan.output,
            ),
        )
    )


def _validate_plan_classifications(
    profile: VerificationProfile, classifications: tuple[PlanClassification, ...]
) -> None:
    keys = tuple(
        (classification.kernel, classification.plan)
        for classification in classifications
    )
    if len(set(keys)) != len(keys):
        raise ValueError("profile classifications contain a duplicate logical plan")
    for classification in classifications:
        if classification.kernel.profile_id != profile.profile_id:
            raise ValueError("plan classification contains a mismatched profile")
        if classification.kernel.operation != classification.plan.operation:
            raise ValueError("plan classification operation does not match its kernel")
        if classification.disposition is ClassificationDisposition.ACTIVE:
            if not classification.classes:
                raise ValueError(
                    "active plan classification has no verification classes"
                )
            if VerificationClass.DEFERRED in classification.classes:
                raise ValueError("active plan classification cannot require deferred")
            if len(set(classification.classes)) != len(classification.classes):
                raise ValueError("active plan classification has duplicate classes")
            if classification.deferred_reason is not None:
                raise ValueError("active plan classification has a deferred reason")
            continue
        if classification.disposition is ClassificationDisposition.DEFERRED:
            if classification.classes:
                raise ValueError("deferred plan classification has active classes")
            if not classification.deferred_reason:
                raise ValueError("deferred plan classification has no reason")
            continue
        raise ValueError("plan classification has an unknown disposition")
