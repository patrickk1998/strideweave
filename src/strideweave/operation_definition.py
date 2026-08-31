"""Carrier-neutral operation definitions for definition-backed dispatch."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Final, cast

from .carriers.dtype import DType
from .carriers.operation_capability import UnsupportedOperationPlan
from .carriers.operation_policy import OperandRole, OperationPlan
from .layout import Layout, Shape
from .tensor import Tensor


class ProviderContractError(RuntimeError):
    """Report provider output that conflicts with resolved operation semantics."""


class OperandKind(str, Enum):
    """Classify one public operation operand before semantic resolution."""

    TENSOR = "tensor"
    WEAK_SCALAR = "weak_scalar"
    STRUCTURAL = "structural"


class _Marker:
    """Represent one immutable public operation-contract marker."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def __repr__(self) -> str:
        return self._name

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("operation markers are immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError("operation markers are immutable")


REQUIRED: Final = _Marker("REQUIRED")
NON_DIFFERENTIABLE: Final = _Marker("NON_DIFFERENTIABLE")
_VJP_OMITTED: Final = _Marker("_VJP_OMITTED")


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or not value.isidentifier():
        raise ValueError(f"{name} must be a non-empty Python identifier")
    return value


def _materialize(value: object, name: str) -> tuple[object, ...]:
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be a finite iterable") from error


def _require_index_tuple(value: object, name: str) -> tuple[int, ...]:
    entries = _materialize(value, name)
    for entry in entries:
        if type(entry) is not int:
            raise TypeError(f"{name} must contain integers")
        if entry < 0:
            raise ValueError(f"{name} must contain non-negative indices")
    normalized = cast(tuple[int, ...], entries)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must not contain duplicate indices")
    return normalized


def _is_weak_scalar(value: object) -> bool:
    return isinstance(value, Real)


@dataclass(frozen=True, slots=True)
class OperandSpec:
    """Describe one positionally bound operation operand."""

    name: str
    kind: OperandKind
    differentiable: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.name, "name")
        if not isinstance(self.kind, OperandKind):
            raise TypeError("kind must be an OperandKind")
        if type(self.differentiable) is not bool:
            raise TypeError("differentiable must be a bool")
        if self.differentiable and self.kind is not OperandKind.TENSOR:
            raise ValueError("only a Tensor operand may be differentiable")


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """Describe one exactly named operation option and its optional default."""

    name: str
    default: object = REQUIRED

    def __post_init__(self) -> None:
        _require_identifier(self.name, "name")


@dataclass(frozen=True, slots=True, init=False)
class OperationSchema:
    """Materialize the ordered operand and option schema for an operation."""

    operands: tuple[OperandSpec, ...]
    options: tuple[OptionSpec, ...]

    def __init__(
        self,
        operands: Iterable[OperandSpec],
        options: Iterable[OptionSpec] = (),
    ) -> None:
        normalized_operands = _materialize(operands, "operands")
        normalized_options = _materialize(options, "options")
        if not normalized_operands:
            raise ValueError("operands must not be empty")
        if any(not isinstance(entry, OperandSpec) for entry in normalized_operands):
            raise TypeError("operands must contain OperandSpec values")
        if any(not isinstance(entry, OptionSpec) for entry in normalized_options):
            raise TypeError("options must contain OptionSpec values")
        typed_operands = cast(tuple[OperandSpec, ...], normalized_operands)
        typed_options = cast(tuple[OptionSpec, ...], normalized_options)
        names = [entry.name for entry in (*typed_operands, *typed_options)]
        if len(names) != len(set(names)):
            raise ValueError("operand and option names must be unique")
        object.__setattr__(self, "operands", typed_operands)
        object.__setattr__(self, "options", typed_options)


@dataclass(frozen=True, slots=True, init=False)
class OperationCall:
    """Capture one unbound operation call before schema validation."""

    operands: tuple[object, ...]
    options: Mapping[str, object]

    def __init__(
        self, operands: Iterable[object], options: Mapping[str, object] | None = None
    ) -> None:
        normalized_operands = _materialize(operands, "operands")
        if options is None:
            normalized_options: dict[str, object] = {}
        elif isinstance(options, Mapping):
            normalized_options = dict(options)
        else:
            raise TypeError("options must be a mapping or None")
        if any(not isinstance(name, str) for name in normalized_options):
            raise TypeError("option names must be strings")
        object.__setattr__(self, "operands", normalized_operands)
        object.__setattr__(self, "options", MappingProxyType(normalized_options))


_BOUND_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class BoundOperationCall:
    """Expose one framework-bound immutable call to semantic resolution."""

    definition: OperationDefinition
    operands: tuple[object, ...]
    options: Mapping[str, object]
    operand_kinds: tuple[OperandKind, ...]

    def __init__(
        self,
        definition: OperationDefinition,
        operands: tuple[object, ...],
        options: Mapping[str, object],
        operand_kinds: tuple[OperandKind, ...],
        *,
        _token: object = None,
    ) -> None:
        if _token is not _BOUND_TOKEN:
            raise TypeError("BoundOperationCall values are created by the framework")
        object.__setattr__(self, "definition", definition)
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "options", MappingProxyType(dict(options)))
        object.__setattr__(self, "operand_kinds", operand_kinds)


def _normalized_operand_kind(spec: OperandSpec, value: object) -> OperandKind:
    if spec.kind is OperandKind.TENSOR:
        if not isinstance(value, Tensor):
            raise TypeError(f"{spec.name} must be a Tensor")
        return OperandKind.TENSOR
    if spec.kind is OperandKind.WEAK_SCALAR:
        if isinstance(value, Tensor) or not _is_weak_scalar(value):
            raise TypeError(f"{spec.name} must be a real Python weak scalar")
        return OperandKind.WEAK_SCALAR
    if isinstance(value, Tensor):
        return OperandKind.TENSOR
    if _is_weak_scalar(value):
        return OperandKind.WEAK_SCALAR
    return OperandKind.STRUCTURAL


def _bind(definition: OperationDefinition, call: OperationCall) -> BoundOperationCall:
    schema = definition.schema
    expected = len(schema.operands)
    actual = len(call.operands)
    if actual != expected:
        raise TypeError(
            f"{definition.name} expects {expected} operands but received {actual}"
        )
    declared_options = {entry.name: entry for entry in schema.options}
    unknown = tuple(name for name in call.options if name not in declared_options)
    if unknown:
        raise TypeError(f"unknown option {unknown[0]!r} for {definition.name}")
    bound_options: dict[str, object] = {}
    for option in schema.options:
        if option.name in call.options:
            bound_options[option.name] = call.options[option.name]
        elif option.default is REQUIRED:
            raise TypeError(f"missing required option {option.name!r}")
        else:
            bound_options[option.name] = option.default
    kinds = tuple(
        _normalized_operand_kind(spec, value)
        for spec, value in zip(schema.operands, call.operands, strict=True)
    )
    return BoundOperationCall(
        definition,
        call.operands,
        bound_options,
        kinds,
        _token=_BOUND_TOKEN,
    )


def _require_tolerance(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


@dataclass(frozen=True, slots=True)
class ResultSpec:
    """Declare one result's dtype, logical layout, aliasing, and tolerance."""

    dtype: DType
    shape: Shape
    layout: Layout
    alias_of: int | None = None
    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DType):
            raise TypeError("dtype must be a DType")
        if not isinstance(self.shape, Shape):
            raise TypeError("shape must be a Shape")
        if not isinstance(self.layout, Layout):
            raise TypeError("layout must be a Layout")
        if self.layout.shape != self.shape:
            raise ValueError("layout Shape must match shape")
        if self.alias_of is not None:
            if type(self.alias_of) is not int:
                raise TypeError("alias_of must be an integer or None")
            if self.alias_of < 0:
                raise ValueError("alias_of must be non-negative")
        object.__setattr__(self, "atol", _require_tolerance(self.atol, "atol"))
        object.__setattr__(self, "rtol", _require_tolerance(self.rtol, "rtol"))


def _validate_index(call: BoundOperationCall, index: int, field: str) -> Tensor:
    if index >= len(call.operands):
        raise ValueError(f"{field} contains an out-of-range operand index")
    operand = call.operands[index]
    if not isinstance(operand, Tensor):
        raise ValueError(f"{field} must identify Tensor operands")
    return operand


def _planned_operand_kinds(call: BoundOperationCall) -> tuple[OperandKind, ...]:
    return tuple(
        kind for kind in call.operand_kinds if kind is not OperandKind.STRUCTURAL
    )


@dataclass(frozen=True, slots=True, init=False)
class ResolvedInvocation:
    """Freeze every semantic and result obligation before carrier selection."""

    call: BoundOperationCall
    plan: OperationPlan
    results: tuple[ResultSpec, ...]
    mutated_operands: tuple[int, ...]
    saved_operands: tuple[int, ...]
    kernel_id: str

    def __init__(
        self,
        call: BoundOperationCall,
        plan: OperationPlan,
        results: Iterable[ResultSpec],
        mutated_operands: Iterable[int] = (),
        saved_operands: Iterable[int] = (),
        kernel_id: str | None = None,
    ) -> None:
        if not isinstance(call, BoundOperationCall):
            raise TypeError("call must be a framework-created BoundOperationCall")
        if not isinstance(plan, OperationPlan):
            raise TypeError("plan must be an OperationPlan")
        normalized_results = _materialize(results, "results")
        if not normalized_results:
            raise ValueError("results must not be empty")
        if any(not isinstance(entry, ResultSpec) for entry in normalized_results):
            raise TypeError("results must contain ResultSpec values")
        normalized_mutated = _require_index_tuple(mutated_operands, "mutated_operands")
        normalized_saved = _require_index_tuple(saved_operands, "saved_operands")
        if plan.operation != call.definition.name:
            raise ValueError("plan operation must match the operation definition")
        roles = tuple(
            OperandKind.TENSOR
            if entry.role is OperandRole.TENSOR
            else OperandKind.WEAK_SCALAR
            for entry in plan.operands
        )
        if roles != _planned_operand_kinds(call):
            raise ValueError("plan operand roles must match the bound call")
        for index in (*normalized_mutated, *normalized_saved):
            _validate_index(call, index, "operand indices")
        typed_results = cast(tuple[ResultSpec, ...], normalized_results)
        for result in typed_results:
            if result.alias_of is not None:
                _validate_index(call, result.alias_of, "alias_of")
        resolved_kernel_id = call.definition.name if kernel_id is None else kernel_id
        if not isinstance(resolved_kernel_id, str):
            raise TypeError("kernel_id must be a string or None")
        if not resolved_kernel_id:
            raise ValueError("kernel_id must not be empty")
        object.__setattr__(self, "call", call)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "results", typed_results)
        object.__setattr__(self, "mutated_operands", normalized_mutated)
        object.__setattr__(self, "saved_operands", normalized_saved)
        object.__setattr__(self, "kernel_id", resolved_kernel_id)

    @property
    def definition(self) -> OperationDefinition:
        """Return the semantic definition that owns this invocation."""

        return self.call.definition

    @property
    def name(self) -> str:
        """Return the canonical operation name."""

        return self.definition.name

    @property
    def operands(self) -> tuple[object, ...]:
        """Return normalized operands in public schema order."""

        return self.call.operands

    @property
    def options(self) -> Mapping[str, object]:
        """Return the read-only execution-option mapping."""

        return self.call.options


@dataclass(frozen=True, slots=True, init=False)
class VJPContext:
    """Expose immutable forward state to one semantic VJP callback."""

    invocation: ResolvedInvocation
    outputs: tuple[Tensor, ...]
    saved_operands: tuple[Tensor, ...]

    def __init__(
        self,
        invocation: ResolvedInvocation,
        outputs: Iterable[Tensor],
        saved_operands: Iterable[Tensor],
    ) -> None:
        if not isinstance(invocation, ResolvedInvocation):
            raise TypeError("invocation must be a ResolvedInvocation")
        normalized_outputs = _materialize(outputs, "outputs")
        normalized_saved = _materialize(saved_operands, "saved_operands")
        if any(not isinstance(entry, Tensor) for entry in normalized_outputs):
            raise TypeError("outputs must contain Tensor values")
        if any(not isinstance(entry, Tensor) for entry in normalized_saved):
            raise TypeError("saved_operands must contain Tensor values")
        if len(normalized_outputs) != len(invocation.results):
            raise ValueError("outputs must contain one Tensor per ResultSpec")
        expected_saved = tuple(
            _validate_index(invocation.call, index, "saved_operands")
            for index in invocation.saved_operands
        )
        if normalized_saved != expected_saved:
            raise ValueError("saved_operands must follow the invocation save recipe")
        object.__setattr__(self, "invocation", invocation)
        object.__setattr__(self, "outputs", normalized_outputs)
        object.__setattr__(self, "saved_operands", normalized_saved)


ResolveCallback = Callable[[BoundOperationCall], ResolvedInvocation]
ReferenceCallback = Callable[[ResolvedInvocation], Tensor | tuple[Tensor, ...]]
VJPCallback = Callable[
    [VJPContext, tuple[Tensor | None, ...]], tuple[Tensor | None, ...]
]


@dataclass(frozen=True, slots=True)
class OperationDefinition:
    """Own one operation's binding, semantics, reference, and gradient rule.

    Args:
        name: Registered built-in or namespaced custom operation name.
        schema: Immutable operand and option schema.
        resolve: Callback producing one complete ``ResolvedInvocation``.
        reference: Optional carrier-neutral reference callback.
        vjp: Explicit VJP callback or ``NON_DIFFERENTIABLE`` marker.

    Examples:
        >>> callable(OperationDefinition.resolve_call)
        True
    """

    name: str
    schema: OperationSchema
    resolve: ResolveCallback
    reference: ReferenceCallback | None
    vjp: VJPCallback | _Marker

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name:
            raise ValueError("name must not be empty")
        if not isinstance(self.schema, OperationSchema):
            raise TypeError("schema must be an OperationSchema")
        if not callable(self.resolve):
            raise TypeError("resolve must be callable")
        if self.reference is not None and not callable(self.reference):
            raise TypeError("reference must be callable or None")
        if self.vjp is not NON_DIFFERENTIABLE and not callable(self.vjp):
            raise TypeError("vjp must be callable or NON_DIFFERENTIABLE")

    def bind(self, *operands: object, **options: object) -> BoundOperationCall:
        """Bind operands and options without resolving or executing.

        Args:
            operands: Positional values checked against the operand schema.
            options: Named values checked against the option schema.

        Returns:
            Immutable bound call with defaults applied.

        Examples:
            >>> callable(OperationDefinition.bind)
            True
        """

        return _bind(self, OperationCall(operands, options))

    def resolve_call(self, *operands: object, **options: object) -> ResolvedInvocation:
        """Bind and resolve one call before carrier capability work.

        Args:
            operands: Positional values checked against the operand schema.
            options: Named values checked against the option schema.

        Returns:
            Complete immutable invocation consumed by a provider.

        Examples:
            >>> callable(OperationDefinition.resolve_call)
            True
        """

        call = self.bind(*operands, **options)
        invocation = self.resolve(call)
        if not isinstance(invocation, ResolvedInvocation):
            raise TypeError("resolve must return a ResolvedInvocation")
        if invocation.call is not call:
            raise ValueError("resolved invocation must preserve bound-call identity")
        if invocation.definition is not self:
            raise ValueError("resolved invocation must preserve definition identity")
        return invocation

    def __call__(self, *operands: object, **options: object) -> object:
        """Resolve and execute one definition-backed operation call.

        Args:
            operands: Positional operation operands.
            options: Named semantic options declared by the schema.

        Returns:
            One Tensor or an ordered result tuple as declared by the resolver.

        Examples:
            >>> callable(OperationDefinition.__call__)
            True
        """

        # The ordinary no-option form goes through Carrier.dispatch_op so it
        # receives fresh dispatch metadata, profiling, and autograd behavior.
        # Custom semantic options cannot be expressed through the legacy
        # native ``options=OperationExecutionOptions`` slot, so those resolve
        # once here and enter the same provider executor directly.
        if options:
            invocation = self.resolve_call(*operands, **options)
            normalized_operands = invocation.operands
        else:
            call = self.bind(*operands)
            normalized_operands = call.operands
            invocation = None
        tensor_operands = tuple(
            operand for operand in normalized_operands if isinstance(operand, Tensor)
        )
        if not tensor_operands:
            raise UnsupportedOperationPlan(
                f"operation {self.name!r} has no Tensor dispatch operand"
            )
        carrier = tensor_operands[0].carrier
        if any(
            type(operand.carrier) is not type(carrier) for operand in tensor_operands
        ):
            raise TypeError("Tensor backing carriers must match")
        if invocation is None:
            return carrier.dispatch_op(self.name).forward(*operands)
        execute = getattr(carrier, "_execute_resolved_invocation", None)
        if execute is None:
            raise UnsupportedOperationPlan(
                f"{type(carrier).__name__} has no definition-backed execution path "
                f"for operation {self.name!r}"
            )
        return execute(invocation)


_REGISTRY: dict[str, OperationDefinition] = {}
_REGISTRY_LOCK: Final = threading.RLock()
_BUILTIN_NAMES: Final = frozenset(
    {"add", "elementwise_mul", "mul", "relu", "reduce_sum", "matmul"}
)


def _validate_custom_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    segments = name.split(".")
    if len(segments) < 2 or any(not segment.isidentifier() for segment in segments):
        raise ValueError("name must be a dot-namespaced Python identifier")
    if segments[0] == "strideweave" or segments[0] in _BUILTIN_NAMES:
        raise ValueError("name uses a reserved StrideWeave or built-in namespace")
    return name


def define_operation(
    name: str,
    schema: OperationSchema,
    resolve: ResolveCallback,
    reference: ReferenceCallback | None = None,
    vjp: VJPCallback | _Marker = _VJP_OMITTED,
) -> OperationDefinition:
    """Atomically define one immutable namespaced custom operation.

    Args:
        name: Dot-namespaced name whose first segment is extension-owned.
        schema: Ordered operand and option binding contract.
        resolve: Callback producing a complete resolved invocation.
        reference: Optional carrier-neutral conformance oracle.
        vjp: Explicit gradient callback or ``NON_DIFFERENTIABLE``.

    Returns:
        The registered callable operation definition.

    Examples:
        >>> import strideweave as sw
        >>> schema = sw.OperationSchema((sw.OperandSpec("x", sw.OperandKind.TENSOR),))
        >>> op = sw.define_operation(
        ...     "example.identity",
        ...     schema,
        ...     lambda call: (_ for _ in ()).throw(NotImplementedError()),
        ...     vjp=sw.NON_DIFFERENTIABLE,
        ... )
        >>> op.name
        'example.identity'
    """

    normalized_name = _validate_custom_name(name)
    if not isinstance(schema, OperationSchema):
        raise TypeError("schema must be an OperationSchema")
    if not callable(resolve):
        raise TypeError("resolve must be callable")
    if reference is not None and not callable(reference):
        raise TypeError("reference must be callable or None")
    if vjp is _VJP_OMITTED:
        raise ValueError("vjp must be supplied explicitly")
    if vjp is not NON_DIFFERENTIABLE and not callable(vjp):
        raise TypeError("vjp must be callable or NON_DIFFERENTIABLE")
    definition = OperationDefinition(normalized_name, schema, resolve, reference, vjp)
    with _REGISTRY_LOCK:
        if normalized_name in _REGISTRY:
            raise ValueError(f"operation {normalized_name!r} is already defined")
        _REGISTRY[normalized_name] = definition
    return definition


def _install_builtin(definition: OperationDefinition) -> None:
    if definition.name not in _BUILTIN_NAMES:
        raise ValueError("definition is not one of the v0 built-in operations")
    with _REGISTRY_LOCK:
        if definition.name in _REGISTRY:
            raise RuntimeError(f"built-in operation {definition.name!r} is installed")
        _REGISTRY[definition.name] = definition


def _operation_definition(name: str) -> OperationDefinition:
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    with _REGISTRY_LOCK:
        definition = _REGISTRY.get(name)
    if definition is None:
        raise LookupError(f"operation {name!r} is not defined")
    return definition


def _validate_result_tensor(
    tensor: object, spec: ResultSpec, index: int, *, prefix: str
) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{prefix} result {index} must be a Tensor")
    if tensor.dtype() is not spec.dtype:
        raise ProviderContractError(f"{prefix} result {index} dtype conflicts")
    if tensor.layout.shape != spec.shape:
        raise ProviderContractError(f"{prefix} result {index} shape conflicts")
    if tensor.layout != spec.layout:
        raise ProviderContractError(f"{prefix} result {index} layout conflicts")
    return tensor


def _execute_reference(invocation: ResolvedInvocation) -> tuple[Tensor, ...]:
    reference = invocation.definition.reference
    if reference is None:
        raise UnsupportedOperationPlan(
            f"operation {invocation.name!r} has no reference execution"
        )
    returned = reference(invocation)
    if isinstance(returned, Tensor):
        results = (returned,)
    elif isinstance(returned, tuple):
        results = returned
    else:
        raise TypeError("reference must return a Tensor or tuple of Tensors")
    if not results:
        raise ProviderContractError("reference returned no results")
    if len(results) != len(invocation.results):
        raise ProviderContractError("reference result count conflicts")
    validated = tuple(
        _validate_result_tensor(result, spec, index, prefix="reference")
        for index, (result, spec) in enumerate(
            zip(results, invocation.results, strict=True)
        )
    )
    tensor_operands = tuple(
        operand for operand in invocation.operands if isinstance(operand, Tensor)
    )
    for index, (result, spec) in enumerate(
        zip(validated, invocation.results, strict=True)
    ):
        if spec.alias_of is not None:
            if result is not invocation.operands[spec.alias_of]:
                raise ProviderContractError(
                    f"reference result {index} aliasing conflicts"
                )
        elif any(result is operand for operand in tensor_operands):
            raise ProviderContractError(
                f"reference result {index} unexpectedly aliases an operand"
            )
    return validated


def _differentiable_indices(invocation: ResolvedInvocation) -> tuple[int, ...]:
    if invocation.name == "mul":
        return tuple(
            index
            for index, operand in enumerate(invocation.operands)
            if isinstance(operand, Tensor)
        )
    return tuple(
        index
        for index, spec in enumerate(invocation.definition.schema.operands)
        if spec.differentiable
    )


def _call_vjp(
    context: VJPContext, cotangents: tuple[Tensor | None, ...]
) -> tuple[Tensor | None, ...]:
    """Validate a semantic VJP call and its returned operand gradients."""

    if not isinstance(context, VJPContext):
        raise TypeError("context must be a VJPContext")
    if not isinstance(cotangents, tuple):
        raise TypeError("cotangents must be a tuple")
    invocation = context.invocation
    if len(cotangents) != len(invocation.results):
        raise ValueError("cotangents must contain one entry per result")
    for index, (cotangent, spec) in enumerate(
        zip(cotangents, invocation.results, strict=True)
    ):
        if cotangent is None:
            continue
        if not isinstance(cotangent, Tensor):
            raise TypeError("cotangents must contain Tensor values or None")
        if cotangent.dtype() is not spec.dtype:
            raise TypeError(f"cotangent {index} dtype is incompatible")
        if cotangent.layout.shape != spec.shape:
            raise ValueError(f"cotangent {index} Shape is incompatible")
    callback = invocation.definition.vjp
    if callback is NON_DIFFERENTIABLE:
        raise RuntimeError("a NON_DIFFERENTIABLE operation has no VJP")
    if not callable(callback):
        raise RuntimeError("operation definition has no callable VJP")
    returned = callback(context, cotangents)
    if not isinstance(returned, tuple):
        raise TypeError("vjp must return a tuple")
    differentiable = _differentiable_indices(invocation)
    if len(returned) != len(differentiable):
        raise ValueError("vjp must return one entry per differentiable schema operand")
    for gradient, operand_index in zip(returned, differentiable, strict=True):
        if gradient is None:
            continue
        if not isinstance(gradient, Tensor):
            raise TypeError("vjp entries must be Tensor values or None")
        operand = _validate_index(invocation.call, operand_index, "vjp")
        if gradient.dtype() is not operand.dtype():
            raise TypeError("vjp gradient dtype is incompatible")
        if gradient.layout.shape != operand.layout.shape:
            raise ValueError("vjp gradient Shape is incompatible")
    return returned


__all__ = [
    "NON_DIFFERENTIABLE",
    "REQUIRED",
    "BoundOperationCall",
    "OperandKind",
    "OperandSpec",
    "OperationCall",
    "OperationDefinition",
    "OperationPlan",
    "OperationSchema",
    "OptionSpec",
    "ProviderContractError",
    "ResolvedInvocation",
    "ResultSpec",
    "VJPContext",
    "define_operation",
]
