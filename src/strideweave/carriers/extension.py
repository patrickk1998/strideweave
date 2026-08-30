"""Public exact-class carrier-definition and provider contracts."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from ..operation_definition import OperationDefinition, ResolvedInvocation
    from .base import Carrier
    from .dtype import DType
    from .operation_capability import OperationCapability

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


def _materialize(value: Iterable[T], name: str) -> tuple[T, ...]:
    try:
        return tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a finite iterable") from error


def _require_callable(value: object, name: str) -> Callable[..., object]:
    if not callable(value):
        raise TypeError(f"{name} must be callable")
    return value


def _carrier_type() -> type:
    from .base import Carrier

    return Carrier


def _dependent_carrier_type() -> type:
    from .operation_capability import DependentCarrier

    return DependentCarrier


def _require_concrete_carrier_class(value: object, name: str) -> type[Carrier]:
    carrier_type = _carrier_type()
    if (
        not isinstance(value, type)
        or value is carrier_type
        or not issubclass(value, carrier_type)
    ):
        raise TypeError(f"{name} must be a concrete Carrier subclass")
    if inspect.isabstract(value):
        raise TypeError(f"{name} must be a concrete Carrier subclass")
    return value


@dataclass(frozen=True, slots=True)
class Unsupported:
    """Represent one non-terminal provider preparation rejection.

    Args:
        reason: Non-empty explanation for why this candidate cannot prepare.

    Examples:
        >>> from strideweave.carriers.extension import Unsupported
        >>> Unsupported("alignment is unsupported").reason
        'alignment is unsupported'
    """

    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        if not self.reason:
            raise ValueError("reason must not be empty")


class CarrierFacet:
    """Mark an immutable provider for one carrier-specific control surface.

    Examples:
        >>> from dataclasses import dataclass
        >>> from strideweave.carriers.extension import CarrierFacet
        >>> @dataclass(frozen=True)
        ... class Metrics(CarrierFacet):
        ...     label: str
        >>> Metrics("device").label
        'device'
    """

    __slots__ = ()


class KernelExecutionInterface:
    """Identify one versioned carrier-specific kernel execution context.

    Args:
        version: Non-empty compatibility version for this exact interface type.

    Examples:
        >>> from strideweave.carriers.extension import KernelExecutionInterface
        >>> KernelExecutionInterface("v1").compatibility_identity
        (<class 'strideweave.carriers.extension.KernelExecutionInterface'>, 'v1')
    """

    __slots__ = ("_version",)

    def __init__(self, version: str) -> None:
        if not isinstance(version, str):
            raise TypeError("version must be a string")
        if not version:
            raise ValueError("version must not be empty")
        object.__setattr__(self, "_version", version)

    @property
    def version(self) -> str:
        """Return this interface's compatibility version."""

        return self._version

    @property
    def compatibility_identity(self) -> tuple[type[KernelExecutionInterface], str]:
        """Return the exact interface type and version used for compatibility."""

        return (type(self), self.version)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("kernel execution interfaces are immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError("kernel execution interfaces are immutable")


@dataclass(frozen=True, slots=True)
class StorageProvider:
    """Define storage mechanics used by the generic carrier path.

    Args:
        supports_dtype: Callback deciding whether the exact carrier instance
            can store one dtype.
        physical_size: Callback reporting a carrier's physical slot count.
        allocate: Callback allocating a non-negative slot extent.
        release: Callback relinquishing provider-owned storage.
        read: Callback reading one physical slot.
        write: Callback writing one physical slot.
        elementwise_fallback: Whether definition-backed movement may fall back
            to framework elementwise copying when both endpoints opt in.

    Examples:
        >>> from strideweave.carriers.extension import StorageProvider
        >>> provider = StorageProvider(
        ...     lambda carrier, dtype: True,
        ...     lambda carrier: len(carrier.values),
        ...     lambda carrier, slots: setattr(carrier, "values", [None] * slots),
        ...     lambda carrier: carrier.values.clear(),
        ...     lambda carrier, index: carrier.values[index],
        ...     lambda carrier, index, value: carrier.values.__setitem__(index, value),
        ... )
        >>> provider.elementwise_fallback
        False
    """

    supports_dtype: Callable[[Carrier, DType], bool]
    physical_size: Callable[[object], int]
    allocate: Callable[[object, int], None]
    release: Callable[[object], None]
    read: Callable[[object, int], object]
    write: Callable[[object, int, object], None]
    elementwise_fallback: bool = False

    def __post_init__(self) -> None:
        for name in (
            "supports_dtype",
            "physical_size",
            "allocate",
            "release",
            "read",
            "write",
        ):
            _require_callable(getattr(self, name), name)
        if type(self.elementwise_fallback) is not bool:
            raise TypeError("elementwise_fallback must be a bool")


@dataclass(frozen=True, slots=True, init=False)
class KernelProvider:
    """Describe one independent carrier's built-in kernel provider.

    Args:
        interface: Versioned execution context supplied to kernel preparation.
        patterns: Finite iterable of built-in KernelPattern values. Pattern
            execution is completed by the kernel-pack implementation layer.

    Examples:
        >>> from strideweave.carriers.extension import (
        ...     KernelExecutionInterface,
        ...     KernelProvider,
        ... )
        >>> KernelProvider(KernelExecutionInterface("v1")).patterns
        ()
    """

    interface: KernelExecutionInterface
    patterns: tuple[KernelPattern, ...]

    def __init__(
        self,
        interface: KernelExecutionInterface,
        patterns: Iterable[object] = (),
    ) -> None:
        if not isinstance(interface, KernelExecutionInterface):
            raise TypeError("interface must be a KernelExecutionInterface")
        materialized = _materialize(patterns, "patterns")
        for pattern in materialized:
            if not isinstance(pattern, KernelPattern):
                raise TypeError("patterns must contain KernelPattern values")
        kernel_ids = [getattr(pattern, "kernel_id") for pattern in materialized]
        if len(kernel_ids) != len(set(kernel_ids)):
            raise ValueError("kernel provider contains duplicate kernel_id values")
        object.__setattr__(self, "interface", interface)
        object.__setattr__(self, "patterns", materialized)


@dataclass(frozen=True, slots=True, init=False)
class KernelPattern:
    """Describe one statically authorized kernel preparation candidate.

    Args:
        operation: Registered immutable operation definition this pattern runs.
        plans: Non-empty finite iterable of exact executable plan capabilities.
        kernel_id: Stable non-empty identifier within the target carrier.
        prepare: Callback receiving an invocation and execution interface.
        preference: Signed priority; larger values prepare first.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave.carriers.operation_capability import OperationCapability
        >>> from strideweave.carriers.operation_policy import resolve_operation_plan
        >>> from strideweave.operation_definition import _operation_definition
        >>> pattern = KernelPattern(
        ...     _operation_definition("relu"),
        ...     (OperationCapability.from_plan(
        ...         resolve_operation_plan("relu", sw.DType.Float32)
        ...     ),),
        ...     "relu.f32",
        ...     lambda invocation, interface: Unsupported("example"),
        ... )
        >>> pattern.kernel_id
        'relu.f32'
    """

    operation: OperationDefinition
    plans: tuple[OperationCapability, ...]
    kernel_id: str
    prepare: Callable[[ResolvedInvocation, KernelExecutionInterface], object]
    preference: int

    def __init__(
        self,
        operation: OperationDefinition,
        plans: Iterable[OperationCapability],
        kernel_id: str,
        prepare: Callable[[ResolvedInvocation, KernelExecutionInterface], object],
        preference: int = 0,
    ) -> None:
        from ..operation_definition import (
            OperationDefinition,
            _operation_definition,
        )
        from .operation_capability import OperationCapability

        if not isinstance(operation, OperationDefinition):
            raise TypeError("operation must be an OperationDefinition")
        try:
            registered = _operation_definition(operation.name)
        except LookupError as error:
            raise ValueError(
                "operation must be a registered OperationDefinition"
            ) from error
        if registered is not operation:
            raise ValueError("operation must be the registered definition identity")
        materialized = _materialize(plans, "plans")
        if not materialized:
            raise ValueError("plans must not be empty")
        if any(not isinstance(plan, OperationCapability) for plan in materialized):
            raise TypeError("plans must contain OperationCapability values")
        if len(materialized) != len(set(materialized)):
            raise ValueError("plans must not contain duplicate capabilities")
        if any(plan.operation != operation.name for plan in materialized):
            raise ValueError("every plan operation must match operation.name")
        if not isinstance(kernel_id, str):
            raise TypeError("kernel_id must be a string")
        if not kernel_id:
            raise ValueError("kernel_id must not be empty")
        _require_callable(prepare, "prepare")
        if type(preference) is not int:
            raise TypeError("preference must be a signed integer other than bool")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "plans", materialized)
        object.__setattr__(self, "kernel_id", kernel_id)
        object.__setattr__(self, "prepare", prepare)
        object.__setattr__(self, "preference", preference)


@dataclass(frozen=True, slots=True, init=False)
class KernelPack:
    """Collect kernel patterns for one exact independent carrier interface.

    Args:
        namespace: Non-empty extension-owned pack namespace.
        version: Non-empty pack version.
        carrier_class: Exact concrete carrier class the pack targets.
        execution_interface: Interface identity required by every pattern.
        patterns: Non-empty finite iterable of kernel patterns.

    Examples:
        >>> import strideweave as sw
        >>> class Accelerator(sw.Carrier): ...
        >>> interface = KernelExecutionInterface("v1")
        >>> # Pack eligibility is validated atomically by register_kernel_pack.
        >>> KernelPack
        <class 'strideweave.carriers.extension.KernelPack'>
    """

    namespace: str
    version: str
    carrier_class: type
    execution_interface: KernelExecutionInterface
    patterns: tuple[KernelPattern, ...]

    def __init__(
        self,
        namespace: str,
        version: str,
        carrier_class: type,
        execution_interface: KernelExecutionInterface,
        patterns: Iterable[KernelPattern],
    ) -> None:
        if not isinstance(namespace, str):
            raise TypeError("namespace must be a string")
        if not namespace:
            raise ValueError("namespace must not be empty")
        if not isinstance(version, str):
            raise TypeError("version must be a string")
        if not version:
            raise ValueError("version must not be empty")
        normalized_class = _require_concrete_carrier_class(
            carrier_class, "carrier_class"
        )
        if not isinstance(execution_interface, KernelExecutionInterface):
            raise TypeError("execution_interface must be a KernelExecutionInterface")
        materialized = _materialize(patterns, "patterns")
        if not materialized:
            raise ValueError("patterns must not be empty")
        if any(not isinstance(pattern, KernelPattern) for pattern in materialized):
            raise TypeError("patterns must contain KernelPattern values")
        kernel_ids = [pattern.kernel_id for pattern in materialized]
        if len(kernel_ids) != len(set(kernel_ids)):
            raise ValueError("pack contains duplicate kernel_id values")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "carrier_class", normalized_class)
        object.__setattr__(self, "execution_interface", execution_interface)
        object.__setattr__(self, "patterns", materialized)


@dataclass(frozen=True, slots=True)
class TransferProvider:
    """Prepare one exact directional transfer route.

    Args:
        prepare: Callback accepting a TransferRequest and returning prepared
            transfer work or Unsupported.

    Examples:
        >>> from strideweave.carriers.extension import TransferProvider
        >>> callable(TransferProvider(lambda request: None).prepare)
        True
    """

    prepare: Callable[[TransferRequest], PreparedTransfer | Unsupported]

    def __post_init__(self) -> None:
        _require_callable(self.prepare, "prepare")


@dataclass(frozen=True, slots=True)
class TransferRoute:
    """Describe one exact-class directional transfer route.

    Args:
        source_class: Exact source Carrier implementation.
        destination_class: Exact destination Carrier implementation.
        route_id: Stable non-empty route identifier.
        provider: Provider that prepares this route's work.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave.carriers.extension import TransferProvider, TransferRoute
        >>> class Source(sw.Carrier): ...
        >>> class Destination(sw.Carrier): ...
        >>> route = TransferRoute(
        ...     Source, Destination, "copy", TransferProvider(lambda request: None)
        ... )
        >>> route.route_id
        'copy'
    """

    source_class: type
    destination_class: type
    route_id: str
    provider: TransferProvider

    def __post_init__(self) -> None:
        _require_concrete_carrier_class(self.source_class, "source_class")
        _require_concrete_carrier_class(self.destination_class, "destination_class")
        if not isinstance(self.route_id, str):
            raise TypeError("route_id must be a string")
        if not self.route_id:
            raise ValueError("route_id must not be empty")
        if not isinstance(self.provider, TransferProvider):
            raise TypeError("provider must be a TransferProvider")


@dataclass(frozen=True, slots=True)
class CompositeProvider:
    """Define dependent-carrier capability and execution translation.

    Args:
        capabilities: Callback deriving one instance's complete capabilities.
        result_carriers: Callback naming exact result Carrier classes.
        prepare: Callback preparing composite execution.

    Examples:
        >>> from strideweave.carriers.extension import CompositeProvider
        >>> provider = CompositeProvider(
        ...     lambda carrier: (), lambda carrier, invocation: (),
        ...     lambda carrier, invocation: None,
        ... )
        >>> callable(provider.capabilities)
        True
    """

    capabilities: Callable[[object], Iterable[OperationCapability]]
    result_carriers: Callable[[object, object], tuple[type, ...]]
    prepare: Callable[[object, object], PreparedComposite | Unsupported]

    def __post_init__(self) -> None:
        for name in ("capabilities", "result_carriers", "prepare"):
            _require_callable(getattr(self, name), name)


@dataclass(frozen=True, slots=True, init=False)
class CarrierDefinition:
    """Describe one exact Carrier implementation's provider responsibilities.

    Args:
        storage: Complete storage provider for the carrier.
        kernels: Independent kernel provider, or ``None``.
        transfers: Finite iterable of exact directional routes.
        composite: Dependent composition provider, or ``None``.
        facets: Finite iterable of immutable carrier-specific facets.

    Examples:
        >>> from strideweave.carriers.extension import (
        ...     CarrierDefinition,
        ...     StorageProvider,
        ... )
        >>> storage = StorageProvider(
        ...     lambda carrier, dtype: True, lambda carrier: 0,
        ...     lambda carrier, slots: None, lambda carrier: None,
        ...     lambda carrier, index: None,
        ...     lambda carrier, index, value: None,
        ... )
        >>> CarrierDefinition(storage).transfers
        ()
    """

    storage: StorageProvider
    kernels: KernelProvider | None
    transfers: tuple[TransferRoute, ...]
    composite: CompositeProvider | None
    facets: tuple[CarrierFacet, ...]

    def __init__(
        self,
        storage: StorageProvider,
        kernels: KernelProvider | None = None,
        transfers: Iterable[TransferRoute] = (),
        composite: CompositeProvider | None = None,
        facets: Iterable[CarrierFacet] = (),
    ) -> None:
        if not isinstance(storage, StorageProvider):
            raise TypeError("storage must be a StorageProvider")
        if kernels is not None and not isinstance(kernels, KernelProvider):
            raise TypeError("kernels must be a KernelProvider or None")
        if composite is not None and not isinstance(composite, CompositeProvider):
            raise TypeError("composite must be a CompositeProvider or None")

        materialized_routes = _materialize(transfers, "transfers")
        route_keys: set[tuple[type, type]] = set()
        for route in materialized_routes:
            if not isinstance(route, TransferRoute):
                raise TypeError("transfers must contain TransferRoute values")
            key = (route.source_class, route.destination_class)
            if key in route_keys:
                raise ValueError("definition contains duplicate exact transfer routes")
            route_keys.add(key)

        materialized_facets = _materialize(facets, "facets")
        facet_types: set[type] = set()
        for facet in materialized_facets:
            if not isinstance(facet, CarrierFacet):
                raise TypeError("facets must contain CarrierFacet instances")
            facet_type = type(facet)
            if facet_type in facet_types:
                raise ValueError("definition contains duplicate exact facet types")
            facet_types.add(facet_type)

        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "transfers", materialized_routes)
        object.__setattr__(self, "composite", composite)
        object.__setattr__(self, "facets", materialized_facets)


@runtime_checkable
class ProviderCompletion(Protocol, Generic[T_co]):
    """Blocking-only structural completion returned by provider submission."""

    @property
    def done(self) -> bool: ...

    def wait(self) -> T_co: ...


def _require_provider_completion(value: object) -> ProviderCompletion[object]:
    """Validate one provider completion before framework acceptance."""

    if not isinstance(value, ProviderCompletion):
        raise TypeError("provider completion must expose done and wait()")
    if type(value.done) is not bool:
        raise TypeError("provider completion done must be a bool")
    if hasattr(value, "__await__"):
        raise TypeError("provider completion must not expose a coroutine protocol")
    return value


@dataclass(frozen=True, slots=True, init=False)
class ProviderResult:
    """Capture one non-empty immutable tuple of provider output Tensors.

    Args:
        outputs: Non-empty finite iterable of Tensor results.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave.carriers.extension import ProviderResult
        >>> tensor = sw.Tensor(
        ...     sw.Generic([1.0]), 0, sw.Layout(sw.Shape(1), sw.Stride(1))
        ... )
        >>> ProviderResult([tensor]).outputs == (tensor,)
        True
    """

    outputs: tuple[object, ...]

    def __init__(self, outputs: Iterable[object]) -> None:
        from ..core.tensor import Tensor

        materialized = _materialize(outputs, "outputs")
        if not materialized:
            raise ValueError("outputs must not be empty")
        if any(not isinstance(output, Tensor) for output in materialized):
            raise TypeError("outputs must contain Tensor values")
        object.__setattr__(self, "outputs", materialized)


@dataclass(frozen=True, slots=True)
class PreparedKernel:
    """Capture validated kernel work for one later submission.

    Args:
        interface: Exact execution interface used during preparation.
        submit: Zero-argument submission callback.

    Examples:
        >>> from strideweave.carriers.extension import (
        ...     KernelExecutionInterface,
        ...     PreparedKernel,
        ... )
        >>> prepared = PreparedKernel(KernelExecutionInterface("v1"), lambda: None)
        >>> callable(prepared.submit)
        True
    """

    interface: KernelExecutionInterface
    submit: Callable[[], ProviderResult | ProviderCompletion[ProviderResult]]

    def __post_init__(self) -> None:
        if not isinstance(self.interface, KernelExecutionInterface):
            raise TypeError("interface must be a KernelExecutionInterface")
        _require_callable(self.submit, "submit")


@dataclass(frozen=True, slots=True)
class PreparedTransfer:
    """Capture validated transfer work for one later submission.

    Args:
        submit: Zero-argument submission callback.

    Examples:
        >>> from strideweave.carriers.extension import PreparedTransfer
        >>> callable(PreparedTransfer(lambda: None).submit)
        True
    """

    submit: Callable[[], None | ProviderCompletion[None]]

    def __post_init__(self) -> None:
        _require_callable(self.submit, "submit")


@dataclass(frozen=True, slots=True)
class PreparedComposite:
    """Capture validated composite work for one later submission.

    Args:
        submit: Zero-argument submission callback.

    Examples:
        >>> from strideweave.carriers.extension import PreparedComposite
        >>> callable(PreparedComposite(lambda: None).submit)
        True
    """

    submit: Callable[[], ProviderResult | ProviderCompletion[ProviderResult]]

    def __post_init__(self) -> None:
        _require_callable(self.submit, "submit")


@dataclass(frozen=True, slots=True)
class TransferRequest:
    """Describe one framework-validated exact-class Tensor transfer.

    Args:
        tensor: Source Tensor.
        destination: Destination Carrier.
        source_class: Exact runtime class of the source carrier.
        destination_class: Exact runtime class of ``destination``.
        dtype: Source dtype identity.
        layout: Source Tensor Layout.
        physical_span: Physical span, exactly ``layout.cosize``.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave.carriers.extension import TransferRequest
        >>> tensor = sw.Tensor(
        ...     sw.Generic([1.0]), 0, sw.Layout(sw.Shape(1), sw.Stride(1))
        ... )
        >>> destination = sw.Generic([None])
        >>> request = TransferRequest(
        ...     tensor, destination, sw.Generic, sw.Generic,
        ...     tensor.dtype(), tensor.layout, tensor.layout.cosize,
        ... )
        >>> request.physical_span
        1
    """

    tensor: object
    destination: object
    source_class: type
    destination_class: type
    dtype: object
    layout: object
    physical_span: int

    def __post_init__(self) -> None:
        from ..core.layout import Layout
        from ..core.tensor import Tensor
        from .dtype import DType

        carrier_type = _carrier_type()
        if not isinstance(self.tensor, Tensor):
            raise TypeError("tensor must be a Tensor")
        if not isinstance(self.destination, carrier_type):
            raise TypeError("destination must be a Carrier")
        if self.source_class is not type(self.tensor.carrier):
            raise TypeError("source_class must equal the source carrier's exact class")
        if self.destination_class is not type(self.destination):
            raise TypeError(
                "destination_class must equal the destination carrier's exact class"
            )
        if not isinstance(self.dtype, DType) or self.dtype is not self.tensor.dtype():
            raise TypeError("dtype must equal the source Tensor dtype identity")
        if not isinstance(self.layout, Layout) or self.layout is not self.tensor.layout:
            raise TypeError("layout must be the source Tensor Layout")
        if type(self.physical_span) is not int:
            raise TypeError("physical_span must be an integer")
        if self.physical_span != self.layout.cosize:
            raise ValueError("physical_span must equal layout.cosize")


_LOCK = threading.RLock()
_DEFINITIONS: dict[type, CarrierDefinition] = {}
_OBSERVED: set[type] = set()
_PACKS: dict[type, dict[tuple[str, str], KernelPack]] = {}
_FROZEN_KERNEL_PATTERNS: dict[type, tuple[KernelPattern, ...]] = {}


def _shipped_carrier_classes() -> tuple[type, ...]:
    from .block_device import BlockDeviceCarrier
    from .cpu import CPU
    from .evictable import Evictable
    from .file_backed import FileBacked
    from .generic import Generic
    from .metal import Metal

    return (Generic, CPU, Metal, FileBacked, BlockDeviceCarrier, Evictable)


def register_carrier_definition(
    carrier_class: type,
    definition: CarrierDefinition,
) -> None:
    """Register one complete immutable definition for an exact Carrier class.

    Args:
        carrier_class: Concrete custom Carrier implementation to define.
        definition: Complete provider definition for that exact class.

    Returns:
        None.

    Examples:
        >>> import strideweave as sw
        >>> from strideweave.carriers.extension import (
        ...     CarrierDefinition,
        ...     StorageProvider,
        ...     register_carrier_definition,
        ... )
        >>> class EmptyStorage(sw.Carrier): ...
        >>> storage = StorageProvider(
        ...     lambda carrier, dtype: True,
        ...     lambda carrier: len(carrier.values),
        ...     lambda carrier, slots: setattr(carrier, "values", [None] * slots),
        ...     lambda carrier: carrier.values.clear(),
        ...     lambda carrier, index: carrier.values[index],
        ...     lambda carrier, index, value: carrier.values.__setitem__(index, value),
        ... )
        >>> register_carrier_definition(EmptyStorage, CarrierDefinition(storage))
        >>> EmptyStorage(1, dtype=sw.DType.Any).size()
        1
    """

    normalized_class = _require_concrete_carrier_class(carrier_class, "carrier_class")
    if not isinstance(definition, CarrierDefinition):
        raise TypeError("definition must be a CarrierDefinition")
    if normalized_class in _shipped_carrier_classes():
        raise TypeError(
            f"{normalized_class.__name__} is a definition-free shipped carrier"
        )
    for route in definition.transfers:
        if route.source_class is not normalized_class:
            raise TypeError(
                "every definition transfer route must use carrier_class as its "
                "exact source_class"
            )

    is_dependent = issubclass(normalized_class, _dependent_carrier_type())
    if is_dependent:
        if definition.composite is None or definition.kernels is not None:
            raise TypeError(
                "a dependent Carrier definition requires composite and forbids kernels"
            )
    elif definition.composite is not None:
        raise TypeError(
            "an independent Carrier definition must not supply a composite provider"
        )

    from .operation_capability import _LOCK as capability_lock
    from .operation_capability import _is_sealed

    # Definition and legacy capability authority are one exact-class choice.
    # Both registries publish under this lock order, so concurrent attempts can
    # never each pass validation and install two authorities.
    with _LOCK:
        with capability_lock:
            if _is_sealed(normalized_class):
                raise TypeError(
                    f"{normalized_class.__name__} already has a definition-free "
                    "capability authority"
                )
            if normalized_class in _OBSERVED:
                raise TypeError(
                    f"{normalized_class.__name__} was already observed and its exact "
                    "definition is sealed"
                )
            if normalized_class in _DEFINITIONS:
                raise TypeError(
                    f"{normalized_class.__name__} already has a CarrierDefinition"
                )
            _DEFINITIONS[normalized_class] = definition


def register_kernel_pack(pack: KernelPack) -> None:
    """Attach one complete kernel pack before its exact carrier is observed.

    Args:
        pack: Immutable pack targeting one defined independent carrier class.

    Returns:
        None.

    Examples:
        >>> import strideweave as sw
        >>> callable(sw.register_kernel_pack)
        True
    """

    if not isinstance(pack, KernelPack):
        raise TypeError("pack must be a KernelPack")
    carrier_class = pack.carrier_class
    with _LOCK:
        definition = _DEFINITIONS.get(carrier_class)
        if definition is None:
            raise TypeError("kernel pack target must have an exact CarrierDefinition")
        if issubclass(carrier_class, _dependent_carrier_type()):
            raise TypeError("kernel packs cannot target a DependentCarrier class")
        provider = definition.kernels
        if provider is None:
            raise TypeError("kernel pack target is a storage-only carrier")
        if carrier_class in _OBSERVED:
            raise TypeError(
                f"{carrier_class.__name__} was already observed; its kernel set "
                "is sealed"
            )
        if (
            pack.execution_interface.compatibility_identity
            != provider.interface.compatibility_identity
        ):
            raise TypeError(
                "kernel pack execution_interface does not match the target "
                "KernelProvider"
            )
        identity = (pack.namespace, pack.version)
        registered = _PACKS.setdefault(carrier_class, {})
        if identity in registered:
            raise ValueError(
                "kernel pack namespace and version are already registered for "
                "this exact carrier"
            )
        existing_ids = {pattern.kernel_id for pattern in provider.patterns}
        existing_ids.update(
            pattern.kernel_id
            for existing in registered.values()
            for pattern in existing.patterns
        )
        duplicate_ids = existing_ids.intersection(
            pattern.kernel_id for pattern in pack.patterns
        )
        if duplicate_ids:
            duplicate = sorted(duplicate_ids)[0]
            raise ValueError(
                f"kernel_id {duplicate!r} is already registered for this exact carrier"
            )
        registered[identity] = pack


def _freeze_kernel_patterns_locked(
    carrier_class: type, definition: CarrierDefinition
) -> tuple[KernelPattern, ...]:
    frozen = _FROZEN_KERNEL_PATTERNS.get(carrier_class)
    if frozen is not None:
        return frozen
    provider = definition.kernels
    if provider is None:
        frozen = ()
    else:
        patterns = list(provider.patterns)
        for identity in sorted(_PACKS.get(carrier_class, {})):
            patterns.extend(_PACKS[carrier_class][identity].patterns)
        frozen = tuple(patterns)
    _FROZEN_KERNEL_PATTERNS[carrier_class] = frozen
    return frozen


def _definition_kernel_patterns(carrier_class: type) -> tuple[KernelPattern, ...]:
    """Observe and return one exact class's frozen ordered kernel set."""

    with _LOCK:
        definition = _DEFINITIONS.get(carrier_class)
        if definition is None:
            return ()
        _OBSERVED.add(carrier_class)
        return _freeze_kernel_patterns_locked(carrier_class, definition)


def _definition_kernel_capabilities(
    carrier_class: type,
) -> tuple[OperationCapability, ...]:
    """Return the duplicate-free capability union of the frozen kernel set."""

    from .operation_capability import _sort_key

    unique: dict[OperationCapability, OperationCapability] = {}
    for pattern in _definition_kernel_patterns(carrier_class):
        for capability in pattern.plans:
            unique.setdefault(capability, capability)
    return tuple(sorted(unique.values(), key=_sort_key))


def _peek_carrier_definition(carrier_class: type) -> CarrierDefinition | None:
    """Return an exact class's registered definition without observing it."""

    with _LOCK:
        return _DEFINITIONS.get(carrier_class)


def _observe_carrier_definition(carrier_class: type) -> CarrierDefinition | None:
    """Seal and return an exact class's definition, without base traversal."""

    with _LOCK:
        _OBSERVED.add(carrier_class)
        definition = _DEFINITIONS.get(carrier_class)
        if definition is not None:
            _freeze_kernel_patterns_locked(carrier_class, definition)
        return definition


def _has_carrier_definition(carrier_class: type) -> bool:
    """Return whether this exact class has a registered definition."""

    return _peek_carrier_definition(carrier_class) is not None


def _is_carrier_definition_observed(carrier_class: type) -> bool:
    """Return whether exact-class construction or lookup sealed authority."""

    with _LOCK:
        return carrier_class in _OBSERVED


def _provider_size(carrier: object, definition: CarrierDefinition) -> int:
    size = definition.storage.physical_size(carrier)
    if type(size) is not int:
        raise TypeError("StorageProvider.physical_size must return an integer")
    if size < 0:
        raise ValueError(
            "StorageProvider.physical_size must not return a negative size"
        )
    return size


def _initialize_definition_storage(
    carrier: Carrier,
    definition: CarrierDefinition,
    slots: object,
    dtype: object,
    mutable: object,
) -> None:
    from .dtype import DType

    if type(slots) is not int:
        raise TypeError("definition-backed carrier slots must be an integer")
    if slots < 0:
        raise ValueError("definition-backed carrier slots must be non-negative")
    if not isinstance(dtype, DType):
        raise TypeError("definition-backed carrier dtype must be a DType")
    if type(mutable) is not bool:
        raise TypeError("definition-backed carrier mutable must be a bool")
    supported = definition.storage.supports_dtype(carrier, dtype)
    if type(supported) is not bool:
        raise TypeError("StorageProvider.supports_dtype must return a bool")
    if not supported:
        raise TypeError(
            f"{type(carrier).__name__} does not support storage dtype {dtype.name}"
        )
    object.__setattr__(carrier, "_strideweave_definition_dtype", dtype)
    object.__setattr__(carrier, "_strideweave_definition_mutable", mutable)
    result = definition.storage.allocate(carrier, slots)
    if result is not None:
        raise TypeError("StorageProvider.allocate must return None")
    if _provider_size(carrier, definition) != slots:
        raise ValueError(
            "StorageProvider.allocate must establish the requested physical size"
        )


def _definition_for_instance(carrier: object) -> CarrierDefinition | None:
    return _peek_carrier_definition(type(carrier))


def _definition_dtype(carrier: object) -> object:
    return getattr(carrier, "_strideweave_definition_dtype")


def _definition_is_mutable(carrier: object) -> bool:
    return bool(getattr(carrier, "_strideweave_definition_mutable"))


def _definition_facet(carrier: object, facet_type: object) -> CarrierFacet | None:
    if not isinstance(facet_type, type) or not issubclass(facet_type, CarrierFacet):
        raise TypeError("facet_type must be a CarrierFacet subclass")
    definition = _definition_for_instance(carrier)
    if definition is None:
        return None
    return next(
        (facet for facet in definition.facets if type(facet) is facet_type), None
    )


__all__ = [
    "CarrierDefinition",
    "CarrierFacet",
    "CompositeProvider",
    "KernelExecutionInterface",
    "KernelPack",
    "KernelPattern",
    "KernelProvider",
    "PreparedComposite",
    "PreparedKernel",
    "PreparedTransfer",
    "ProviderCompletion",
    "ProviderResult",
    "StorageProvider",
    "TransferProvider",
    "TransferRequest",
    "TransferRoute",
    "Unsupported",
    "register_carrier_definition",
    "register_kernel_pack",
]
