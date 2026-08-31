from collections.abc import Callable, Iterable
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from ..core.layout import Layout
from ..core.tensor import Tensor
from ..operation_definition import OperationDefinition, ResolvedInvocation
from .base import Carrier
from .dtype import DType
from .operation_capability import OperationCapability

T_co = TypeVar("T_co", covariant=True)

_LOCK: Any

class Unsupported:
    reason: str
    def __init__(self, reason: str) -> None: ...

class CarrierFacet: ...

class KernelExecutionInterface:
    def __init__(self, version: str) -> None: ...
    @property
    def version(self) -> str: ...
    @property
    def compatibility_identity(
        self,
    ) -> tuple[type[KernelExecutionInterface], str]: ...

class StorageProvider:
    supports_dtype: Callable[[Carrier, DType], bool]
    physical_size: Callable[[object], int]
    allocate: Callable[[object, int], None]
    release: Callable[[object], None]
    read: Callable[[object, int], object]
    write: Callable[[object, int, object], None]
    elementwise_fallback: bool
    def __init__(
        self,
        supports_dtype: Callable[[Carrier, DType], bool],
        physical_size: Callable[[object], int],
        allocate: Callable[[object, int], None],
        release: Callable[[object], None],
        read: Callable[[object, int], object],
        write: Callable[[object, int, object], None],
        elementwise_fallback: bool = ...,
    ) -> None: ...

class KernelProvider:
    interface: KernelExecutionInterface
    patterns: tuple[KernelPattern, ...]
    def __init__(
        self,
        interface: KernelExecutionInterface,
        patterns: Iterable[object] = ...,
    ) -> None: ...

class KernelPattern:
    operation: OperationDefinition
    plans: tuple[OperationCapability, ...]
    kernel_id: str
    prepare: Callable[
        [ResolvedInvocation, KernelExecutionInterface], PreparedKernel | Unsupported
    ]
    preference: int
    def __init__(
        self,
        operation: OperationDefinition,
        plans: Iterable[OperationCapability],
        kernel_id: str,
        prepare: Callable[
            [ResolvedInvocation, KernelExecutionInterface], PreparedKernel | Unsupported
        ],
        preference: int = ...,
    ) -> None: ...

class KernelPack:
    namespace: str
    version: str
    carrier_class: type[Carrier]
    execution_interface: KernelExecutionInterface
    patterns: tuple[KernelPattern, ...]
    def __init__(
        self,
        namespace: str,
        version: str,
        carrier_class: type[Carrier],
        execution_interface: KernelExecutionInterface,
        patterns: Iterable[KernelPattern],
    ) -> None: ...

class TransferProvider:
    prepare: Callable[[TransferRequest], PreparedTransfer | Unsupported]
    def __init__(
        self,
        prepare: Callable[[TransferRequest], PreparedTransfer | Unsupported],
    ) -> None: ...

class TransferRoute:
    source_class: type[Carrier]
    destination_class: type[Carrier]
    route_id: str
    provider: TransferProvider
    def __init__(
        self,
        source_class: type[Carrier],
        destination_class: type[Carrier],
        route_id: str,
        provider: TransferProvider,
    ) -> None: ...

class CompositeProvider:
    capabilities: Callable[[object], Iterable[OperationCapability]]
    result_carriers: Callable[[object, object], tuple[type[Carrier], ...]]
    prepare: Callable[[object, object], PreparedComposite | Unsupported]
    def __init__(
        self,
        capabilities: Callable[[object], Iterable[OperationCapability]],
        result_carriers: Callable[[object, object], tuple[type[Carrier], ...]],
        prepare: Callable[[object, object], PreparedComposite | Unsupported],
    ) -> None: ...

class CarrierDefinition:
    storage: StorageProvider
    kernels: KernelProvider | None
    transfers: tuple[TransferRoute, ...]
    composite: CompositeProvider | None
    facets: tuple[CarrierFacet, ...]
    def __init__(
        self,
        storage: StorageProvider,
        kernels: KernelProvider | None = ...,
        transfers: Iterable[TransferRoute] = ...,
        composite: CompositeProvider | None = ...,
        facets: Iterable[CarrierFacet] = ...,
    ) -> None: ...

@runtime_checkable
class ProviderCompletion(Protocol, Generic[T_co]):
    @property
    def done(self) -> bool: ...
    def wait(self) -> T_co: ...

class ProviderResult:
    outputs: tuple[Tensor, ...]
    def __init__(self, outputs: Iterable[Tensor]) -> None: ...

class PreparedKernel:
    interface: KernelExecutionInterface
    submit: Callable[[], ProviderResult | ProviderCompletion[ProviderResult]]
    def __init__(
        self,
        interface: KernelExecutionInterface,
        submit: Callable[[], ProviderResult | ProviderCompletion[ProviderResult]],
    ) -> None: ...

class PreparedTransfer:
    submit: Callable[[], None | ProviderCompletion[None]]
    def __init__(
        self,
        submit: Callable[[], None | ProviderCompletion[None]],
    ) -> None: ...

class PreparedComposite:
    submit: Callable[[], ProviderResult | ProviderCompletion[ProviderResult]]
    def __init__(
        self,
        submit: Callable[[], ProviderResult | ProviderCompletion[ProviderResult]],
    ) -> None: ...

class TransferRequest:
    tensor: Tensor
    destination: Carrier
    source_class: type[Carrier]
    destination_class: type[Carrier]
    dtype: DType
    layout: Layout
    physical_span: int
    def __init__(
        self,
        tensor: Tensor,
        destination: Carrier,
        source_class: type[Carrier],
        destination_class: type[Carrier],
        dtype: DType,
        layout: Layout,
        physical_span: int,
    ) -> None: ...

def register_carrier_definition(
    carrier_class: type[Carrier], definition: CarrierDefinition
) -> None: ...
def register_kernel_pack(pack: KernelPack) -> None: ...
def _peek_carrier_definition(
    carrier_class: type[Carrier],
) -> CarrierDefinition | None: ...
def _observe_carrier_definition(
    carrier_class: type[Carrier],
) -> CarrierDefinition | None: ...
def _has_carrier_definition(carrier_class: type[Carrier]) -> bool: ...
def _is_carrier_definition_observed(carrier_class: type[Carrier]) -> bool: ...
def _definition_kernel_patterns(
    carrier_class: type[Carrier],
) -> tuple[KernelPattern, ...]: ...
def _definition_kernel_capabilities(
    carrier_class: type[Carrier],
) -> tuple[OperationCapability, ...]: ...
def _provider_size(carrier: object, definition: CarrierDefinition) -> int: ...
def _initialize_definition_storage(
    carrier: Carrier,
    definition: CarrierDefinition,
    slots: object,
    dtype: object,
    mutable: object,
) -> None: ...
def _definition_for_instance(carrier: object) -> CarrierDefinition | None: ...
def _definition_dtype(carrier: object) -> object: ...
def _definition_is_mutable(carrier: object) -> bool: ...
def _definition_facet(carrier: object, facet_type: object) -> CarrierFacet | None: ...
def _require_provider_completion(value: object) -> ProviderCompletion[object]: ...

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
