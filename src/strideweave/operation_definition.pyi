from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from typing import Final, TypeAlias

from .carriers.dtype import DType
from .carriers.operation_policy import OperationPlan as OperationPlan
from .layout import Layout, Shape
from .tensor import Tensor

class ProviderContractError(RuntimeError): ...

class OperandKind(str, Enum):
    TENSOR: OperandKind
    WEAK_SCALAR: OperandKind
    STRUCTURAL: OperandKind

REQUIRED: Final[object]
NON_DIFFERENTIABLE: Final[object]

class OperandSpec:
    name: str
    kind: OperandKind
    differentiable: bool
    def __init__(
        self, name: str, kind: OperandKind, differentiable: bool = ...
    ) -> None: ...

class OptionSpec:
    name: str
    default: object
    def __init__(self, name: str, default: object = ...) -> None: ...

class OperationSchema:
    operands: tuple[OperandSpec, ...]
    options: tuple[OptionSpec, ...]
    def __init__(
        self,
        operands: Iterable[OperandSpec],
        options: Iterable[OptionSpec] = ...,
    ) -> None: ...

class OperationCall:
    operands: tuple[object, ...]
    options: Mapping[str, object]
    def __init__(
        self,
        operands: Iterable[object],
        options: Mapping[str, object] | None = ...,
    ) -> None: ...

class BoundOperationCall:
    definition: OperationDefinition
    operands: tuple[object, ...]
    options: Mapping[str, object]
    operand_kinds: tuple[OperandKind, ...]

class ResultSpec:
    dtype: DType
    shape: Shape
    layout: Layout
    alias_of: int | None
    atol: float
    rtol: float
    def __init__(
        self,
        dtype: DType,
        shape: Shape,
        layout: Layout,
        alias_of: int | None = ...,
        *,
        atol: float = ...,
        rtol: float = ...,
    ) -> None: ...

class ResolvedInvocation:
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
        mutated_operands: Iterable[int] = ...,
        saved_operands: Iterable[int] = ...,
        kernel_id: str | None = ...,
    ) -> None: ...
    @property
    def definition(self) -> OperationDefinition: ...
    @property
    def name(self) -> str: ...
    @property
    def operands(self) -> tuple[object, ...]: ...
    @property
    def options(self) -> Mapping[str, object]: ...

class VJPContext:
    invocation: ResolvedInvocation
    outputs: tuple[Tensor, ...]
    saved_operands: tuple[Tensor, ...]
    def __init__(
        self,
        invocation: ResolvedInvocation,
        outputs: Iterable[Tensor],
        saved_operands: Iterable[Tensor],
    ) -> None: ...

ResolveCallback: TypeAlias = Callable[[BoundOperationCall], ResolvedInvocation]
ReferenceCallback: TypeAlias = Callable[
    [ResolvedInvocation], Tensor | tuple[Tensor, ...]
]
VJPCallback: TypeAlias = Callable[
    [VJPContext, tuple[Tensor | None, ...]], tuple[Tensor | None, ...]
]

class OperationDefinition:
    name: str
    schema: OperationSchema
    resolve: ResolveCallback
    reference: ReferenceCallback | None
    vjp: VJPCallback | object
    def __init__(
        self,
        name: str,
        schema: OperationSchema,
        resolve: ResolveCallback,
        reference: ReferenceCallback | None,
        vjp: VJPCallback | object,
    ) -> None: ...
    def bind(self, *operands: object, **options: object) -> BoundOperationCall: ...
    def resolve_call(
        self, *operands: object, **options: object
    ) -> ResolvedInvocation: ...
    def __call__(self, *operands: object, **options: object) -> object: ...

def define_operation(
    name: str,
    schema: OperationSchema,
    resolve: ResolveCallback,
    reference: ReferenceCallback | None = ...,
    vjp: VJPCallback | object = ...,
) -> OperationDefinition: ...
def _install_builtin(definition: OperationDefinition) -> None: ...
def _operation_definition(name: str) -> OperationDefinition: ...
def _execute_reference(invocation: ResolvedInvocation) -> tuple[Tensor, ...]: ...
def _call_vjp(
    context: VJPContext, cotangents: tuple[Tensor | None, ...]
) -> tuple[Tensor | None, ...]: ...
def _differentiable_indices(
    invocation: ResolvedInvocation,
) -> tuple[int, ...]: ...

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
