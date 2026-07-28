import threading
from collections.abc import Callable, Iterable
from typing import Any, ClassVar, Final, Self

__all__ = [
    "BlockScaledDType",
    "CompoundDType",
    "DType",
    "DTypeCategory",
    "Level",
    "SimpleDType",
    "SymbolicBits",
    "Whole",
    "WholeExtent",
    "accepts_storage_dtype",
    "storage_zero",
    "validate_storage_dtype",
]

_REGISTRY: dict[str, DType]
_STRUCTURES: dict[object, DType]
_REGISTRY_LOCK: Final[threading.RLock]

class _ContractSpec:
    owned: frozenset[str]
    layer: Callable[[Any], tuple[object, ...]]
    validate: Callable[[Any], None] | None
    structural_conflict: Callable[[Any], str] | None

    def __init__(
        self,
        owned: frozenset[str],
        layer: Callable[[Any], tuple[object, ...]],
        validate: Callable[[Any], None] | None = ...,
        structural_conflict: Callable[[Any], str] | None = ...,
    ) -> None: ...

_CONTRACT_SPECS: Final[dict[type, _ContractSpec]]

def accepts_storage_dtype(dtype: DType, accepted: tuple[DType, ...]) -> bool: ...
def storage_zero(dtype: DType) -> object: ...
def validate_storage_dtype(
    dtype: object, *, carrier: str, accepted: tuple[DType, ...]
) -> DType: ...
def _unpickle_dtype(name: str, structure: tuple[object, ...]) -> DType: ...

class _DTypeNamespace(type):
    def _install_builtin(cls, name: str, dtype: DType) -> None: ...

class DType(metaclass=_DTypeNamespace):
    Any: ClassVar[DTypeCategory]
    Floating: ClassVar[DTypeCategory]
    Integer: ClassVar[DTypeCategory]
    Float32: ClassVar[SimpleDType]
    Int32: ClassVar[SimpleDType]
    Int8: ClassVar[SimpleDType]
    E8M0: ClassVar[SimpleDType]
    E5M2: ClassVar[SimpleDType]
    E4M3: ClassVar[SimpleDType]
    E3M2: ClassVar[SimpleDType]
    E2M3: ClassVar[SimpleDType]
    E2M1: ClassVar[SimpleDType]
    MXFP8_E4M3: ClassVar[BlockScaledDType]
    MXFP8_E5M2: ClassVar[BlockScaledDType]
    MXFP6_E3M2: ClassVar[BlockScaledDType]
    MXFP6_E2M3: ClassVar[BlockScaledDType]
    MXFP4: ClassVar[BlockScaledDType]
    MXINT8: ClassVar[BlockScaledDType]
    NVFP4: ClassVar[BlockScaledDType]

    _structure: tuple[object, ...]

    def __init_subclass__(cls, *, abstract: bool = ..., **kwargs: object) -> None: ...
    def __init__(self, name: str, *, supertype: DTypeCategory | None = ...) -> None: ...
    def structure(self) -> tuple[object, ...]: ...
    def structure_extension(self) -> tuple[object, ...]: ...
    @classmethod
    def registered(cls) -> tuple[Self, ...]: ...
    @classmethod
    def from_name(cls, name: str) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> str: ...
    @property
    def supertype(self) -> DTypeCategory | None: ...
    def supertypes(self) -> tuple[DTypeCategory, ...]: ...
    def is_simple(self) -> bool: ...
    def is_category(self) -> bool: ...
    def is_compound(self) -> bool: ...
    def is_opaque_storage(self) -> bool: ...
    def is_subtype_of(self, other: DType) -> bool: ...

class DTypeCategory(DType):
    def __init__(
        self,
        name: str,
        *,
        supertype: DTypeCategory | None = ...,
        opaque_storage: bool = ...,
    ) -> None: ...

class SimpleDType(DType):
    def __init__(self, name: str, *, bits: int, supertype: DTypeCategory) -> None: ...
    @property
    def bits(self) -> int: ...

class WholeExtent: ...

Whole: Final[WholeExtent]

class Level:
    scale: SimpleDType
    block: int | WholeExtent

    def __init__(self, scale: SimpleDType, block: int | WholeExtent) -> None: ...
    def is_whole(self) -> bool: ...

class SymbolicBits:
    constant: float
    whole_scale_bits: int

    def __init__(self, constant: float, whole_scale_bits: int) -> None: ...
    def evaluate(self, element_count: int) -> float: ...

class CompoundDType(DType):
    def __init_subclass__(cls, *, abstract: bool = ..., **kwargs: object) -> None: ...
    def __init__(
        self,
        name: str,
        *,
        supertype: DTypeCategory | None = ...,
        simple_types: Iterable[SimpleDType],
    ) -> None: ...
    @property
    def simple_types(self) -> tuple[SimpleDType, ...]: ...
    @property
    def num_carriers(self) -> int: ...

class BlockScaledDType(CompoundDType):
    def __init__(
        self, name: str, *, element: SimpleDType, levels: tuple[Level, ...]
    ) -> None: ...
    @property
    def element(self) -> SimpleDType: ...
    @property
    def levels(self) -> tuple[Level, ...]: ...
    @property
    def num_axes(self) -> int: ...
    @property
    def bits_per_element(self) -> float | SymbolicBits: ...
