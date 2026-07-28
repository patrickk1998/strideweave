"""Carrier implementations and dtype tags for tensor storage and dispatch."""

from ._built_in_capabilities import _initialize_built_in_capabilities
from .base import Carrier
from .cpu import CPU
from .dtype import (
    BlockScaledDType,
    CompoundDType,
    DType,
    DTypeCategory,
    Level,
    SimpleDType,
    SymbolicBits,
    Whole,
    WholeExtent,
)
from .evictable import Evictable, EvictableOperation
from .file_backed import FileBacked
from .generic import Generic
from .operation_capability import (
    DependentCarrier,
    OperandCapability,
    OperationCapability,
    UnsupportedOperationPlan,
)

__all__ = [
    "CPU",
    "BlockScaledDType",
    "Carrier",
    "CompoundDType",
    "DType",
    "DTypeCategory",
    "DependentCarrier",
    "Evictable",
    "EvictableOperation",
    "FileBacked",
    "Generic",
    "Level",
    "OperandCapability",
    "OperationCapability",
    "SimpleDType",
    "SymbolicBits",
    "UnsupportedOperationPlan",
    "Whole",
    "WholeExtent",
]

# Every shipped carrier declares and seals its executable plan shapes here, once
# the classes exist and before any of them can be constructed, so no shipped
# backend is ever reachable in an unsealed state.
_initialize_built_in_capabilities()
