"""Carrier implementations and dtype tags for tensor storage and dispatch."""

from ._built_in_capabilities import _initialize_built_in_capabilities
from .base import Carrier
from .block_device import BlockDevice, BlockDeviceCarrier
from .cpu import CPU
from .dtype import (
    BlockScaledDType,
    CompoundDType,
    DType,
    DTypeCategory,
    Level,
    LevelExtent,
    RepresentationRule,
    RepresentationValidationContext,
    SimpleDType,
    SymbolicBits,
    Whole,
    WholeExtent,
)
from .evictable import Evictable, EvictableOperation
from .extension import (
    CarrierDefinition,
    CarrierFacet,
    CompositeProvider,
    KernelExecutionInterface,
    KernelPack,
    KernelPattern,
    KernelProvider,
    StorageProvider,
    TransferRoute,
    Unsupported,
    register_carrier_definition,
    register_kernel_pack,
)
from .file_backed import FileBacked
from .generic import Generic
from .metal import Metal
from .operation_capability import (
    DependentCarrier,
    OperandCapability,
    OperationCapability,
    UnsupportedOperationPlan,
)
from .tiled_evictable import (
    AwaitProjection,
    AwaitResidency,
    ResidencyPlan,
    ResidencyPolicy,
    TiledEvictable,
    TiledResidencyFacet,
    TileSelection,
    TileSet,
)

__all__ = [
    "CPU",
    "AwaitProjection",
    "AwaitResidency",
    "BlockDevice",
    "BlockDeviceCarrier",
    "BlockScaledDType",
    "Carrier",
    "CarrierDefinition",
    "CarrierFacet",
    "CompositeProvider",
    "CompoundDType",
    "DType",
    "DTypeCategory",
    "DependentCarrier",
    "Evictable",
    "EvictableOperation",
    "FileBacked",
    "Generic",
    "KernelExecutionInterface",
    "KernelPack",
    "KernelPattern",
    "KernelProvider",
    "Level",
    "LevelExtent",
    "Metal",
    "OperandCapability",
    "OperationCapability",
    "RepresentationRule",
    "RepresentationValidationContext",
    "ResidencyPlan",
    "ResidencyPolicy",
    "SimpleDType",
    "StorageProvider",
    "SymbolicBits",
    "TileSelection",
    "TileSet",
    "TiledEvictable",
    "TiledResidencyFacet",
    "TransferRoute",
    "Unsupported",
    "UnsupportedOperationPlan",
    "Whole",
    "WholeExtent",
    "register_carrier_definition",
    "register_kernel_pack",
]

# Every shipped carrier declares and seals its executable plan shapes here, once
# the classes exist and before any of them can be constructed, so no shipped
# backend is ever reachable in an unsealed state.
_initialize_built_in_capabilities()
