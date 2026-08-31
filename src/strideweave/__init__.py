"""Public StrideWeave API for carriers, tensors, layouts, and autograd."""

from .autograd import grad as grad
from .carriers import (
    CPU as CPU,
)
from .carriers import (
    AwaitProjection as AwaitProjection,
)
from .carriers import (
    AwaitResidency as AwaitResidency,
)
from .carriers import (
    BlockDevice as BlockDevice,
)
from .carriers import (
    BlockDeviceCarrier as BlockDeviceCarrier,
)
from .carriers import (
    BlockScaledDType as BlockScaledDType,
)
from .carriers import (
    Carrier as Carrier,
)
from .carriers import (
    CarrierDefinition as CarrierDefinition,
)
from .carriers import (
    CarrierFacet as CarrierFacet,
)
from .carriers import (
    CompositeProvider as CompositeProvider,
)
from .carriers import (
    CompoundDType as CompoundDType,
)
from .carriers import (
    DependentCarrier as DependentCarrier,
)
from .carriers import (
    DType as DType,
)
from .carriers import (
    DTypeCategory as DTypeCategory,
)
from .carriers import (
    Evictable as Evictable,
)
from .carriers import (
    FileBacked as FileBacked,
)
from .carriers import (
    Generic as Generic,
)
from .carriers import (
    KernelExecutionInterface as KernelExecutionInterface,
)
from .carriers import (
    KernelPack as KernelPack,
)
from .carriers import (
    KernelPattern as KernelPattern,
)
from .carriers import (
    KernelProvider as KernelProvider,
)
from .carriers import (
    Level as Level,
)
from .carriers import (
    LevelExtent as LevelExtent,
)
from .carriers import (
    Metal as Metal,
)
from .carriers import (
    OperandCapability as OperandCapability,
)
from .carriers import (
    OperationCapability as OperationCapability,
)
from .carriers import (
    RepresentationRule as RepresentationRule,
)
from .carriers import (
    RepresentationValidationContext as RepresentationValidationContext,
)
from .carriers import (
    ResidencyPlan as ResidencyPlan,
)
from .carriers import (
    ResidencyPolicy as ResidencyPolicy,
)
from .carriers import (
    SimpleDType as SimpleDType,
)
from .carriers import (
    StorageProvider as StorageProvider,
)
from .carriers import (
    SymbolicBits as SymbolicBits,
)
from .carriers import (
    TiledEvictable as TiledEvictable,
)
from .carriers import (
    TiledResidencyFacet as TiledResidencyFacet,
)
from .carriers import (
    TileSelection as TileSelection,
)
from .carriers import (
    TileSet as TileSet,
)
from .carriers import (
    TransferRoute as TransferRoute,
)
from .carriers import (
    Unsupported as Unsupported,
)
from .carriers import (
    UnsupportedOperationPlan as UnsupportedOperationPlan,
)
from .carriers import (
    Whole as Whole,
)
from .carriers import (
    WholeExtent as WholeExtent,
)
from .carriers import (
    register_carrier_definition as register_carrier_definition,
)
from .carriers import (
    register_kernel_pack as register_kernel_pack,
)
from .carriers.move import AwaitMove as AwaitMove
from .carriers.move import AwaitResult as AwaitResult
from .layout import (
    IndexMap as IndexMap,
)
from .layout import (
    Layout as Layout,
)
from .layout import (
    Node as Node,
)
from .layout import (
    Permutation as Permutation,
)
from .layout import (
    Product as Product,
)
from .layout import (
    Shape as Shape,
)
from .layout import (
    Stride as Stride,
)
from .layout import (
    Swizzle as Swizzle,
)
from .layout import (
    SwizzleStage as SwizzleStage,
)
from .layout import (
    Tiler as Tiler,
)
from .layout import (
    Tree as Tree,
)
from .module import Module as Module
from .module import Parameter as Parameter
from .operation import *  # noqa: F403
from .operation import __all__ as _operation_all
from .tensor import Tensor as Tensor
from .verification.api import verify_backend as verify_backend

_CORE_EXPORTS = [
    "AwaitMove",
    "AwaitProjection",
    "AwaitResidency",
    "AwaitResult",
    "BlockScaledDType",
    "BlockDevice",
    "BlockDeviceCarrier",
    "CPU",
    "Carrier",
    "CarrierDefinition",
    "CarrierFacet",
    "CompoundDType",
    "CompositeProvider",
    "DType",
    "DTypeCategory",
    "DependentCarrier",
    "Evictable",
    "FileBacked",
    "Generic",
    "grad",
    "IndexMap",
    "Layout",
    "Level",
    "LevelExtent",
    "KernelExecutionInterface",
    "KernelPack",
    "KernelPattern",
    "KernelProvider",
    "Metal",
    "Module",
    "Node",
    "OperandCapability",
    "OperationCapability",
    "Parameter",
    "Permutation",
    "Product",
    "RepresentationRule",
    "RepresentationValidationContext",
    "ResidencyPlan",
    "ResidencyPolicy",
    "Shape",
    "SimpleDType",
    "StorageProvider",
    "Stride",
    "SymbolicBits",
    "Swizzle",
    "SwizzleStage",
    "Tensor",
    "TileSelection",
    "TileSet",
    "TiledEvictable",
    "TiledResidencyFacet",
    "TransferRoute",
    "verify_backend",
    "Tiler",
    "Tree",
    "UnsupportedOperationPlan",
    "Unsupported",
    "Whole",
    "WholeExtent",
    "register_carrier_definition",
    "register_kernel_pack",
]

_OPERATION_MODULE_ONLY_EXPORTS = {
    "BoundOperationCall",
    "OperationCall",
    "OperationPlan",
    "ResolvedInvocation",
    "VJPContext",
}

_TOP_LEVEL_EXPORTS = [
    *_CORE_EXPORTS,
    *(name for name in _operation_all if name not in _OPERATION_MODULE_ONLY_EXPORTS),
]
__all__ = _TOP_LEVEL_EXPORTS  # pyright: ignore[reportUnsupportedDunderAll]
