"""Compatibility exports for tensor operation classes and functions.

Public operation functions live in ``strideweave.functional``. Carrier-owned
operation classes live under ``strideweave.carriers``. This module keeps the historic
``strideweave.operation`` public import path stable.
"""

# This module intentionally re-exports the imported operation classes.
# ruff: noqa: F401

from __future__ import annotations

from .carriers.evictable import EvictableOperation
from .carriers.generic.as_strided_ops import GenericAsStridedOperation
from .carriers.generic.convolution_ops import GenericConvGeneralOperation
from .carriers.generic.indexing_ops import (
    GenericGatherOperation,
    GenericScatterAddOperation,
    GenericScatterOperation,
)
from .carriers.generic.ops import (
    GenericAbsOperation,
    GenericAddOperation,
    GenericCeilOperation,
    GenericCosOperation,
    GenericDivOperation,
    GenericElementwiseMulOperation,
    GenericELUOperation,
    GenericErfOperation,
    GenericExp2Operation,
    GenericExpOperation,
    GenericFloorOperation,
    GenericGELUOperation,
    GenericLeakyReLUOperation,
    GenericLog2Operation,
    GenericLogOperation,
    GenericMatmulOperation,
    GenericMaximumOperation,
    GenericMinimumOperation,
    GenericNegOperation,
    GenericPowOperation,
    GenericRecipOperation,
    GenericReLUOperation,
    GenericRemOperation,
    GenericRoundOperation,
    GenericRsqrtOperation,
    GenericScalarMulOperation,
    GenericSigmoidOperation,
    GenericSignOperation,
    GenericSiLUOperation,
    GenericSinOperation,
    GenericSoftplusOperation,
    GenericSqrtOperation,
    GenericSubOperation,
    GenericTanhOperation,
)
from .carriers.generic.predicate_ops import (
    GenericEqOperation,
    GenericLeOperation,
    GenericLogicalNotOperation,
    GenericLtOperation,
    GenericNeOperation,
)
from .carriers.generic.reduction_ops import (
    GenericArgMaxOperation,
    GenericArgMinOperation,
    GenericCumsumOperation,
    GenericReduceMaxOperation,
    GenericReduceMinOperation,
    GenericReduceProdOperation,
    GenericReduceSumOperation,
)
from .carriers.generic.selection_ops import (
    GenericSortIndicesOperation,
    GenericSortValuesOperation,
    GenericTopKIndicesOperation,
    GenericTopKValuesOperation,
)
from .carriers.generic.ternary_ops import (
    GenericClampOperation,
    GenericSelectOperation,
)
from .carriers.move.ops import (
    CpuToFileBackedMoveOperation,
    ElementwiseMoveOperation,
    FileBackedToCpuMoveOperation,
    MoveOperation,
)
from .carriers.operation_helpers import Operation
from .carriers.shared_ops import (
    BroadcastOperation,
    GenericViewOperation,
    PermuteOperation,
    RearrangeOperation,
    ReshapeOperation,
    SqueezeOperation,
    UnsqueezeOperation,
)
from .functional import *  # noqa: F403
from .functional import __all__ as _functional_all
from .profiling import Profiler, ProfilerAggregate, ProfilerEvent, profile

_OPERATION_CLASS_EXPORTS = [
    "BroadcastOperation",
    "CpuToFileBackedMoveOperation",
    "ElementwiseMoveOperation",
    "EvictableOperation",
    "FileBackedToCpuMoveOperation",
    "GenericAbsOperation",
    "GenericAddOperation",
    "GenericArgMaxOperation",
    "GenericArgMinOperation",
    "GenericAsStridedOperation",
    "GenericCeilOperation",
    "GenericClampOperation",
    "GenericConvGeneralOperation",
    "GenericCosOperation",
    "GenericCumsumOperation",
    "GenericDivOperation",
    "GenericELUOperation",
    "GenericElementwiseMulOperation",
    "GenericEqOperation",
    "GenericErfOperation",
    "GenericExpOperation",
    "GenericExp2Operation",
    "GenericFloorOperation",
    "GenericGELUOperation",
    "GenericGatherOperation",
    "GenericLeakyReLUOperation",
    "GenericLeOperation",
    "GenericLogicalNotOperation",
    "GenericLogOperation",
    "GenericLog2Operation",
    "GenericLtOperation",
    "GenericMatmulOperation",
    "GenericMaximumOperation",
    "GenericMinimumOperation",
    "GenericNegOperation",
    "GenericNeOperation",
    "GenericPowOperation",
    "GenericReLUOperation",
    "GenericRecipOperation",
    "GenericReduceMaxOperation",
    "GenericReduceMinOperation",
    "GenericReduceProdOperation",
    "GenericReduceSumOperation",
    "GenericRemOperation",
    "GenericRoundOperation",
    "GenericRsqrtOperation",
    "GenericScalarMulOperation",
    "GenericScatterAddOperation",
    "GenericScatterOperation",
    "GenericSelectOperation",
    "GenericSignOperation",
    "GenericSinOperation",
    "GenericSiLUOperation",
    "GenericSigmoidOperation",
    "GenericSoftplusOperation",
    "GenericSortIndicesOperation",
    "GenericSortValuesOperation",
    "GenericSqrtOperation",
    "GenericSubOperation",
    "GenericTanhOperation",
    "GenericTopKIndicesOperation",
    "GenericTopKValuesOperation",
    "GenericViewOperation",
    "MoveOperation",
    "Operation",
    "PermuteOperation",
    "Profiler",
    "ProfilerAggregate",
    "ProfilerEvent",
    "RearrangeOperation",
    "ReshapeOperation",
    "SqueezeOperation",
    "UnsqueezeOperation",
]

_OPERATION_EXPORTS = sorted([*_OPERATION_CLASS_EXPORTS, "profile", *_functional_all])
__all__ = _OPERATION_EXPORTS  # pyright: ignore[reportUnsupportedDunderAll]
