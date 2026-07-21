"""Compatibility exports for tensor operation classes and functions.

Public operation functions live in ``strideweave.functional``. Carrier-owned
operation classes live under ``strideweave.carriers``. This module keeps the historic
``strideweave.operation`` public import path stable.
"""

# This module intentionally re-exports the imported operation classes.
# ruff: noqa: F401

from __future__ import annotations

from .carriers.evictable import EvictableOperation
from .carriers.generic.ops import (
    GenericAddOperation,
    GenericDivOperation,
    GenericElementwiseMulOperation,
    GenericELUOperation,
    GenericExpOperation,
    GenericGELUOperation,
    GenericLeakyReLUOperation,
    GenericMatmulOperation,
    GenericPowOperation,
    GenericReduceSumOperation,
    GenericReLUOperation,
    GenericScalarMulOperation,
    GenericSigmoidOperation,
    GenericSiLUOperation,
    GenericSoftplusOperation,
    GenericSubOperation,
    GenericTanhOperation,
)
from .carriers.move.ops import (
    CpuToFileBackedMoveOperation,
    ElementwiseMoveOperation,
    FileBackedToCpuMoveOperation,
    MoveOperation,
)
from .carriers.operation_helpers import Operation
from .carriers.shared_ops import (
    GenericViewOperation,
    PermuteOperation,
    RearrangeOperation,
)
from .functional import *  # noqa: F403
from .functional import __all__ as _functional_all

_OPERATION_CLASS_EXPORTS = [
    "CpuToFileBackedMoveOperation",
    "ElementwiseMoveOperation",
    "EvictableOperation",
    "FileBackedToCpuMoveOperation",
    "GenericAddOperation",
    "GenericDivOperation",
    "GenericELUOperation",
    "GenericElementwiseMulOperation",
    "GenericExpOperation",
    "GenericGELUOperation",
    "GenericLeakyReLUOperation",
    "GenericMatmulOperation",
    "GenericPowOperation",
    "GenericReLUOperation",
    "GenericReduceSumOperation",
    "GenericScalarMulOperation",
    "GenericSiLUOperation",
    "GenericSigmoidOperation",
    "GenericSoftplusOperation",
    "GenericSubOperation",
    "GenericTanhOperation",
    "GenericViewOperation",
    "MoveOperation",
    "Operation",
    "PermuteOperation",
    "RearrangeOperation",
]

_OPERATION_EXPORTS = [*_OPERATION_CLASS_EXPORTS, *_functional_all]
__all__ = _OPERATION_EXPORTS  # pyright: ignore[reportUnsupportedDunderAll]
