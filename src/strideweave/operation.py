"""Compatibility exports for tensor operation classes and functions.

Public operation functions live in ``strideweave.functional``. Carrier-owned
operation classes live under ``strideweave.carriers``. This module keeps the historic
``strideweave.operation`` public import path stable.
"""

from __future__ import annotations

from .carriers.evictable import EvictableOperation as EvictableOperation
from .carriers.generic.ops import (
    GenericAddOperation as GenericAddOperation,
)
from .carriers.generic.ops import (
    GenericDivOperation as GenericDivOperation,
)
from .carriers.generic.ops import (
    GenericElementwiseMulOperation as GenericElementwiseMulOperation,
)
from .carriers.generic.ops import (
    GenericELUOperation as GenericELUOperation,
)
from .carriers.generic.ops import (
    GenericExpOperation as GenericExpOperation,
)
from .carriers.generic.ops import (
    GenericGELUOperation as GenericGELUOperation,
)
from .carriers.generic.ops import (
    GenericLeakyReLUOperation as GenericLeakyReLUOperation,
)
from .carriers.generic.ops import (
    GenericMatmulOperation as GenericMatmulOperation,
)
from .carriers.generic.ops import (
    GenericPowOperation as GenericPowOperation,
)
from .carriers.generic.ops import (
    GenericReduceSumOperation as GenericReduceSumOperation,
)
from .carriers.generic.ops import (
    GenericReLUOperation as GenericReLUOperation,
)
from .carriers.generic.ops import (
    GenericScalarMulOperation as GenericScalarMulOperation,
)
from .carriers.generic.ops import (
    GenericSigmoidOperation as GenericSigmoidOperation,
)
from .carriers.generic.ops import (
    GenericSiLUOperation as GenericSiLUOperation,
)
from .carriers.generic.ops import (
    GenericSoftplusOperation as GenericSoftplusOperation,
)
from .carriers.generic.ops import (
    GenericSubOperation as GenericSubOperation,
)
from .carriers.generic.ops import (
    GenericTanhOperation as GenericTanhOperation,
)
from .carriers.move.ops import (
    CpuToFileBackedMoveOperation as CpuToFileBackedMoveOperation,
)
from .carriers.move.ops import (
    ElementwiseMoveOperation as ElementwiseMoveOperation,
)
from .carriers.move.ops import (
    FileBackedToCpuMoveOperation as FileBackedToCpuMoveOperation,
)
from .carriers.move.ops import (
    MoveOperation as MoveOperation,
)
from .carriers.operation_helpers import Operation as Operation
from .carriers.shared_ops import (
    GenericViewOperation as GenericViewOperation,
)
from .carriers.shared_ops import (
    PermuteOperation as PermuteOperation,
)
from .carriers.shared_ops import (
    RearrangeOperation as RearrangeOperation,
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
