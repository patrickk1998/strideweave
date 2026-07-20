"""Compatibility exports for tensor operation classes and functions.

Public operation functions live in ``neotorch.functional``. Data-owned
operation classes live under ``neotorch.data``. This module keeps the historic
``neotorch.operation`` public import path stable.
"""

from __future__ import annotations

from .data.evictable import EvictableOperation as EvictableOperation
from .data.generic.ops import (
    GenericAddOperation as GenericAddOperation,
)
from .data.generic.ops import (
    GenericDivOperation as GenericDivOperation,
)
from .data.generic.ops import (
    GenericElementwiseMulOperation as GenericElementwiseMulOperation,
)
from .data.generic.ops import (
    GenericELUOperation as GenericELUOperation,
)
from .data.generic.ops import (
    GenericExpOperation as GenericExpOperation,
)
from .data.generic.ops import (
    GenericGELUOperation as GenericGELUOperation,
)
from .data.generic.ops import (
    GenericLeakyReLUOperation as GenericLeakyReLUOperation,
)
from .data.generic.ops import (
    GenericMatmulOperation as GenericMatmulOperation,
)
from .data.generic.ops import (
    GenericPowOperation as GenericPowOperation,
)
from .data.generic.ops import (
    GenericReduceSumOperation as GenericReduceSumOperation,
)
from .data.generic.ops import (
    GenericReLUOperation as GenericReLUOperation,
)
from .data.generic.ops import (
    GenericScalarMulOperation as GenericScalarMulOperation,
)
from .data.generic.ops import (
    GenericSigmoidOperation as GenericSigmoidOperation,
)
from .data.generic.ops import (
    GenericSiLUOperation as GenericSiLUOperation,
)
from .data.generic.ops import (
    GenericSoftplusOperation as GenericSoftplusOperation,
)
from .data.generic.ops import (
    GenericSubOperation as GenericSubOperation,
)
from .data.generic.ops import (
    GenericTanhOperation as GenericTanhOperation,
)
from .data.move.ops import (
    CpuToFileBackedMoveOperation as CpuToFileBackedMoveOperation,
)
from .data.move.ops import (
    ElementwiseMoveOperation as ElementwiseMoveOperation,
)
from .data.move.ops import (
    FileBackedToCpuMoveOperation as FileBackedToCpuMoveOperation,
)
from .data.move.ops import (
    MoveOperation as MoveOperation,
)
from .data.operation_helpers import Operation as Operation
from .data.shared_ops import (
    GenericViewOperation as GenericViewOperation,
)
from .data.shared_ops import (
    PermuteOperation as PermuteOperation,
)
from .data.shared_ops import (
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
