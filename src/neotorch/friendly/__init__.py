"""Ergonomic helpers for creating and inspecting neotorch tensors.

``neotorch.friendly`` wraps the composable primitives — data classes,
layouts, and functional operations — behind the factories and conveniences a
training script wants: ``tensor``/``zeros``/``ones``/``rand``-style creation
over the CPU backend, compact layout builders, scalar reductions, and value
extraction. The package is submodule-only — import it explicitly, e.g.
``import neotorch.friendly as F`` — so the top-level ``neotorch`` namespace
stays restricted to the primitives.
"""

from .creation import (
    arange as arange,
)
from .creation import (
    full as full,
)
from .creation import (
    ones as ones,
)
from .creation import (
    rand as rand,
)
from .creation import (
    randn as randn,
)
from .creation import (
    tensor as tensor,
)
from .creation import (
    zeros as zeros,
)
from .layouts import (
    column_major as column_major,
)
from .layouts import (
    row_major as row_major,
)
from .ops import (
    item as item,
)
from .ops import (
    mean as mean,
)
from .ops import (
    sum as sum,
)
from .ops import (
    to_list as to_list,
)

__all__ = [
    "arange",
    "column_major",
    "full",
    "item",
    "mean",
    "ones",
    "rand",
    "randn",
    "row_major",
    "sum",
    "tensor",
    "to_list",
    "zeros",
]
