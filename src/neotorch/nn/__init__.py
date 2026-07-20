"""PyTorch-style neural network modules composed from neotorch primitives.

``neotorch.nn`` layers keep the core data classes untouched: every module is
built from the public functional operations and the ``Module``/``Parameter``
registration system. The package is submodule-only — import it explicitly as
``import neotorch.nn as nn`` — so the top-level ``neotorch`` namespace stays
restricted to composable primitives.
"""

from .modules import (
    ELU as ELU,
)
from .modules import (
    GELU as GELU,
)
from .modules import (
    LeakyReLU as LeakyReLU,
)
from .modules import (
    Linear as Linear,
)
from .modules import (
    MSELoss as MSELoss,
)
from .modules import (
    ReLU as ReLU,
)
from .modules import (
    Sigmoid as Sigmoid,
)
from .modules import (
    SiLU as SiLU,
)
from .modules import (
    Softplus as Softplus,
)
from .modules import (
    Tanh as Tanh,
)
from .optim import SGD as SGD

__all__ = [
    "ELU",
    "GELU",
    "LeakyReLU",
    "Linear",
    "MSELoss",
    "ReLU",
    "SGD",
    "SiLU",
    "Sigmoid",
    "Softplus",
    "Tanh",
]
