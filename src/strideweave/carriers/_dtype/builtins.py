"""Installation of the built-in dtype descriptor graph."""

from __future__ import annotations

from . import model
from .block_scaled import BlockScaledDType, Level
from .model import DType, DTypeCategory, SimpleDType
from .structure import Whole


def _install(name: str, dtype: DType) -> None:
    """Install ``dtype`` as the protected built-in binding ``DType.<name>``."""
    DType._install_builtin(name, dtype)


_install("Any", DTypeCategory("Any", opaque_storage=True))
_install(
    "Floating", DTypeCategory("Floating", supertype=DType.Any, opaque_storage=True)
)
_install("Integer", DTypeCategory("Integer", supertype=DType.Any))
_install("Float32", SimpleDType("Float32", bits=32, supertype=DType.Floating))
_install("Float64", SimpleDType("Float64", bits=64, supertype=DType.Floating))
_install("Int32", SimpleDType("Int32", bits=32, supertype=DType.Integer))
# Bool is a concrete logical encoding, deliberately outside the numeric
# Floating/Integer categories.  It is rooted at Any so it remains a first-class
# simple representation without participating in numeric promotion.
_install("Bool", SimpleDType("Bool", bits=8, supertype=DType.Any))

# Narrow encodings used by the block-scaled formats below. They are structural
# descriptors only: no carrier stores them and no kernel interprets them yet.
_install("Int8", SimpleDType("Int8", bits=8, supertype=DType.Integer))
_install("E8M0", SimpleDType("E8M0", bits=8, supertype=DType.Floating))
_install("E5M2", SimpleDType("E5M2", bits=8, supertype=DType.Floating))
_install("E4M3", SimpleDType("E4M3", bits=8, supertype=DType.Floating))
_install("E3M2", SimpleDType("E3M2", bits=6, supertype=DType.Floating))
_install("E2M3", SimpleDType("E2M3", bits=6, supertype=DType.Floating))
_install("E2M1", SimpleDType("E2M1", bits=4, supertype=DType.Floating))

# OCP MX v1.0 formats: 32 elements per E8M0 scale. The block extent is fixed by
# the format, so it is never a caller-supplied argument.
_MX_LEVELS = (Level(scale=DType.E8M0, block=32),)
for _mx_name, _mx_element in (
    ("MXFP8_E4M3", DType.E4M3),
    ("MXFP8_E5M2", DType.E5M2),
    ("MXFP6_E3M2", DType.E3M2),
    ("MXFP6_E2M3", DType.E2M3),
    ("MXFP4", DType.E2M1),
    ("MXINT8", DType.Int8),
):
    _install(
        _mx_name, BlockScaledDType(_mx_name, element=_mx_element, levels=_MX_LEVELS)
    )

# NVFP4: 16 elements per E4M3 scale, then one Float32 scale for the tensor.
_install(
    "NVFP4",
    BlockScaledDType(
        "NVFP4",
        element=DType.E2M1,
        levels=(
            Level(scale=DType.E4M3, block=16),
            Level(scale=DType.Float32, block=Whole),
        ),
    ),
)

# The built-in surface is complete: from here the DType namespace is exactly
# these bindings, and every later registration lives in the registry alone.
model._BUILTINS_SEALED = True
