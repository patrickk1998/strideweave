"""Logical dtype tags shared by StrideWeave carriers."""

from enum import Enum


class DType(Enum):
    """Logical value-type tags used by StrideWeave carriers."""

    Any = "Any"
    Floating = "Floating"
    Float32 = "Float32"
    Int32 = "Int32"


__all__ = [
    "DType",
]
