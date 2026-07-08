"""Logical dtype tags shared by Neotorch data backends."""

from enum import Enum


class DataType(Enum):
    """Logical data type tags used by Neotorch data backends."""

    Any = "Any"
    Floating = "Floating"
    Float32 = "Float32"
    Int32 = "Int32"


__all__ = [
    "DataType",
]
