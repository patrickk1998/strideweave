"""Generic Python-backed carrier."""

from .carrier import Generic

__all__ = [
    "Generic",
]

# Generic's capabilities are declared and sealed by the shipped-carrier bootstrap
# in `strideweave.carriers._built_in_capabilities`, which runs while the parent
# package is imported and therefore before this module can be reached.
