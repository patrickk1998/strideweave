"""CPU carrier."""

from .carrier import CPU

__all__ = [
    "CPU",
]

# A kernel is only reached through a capability the backend declared. CPU's
# capabilities are declared and sealed by the shipped-carrier bootstrap in
# `strideweave.carriers._built_in_capabilities`, which runs while the parent
# package is imported and therefore before any kernel can be reached.
