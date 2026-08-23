"""Compatibility exports for StrideWeave layout infrastructure."""

from .core.index_map import IndexMap
from .core.layout import Layout, Node, Shape, Stride, Tiler, Tree
from .core.permutation import Permutation
from .core.product import Product
from .core.swizzle import Swizzle, SwizzleStage

__all__ = [
    "IndexMap",
    "Layout",
    "Node",
    "Permutation",
    "Product",
    "Shape",
    "Stride",
    "Swizzle",
    "SwizzleStage",
    "Tiler",
    "Tree",
]
