from importlib import import_module
from typing import Any, cast

Tensor = cast(type[Any], import_module("strideweave._tensor").Tensor)

__all__ = [
    "Tensor",
]
