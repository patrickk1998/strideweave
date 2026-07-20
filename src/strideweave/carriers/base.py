"""Base carrier class export."""

from importlib import import_module
from typing import Any, cast

_carrier = import_module("strideweave._carrier")
Carrier = cast(type[Any], _carrier.Carrier)

__all__ = [
    "Carrier",
]
