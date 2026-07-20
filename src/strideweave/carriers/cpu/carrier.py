"""CPU carrier export."""

from importlib import import_module
from typing import Any, cast

_carrier = import_module("strideweave._carrier")
CPU = cast(type[Any], _carrier.CPU)

__all__ = [
    "CPU",
]
