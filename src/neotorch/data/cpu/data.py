"""CPU data backend export."""

from importlib import import_module
from typing import Any, cast

_data = import_module("neotorch._data")
CPU = cast(type[Any], _data.CPU)

__all__ = [
    "CPU",
]
