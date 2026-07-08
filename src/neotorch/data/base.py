"""Base data class export."""

from importlib import import_module
from typing import Any, cast

_data = import_module("neotorch._data")
Data = cast(type[Any], _data.Data)

__all__ = [
    "Data",
]
