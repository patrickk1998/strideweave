from enum import Enum
from importlib import import_module
from typing import Any, cast

Data = cast(type[Any], import_module("neotorch._data").Data)


class DataType(Enum):
    Any = "Any"


__all__ = [
    "Data",
    "DataType",
]
