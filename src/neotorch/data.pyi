from collections.abc import Iterable
from enum import Enum
from typing import Any

from ._data import Data as Data

class DataType(Enum):
    Any = "Any"

class Generic(Data):
    def __init__(self, values: Iterable[Any]) -> None: ...
    def size(self) -> int: ...
    def type(self) -> DataType: ...
    def get_value(self, index: int) -> Any: ...
