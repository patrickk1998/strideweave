from typing import NoReturn

from .._carrier import Carrier as Carrier

CLOSED_CARRIER_MESSAGE: str

def reject_carrier_subclass(name: str) -> NoReturn: ...
