from typing import Protocol, runtime_checkable

from ..extension import CarrierFacet
from ..move.await_result import AwaitResult
from .carrier import TiledEvictable
from .values import ResidencyPlan, TileSet

@runtime_checkable
class ResidencyPolicy(Protocol):
    def plan(self, required: TileSet, purpose: str) -> ResidencyPlan: ...

class AwaitResidency(AwaitResult[None]): ...

class TiledResidencyFacet(CarrierFacet):
    @staticmethod
    def _require_carrier(carrier: object) -> TiledEvictable: ...
    def promote_async(
        self, carrier: TiledEvictable, tiles: TileSet
    ) -> AwaitResidency: ...
    def evict_async(
        self, carrier: TiledEvictable, tiles: TileSet
    ) -> AwaitResidency: ...
    def promote(self, carrier: TiledEvictable, tiles: TileSet) -> None: ...
    def evict(self, carrier: TiledEvictable, tiles: TileSet) -> None: ...

__all__ = ["AwaitResidency", "ResidencyPolicy", "TiledResidencyFacet"]
