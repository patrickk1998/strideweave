"""Residency policy and facet contracts for the tiled carrier."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from ..extension import CarrierFacet
from ..move.await_result import AwaitResult
from .values import ResidencyPlan, TileSet

if TYPE_CHECKING:
    from .carrier import TiledEvictable


@runtime_checkable
class ResidencyPolicy(Protocol):
    """Describe an optional synchronous policy for tiled residency requests.

    A policy may optimize which tiles remain in primary storage, but it cannot
    change the values or correctness of a request.  The framework supplies an
    immutable requirement set and one of the supported request purposes.

    Args:
        required: Immutable tile set that the request must be able to read or
            write.
        purpose: Request kind: ``"operation"``, ``"projection"``,
            ``"scatter"``, or ``"backward"``.

    Returns:
        A disjoint :class:`ResidencyPlan` for the request.

    Examples:
        >>> import strideweave as sw
        >>> class KeepRequired:
        ...     def plan(self, required, purpose):
        ...         return sw.ResidencyPlan(promote=required)
        >>> isinstance(KeepRequired(), sw.ResidencyPolicy)
        True
    """

    def plan(self, required: TileSet, purpose: str) -> ResidencyPlan:
        """Return a residency plan for one immutable request requirement."""

        ...


class AwaitResidency(AwaitResult[None]):
    """Represent one framework-created asynchronous residency transition.

    Instances are returned by ``TiledResidencyFacet.promote_async`` and
    ``TiledResidencyFacet.evict_async``.  They expose only blocking ``wait``
    and the terminal ``done`` flag; they are not coroutine or Tensor objects.

    Examples:
        >>> hasattr(AwaitResidency, "__await__")
        False
    """

    __slots__ = ()


class TiledResidencyFacet(CarrierFacet):
    """Provide residency transitions for one exact tiled carrier.

    The facet is an immutable, definition-owned adapter.  Its methods receive
    the exact ``TiledEvictable`` instance because one shared facet value may be
    registered on the carrier definition while residency state belongs to
    each constructed carrier.

    Examples:
        >>> from strideweave.carriers.tiled_evictable.residency import (
        ...     TiledResidencyFacet,
        ... )
        >>> isinstance(TiledResidencyFacet(), TiledResidencyFacet)
        True
    """

    __slots__ = ()

    @staticmethod
    def _require_carrier(carrier: object) -> TiledEvictable:
        # Import lazily: the carrier's definition installs this facet while
        # importing the tiled package, so importing it at module load time
        # would create a cycle.
        from .carrier import TiledEvictable

        if type(carrier) is not TiledEvictable:
            raise TypeError("carrier must be the exact TiledEvictable instance")
        return carrier

    def promote_async(self, carrier: TiledEvictable, tiles: TileSet) -> AwaitResidency:
        """Eagerly promote ``tiles`` on ``carrier`` and return a handle.

        Args:
            carrier: Exact tiled carrier whose residency is changed.
            tiles: Immutable full-grid tile set to make primary-resident.

        Returns:
            A blocking-only handle that completes with ``None``.

        Examples:
            >>> callable(TiledResidencyFacet.promote_async)
            True
        """

        target = self._require_carrier(carrier)
        return cast(AwaitResidency, cast(Any, target)._promote_async(tiles))

    def evict_async(self, carrier: TiledEvictable, tiles: TileSet) -> AwaitResidency:
        """Eagerly evict ``tiles`` from ``carrier`` and return a handle.

        Args:
            carrier: Exact tiled carrier whose residency is changed.
            tiles: Immutable full-grid tile set to move out of primary storage.

        Returns:
            A blocking-only handle that completes with ``None``.

        Examples:
            >>> callable(TiledResidencyFacet.evict_async)
            True
        """

        target = self._require_carrier(carrier)
        return cast(AwaitResidency, cast(Any, target)._evict_async(tiles))

    def promote(self, carrier: TiledEvictable, tiles: TileSet) -> None:
        """Promote ``tiles`` synchronously, returning after completion.

        Args:
            carrier: Exact tiled carrier whose residency is changed.
            tiles: Immutable full-grid tile set to make primary-resident.

        Returns:
            ``None`` after the corresponding asynchronous request completes.

        Examples:
            >>> callable(TiledResidencyFacet.promote)
            True
        """

        self.promote_async(carrier, tiles).wait()

    def evict(self, carrier: TiledEvictable, tiles: TileSet) -> None:
        """Evict ``tiles`` synchronously, returning after completion.

        Args:
            carrier: Exact tiled carrier whose residency is changed.
            tiles: Immutable full-grid tile set to move out of primary storage.

        Returns:
            ``None`` after the corresponding asynchronous request completes.

        Examples:
            >>> callable(TiledResidencyFacet.evict)
            True
        """

        self.evict_async(carrier, tiles).wait()


__all__ = ["AwaitResidency", "ResidencyPolicy", "TiledResidencyFacet"]
