"""Completion handle for eager tiled projections."""

from __future__ import annotations

from ...tensor import Tensor
from ..move.await_result import AwaitResult


class AwaitProjection(AwaitResult[Tensor]):
    """Represent one framework-created asynchronous tiled projection.

    ``AwaitProjection`` exposes the common blocking completion interface and
    completes with the ordinary compact :class:`~strideweave.Tensor` produced
    by projection.  Handles are created by ``TiledEvictable.project``; callers
    inspect ``done`` or call ``wait()`` and never await the handle as a
    coroutine.

    Examples:
        >>> hasattr(AwaitProjection, "__await__")
        False
    """

    __slots__ = ()


__all__ = ["AwaitProjection"]
