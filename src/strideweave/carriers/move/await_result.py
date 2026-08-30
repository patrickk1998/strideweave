"""Thread-safe blocking completion handles shared by carrier movement APIs."""

from __future__ import annotations

import threading
from typing import Generic, TypeVar, cast

T = TypeVar("T")
_UNSET = object()
_CREATE_TOKEN = object()


class _CompletionState(Generic[T]):
    __slots__ = ("_condition", "_done", "_error", "_result")

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._done = False
        self._result: object = _UNSET
        self._error: BaseException | None = None

    @property
    def done(self) -> bool:
        with self._condition:
            return self._done

    def succeed(self, result: T) -> None:
        with self._condition:
            if self._done:
                raise RuntimeError("completion already has a terminal outcome")
            self._result = result
            self._done = True
            self._condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self._condition:
            if self._done:
                raise RuntimeError("completion already has a terminal outcome")
            self._error = error
            self._done = True
            self._condition.notify_all()

    def wait(self) -> T:
        with self._condition:
            while not self._done:
                self._condition.wait()
            if self._error is not None:
                raise self._error
            if self._result is _UNSET:
                raise RuntimeError("successful completion has no result")
            return cast(T, self._result)


class AwaitResult(Generic[T]):
    """Expose one framework-owned, thread-safe blocking result.

    The handle has deliberately small semantics: ``done`` reports terminal
    completion and ``wait()`` blocks for the identical result or exception.
    It is not a coroutine and does not support cancellation.

    Examples:
        >>> import strideweave as sw
        >>> hasattr(sw.AwaitResult, "__await__")
        False
    """

    __slots__ = ("_state", "__weakref__")

    def __init__(
        self,
        state: _CompletionState[T] | None = None,
        *,
        _token: object = None,
    ) -> None:
        if _token is not _CREATE_TOKEN or state is None:
            raise TypeError("AwaitResult handles are created by the framework")
        self._state = state

    @classmethod
    def _create(cls, state: _CompletionState[T]) -> AwaitResult[T]:
        return cls(state, _token=_CREATE_TOKEN)

    @property
    def done(self) -> bool:
        """Return whether this handle has recorded a terminal outcome."""

        return self._state.done

    def wait(self) -> T:
        """Block until completion, then return or raise the stored outcome.

        Returns:
            The completed value. Repeated calls return the identical object.

        Examples:
            ``wait()`` is normally called on a handle returned by
            :func:`strideweave.move_async`.
        """

        return self._state.wait()


class AwaitMove(AwaitResult[T]):
    """Represent one eagerly initiated single-value movement completion.

    Instances are returned by :func:`strideweave.move_async`; callers can
    inspect ``done`` or block with ``wait()``.

    Examples:
        >>> import strideweave as sw
        >>> source = sw.Tensor(
        ...     sw.Generic([1.0]), 0, sw.Layout(sw.Shape(1), sw.Stride(1))
        ... )
        >>> handle = sw.move_async(source, sw.Generic([0.0]))
        >>> handle.wait()[0]
        1.0
    """

    __slots__ = ()


__all__ = ["AwaitMove", "AwaitResult"]
