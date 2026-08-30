"""Bounded daemon executors for internal carrier completion work."""

from __future__ import annotations

import threading
from collections.abc import Callable
from queue import SimpleQueue


class _DaemonExecutor:
    __slots__ = ("_lock", "_max_workers", "_name", "_tasks", "_threads")

    def __init__(self, max_workers: int, name: str) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._max_workers = max_workers
        self._name = name
        self._lock = threading.Lock()
        self._tasks: SimpleQueue[Callable[[], None]] = SimpleQueue()
        self._threads: list[threading.Thread] = []

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def thread_name_prefix(self) -> str:
        return self._name

    def submit(self, task: Callable[[], None]) -> None:
        if not callable(task):
            raise TypeError("background task must be callable")
        with self._lock:
            if len(self._threads) < self._max_workers:
                worker = threading.Thread(
                    target=self._run,
                    name=f"{self._name}-{len(self._threads)}",
                    daemon=True,
                )
                try:
                    worker.start()
                except BaseException:
                    if not self._threads:
                        raise
                else:
                    self._threads.append(worker)
            self._tasks.put(task)

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            try:
                task()
            except BaseException:
                # Every submitted carrier task owns its terminal publication.
                # Contain a broken task so the bounded worker remains usable.
                pass
            finally:
                del task


_MOVE_BACKGROUND_EXECUTOR = _DaemonExecutor(8, "strideweave-move-worker")
_TILED_RESIDENCY_EXECUTOR = _DaemonExecutor(4, "strideweave-tiled-residency")
_TILED_SELECTION_EXECUTOR = _DaemonExecutor(4, "strideweave-tiled-selection")


__all__: list[str] = []
