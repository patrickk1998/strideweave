from __future__ import annotations

import gc
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any, cast

import pytest

from strideweave.carriers._background import _DaemonExecutor


def _wait_until_collected(reference: Callable[[], object | None]) -> None:
    deadline = time.monotonic() + 2
    while reference() is not None and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.001)
    assert reference() is None


@pytest.mark.parametrize("raises", [False, True])
def test_daemon_executor_releases_completed_task_before_idling(raises: bool) -> None:
    executor = _DaemonExecutor(1, f"strideweave-test-release-{raises}")
    first_task_finishing = threading.Event()

    class Payload:
        pass

    payload = Payload()
    payload_reference = weakref.ref(payload)

    def task(retained_payload: Payload = payload) -> None:
        assert retained_payload is not None
        first_task_finishing.set()
        if raises:
            raise KeyboardInterrupt("contained task failure")

    executor.submit(task)
    del task
    del payload

    assert first_task_finishing.wait(timeout=2)
    _wait_until_collected(payload_reference)

    later_task_count = 8
    later_tasks_finished = threading.Event()
    count_lock = threading.Lock()
    completed = 0

    def later_task() -> None:
        nonlocal completed
        with count_lock:
            completed += 1
            if completed == later_task_count:
                later_tasks_finished.set()

    for _ in range(later_task_count):
        executor.submit(later_task)

    assert later_tasks_finished.wait(timeout=2)
    threads = cast(Any, executor)._threads
    assert len(threads) == executor.max_workers == 1
    assert threads[0].is_alive()
