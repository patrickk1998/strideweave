"""The one bootstrap that declares and seals every shipped carrier's reach.

Sealing is framework-internal: a shipped backend's advertised reach must follow
from its implementation, never from a later call. That holds only if *every*
shipped class is sealed, so this module names all four of them in one place and
runs once, at the end of :mod:`strideweave.carriers`' initialization, before any
carrier can be constructed or any plan executed.

``Generic`` and ``CPU`` declare the plan shapes their reference implementation
and their kernels execute. ``FileBacked`` declares the empty set: it is a
storage carrier that plans no operation of its own. An empty declaration is a
statement, not an omission — it is what makes "this backend executes no planned
operation" a declared fact rather than one inferred from silence.

``Evictable`` is deliberately absent. What a hierarchy executes depends on the
carriers it was handed, so it is a ``DependentCarrier`` that generates and
freezes its capabilities per instance; a class-level declaration here could
only describe hierarchies it has never seen, and the registry rejects one.

The declaration path itself is private and exported nowhere, so this module is
the only way a shipped class is ever declared for.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from .cpu import CPU
from .cpu.capabilities import cpu_capabilities
from .file_backed import FileBacked
from .generic import Generic
from .generic.capabilities import generic_capabilities
from .operation_capability import (
    OperationCapability,
    _declare_built_in_capabilities,
    _is_sealed,
)

__all__: list[str] = []


def _no_capabilities() -> tuple[OperationCapability, ...]:
    """Return the intentional empty set of a storage carrier."""
    return ()


# Every shipped independent carrier class, each paired with the capabilities it
# declares. A new shipped independent carrier belongs here; a class left out
# would ship undeclared. A shipped dependent carrier does not: it finalizes its
# own instances.
_BUILT_INS: Final[
    tuple[tuple[type, Callable[[], tuple[OperationCapability, ...]]], ...]
] = (
    (Generic, generic_capabilities),
    (CPU, cpu_capabilities),
    (FileBacked, _no_capabilities),
)


def _initialize_built_in_capabilities() -> None:
    """Declare and seal every shipped independent carrier's capabilities, once.

    Called while :mod:`strideweave.carriers` is imported, because a backend
    cannot execute a planned operation before it has said which plans it
    executes. Re-running it is a no-op: a class already sealed keeps the entries
    it declared, so an unusual import order or a reimported module cannot
    disturb what a shipped backend advertises.
    """
    for carrier_class, capabilities in _BUILT_INS:
        if _is_sealed(carrier_class):
            continue
        _declare_built_in_capabilities(carrier_class, capabilities())
