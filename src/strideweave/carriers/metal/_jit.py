"""Failure-atomic in-memory specialization cache for TileLang Metal kernels."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from threading import RLock
from typing import Literal

type CanonicalValue = (
    tuple[Literal["none"]]
    | tuple[Literal["bool"], bool]
    | tuple[Literal["int"], int]
    | tuple[Literal["float"], str]
    | tuple[Literal["str"], str]
    | tuple[Literal["tuple"], tuple[CanonicalValue, ...]]
)


def _canonical_value(value: object) -> CanonicalValue:
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("specialization float axes must be finite")
        return ("float", value.hex())
    if type(value) is str:
        return ("str", value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_canonical_value(item) for item in value))
    raise TypeError(
        "specialization axes must contain only None, bool, int, finite float, "
        "str, or tuples of those values"
    )


@dataclass(frozen=True, slots=True)
class LogicalKernel:
    """Finite semantic identity shared by all compile-time specializations."""

    operation: str
    variant: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("logical kernel operation must be a non-empty string")
        if not isinstance(self.variant, str) or not self.variant:
            raise ValueError("logical kernel variant must be a non-empty string")


@dataclass(frozen=True, slots=True)
class MetalSpecializationKey:
    """Canonical code-changing axes for one logical Metal kernel."""

    logical_kernel: LogicalKernel
    axes: tuple[tuple[str, CanonicalValue], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.logical_kernel, LogicalKernel):
            raise TypeError("logical_kernel must be a LogicalKernel")
        names = tuple(name for name, _value in self.axes)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("specialization axis names must be non-empty strings")
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("specialization axes must be unique and sorted by name")

    @classmethod
    def from_axes(
        cls, logical_kernel: LogicalKernel, axes: Mapping[str, object]
    ) -> MetalSpecializationKey:
        if not isinstance(axes, Mapping):
            raise TypeError("axes must be a mapping")
        if any(not isinstance(name, str) for name in axes):
            raise TypeError("specialization axis names must be strings")
        canonical = tuple(
            (name, _canonical_value(value))
            for name, value in sorted(axes.items(), key=lambda item: item[0])
        )
        return cls(logical_kernel, canonical)


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class GeneratedMetalFacts:
    """Complete provider facts exposed after one successful Metal compilation."""

    provider: str
    target: str
    execution_backend: str
    tilelang_version: str
    torch_version: str
    device_source: str = field(repr=False)
    host_source: str = field(repr=False)
    runtime_artifact_digest: str | None = None
    device_source_digest: str = field(init=False)
    host_source_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "provider",
            "target",
            "execution_backend",
            "tilelang_version",
            "torch_version",
            "device_source",
            "host_source",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.runtime_artifact_digest is not None:
            _validate_digest(self.runtime_artifact_digest, "runtime_artifact_digest")
        object.__setattr__(
            self,
            "device_source_digest",
            sha256(self.device_source.encode()).hexdigest(),
        )
        object.__setattr__(
            self,
            "host_source_digest",
            sha256(self.host_source.encode()).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class CompiledMetalKernel:
    """Callable TileLang specialization paired with immutable generated facts."""

    executable: Callable[..., object] = field(compare=False, repr=False)
    facts: GeneratedMetalFacts

    def __post_init__(self) -> None:
        if not callable(self.executable):
            raise TypeError("compiled Metal executable must be callable")
        if not isinstance(self.facts, GeneratedMetalFacts):
            raise TypeError("compiled Metal facts must be GeneratedMetalFacts")


class MetalJITCache:
    """Bounded, failure-atomic cache over a fixed logical-kernel manifest."""

    def __init__(
        self,
        logical_kernels: tuple[LogicalKernel, ...],
        *,
        max_specializations: int = 256,
    ) -> None:
        if type(logical_kernels) is not tuple:
            raise TypeError("logical_kernels must be a tuple")
        if not logical_kernels or len(logical_kernels) != len(set(logical_kernels)):
            raise ValueError("logical_kernels must be a non-empty unique tuple")
        if max_specializations <= 0:
            raise ValueError("max_specializations must be positive")
        self._logical_kernels = logical_kernels
        self._known_logical_kernels = frozenset(logical_kernels)
        self._max_specializations = max_specializations
        self._entries: OrderedDict[MetalSpecializationKey, CompiledMetalKernel] = (
            OrderedDict()
        )
        self._lock = RLock()

    @property
    def logical_kernels(self) -> tuple[LogicalKernel, ...]:
        return self._logical_kernels

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_or_compile(
        self,
        key: MetalSpecializationKey,
        compiler: Callable[[MetalSpecializationKey], CompiledMetalKernel],
    ) -> CompiledMetalKernel:
        if not isinstance(key, MetalSpecializationKey):
            raise TypeError("key must be a MetalSpecializationKey")
        if key.logical_kernel not in self._known_logical_kernels:
            raise KeyError("specialization uses an undeclared logical kernel")
        if not callable(compiler):
            raise TypeError("compiler must be callable")
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached

            compiled = compiler(key)
            if not isinstance(compiled, CompiledMetalKernel):
                raise TypeError("compiler must return a CompiledMetalKernel")
            self._entries[key] = compiled
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_specializations:
                self._entries.popitem(last=False)
            return compiled

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


__all__: list[str] = []
