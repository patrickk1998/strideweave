"""Failure-atomic in-memory specialization cache for TileLang Metal kernels."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import import_module
from threading import RLock
from typing import Any, Literal, cast

type CanonicalValue = (
    tuple[Literal["none"]]
    | tuple[Literal["bool"], bool]
    | tuple[Literal["int"], int]
    | tuple[Literal["float"], str]
    | tuple[Literal["str"], str]
    | tuple[Literal["tuple"], tuple[CanonicalValue, ...]]
)


class MetalRecipeError(ValueError):
    """A provider-owned Metal reconstruction recipe is malformed."""


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


def _decoded_value(value: CanonicalValue) -> object:
    raw = cast(object, value)
    if type(raw) is not tuple or not raw:
        raise ValueError("canonical specialization value must be a tagged tuple")
    kind = raw[0]
    if type(kind) is not str:
        raise ValueError("canonical specialization value tag must be a string")
    if kind == "none":
        if len(raw) != 1:
            raise ValueError("canonical none specialization value has invalid arity")
        return None
    if kind == "bool":
        if len(raw) != 2 or type(raw[1]) is not bool:
            raise ValueError("canonical bool specialization value is invalid")
        return raw[1]
    if kind == "int":
        if len(raw) != 2 or type(raw[1]) is not int:
            raise ValueError("canonical int specialization value is invalid")
        return raw[1]
    if kind == "str":
        if len(raw) != 2 or type(raw[1]) is not str:
            raise ValueError("canonical string specialization value is invalid")
        return raw[1]
    if kind == "float":
        if len(raw) != 2 or type(raw[1]) is not str:
            raise ValueError("canonical float specialization value is invalid")
        try:
            decoded = float.fromhex(raw[1])
        except (OverflowError, ValueError) as exc:
            raise MetalRecipeError(
                "canonical float specialization value is invalid"
            ) from exc
        if not math.isfinite(decoded) or decoded.hex() != raw[1]:
            raise ValueError("canonical float specialization value is invalid")
        return decoded
    if kind == "tuple":
        if len(raw) != 2 or type(raw[1]) is not tuple:
            raise ValueError("canonical tuple specialization value is invalid")
        return tuple(_decoded_value(cast(CanonicalValue, item)) for item in raw[1])
    raise ValueError(f"unknown canonical specialization value kind {kind!r}")


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
        if type(self.axes) is not tuple or any(
            type(axis) is not tuple or len(axis) != 2 for axis in self.axes
        ):
            raise ValueError("specialization axes must be name/value pairs")
        names = tuple(name for name, _value in self.axes)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("specialization axis names must be non-empty strings")
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("specialization axes must be unique and sorted by name")
        for _name, value in self.axes:
            _decoded_value(value)

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


type CompilationObserver = Callable[[MetalSpecializationKey, CompiledMetalKernel], None]
type MetalCompiler = Callable[[MetalSpecializationKey], CompiledMetalKernel]
type RecipeParser = Callable[[MetalSpecializationKey], object]
type RecipeCompiler = Callable[
    [Any, MetalSpecializationKey, object], CompiledMetalKernel
]

_COMPILATION_OBSERVERS: ContextVar[tuple[CompilationObserver, ...]] = ContextVar(
    "strideweave_metal_compilation_observers", default=()
)

_RECIPE_MODULES = {
    "metal.pointwise": "strideweave.carriers.metal._executor",
    "metal.reduction": "strideweave.carriers.metal._reduction_executor",
    "metal.scan": "strideweave.carriers.metal._reduction_executor",
    "metal.indexing": "strideweave.carriers.metal._indexing_executor",
    "metal.selection": "strideweave.carriers.metal._indexing_executor",
    "metal.matmul": "strideweave.carriers.metal._contraction_executor",
    "metal.conv_general": "strideweave.carriers.metal._contraction_executor",
}


@contextmanager
def observe_compilations(observer: CompilationObserver):
    """Observe exact cached or newly compiled specializations before launch."""
    if not callable(observer):
        raise TypeError("compilation observer must be callable")
    token = _COMPILATION_OBSERVERS.set((*_COMPILATION_OBSERVERS.get(), observer))
    try:
        yield
    finally:
        _COMPILATION_OBSERVERS.reset(token)


def _publish_compilation(
    key: MetalSpecializationKey, compiled: CompiledMetalKernel
) -> None:
    for observer in _COMPILATION_OBSERVERS.get():
        observer(key, compiled)


def specialization_axes(
    key: MetalSpecializationKey,
    expected_names: frozenset[str],
) -> dict[str, object]:
    """Decode one canonical recipe while requiring its exact axis vocabulary."""
    if not isinstance(key, MetalSpecializationKey):
        raise TypeError("key must be a MetalSpecializationKey")
    axes = {name: _decoded_value(value) for name, value in key.axes}
    if axes.keys() != expected_names:
        missing = sorted(expected_names - axes.keys())
        extra = sorted(axes.keys() - expected_names)
        raise MetalRecipeError(
            "Metal specialization recipe has unexpected axes; "
            f"missing={missing}, extra={extra}"
        )
    return axes


def _validated_recipe(key: MetalSpecializationKey) -> tuple[Any, object]:
    if not isinstance(key, MetalSpecializationKey):
        raise TypeError("key must be a MetalSpecializationKey")
    try:
        module_name = _RECIPE_MODULES[key.logical_kernel.operation]
    except KeyError as exc:
        raise RuntimeError(
            "Metal specialization has no provider-owned reconstruction recipe: "
            f"{key.logical_kernel.operation}/{key.logical_kernel.variant}"
        ) from exc
    module = import_module(module_name)
    parser = cast(RecipeParser, getattr(module, "_recipe_from_key", None))
    compiler = cast(RecipeCompiler, getattr(module, "_recompile_recipe", None))
    if not callable(parser) or not callable(compiler):
        raise RuntimeError(
            "Metal specialization family has no callable reconstruction recipe: "
            f"{key.logical_kernel.operation}"
        )
    return module, parser(key)


def validate_specialization_recipe(key: MetalSpecializationKey) -> None:
    """Validate provider recipe data without importing the optional runtime."""
    _validated_recipe(key)


def recompile_specialization(key: MetalSpecializationKey) -> CompiledMetalKernel:
    """Regenerate current facts from a provider-owned recipe without launch."""
    module, recipe = _validated_recipe(key)
    compiler = cast(RecipeCompiler, getattr(module, "_recompile_recipe"))
    from ._runtime import load_metal_runtime

    compiled = compiler(load_metal_runtime(), key, recipe)
    if not isinstance(compiled, CompiledMetalKernel):
        raise TypeError("Metal specialization recipe returned an invalid kernel")
    return compiled


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
        compiler: MetalCompiler,
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
                compiled = cached
            else:
                compiled = compiler(key)
                if not isinstance(compiled, CompiledMetalKernel):
                    raise TypeError("compiler must return a CompiledMetalKernel")
                self._entries[key] = compiled
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_specializations:
                    self._entries.popitem(last=False)
        # Provenance observers run after the cache transaction but before the
        # caller can launch the returned executable. An observer that cannot
        # bind complete facts therefore fails closed without publishing
        # verification evidence or entering the kernel.
        _publish_compilation(key, compiled)
        return compiled

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


__all__: list[str] = []
