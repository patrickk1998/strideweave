"""Lazy availability checks for the optional TileLang Metal runtime."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

_OPTIONAL_PACKAGES = ("torch", "tilelang")
_INSTALL_HINT = (
    "install the optional dependency set with `pip install 'strideweave[metal]'`"
)


@dataclass(frozen=True, slots=True)
class MetalRuntime:
    """Validated optional modules required by the TileLang Metal boundary."""

    torch: ModuleType
    tilelang: ModuleType

    def synchronize(self) -> None:
        """Wait for all queued MPS work before a host observation."""
        self.torch.mps.synchronize()


def load_metal_runtime() -> MetalRuntime:
    """Load and validate the optional TileLang Metal dependency boundary."""
    system = platform.system()
    machine = platform.machine()
    if system != "Darwin" or machine != "arm64":
        raise RuntimeError(
            "Metal requires macOS on Apple silicon; "
            f"found {system or '<unknown>'} on {machine or '<unknown>'}"
        )

    missing = tuple(name for name in _OPTIONAL_PACKAGES if find_spec(name) is None)
    if missing:
        packages = ", ".join(repr(name) for name in missing)
        raise RuntimeError(
            f"Metal is missing optional dependency {packages}; {_INSTALL_HINT}"
        )

    try:
        torch = import_module("torch")
        tilelang = import_module("tilelang")
    except ImportError as exc:
        raise RuntimeError(
            "Metal could not load its optional runtime dependencies; "
            f"{_INSTALL_HINT}. Original import error: {exc}"
        ) from exc

    torch_version = getattr(torch, "__version__", "<unknown>")
    tilelang_version = getattr(tilelang, "__version__", "<unknown>")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    is_built_method = getattr(mps_backend, "is_built", None)
    is_available_method = getattr(mps_backend, "is_available", None)
    is_built = callable(is_built_method) and bool(is_built_method())
    is_available = callable(is_available_method) and bool(is_available_method())
    facts = f"torch={torch_version}, tilelang={tilelang_version}, mps_built={is_built}"
    if not is_built:
        raise RuntimeError("Metal requires a PyTorch build with MPS support; " + facts)
    if not is_available:
        raise RuntimeError(
            "Metal requires an available MPS device, but PyTorch cannot use one "
            "on this host. Use Apple Metal hardware and a PyTorch build supported "
            f"by this macOS release; {facts}"
        )

    mps_module = getattr(torch, "mps", None)
    if not callable(getattr(mps_module, "compile_shader", None)):
        raise RuntimeError(
            "Metal requires torch.mps.compile_shader, which this PyTorch runtime "
            f"does not expose; {facts}"
        )
    if not callable(getattr(mps_module, "synchronize", None)):
        raise RuntimeError(
            "Metal requires torch.mps.synchronize for deterministic host "
            f"observation; {facts}"
        )

    return MetalRuntime(torch=torch, tilelang=tilelang)


__all__: list[str] = []
