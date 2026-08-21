"""Shared lazy transport helpers for TileLang Metal kernels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

from ...layout import Layout
from ...tensor import Tensor
from ..dtype import DType
from ..operation_policy import OperationPlan
from ._address_plan import address_plan
from ._jit import CompiledMetalKernel, GeneratedMetalFacts

_BUFFER_PATTERN = re.compile(
    r"\b(?P<name>status|output|output_addresses|input\d+|addresses\d+)\b\s*"
    r"\[\[\s*buffer\((?P<index>\d+)\)\s*\]\]"
)


@dataclass(frozen=True, slots=True)
class PreparedTensor:
    """Private storage and immutable logical address facts for one tensor."""

    source: Any
    storage_dtype: DType
    storage_size: int
    addresses: tuple[int, ...]
    address_key: object


def dtype_name(dtype: DType) -> str:
    """Return TileLang's spelling for one Metal storage dtype."""
    if dtype is DType.Float32:
        return "float32"
    if dtype is DType.Int32:
        return "int32"
    if dtype is DType.Bool:
        return "bool"
    raise TypeError(f"Metal kernels do not support DType.{dtype.name}")


def prepare_tensor(tensor: Tensor, *, family: str) -> PreparedTensor:
    """Validate and snapshot a tensor's private Metal storage/address boundary."""
    carrier = cast(Any, tensor.carrier)
    plan = address_plan(tensor.layout, offset=tensor.offset)
    if any(
        address < 0 or address >= carrier.size() or address > 2**31 - 1
        for address in plan.addresses
    ):
        raise RuntimeError(
            f"Metal {family} address plan contains an address outside source "
            "storage or the Int32 map range"
        )
    return PreparedTensor(
        source=carrier._require_storage(),
        storage_dtype=tensor.dtype(),
        storage_size=carrier.size(),
        addresses=plan.addresses,
        address_key=plan.key,
    )


def compile_kernel(
    runtime: Any,
    prim_func: Any,
    output_index: int,
    expected_buffers: tuple[str, ...],
    *,
    family: str,
    pass_configs: tuple[tuple[str, object], ...],
) -> CompiledMetalKernel:
    """Compile one direct JIT kernel and bind its emitted Metal buffer ABI."""
    kernel = runtime.tilelang.JITKernel(
        prim_func,
        out_idx=[output_index],
        target="metal",
        execution_backend="torch",
        pass_configs=dict(pass_configs),
    )
    device_source = kernel.kernel_source
    bindings: dict[str, int] = {}
    for match in _BUFFER_PATTERN.finditer(device_source):
        name = match.group("name")
        index = int(match.group("index"))
        if name in bindings and bindings[name] != index:
            raise RuntimeError(f"TileLang emitted ambiguous Metal buffer {name!r}")
        bindings[name] = index
    if set(bindings) != set(expected_buffers) or set(bindings.values()) != set(
        range(len(expected_buffers))
    ):
        raise RuntimeError(
            f"TileLang emitted an unexpected Metal {family} buffer ABI; "
            f"expected={sorted(expected_buffers)}, bindings={bindings}"
        )
    order = tuple(
        name for name, _ in sorted(bindings.items(), key=lambda item: item[1])
    )

    def executable(**buffers: Any) -> object:
        return kernel(*(buffers[name] for name in order))

    return CompiledMetalKernel(
        executable,
        GeneratedMetalFacts(
            provider="tilelang.JITKernel",
            target="metal",
            execution_backend="torch",
            tilelang_version=str(runtime.tilelang.__version__),
            torch_version=str(runtime.torch.__version__),
            device_source=device_source,
            host_source=kernel.host_source,
            runtime_artifact_digest=None,
        ),
    )


def plan_axis(plan: OperationPlan | None, variant: str) -> tuple[object, ...]:
    """Return the complete central-plan identity used by a specialization."""
    if plan is None:
        return ("internal-gradient", variant)
    return (
        plan.operation,
        tuple(
            (
                operand.role.value,
                operand.dtype.name if operand.dtype is not None else None,
                operand.convert_to.name,
            )
            for operand in plan.operands
        ),
        plan.compute.value,
        plan.accumulation.value if plan.accumulation is not None else None,
        plan.accumulator_dtype.name if plan.accumulator_dtype is not None else None,
        plan.output.name,
    )


def address_tensor(runtime: Any, prepared: PreparedTensor) -> Any:
    """Upload one immutable logical-to-physical map to MPS."""
    return runtime.torch.tensor(
        prepared.addresses,
        dtype=runtime.torch.int32,
        device="mps",
    )


def layout_address_tensor(
    runtime: Any, layout: Layout, *, family: str
) -> tuple[object, Any]:
    """Validate and upload an injective result layout's address map."""
    plan = address_plan(layout)
    if not plan.is_injective:
        raise RuntimeError(f"Metal {family} results require an injective layout")
    if any(address < 0 or address > 2**31 - 1 for address in plan.addresses):
        raise RuntimeError(
            f"Metal {family} result address plan exceeds the Int32 map range"
        )
    return plan.key, runtime.torch.tensor(
        plan.addresses,
        dtype=runtime.torch.int32,
        device="mps",
    )


__all__: list[str] = []
