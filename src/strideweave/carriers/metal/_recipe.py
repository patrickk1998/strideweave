"""Typed validation boundary for provider-owned Metal JIT recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

from ..dtype import DType, SimpleDType
from ..operation_policy import (
    Accumulation,
    Arithmetic,
    OperandPlan,
    OperandRole,
    OperationPlan,
)
from ._jit import MetalRecipeError, MetalSpecializationKey, specialization_axes
from ._kernel_support import (
    PreparedTensor,
    dtype_from_name,
    prepared_tensor_from_recipe,
)

_DTYPE_NAMES = frozenset({DType.Float32.name, DType.Int32.name, DType.Bool.name})
_ARITHMETIC_NAMES = frozenset(item.value for item in Arithmetic)
_ACCUMULATION_NAMES = frozenset(item.value for item in Accumulation)
_ROLE_NAMES = frozenset(item.value for item in OperandRole)
_SCALAR_ADDRESS_KEY = ("strideweave.metal.scalar-address.v1",)


def _fail(message: str) -> MetalRecipeError:
    return MetalRecipeError(message)


def _axes(key: MetalSpecializationKey, names: frozenset[str]) -> dict[str, object]:
    return specialization_axes(key, names)


def _exact_string(value: object, expected: str, name: str) -> str:
    if type(value) is not str or value != expected:
        raise _fail(f"Metal recipe {name} is inconsistent")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise _fail(f"Metal recipe {name} must be a positive integer")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise _fail(f"Metal recipe {name} must be boolean")
    return value


def _tuple(value: object, name: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise _fail(f"Metal recipe {name} must be a tuple")
    return cast(tuple[object, ...], value)


def _exact_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is tuple:
        actual_items = cast(tuple[object, ...], actual)
        expected_items = cast(tuple[object, ...], expected)
        return len(actual_items) == len(expected_items) and all(
            _exact_value(left, right)
            for left, right in zip(actual_items, expected_items, strict=True)
        )
    return actual == expected


def _tensor(
    address_key: object,
    storage_size: object,
    dtype: DType,
    name: str,
) -> PreparedTensor:
    try:
        return prepared_tensor_from_recipe(
            address_key,
            storage_size,
            storage_dtype=dtype,
            cache_address_plan=False,
        )
    except (TypeError, ValueError) as exc:
        raise _fail(f"Metal recipe {name} tensor is invalid: {exc}") from exc


def _output_tensor(
    address_key: object,
    storage_size: object,
    dtype: DType,
    name: str,
) -> PreparedTensor:
    tensor = _tensor(address_key, storage_size, dtype, name)
    if not tensor.is_injective:
        raise _fail(f"Metal recipe {name} address plan must be injective")
    return tensor


def _require_count(tensor: PreparedTensor, expected: int, name: str) -> None:
    if len(tensor.addresses) != expected:
        raise _fail(
            f"Metal recipe {name} address cardinality is inconsistent; "
            f"expected={expected}, actual={len(tensor.addresses)}"
        )


def _require_axis_groups(size: int, stride: int, extent: int, name: str) -> int:
    group = stride * extent
    if size % group:
        raise _fail(
            f"Metal recipe {name} cardinality is inconsistent with axis dimensions"
        )
    return size // group


@dataclass(frozen=True, slots=True)
class PlanRecipe:
    """Validated central operation-plan identity."""

    operation: str
    operands: tuple[tuple[str, str | None, str], ...]
    compute: str
    accumulation: str | None
    accumulator_dtype: str | None
    output: str


def _plan_dtype(name: str) -> SimpleDType:
    return cast(SimpleDType, dtype_from_name(name))


def _require_current_metal_plan(plan: PlanRecipe) -> None:
    """Require every decoded field to match one current central Metal plan."""
    from .capabilities import metal_capabilities

    resolved = OperationPlan(
        operation=plan.operation,
        operands=tuple(
            OperandPlan(
                role=OperandRole(role),
                dtype=None if dtype is None else _plan_dtype(dtype),
                convert_to=_plan_dtype(convert_to),
            )
            for role, dtype, convert_to in plan.operands
        ),
        compute=Arithmetic(plan.compute),
        accumulation=(
            None if plan.accumulation is None else Accumulation(plan.accumulation)
        ),
        accumulator_dtype=(
            None
            if plan.accumulator_dtype is None
            else _plan_dtype(plan.accumulator_dtype)
        ),
        output=_plan_dtype(plan.output),
    )
    if not any(capability.matches(resolved) for capability in metal_capabilities()):
        raise _fail("Metal recipe plan is not a current executable operation plan")


def _plan(
    value: object,
    variant: str,
    *,
    expected_operation: str | None = None,
    allow_internal: bool,
    require_internal: bool = False,
) -> PlanRecipe | None:
    raw = _tuple(value, "plan")
    if raw == ("internal-gradient", variant):
        if not allow_internal:
            raise _fail("Metal recipe unexpectedly uses an internal-gradient plan")
        return None
    if require_internal:
        raise _fail("Metal recipe internal-gradient plan is inconsistent")
    if len(raw) != 6:
        raise _fail("Metal recipe plan has invalid arity")
    operation, operand_value, compute, accumulation, accumulator_dtype, output = raw
    if type(operation) is not str or not operation:
        raise _fail("Metal recipe plan operation is invalid")
    if expected_operation is not None and operation != expected_operation:
        raise _fail("Metal recipe plan operation is inconsistent")
    operand_items = _tuple(operand_value, "plan operands")
    operands: list[tuple[str, str | None, str]] = []
    for item in operand_items:
        fields = _tuple(item, "plan operand")
        if len(fields) != 3:
            raise _fail("Metal recipe plan operand has invalid arity")
        role, dtype, convert_to = fields
        if type(role) is not str or role not in _ROLE_NAMES:
            raise _fail("Metal recipe plan operand role is invalid")
        if dtype is not None and (type(dtype) is not str or dtype not in _DTYPE_NAMES):
            raise _fail("Metal recipe plan operand dtype is invalid")
        if type(convert_to) is not str or convert_to not in _DTYPE_NAMES:
            raise _fail("Metal recipe plan operand conversion dtype is invalid")
        if (role == OperandRole.TENSOR.value) != (dtype is not None):
            raise _fail("Metal recipe plan operand role/dtype pair is inconsistent")
        operands.append((role, cast(str | None, dtype), convert_to))
    if type(compute) is not str or compute not in _ARITHMETIC_NAMES:
        raise _fail("Metal recipe plan arithmetic is invalid")
    if accumulation is not None and (
        type(accumulation) is not str or accumulation not in _ACCUMULATION_NAMES
    ):
        raise _fail("Metal recipe plan accumulation is invalid")
    if accumulator_dtype is not None and (
        type(accumulator_dtype) is not str or accumulator_dtype not in _DTYPE_NAMES
    ):
        raise _fail("Metal recipe plan accumulator dtype is invalid")
    if type(output) is not str or output not in _DTYPE_NAMES:
        raise _fail("Metal recipe plan output dtype is invalid")
    if (accumulation == Accumulation.FLOATING.value) != (accumulator_dtype is not None):
        raise _fail("Metal recipe plan accumulation dtype is inconsistent")
    plan = PlanRecipe(
        operation,
        tuple(operands),
        compute,
        cast(str | None, accumulation),
        cast(str | None, accumulator_dtype),
        output,
    )
    _require_current_metal_plan(plan)
    return plan


def _require_plan_signature(
    plan: PlanRecipe | None,
    input_dtypes: tuple[DType, ...],
    *,
    output_dtype: DType | None = None,
) -> None:
    if plan is None or len(plan.operands) != len(input_dtypes):
        raise _fail("Metal recipe plan operand arity is inconsistent")
    for operand, dtype in zip(plan.operands, input_dtypes, strict=True):
        role, storage_name, _convert_name = operand
        if role != OperandRole.TENSOR.value or storage_name != dtype.name:
            raise _fail("Metal recipe plan operand dtype is inconsistent")
    if output_dtype is not None and plan.output != output_dtype.name:
        raise _fail("Metal recipe output dtype disagrees with plan")


def _header(
    key: MetalSpecializationKey,
    axes: dict[str, object],
    *,
    template_revision: str,
    pass_configs: tuple[tuple[str, object], ...] | None,
    operation_axis: bool = True,
) -> str:
    variant = key.logical_kernel.variant
    if operation_axis:
        _exact_string(axes["operation"], variant, "operation")
    _exact_string(axes["template_revision"], template_revision, "template revision")
    if pass_configs is not None and not _exact_value(
        axes["pass_configs"], pass_configs
    ):
        raise _fail("Metal recipe pass configuration is unavailable")
    return variant


@dataclass(frozen=True, slots=True)
class PointwiseOperandRecipe:
    storage_dtype: DType
    convert_dtype: DType
    storage_size: int


@dataclass(frozen=True, slots=True)
class PointwiseRecipe:
    expression: str
    operands: tuple[PointwiseOperandRecipe, ...]
    output_dtype: DType
    logical_size: int


_POINTWISE_AXES = frozenset(
    {
        "address_plans",
        "convert_dtypes",
        "expression",
        "logical_size",
        "output_dtype",
        "physical_sizes",
        "plan",
        "storage_dtypes",
        "template_revision",
    }
)


def pointwise_recipe(
    key: MetalSpecializationKey, *, template_revision: str
) -> PointwiseRecipe:
    axes = _axes(key, _POINTWISE_AXES)
    expression = _exact_string(
        axes["expression"], key.logical_kernel.variant, "pointwise expression"
    )
    _exact_string(axes["template_revision"], template_revision, "template revision")
    logical_size = _positive_int(axes["logical_size"], "logical_size")
    address_keys = _tuple(axes["address_plans"], "address_plans")
    storage_names = _tuple(axes["storage_dtypes"], "storage_dtypes")
    convert_names = _tuple(axes["convert_dtypes"], "convert_dtypes")
    physical_sizes = _tuple(axes["physical_sizes"], "physical_sizes")
    arity = len(address_keys)
    if not 1 <= arity <= 4 or any(
        len(items) != arity for items in (storage_names, convert_names, physical_sizes)
    ):
        raise _fail("Metal pointwise recipe operand arity is inconsistent")
    operands: list[PointwiseOperandRecipe] = []
    scalar_flags: list[bool] = []
    for index, (address_key, storage_name, convert_name, physical_size) in enumerate(
        zip(
            address_keys,
            storage_names,
            convert_names,
            physical_sizes,
            strict=True,
        )
    ):
        storage_dtype = dtype_from_name(storage_name)
        convert_dtype = dtype_from_name(convert_name)
        scalar = address_key == _SCALAR_ADDRESS_KEY
        scalar_flags.append(scalar)
        if scalar:
            storage_size = _positive_int(physical_size, f"operand {index} storage_size")
            if storage_size != 1:
                raise _fail(
                    "Metal pointwise scalar recipe must use one storage element"
                )
        else:
            prepared = _tensor(
                address_key,
                physical_size,
                storage_dtype,
                f"pointwise operand {index}",
            )
            _require_count(prepared, logical_size, f"pointwise operand {index}")
            storage_size = prepared.storage_size
        operands.append(
            PointwiseOperandRecipe(storage_dtype, convert_dtype, storage_size)
        )
    output_dtype = dtype_from_name(axes["output_dtype"])
    plan = _plan(
        axes["plan"],
        expression,
        expected_operation=expression,
        allow_internal=expression.startswith("grad_")
        or expression == "elementwise_mul",
        require_internal=expression.startswith("grad_"),
    )
    if plan is not None:
        if len(plan.operands) != arity:
            raise _fail("Metal pointwise recipe plan operand arity is inconsistent")
        for index, (operand, planned, scalar) in enumerate(
            zip(operands, plan.operands, scalar_flags, strict=True)
        ):
            role, dtype_name, convert_name = planned
            expected_role = (
                OperandRole.WEAK_SCALAR.value if scalar else OperandRole.TENSOR.value
            )
            if (
                role != expected_role
                or (dtype_name is not None and dtype_name != operand.storage_dtype.name)
                or convert_name != operand.convert_dtype.name
            ):
                raise _fail(
                    f"Metal pointwise recipe operand {index} disagrees with plan"
                )
        if plan.output != output_dtype.name:
            raise _fail("Metal pointwise recipe output dtype disagrees with plan")
    return PointwiseRecipe(expression, tuple(operands), output_dtype, logical_size)


@dataclass(frozen=True, slots=True)
class ReductionForwardRecipe:
    operation: str
    source: PreparedTensor
    output_dtype: DType
    row_count: int
    fiber_size: int


@dataclass(frozen=True, slots=True)
class ReductionBackwardRecipe:
    operation: str
    source: PreparedTensor
    gradient: PreparedTensor
    output: PreparedTensor
    row_count: int
    fiber_size: int


@dataclass(frozen=True, slots=True)
class ScanRecipe:
    operation: str
    source: PreparedTensor
    output: PreparedTensor
    logical_size: int
    axis_stride: int
    axis_extent: int


ReductionRecipe: TypeAlias = (
    ReductionForwardRecipe | ReductionBackwardRecipe | ScanRecipe
)

_REDUCTION_FORWARD_AXES = frozenset(
    {
        "address_plan",
        "fiber_size",
        "operation",
        "output_dtype",
        "pass_configs",
        "plan",
        "row_count",
        "storage_dtype",
        "storage_size",
        "template_revision",
    }
)
_REDUCTION_BACKWARD_AXES = frozenset(
    {
        "fiber_size",
        "gradient_address_plan",
        "gradient_storage_size",
        "operation",
        "output_address_plan",
        "output_storage_size",
        "pass_configs",
        "plan",
        "row_count",
        "source_address_plan",
        "source_storage_size",
        "template_revision",
    }
)
_SCAN_AXES = frozenset(
    {
        "address_plan",
        "axis_extent",
        "axis_stride",
        "logical_size",
        "operation",
        "output_address_plan",
        "output_storage_size",
        "pass_configs",
        "plan",
        "storage_size",
        "template_revision",
    }
)


def reduction_recipe(
    key: MetalSpecializationKey,
    *,
    template_revision: str,
    pass_configs: tuple[tuple[str, object], ...],
    forward_variants: frozenset[str],
    backward_variants: frozenset[str],
) -> ReductionRecipe:
    variant = key.logical_kernel.variant
    if variant in forward_variants:
        axes = _axes(key, _REDUCTION_FORWARD_AXES)
        _header(
            key, axes, template_revision=template_revision, pass_configs=pass_configs
        )
        rows = _positive_int(axes["row_count"], "row_count")
        fiber = _positive_int(axes["fiber_size"], "fiber_size")
        source = _tensor(
            axes["address_plan"],
            axes["storage_size"],
            dtype_from_name(axes["storage_dtype"]),
            "reduction source",
        )
        _require_count(source, rows * fiber, "reduction source")
        output_dtype = dtype_from_name(axes["output_dtype"])
        plan = _plan(
            axes["plan"], variant, expected_operation=variant, allow_internal=False
        )
        _require_plan_signature(
            plan, (source.storage_dtype,), output_dtype=output_dtype
        )
        return ReductionForwardRecipe(variant, source, output_dtype, rows, fiber)
    if variant in backward_variants:
        axes = _axes(key, _REDUCTION_BACKWARD_AXES)
        _header(
            key, axes, template_revision=template_revision, pass_configs=pass_configs
        )
        rows = _positive_int(axes["row_count"], "row_count")
        fiber = _positive_int(axes["fiber_size"], "fiber_size")
        source = _tensor(
            axes["source_address_plan"],
            axes["source_storage_size"],
            DType.Float32,
            "reduction source",
        )
        gradient = _tensor(
            axes["gradient_address_plan"],
            axes["gradient_storage_size"],
            DType.Float32,
            "reduction gradient",
        )
        output = _output_tensor(
            axes["output_address_plan"],
            axes["output_storage_size"],
            DType.Float32,
            "reduction output",
        )
        _require_count(source, rows * fiber, "reduction source")
        _require_count(gradient, rows, "reduction gradient")
        _require_count(output, rows * fiber, "reduction output")
        _plan(axes["plan"], variant, allow_internal=True, require_internal=True)
        return ReductionBackwardRecipe(variant, source, gradient, output, rows, fiber)
    axes = _axes(key, _SCAN_AXES)
    _header(key, axes, template_revision=template_revision, pass_configs=pass_configs)
    logical_size = _positive_int(axes["logical_size"], "logical_size")
    stride = _positive_int(axes["axis_stride"], "axis_stride")
    extent = _positive_int(axes["axis_extent"], "axis_extent")
    source = _tensor(
        axes["address_plan"], axes["storage_size"], DType.Float32, "scan source"
    )
    output = _output_tensor(
        axes["output_address_plan"],
        axes["output_storage_size"],
        DType.Float32,
        "scan output",
    )
    _require_count(source, logical_size, "scan source")
    _require_count(output, logical_size, "scan output")
    _require_axis_groups(logical_size, stride, extent, "scan")
    plan = _plan(
        axes["plan"],
        variant,
        expected_operation="cumsum" if variant == "cumsum" else None,
        allow_internal=variant != "cumsum",
        require_internal=variant != "cumsum",
    )
    if variant == "cumsum":
        _require_plan_signature(plan, (DType.Float32,), output_dtype=DType.Float32)
    return ScanRecipe(variant, source, output, logical_size, stride, extent)


@dataclass(frozen=True, slots=True)
class ValidateIndicesRecipe:
    indices: PreparedTensor
    extent: int
    unique: bool


@dataclass(frozen=True, slots=True)
class GatherRecipe:
    source: PreparedTensor
    indices: PreparedTensor
    output_size: int
    axis_stride: int
    axis_extent: int


@dataclass(frozen=True, slots=True)
class ScatterRecipe:
    operation: str
    base: PreparedTensor
    indices: PreparedTensor
    updates: PreparedTensor
    axis_stride: int
    axis_extent: int


@dataclass(frozen=True, slots=True)
class GatherBackwardRecipe:
    indices: PreparedTensor
    gradient: PreparedTensor
    output: PreparedTensor
    source_size: int
    axis_stride: int
    axis_extent: int


@dataclass(frozen=True, slots=True)
class ScatterBackwardRecipe:
    operation: str
    indices: PreparedTensor
    gradient: PreparedTensor
    output: PreparedTensor
    axis_stride: int
    axis_extent: int


@dataclass(frozen=True, slots=True)
class SelectionRecipe:
    operation: str
    source: PreparedTensor
    output_dtype: DType
    output_size: int
    axis_stride: int
    axis_extent: int
    k: int
    descending: bool


@dataclass(frozen=True, slots=True)
class SelectionBackwardRecipe:
    source: PreparedTensor
    gradient: PreparedTensor
    output: PreparedTensor
    axis_stride: int
    axis_extent: int
    k: int
    descending: bool


IndexingRecipe: TypeAlias = (
    ValidateIndicesRecipe
    | GatherRecipe
    | ScatterRecipe
    | GatherBackwardRecipe
    | ScatterBackwardRecipe
    | SelectionRecipe
    | SelectionBackwardRecipe
)

_VALIDATE_AXES = frozenset(
    {
        "address_plan",
        "extent",
        "logical_size",
        "pass_configs",
        "storage_size",
        "template_revision",
        "unique",
    }
)
_GATHER_AXES = frozenset(
    {
        "axis_extent",
        "axis_stride",
        "index_address_plan",
        "index_size",
        "index_storage_size",
        "operation",
        "output_size",
        "pass_configs",
        "plan",
        "source_address_plan",
        "source_storage_size",
        "template_revision",
    }
)
_SCATTER_AXES = frozenset(
    {
        "axis_extent",
        "axis_stride",
        "base_address_plan",
        "base_storage_size",
        "index_address_plan",
        "index_storage_size",
        "operation",
        "pass_configs",
        "plan",
        "template_revision",
        "update_address_plan",
        "update_storage_size",
    }
)
_GRAD_GATHER_AXES = frozenset(
    {
        "axis_extent",
        "axis_stride",
        "gradient_address_plan",
        "gradient_storage_size",
        "index_address_plan",
        "index_storage_size",
        "operation",
        "output_address_plan",
        "output_storage_size",
        "pass_configs",
        "source_size",
        "template_revision",
    }
)
_GRAD_SCATTER_AXES = frozenset(
    {
        "axis_extent",
        "axis_stride",
        "gradient_address_plan",
        "gradient_storage_size",
        "index_address_plan",
        "index_storage_size",
        "operation",
        "output_address_plan",
        "output_storage_size",
        "pass_configs",
        "template_revision",
    }
)
_SELECTION_AXES = frozenset(
    {
        "address_plan",
        "axis_extent",
        "axis_stride",
        "descending",
        "k",
        "operation",
        "output_dtype",
        "output_size",
        "pass_configs",
        "plan",
        "storage_size",
        "template_revision",
    }
)
_GRAD_SELECTION_AXES = frozenset(
    {
        "address_plan",
        "axis_extent",
        "axis_stride",
        "descending",
        "gradient_address_plan",
        "gradient_storage_size",
        "k",
        "operation",
        "output_address_plan",
        "output_storage_size",
        "pass_configs",
        "source_storage_size",
        "template_revision",
    }
)


def indexing_recipe(
    key: MetalSpecializationKey,
    *,
    template_revision: str,
    pass_configs: tuple[tuple[str, object], ...],
) -> IndexingRecipe:
    variant = key.logical_kernel.variant
    if variant == "validate_indices":
        axes = _axes(key, _VALIDATE_AXES)
        _header(
            key,
            axes,
            template_revision=template_revision,
            pass_configs=pass_configs,
            operation_axis=False,
        )
        logical = _positive_int(axes["logical_size"], "logical_size")
        indices = _tensor(
            axes["address_plan"],
            axes["storage_size"],
            DType.Int32,
            "index validation input",
        )
        _require_count(indices, logical, "index validation input")
        return ValidateIndicesRecipe(
            indices,
            _positive_int(axes["extent"], "extent"),
            _bool(axes["unique"], "unique"),
        )
    if variant == "gather":
        axes = _axes(key, _GATHER_AXES)
        _header(
            key, axes, template_revision=template_revision, pass_configs=pass_configs
        )
        source = _tensor(
            axes["source_address_plan"],
            axes["source_storage_size"],
            DType.Float32,
            "gather source",
        )
        indices = _tensor(
            axes["index_address_plan"],
            axes["index_storage_size"],
            DType.Int32,
            "gather indices",
        )
        _require_count(
            indices, _positive_int(axes["index_size"], "index_size"), "gather indices"
        )
        output_size = _positive_int(axes["output_size"], "output_size")
        stride = _positive_int(axes["axis_stride"], "axis_stride")
        extent = _positive_int(axes["axis_extent"], "axis_extent")
        groups = _require_axis_groups(
            len(source.addresses), stride, extent, "gather source"
        )
        if output_size != groups * stride * len(indices.addresses):
            raise _fail("Metal gather recipe output cardinality is inconsistent")
        plan = _plan(
            axes["plan"], variant, expected_operation="gather", allow_internal=False
        )
        _require_plan_signature(
            plan, (DType.Float32, DType.Int32), output_dtype=DType.Float32
        )
        return GatherRecipe(source, indices, output_size, stride, extent)
    if variant in {"scatter", "scatter_add"}:
        axes = _axes(key, _SCATTER_AXES)
        _header(
            key, axes, template_revision=template_revision, pass_configs=pass_configs
        )
        base = _tensor(
            axes["base_address_plan"],
            axes["base_storage_size"],
            DType.Float32,
            "scatter base",
        )
        indices = _tensor(
            axes["index_address_plan"],
            axes["index_storage_size"],
            DType.Int32,
            "scatter indices",
        )
        updates = _tensor(
            axes["update_address_plan"],
            axes["update_storage_size"],
            DType.Float32,
            "scatter updates",
        )
        stride = _positive_int(axes["axis_stride"], "axis_stride")
        extent = _positive_int(axes["axis_extent"], "axis_extent")
        groups = _require_axis_groups(
            len(base.addresses), stride, extent, "scatter base"
        )
        if len(updates.addresses) != groups * stride * len(indices.addresses):
            raise _fail("Metal scatter recipe update cardinality is inconsistent")
        plan = _plan(
            axes["plan"], variant, expected_operation=variant, allow_internal=False
        )
        _require_plan_signature(
            plan,
            (DType.Float32, DType.Int32, DType.Float32),
            output_dtype=DType.Float32,
        )
        return ScatterRecipe(variant, base, indices, updates, stride, extent)
    if variant == "grad_gather":
        axes = _axes(key, _GRAD_GATHER_AXES)
        _header(
            key, axes, template_revision=template_revision, pass_configs=pass_configs
        )
        indices = _tensor(
            axes["index_address_plan"],
            axes["index_storage_size"],
            DType.Int32,
            "gather gradient indices",
        )
        gradient = _tensor(
            axes["gradient_address_plan"],
            axes["gradient_storage_size"],
            DType.Float32,
            "gather gradient",
        )
        output = _output_tensor(
            axes["output_address_plan"],
            axes["output_storage_size"],
            DType.Float32,
            "gather gradient output",
        )
        source_size = _positive_int(axes["source_size"], "source_size")
        stride = _positive_int(axes["axis_stride"], "axis_stride")
        extent = _positive_int(axes["axis_extent"], "axis_extent")
        groups = _require_axis_groups(
            source_size, stride, extent, "gather gradient source"
        )
        _require_count(output, source_size, "gather gradient output")
        _require_count(
            gradient, groups * stride * len(indices.addresses), "gather gradient"
        )
        return GatherBackwardRecipe(
            indices, gradient, output, source_size, stride, extent
        )
    if variant in {
        "grad_scatter_base",
        "grad_scatter_add_base",
        "grad_scatter_updates",
    }:
        axes = _axes(key, _GRAD_SCATTER_AXES)
        _header(
            key, axes, template_revision=template_revision, pass_configs=pass_configs
        )
        indices = _tensor(
            axes["index_address_plan"],
            axes["index_storage_size"],
            DType.Int32,
            "scatter gradient indices",
        )
        gradient = _tensor(
            axes["gradient_address_plan"],
            axes["gradient_storage_size"],
            DType.Float32,
            "scatter gradient",
        )
        output = _output_tensor(
            axes["output_address_plan"],
            axes["output_storage_size"],
            DType.Float32,
            "scatter gradient output",
        )
        stride = _positive_int(axes["axis_stride"], "axis_stride")
        extent = _positive_int(axes["axis_extent"], "axis_extent")
        groups = _require_axis_groups(
            len(gradient.addresses), stride, extent, "scatter gradient"
        )
        expected_output = (
            groups * stride * len(indices.addresses)
            if variant == "grad_scatter_updates"
            else len(gradient.addresses)
        )
        _require_count(output, expected_output, "scatter gradient output")
        return ScatterBackwardRecipe(variant, indices, gradient, output, stride, extent)
    if variant == "grad_selection":
        axes = _axes(key, _GRAD_SELECTION_AXES)
        _header(
            key, axes, template_revision=template_revision, pass_configs=pass_configs
        )
        source = _tensor(
            axes["address_plan"],
            axes["source_storage_size"],
            DType.Float32,
            "selection source",
        )
        gradient = _tensor(
            axes["gradient_address_plan"],
            axes["gradient_storage_size"],
            DType.Float32,
            "selection gradient",
        )
        output = _output_tensor(
            axes["output_address_plan"],
            axes["output_storage_size"],
            DType.Float32,
            "selection gradient output",
        )
        stride = _positive_int(axes["axis_stride"], "axis_stride")
        extent = _positive_int(axes["axis_extent"], "axis_extent")
        k = _positive_int(axes["k"], "k")
        if k > extent:
            raise _fail("Metal selection recipe k exceeds axis_extent")
        groups = _require_axis_groups(
            len(source.addresses), stride, extent, "selection source"
        )
        _require_count(gradient, groups * stride * k, "selection gradient")
        _require_count(output, len(source.addresses), "selection gradient output")
        return SelectionBackwardRecipe(
            source,
            gradient,
            output,
            stride,
            extent,
            k,
            _bool(axes["descending"], "descending"),
        )
    axes = _axes(key, _SELECTION_AXES)
    _header(key, axes, template_revision=template_revision, pass_configs=pass_configs)
    source = _tensor(
        axes["address_plan"], axes["storage_size"], DType.Float32, "selection source"
    )
    stride = _positive_int(axes["axis_stride"], "axis_stride")
    extent = _positive_int(axes["axis_extent"], "axis_extent")
    k = _positive_int(axes["k"], "k")
    if k > extent:
        raise _fail("Metal selection recipe k exceeds axis_extent")
    groups = _require_axis_groups(
        len(source.addresses), stride, extent, "selection source"
    )
    output_size = _positive_int(axes["output_size"], "output_size")
    if output_size != groups * stride * k:
        raise _fail("Metal selection recipe output cardinality is inconsistent")
    expected_dtype = DType.Int32 if variant.endswith("_indices") else DType.Float32
    output_dtype = dtype_from_name(axes["output_dtype"])
    if output_dtype is not expected_dtype:
        raise _fail("Metal selection recipe output dtype is inconsistent")
    plan = _plan(
        axes["plan"],
        variant,
        expected_operation=variant,
        allow_internal=False,
    )
    _require_plan_signature(plan, (DType.Float32,), output_dtype=output_dtype)
    return SelectionRecipe(
        variant,
        source,
        output_dtype,
        output_size,
        stride,
        extent,
        k,
        _bool(axes["descending"], "descending"),
    )


@dataclass(frozen=True, slots=True)
class MatmulRecipe:
    operation: str
    first: PreparedTensor
    second: PreparedTensor
    output: PreparedTensor
    n_size: int
    m_size: int
    k_size: int


@dataclass(frozen=True, slots=True)
class ConvolutionGeometryRecipe:
    batch_size: int
    input_features: int
    output_features: int
    channels_per_group: int
    input_spatial: tuple[int, ...]
    kernel_spatial: tuple[int, ...]
    output_spatial: tuple[int, ...]
    strides: tuple[int, ...]
    padding: tuple[tuple[int, int], ...]
    lhs_dilation: tuple[int, ...]
    kernel_dilation: tuple[int, ...]
    feature_groups: int

    @property
    def lhs_count(self) -> int:
        return self.batch_size * self.input_features * _product(self.input_spatial)

    @property
    def kernel_count(self) -> int:
        return (
            self.output_features
            * self.channels_per_group
            * _product(self.kernel_spatial)
        )

    @property
    def output_count(self) -> int:
        return self.batch_size * self.output_features * _product(self.output_spatial)

    @property
    def contraction_count(self) -> int:
        return self.channels_per_group * _product(self.kernel_spatial)


@dataclass(frozen=True, slots=True)
class ConvolutionRecipe:
    operation: str
    first: PreparedTensor
    second: PreparedTensor
    output: PreparedTensor
    geometry: ConvolutionGeometryRecipe


ContractionRecipe: TypeAlias = MatmulRecipe | ConvolutionRecipe

_MATMUL_AXES = frozenset(
    {
        "first_address_plan",
        "first_storage_size",
        "k_size",
        "m_size",
        "n_size",
        "operation",
        "output_address_plan",
        "output_storage_size",
        "pass_configs",
        "plan",
        "second_address_plan",
        "second_storage_size",
        "template_revision",
    }
)
_CONV_AXES = frozenset(
    {
        "convolution_signature",
        "first_address_plan",
        "first_storage_size",
        "operation",
        "output_address_plan",
        "output_storage_size",
        "pass_configs",
        "plan",
        "second_address_plan",
        "second_storage_size",
        "template_revision",
    }
)


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _positive_int_tuple(
    value: object, name: str, rank: int | None = None
) -> tuple[int, ...]:
    raw = _tuple(value, name)
    if rank is not None and len(raw) != rank:
        raise _fail(f"Metal convolution recipe {name} rank is inconsistent")
    return tuple(_positive_int(item, name) for item in raw)


def _conv_geometry(value: object) -> ConvolutionGeometryRecipe:
    signature = _tuple(value, "convolution_signature")
    if len(signature) != 12:
        raise _fail("Metal convolution recipe signature has invalid arity")
    (
        batch,
        input_features,
        output_features,
        channels,
        input_spatial_value,
        kernel_spatial_value,
        output_spatial_value,
        strides_value,
        padding_value,
        lhs_dilation_value,
        kernel_dilation_value,
        groups_value,
    ) = signature
    input_spatial = _positive_int_tuple(input_spatial_value, "input_spatial")
    rank = len(input_spatial)
    if rank < 1:
        raise _fail("Metal convolution recipe requires at least one spatial dimension")
    kernel_spatial = _positive_int_tuple(kernel_spatial_value, "kernel_spatial", rank)
    output_spatial = _positive_int_tuple(output_spatial_value, "output_spatial", rank)
    strides = _positive_int_tuple(strides_value, "strides", rank)
    lhs_dilation = _positive_int_tuple(lhs_dilation_value, "lhs_dilation", rank)
    kernel_dilation = _positive_int_tuple(
        kernel_dilation_value, "kernel_dilation", rank
    )
    padding_raw = _tuple(padding_value, "padding")
    if len(padding_raw) != rank:
        raise _fail("Metal convolution recipe padding rank is inconsistent")
    padding: list[tuple[int, int]] = []
    for pair_value in padding_raw:
        pair = _tuple(pair_value, "padding pair")
        if len(pair) != 2 or any(type(item) is not int or item < 0 for item in pair):
            raise _fail("Metal convolution recipe padding pair is invalid")
        padding.append((cast(int, pair[0]), cast(int, pair[1])))
    geometry = ConvolutionGeometryRecipe(
        _positive_int(batch, "batch_size"),
        _positive_int(input_features, "input_features"),
        _positive_int(output_features, "output_features"),
        _positive_int(channels, "channels_per_group"),
        input_spatial,
        kernel_spatial,
        output_spatial,
        strides,
        tuple(padding),
        lhs_dilation,
        kernel_dilation,
        _positive_int(groups_value, "feature_groups"),
    )
    if (
        geometry.input_features != geometry.channels_per_group * geometry.feature_groups
        or geometry.output_features % geometry.feature_groups
    ):
        raise _fail("Metal convolution recipe feature groups are inconsistent")
    expected_output: list[int] = []
    for dimension in range(rank):
        effective_input = (
            geometry.input_spatial[dimension] - 1
        ) * geometry.lhs_dilation[dimension] + 1
        effective_kernel = (
            geometry.kernel_spatial[dimension] - 1
        ) * geometry.kernel_dilation[dimension] + 1
        numerator = (
            geometry.padding[dimension][0]
            + effective_input
            + geometry.padding[dimension][1]
            - effective_kernel
        )
        expected_output.append(numerator // geometry.strides[dimension] + 1)
    if tuple(expected_output) != geometry.output_spatial:
        raise _fail("Metal convolution recipe output geometry is inconsistent")
    return geometry


def contraction_recipe(
    key: MetalSpecializationKey,
    *,
    template_revision: str,
    pass_configs: tuple[tuple[str, object], ...],
) -> ContractionRecipe:
    variant = key.logical_kernel.variant
    if key.logical_kernel.operation == "metal.matmul":
        axes = _axes(key, _MATMUL_AXES)
        _header(
            key, axes, template_revision=template_revision, pass_configs=pass_configs
        )
        first = _tensor(
            axes["first_address_plan"],
            axes["first_storage_size"],
            DType.Float32,
            "matmul first operand",
        )
        second = _tensor(
            axes["second_address_plan"],
            axes["second_storage_size"],
            DType.Float32,
            "matmul second operand",
        )
        output = _output_tensor(
            axes["output_address_plan"],
            axes["output_storage_size"],
            DType.Float32,
            "matmul output",
        )
        n = _positive_int(axes["n_size"], "n_size")
        m = _positive_int(axes["m_size"], "m_size")
        k = _positive_int(axes["k_size"], "k_size")
        counts = {
            "matmul_forward": (n * k, m * k, n * m),
            "matmul_grad_lhs": (n * m, m * k, n * k),
            "matmul_grad_rhs": (n * m, n * k, m * k),
        }
        try:
            first_count, second_count, output_count = counts[variant]
        except KeyError as exc:
            raise _fail("Metal matmul recipe variant is unavailable") from exc
        _require_count(first, first_count, "matmul first operand")
        _require_count(second, second_count, "matmul second operand")
        _require_count(output, output_count, "matmul output")
        plan = _plan(
            axes["plan"],
            variant,
            expected_operation="matmul" if variant == "matmul_forward" else None,
            allow_internal=variant != "matmul_forward",
            require_internal=variant != "matmul_forward",
        )
        if variant == "matmul_forward":
            _require_plan_signature(
                plan,
                (DType.Float32, DType.Float32),
                output_dtype=DType.Float32,
            )
        return MatmulRecipe(variant, first, second, output, n, m, k)
    axes = _axes(key, _CONV_AXES)
    _header(key, axes, template_revision=template_revision, pass_configs=pass_configs)
    first = _tensor(
        axes["first_address_plan"],
        axes["first_storage_size"],
        DType.Float32,
        "convolution first operand",
    )
    second = _tensor(
        axes["second_address_plan"],
        axes["second_storage_size"],
        DType.Float32,
        "convolution second operand",
    )
    output = _output_tensor(
        axes["output_address_plan"],
        axes["output_storage_size"],
        DType.Float32,
        "convolution output",
    )
    geometry = _conv_geometry(axes["convolution_signature"])
    counts = {
        "conv_forward": (
            geometry.lhs_count,
            geometry.kernel_count,
            geometry.output_count,
        ),
        "conv_grad_lhs": (
            geometry.output_count,
            geometry.kernel_count,
            geometry.lhs_count,
        ),
        "conv_grad_kernel": (
            geometry.output_count,
            geometry.lhs_count,
            geometry.kernel_count,
        ),
    }
    try:
        first_count, second_count, output_count = counts[variant]
    except KeyError as exc:
        raise _fail("Metal convolution recipe variant is unavailable") from exc
    _require_count(first, first_count, "convolution first operand")
    _require_count(second, second_count, "convolution second operand")
    _require_count(output, output_count, "convolution output")
    plan = _plan(
        axes["plan"],
        variant,
        expected_operation="conv_general" if variant == "conv_forward" else None,
        allow_internal=variant != "conv_forward",
        require_internal=variant != "conv_forward",
    )
    if variant == "conv_forward":
        _require_plan_signature(
            plan,
            (DType.Float32, DType.Float32),
            output_dtype=DType.Float32,
        )
    return ConvolutionRecipe(variant, first, second, output, geometry)


MetalRecipe: TypeAlias = (
    PointwiseRecipe | ReductionRecipe | IndexingRecipe | ContractionRecipe
)


def specialization_recipe(
    key: MetalSpecializationKey,
    *,
    template_revision: str,
    pass_configs: tuple[tuple[str, object], ...] | None = None,
    forward_reductions: frozenset[str] = frozenset(),
    backward_reductions: frozenset[str] = frozenset(),
) -> MetalRecipe:
    """Decode one specialization into an immutable provider-owned recipe."""
    operation = key.logical_kernel.operation
    if operation == "metal.pointwise":
        return pointwise_recipe(key, template_revision=template_revision)
    if operation in {"metal.reduction", "metal.scan"}:
        if pass_configs is None:
            raise _fail("Metal reduction recipe pass configuration is unavailable")
        return reduction_recipe(
            key,
            template_revision=template_revision,
            pass_configs=pass_configs,
            forward_variants=forward_reductions,
            backward_variants=backward_reductions,
        )
    if operation in {"metal.indexing", "metal.selection"}:
        if pass_configs is None:
            raise _fail("Metal indexing recipe pass configuration is unavailable")
        return indexing_recipe(
            key, template_revision=template_revision, pass_configs=pass_configs
        )
    if operation in {"metal.matmul", "metal.conv_general"}:
        if pass_configs is None:
            raise _fail("Metal contraction recipe pass configuration is unavailable")
        return contraction_recipe(
            key, template_revision=template_revision, pass_configs=pass_configs
        )
    raise _fail(
        "Metal specialization has no provider-owned reconstruction recipe: "
        f"{operation}/{key.logical_kernel.variant}"
    )


__all__: list[str] = []
