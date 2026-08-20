from __future__ import annotations

from hashlib import sha256

import pytest

from strideweave import Layout, Shape, Stride
from strideweave.carriers.metal._address_plan import address_plan
from strideweave.carriers.metal._jit import (
    CompiledMetalKernel,
    GeneratedMetalFacts,
    LogicalKernel,
    MetalJITCache,
    MetalSpecializationKey,
)


@pytest.mark.parametrize(
    ("layout", "offset"),
    [
        (Layout(Shape([2, 3]), Stride([1, 2])), 0),
        (Layout(Shape([2, [3, 4]]), Stride([1, [2, 6]])), 7),
        (Layout(Shape([2, 3]), Stride([2, 5])), 3),
        (Layout(Shape([4, 3]), Stride([0, 1])), 11),
        (
            Layout.permute(
                Layout(Shape([2, [3, 4], 5]), Stride([1, [2, 6], 24])),
                1,
                0,
                2,
            ),
            5,
        ),
    ],
    ids=["compact", "hierarchical", "holes", "broadcast", "permuted"],
)
def test_address_plan_matches_exact_layout_ordinal_map(
    layout: Layout, offset: int
) -> None:
    plan = address_plan(layout, offset=offset)

    assert plan.offset == offset
    assert plan.logical_size == layout.size
    assert plan.cosize == layout.cosize
    assert plan.addresses == tuple(
        offset + layout.index(index) for index in range(layout.size)
    )
    assert all(offset <= address < offset + layout.cosize for address in plan.addresses)
    assert plan.is_injective is layout.is_injective


def test_address_plan_cache_keys_exact_structure_stride_and_offset() -> None:
    compact = Layout(Shape([2, 3]), Stride([1, 2]))
    equal = Layout(Shape([2, 3]), Stride([1, 2]))
    permuted_stride = Layout(Shape([2, 3]), Stride([3, 1]))
    nested = Layout(Shape([[2, 3]]), Stride([[1, 2]]))

    compact_plan = address_plan(compact, offset=4)
    equal_plan = address_plan(equal, offset=4)

    assert compact_plan is equal_plan
    assert address_plan(compact, offset=5).key != compact_plan.key
    assert address_plan(permuted_stride, offset=4).key != compact_plan.key

    nested_plan = address_plan(nested, offset=4)
    assert nested_plan.addresses == compact_plan.addresses
    assert nested_plan.key != compact_plan.key


def test_address_plan_preserves_holes_and_broadcast_aliases() -> None:
    holes = address_plan(Layout(Shape([2, 3]), Stride([2, 5])), offset=3)
    broadcast = address_plan(Layout(Shape([4, 3]), Stride([0, 1])), offset=9)

    assert holes.addresses == (3, 5, 8, 10, 13, 15)
    assert holes.cosize == 13
    assert holes.is_injective
    assert broadcast.addresses == (9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11)
    assert not broadcast.is_injective


def test_address_plan_inverse_permutation_restores_exact_key() -> None:
    original = Layout(Shape([2, [3, 4], 5]), Stride([1, [2, 6], 24]))
    permuted = Layout.permute(original, 1, 2, 0)
    restored = Layout.permute(permuted, 2, 0, 1)

    assert address_plan(permuted).key != address_plan(original).key
    assert address_plan(restored).key == address_plan(original).key
    assert address_plan(restored).addresses == address_plan(original).addresses


def test_address_plan_validates_layout_and_nonnegative_integer_offset() -> None:
    layout = Layout(Shape(2), Stride(1))

    with pytest.raises(TypeError, match="layout must be a Layout"):
        address_plan(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        address_plan(layout, offset=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="offset must be non-negative"):
        address_plan(layout, offset=-1)


def _facts(tag: str = "base") -> GeneratedMetalFacts:
    return GeneratedMetalFacts(
        provider="tilelang",
        target="metal",
        execution_backend="torch",
        tilelang_version="0.1.12",
        torch_version="2.13.0",
        device_source=f"kernel {tag}",
        host_source=f"host {tag}",
    )


def _compiled(tag: str = "base") -> CompiledMetalKernel:
    def executable(*_args: object) -> object:
        return tag

    return CompiledMetalKernel(executable, _facts(tag))


def _key(logical: LogicalKernel, **axes: object) -> MetalSpecializationKey:
    return MetalSpecializationKey.from_axes(logical, axes)


def test_jit_cache_reuses_an_exact_specialization_once() -> None:
    logical = LogicalKernel("pointwise", "unary")
    cache = MetalJITCache((logical,))
    key = _key(logical, dtype="Float32", logical_size=64)
    calls: list[MetalSpecializationKey] = []

    def compile_one(candidate: MetalSpecializationKey) -> CompiledMetalKernel:
        calls.append(candidate)
        return _compiled()

    first = cache.get_or_compile(key, compile_one)
    second = cache.get_or_compile(key, compile_one)

    assert first is second
    assert calls == [key]
    assert len(cache) == 1


@pytest.mark.parametrize(
    "changed_axes",
    [
        {"dtype": "Int32"},
        {"address_plan": ("nested", 1, 2)},
        {"offset": 7},
        {"scalar": 2.0},
        {"template_revision": "v2"},
    ],
    ids=["dtype", "address", "offset", "scalar", "template"],
)
def test_jit_cache_distinguishes_every_codegen_axis(
    changed_axes: dict[str, object],
) -> None:
    logical = LogicalKernel("pointwise", "binary")
    base_axes: dict[str, object] = {
        "dtype": "Float32",
        "address_plan": ("flat", 0, 1),
        "offset": 0,
        "scalar": 1.0,
        "template_revision": "v1",
    }
    changed = {**base_axes, **changed_axes}
    cache = MetalJITCache((logical,))
    calls = 0

    def compile_one(_candidate: MetalSpecializationKey) -> CompiledMetalKernel:
        nonlocal calls
        calls += 1
        return _compiled(str(calls))

    first = cache.get_or_compile(
        MetalSpecializationKey.from_axes(logical, base_axes), compile_one
    )
    second = cache.get_or_compile(
        MetalSpecializationKey.from_axes(logical, changed), compile_one
    )

    assert first is not second
    assert calls == 2


def test_jit_cache_ignores_values_absent_from_codegen_axes() -> None:
    logical = LogicalKernel("pointwise", "unary")
    cache = MetalJITCache((logical,))
    key = _key(logical, dtype="Float32", logical_size=4)
    runtime_values = ([1.0, 2.0], [999.0, -4.0])
    calls = 0

    def compile_one(_candidate: MetalSpecializationKey) -> CompiledMetalKernel:
        nonlocal calls
        calls += 1
        return _compiled()

    results = [cache.get_or_compile(key, compile_one) for _ in runtime_values]

    assert results[0] is results[1]
    assert calls == 1


def test_jit_cache_failure_and_incomplete_facts_are_not_published() -> None:
    logical = LogicalKernel("reduce", "sum_serial")
    cache = MetalJITCache((logical,))
    key = _key(logical, dtype="Float32", logical_size=16)

    def compiler_error(_candidate: MetalSpecializationKey) -> CompiledMetalKernel:
        raise RuntimeError("compiler failed")

    with pytest.raises(RuntimeError, match="compiler failed"):
        cache.get_or_compile(key, compiler_error)
    assert len(cache) == 0

    def incomplete(_candidate: MetalSpecializationKey) -> CompiledMetalKernel:
        return CompiledMetalKernel(
            lambda: None,
            GeneratedMetalFacts(
                provider="tilelang",
                target="metal",
                execution_backend="torch",
                tilelang_version="0.1.12",
                torch_version="2.13.0",
                device_source="",
                host_source="host",
            ),
        )

    with pytest.raises(ValueError, match="device_source must be a non-empty string"):
        cache.get_or_compile(key, incomplete)
    assert len(cache) == 0

    compiled = cache.get_or_compile(key, lambda _candidate: _compiled("retry"))
    assert compiled.facts.device_source == "kernel retry"
    assert len(cache) == 1


def test_jit_cache_has_fixed_logical_manifest_and_bounded_specializations() -> None:
    logical = LogicalKernel("pointwise", "unary")
    undeclared = LogicalKernel("reduce", "sum_serial")
    cache = MetalJITCache((logical,), max_specializations=3)

    def compiler(key: MetalSpecializationKey) -> CompiledMetalKernel:
        return _compiled(str(key.axes))

    keys = [_key(logical, logical_size=size) for size in range(8)]
    for key in keys:
        cache.get_or_compile(key, compiler)

    assert cache.logical_kernels == (logical,)
    assert all(key.logical_kernel is logical for key in keys)
    assert len(cache) == 3
    with pytest.raises(KeyError, match="undeclared logical kernel"):
        cache.get_or_compile(_key(undeclared, logical_size=1), compiler)


def test_specialization_axes_are_canonical_typed_and_deterministic() -> None:
    logical = LogicalKernel("pointwise", "binary")
    forward = MetalSpecializationKey.from_axes(
        logical, {"z": (1, True, 1.5, None), "a": "Float32"}
    )
    reverse = MetalSpecializationKey.from_axes(
        logical, {"a": "Float32", "z": (1, True, 1.5, None)}
    )

    assert forward == reverse
    assert hash(forward) == hash(reverse)
    assert forward != _key(logical, value=True)
    assert _key(logical, value=True) != _key(logical, value=1)

    with pytest.raises(ValueError, match="finite"):
        _key(logical, value=float("nan"))
    with pytest.raises(TypeError, match="only None"):
        _key(logical, value=[1, 2])
    with pytest.raises(TypeError, match="axis names must be strings"):
        MetalSpecializationKey.from_axes(logical, {1: "bad"})  # type: ignore[dict-item]


def test_generated_facts_digest_every_available_source_without_inventing_artifact() -> (
    None
):
    facts = _facts()

    assert facts.device_source_digest == sha256(b"kernel base").hexdigest()
    assert facts.host_source_digest == sha256(b"host base").hexdigest()
    assert facts.runtime_artifact_digest is None
