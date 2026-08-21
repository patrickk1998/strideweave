from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from strideweave.carriers.metal._runtime import load_metal_runtime

_LENGTH = 37
_THREADS = 32


def _copy_program(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(out_idx=[1], target="metal", execution_backend="torch")
    def copy(dtype: str) -> Any:
        @T.prim_func
        def main(
            source: T.Tensor((_LENGTH,), dtype),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((_LENGTH,), dtype),  # pyright: ignore[reportInvalidTypeForm]
        ):
            with T.Kernel(T.ceildiv(_LENGTH, _THREADS), threads=_THREADS) as block:
                for thread in T.Parallel(_THREADS):
                    index = block * _THREADS + thread
                    if index < _LENGTH:
                        output[index] = source[index]

        return main

    return copy


def _pointwise_program(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(out_idx=[1], target="metal", execution_backend="torch")
    def pointwise() -> Any:
        @T.prim_func
        def main(
            source: T.Tensor((_LENGTH,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((_LENGTH,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            with T.Kernel(T.ceildiv(_LENGTH, _THREADS), threads=_THREADS) as block:
                for thread in T.Parallel(_THREADS):
                    index = block * _THREADS + thread
                    if index < _LENGTH:
                        output[index] = source[index] * 2.0 + 1.0

        return main

    return pointwise


def _serial_sum_program(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(out_idx=[1], target="metal", execution_backend="torch")
    def serial_sum() -> Any:
        @T.prim_func
        def main(
            source: T.Tensor((_LENGTH,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
            output: T.Tensor((1,), "float32"),  # pyright: ignore[reportInvalidTypeForm]
        ):
            with T.Kernel(1, threads=1):
                accumulator = T.alloc_fragment((1,), "float32")
                T.clear(accumulator)
                for index in T.Serial(_LENGTH):
                    accumulator[0] += source[index]
                output[0] = accumulator[0]

        return main

    return serial_sum


def _assert_generated_facts(kernel: Any) -> None:
    device_source = kernel.kernel_source
    host_source = kernel.host_source

    assert "kernel void" in device_source
    assert "output [[ buffer(0) ]]" in device_source
    assert "source [[ buffer(1) ]]" in device_source
    assert len(sha256(device_source.encode()).hexdigest()) == 64
    assert host_source
    assert kernel.artifact.rt_mod is None
    assert not hasattr(kernel.adapter, "artifact_digest")


@pytest.mark.metal
def test_tilelang_metal_explicit_buffer_abi_and_generated_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # TileLang 0.1.12's persistent Metal cache assumes the adapter exposes a
    # native ``libpath`` even though its PyTorch shader adapter does not. Keep
    # this ABI proof on the working in-memory specialization cache; StrideWeave
    # owns its later provenance-capable JIT cache.
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TILELANG_TMP_DIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    try:
        runtime = load_metal_runtime()
    except RuntimeError as exc:
        pytest.skip(str(exc))
        raise AssertionError("pytest.skip must not return")

    torch = runtime.torch
    tilelang = runtime.tilelang
    T = __import__("tilelang.language", fromlist=["language"])

    assert torch.__version__ == "2.13.0"
    assert tilelang.__version__ == "0.1.12"

    copy = _copy_program(tilelang, T)
    dtype_cases = (
        (
            "float32",
            torch.float32,
            [float(index) / 4.0 - 3.0 for index in range(_LENGTH)],
        ),
        ("int32", torch.int32, [index - 18 for index in range(_LENGTH)]),
        ("bool", torch.bool, [index % 3 == 0 for index in range(_LENGTH)]),
    )
    compiled_copy_kernels = []
    for tilelang_dtype, torch_dtype, values in dtype_cases:
        source = torch.tensor(values, dtype=torch_dtype, device="mps")
        output = torch.empty_like(source)
        kernel = copy(tilelang_dtype)
        assert copy(tilelang_dtype) is kernel

        # Metal lowering orders explicit outputs before inputs in generated MSL,
        # and the PyTorch adapter forwards arguments verbatim.
        kernel(output, source)
        compiled_shader = kernel.adapter._kernel
        kernel(output, source)
        assert kernel.adapter._kernel is compiled_shader
        runtime.synchronize()

        assert output.cpu().tolist() == values
        _assert_generated_facts(kernel)
        compiled_copy_kernels.append(kernel)

    pointwise = _pointwise_program(tilelang, T)()
    pointwise_source = torch.arange(_LENGTH, dtype=torch.float32, device="mps")
    pointwise_output = torch.empty_like(pointwise_source)
    pointwise(pointwise_output, pointwise_source)
    runtime.synchronize()
    assert pointwise_output.cpu().tolist() == pytest.approx(
        [float(index * 2 + 1) for index in range(_LENGTH)]
    )
    _assert_generated_facts(pointwise)

    serial_sum = _serial_sum_program(tilelang, T)()
    sum_source = torch.arange(_LENGTH, dtype=torch.float32, device="mps")
    sum_output = torch.empty(1, dtype=torch.float32, device="mps")
    serial_sum(sum_output, sum_source)
    runtime.synchronize()
    assert sum_output.cpu().item() == sum(range(_LENGTH))
    _assert_generated_facts(serial_sum)

    assert len({kernel.kernel_source for kernel in compiled_copy_kernels}) == 3
