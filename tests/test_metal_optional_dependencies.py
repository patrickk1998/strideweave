from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

import strideweave.carriers.metal._runtime as metal_runtime


def test_missing_metal_dependencies_fail_before_any_optional_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    monkeypatch.setattr(metal_runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(metal_runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        metal_runtime,
        "find_spec",
        lambda name: None if name == "tilelang" else object(),
    )
    monkeypatch.setattr(
        metal_runtime,
        "import_module",
        lambda name: imported.append(name),
    )

    with pytest.raises(
        RuntimeError, match=r"'tilelang'.*pip install 'strideweave\[metal\]'"
    ):
        metal_runtime.load_metal_runtime()

    assert imported == []


def test_unavailable_mps_reports_runtime_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = SimpleNamespace(
        __version__="2.13.0",
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_built=lambda: True, is_available=lambda: False)
        ),
        mps=SimpleNamespace(compile_shader=lambda _: None, synchronize=lambda: None),
    )
    tilelang = SimpleNamespace(__version__="0.1.12")

    monkeypatch.setattr(metal_runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(metal_runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(metal_runtime, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        metal_runtime,
        "import_module",
        lambda name: torch if name == "torch" else tilelang,
    )

    with pytest.raises(
        RuntimeError,
        match=r"available MPS device.*torch=2\.13\.0, tilelang=0\.1\.12",
    ):
        metal_runtime.load_metal_runtime()


def test_cpu_only_paths_and_test_collection_do_not_import_optional_runtime() -> None:
    source = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        BLOCKED = {
            "tilelang",
            "torch",
            "torch_c_dlpack_ext",
            "tvm",
            "tvm_ffi",
        }

        class OptionalRuntimeBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in BLOCKED:
                    raise AssertionError(f"unexpected optional Metal import: {fullname}")
                return None

        def assert_help(main):
            try:
                main(["--help"])
            except SystemExit as exc:
                assert exc.code == 0
            else:
                raise AssertionError("help did not exit")

        sys.meta_path.insert(0, OptionalRuntimeBlocker())
        import strideweave as sw
        from strideweave.carriers.operation_capability import (
            capabilities_for_carrier_class,
        )
        from strideweave.carriers.operation_policy import registered_operations

        metal_capabilities = capabilities_for_carrier_class(sw.Metal)
        assert metal_capabilities
        assert {entry.operation for entry in metal_capabilities} == {
            spec.name for spec in registered_operations()
        }

        carrier = sw.CPU(2, dtype=sw.DType.Int32)
        assert carrier.size() == 2

        from strideweave.verification.cli import main as report_main
        from strideweave.verification.status_cli import main as status_main

        assert_help(report_main)
        assert_help(status_main)
        assert sw.test_backend().records

        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        import pytest

        result = pytest.main(["--collect-only", "-q", "tests"])
        assert result == pytest.ExitCode.OK
        assert not ({name.partition(".")[0] for name in sys.modules} & BLOCKED)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_missing_torch_skips_only_reference_cases() -> None:
    source = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        import pytest

        class TorchBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] == "torch":
                    raise ModuleNotFoundError(
                        "blocked PyTorch reference dependency", name=fullname
                    )
                return None

        class Recorder:
            def __init__(self):
                self.reports = []

            def pytest_runtest_logreport(self, report):
                self.reports.append(report)

        sys.meta_path.insert(0, TorchBlocker())
        recorder = Recorder()
        selected = [
            "tests/test_tensor.py::test_tensor_backward_refuses_non_broadcast_aliasing",
            "tests/test_tensor.py::test_broadcast_to_backward_matches_torch_and_restores_input_layout",
            "tests/test_einops.py::test_einops_lex_skips_ascii_whitespace",
            "tests/test_einops.py::test_einops_einsum_batch_only_outer_product_matches_torch",
            "tests/test_activations.py::test_activations_propagate_released_data_errors",
            "tests/test_activations.py::test_relu_activation_matches_pytorch",
        ]
        result = pytest.main(["-q", *selected], plugins=[recorder])
        assert result == pytest.ExitCode.OK

        non_reference = (
            "test_tensor_backward_refuses_non_broadcast_aliasing",
            "test_einops_lex_skips_ascii_whitespace",
            "test_activations_propagate_released_data_errors",
        )
        reference = (
            "test_broadcast_to_backward_matches_torch_and_restores_input_layout",
            "test_einops_einsum_batch_only_outer_product_matches_torch",
            "test_relu_activation_matches_pytorch",
        )
        for name in non_reference:
            matching = [
                report
                for report in recorder.reports
                if name in report.nodeid and report.when == "call"
            ]
            assert matching and all(report.passed for report in matching), name
        for name in reference:
            matching = [
                report
                for report in recorder.reports
                if name in report.nodeid and report.when == "setup"
            ]
            assert matching and all(report.skipped for report in matching), name
            assert all(
                "PyTorch reference dependency is unavailable" in str(report.longrepr)
                for report in matching
            ), name
        assert "torch" not in sys.modules
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
