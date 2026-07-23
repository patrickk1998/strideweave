# StrideWeave Invariant Registry

This registry records cross-cutting constraints that must shape code before it is
written. Read it while planning a change, not only after implementation. Ordinary
feature behavior belongs in `README.md` and its focused tests; this document covers
rules that apply across modules, backends, or future implementations.

## How to Use This Registry

Before changing code:

1. Identify the affected invariant IDs below.
2. Preserve their canonical implementation choices in the design.
3. Run every listed enforcement mechanism relevant to the change.
4. Update this registry in the same change when adding, removing, or materially
   changing a cross-cutting invariant.

Enforcement labels mean:

- **AST** — enforced by `tools/lint_invariants.py`.
- **Ruff** — enforced by the selected rules in `pyproject.toml`.
- **Test** — enforced behaviorally by the referenced tests.
- **Build** — enforced by compiler, formatter, type checker, or sanitizer jobs.
- **Review** — not reliably provable by current automation; authors and reviewers
  must inspect it explicitly.

The evidence column names stable files and test functions rather than line numbers,
which move frequently.

## Python Source Invariants

| ID | Invariant and canonical form | Enforcement | Evidence |
| --- | --- | --- | --- |
| `SW001` | Coordinate indexing uses `tensor[i, j]`. `tensor[[i, j]]` is reserved for intentional compatibility tests or documentation. | AST | `tools/lint_invariants.py`; `tests/test_lint_invariants.py`; intentional cases in `tests/test_tensor.py` |
| `SW002` | Dtype tags are identity values. Compare `DType.*` and zero-argument `.dtype()` results with `is` or `is not`, never `==` or `!=`. | AST | `tools/lint_invariants.py`; `tests/test_lint_invariants.py` |
| `SW003` | `Carrier.is_mutable()` owns the public mutability policy. Python carrier subclasses implement only the `_is_mutable()` backend hook. | AST, Test | `tools/lint_invariants.py`; `tests/test_lint_invariants.py`; carrier mutability tests |
| `SW004` | Tensor logic uses `tensor.size()` for logical element count. Use `layout.size` only when the code genuinely has a layout rather than a tensor or parameter. | AST | `tools/lint_invariants.py`; `tests/test_lint_invariants.py` |
| `PY001` | Every multi-input `zip` call states truncation intent with `strict=True` or `strict=False`. | Ruff (`B905`) | `pyproject.toml` |
| `PY002` | Re-export modules use ordinary imports and sorted literal `__all__` declarations. Wildcard imports remain only in compatibility aggregators whose exports are dynamically composed. | Ruff (`RUF022`, `PLC0414`), Test | `pyproject.toml`; `src/strideweave/operation.py`; `tests/test_docstrings.py::test_operation_runtime_exports_match_stub_declarations` |
| `PY003` | Broad exception assertions name the expected message, contain one simple statement, do not catch `Exception`, and make regular-expression intent explicit. | Ruff (`PT011`, `PT012`, `B017`, `RUF043`) | `pyproject.toml`; test suite |
| `PY004` | Stub declarations use current typing syntax and valid stub defaults rather than executable default expressions. | Ruff (`PYI011`, `PYI026`, `PYI034`, `PYI035`, `UP035`) | `pyproject.toml`; `.pyi` files |

### AST Checker Operation

Run:

```bash
uv run python tools/lint_invariants.py
```

With no arguments it scans `src`, `tests`, and `examples` without importing the
package. It parses each file once, ignores virtual environments, caches, build output,
and recognized generated Python files, and emits deterministic
`path:line:column: SWxxx message` diagnostics.

An intentional exception must be attached to the exact affected line:

```python
tensor[[i, j]]  # strideweave-lint: ignore=SW001
```

Multiple named codes may be comma-separated. Do not use broad file-level suppression.
Exit status is `0` when clean, `1` when diagnostics are present, and `2` for invalid
input, unreadable files, or syntax errors.

## Runtime and API Invariants

| ID | Invariant and canonical form | Enforcement | Evidence |
| --- | --- | --- | --- |
| `RT001` | Operation dispatch is invoked on a carrier instance and follows the documented exact carrier class. `Carrier.dispatch_op` owns shared policy and metadata annotation, and rejects an already-dispatched operation; backends override only `_dispatch_op`, whose factories return fresh operation instances. | AST, Test | `tools/lint_invariants.py`; `tests/test_lint_invariants.py`; fresh and cached-operation dispatch tests in `tests/test_carrier.py`, `tests/test_cpu.py`, and `tests/test_evictable.py` |
| `RT002` | Tensor and operation-result storage covers `layout.cosize`, not merely logical `layout.size`. Logical iteration still uses `tensor.size()`. | Test | `tests/test_tensor.py::test_tensor_storage_validation_uses_cosize_not_logical_size`; hierarchical CPU operation tests; factory allocation tests |
| `RT003` | Every successful public mutation path changes the visible carrier version. Aliases and views observe the same version; storage-only transitions such as eviction do not pretend to be value mutations. | Test | `tests/test_carrier.py::test_generic_public_write_paths_increment_version`; carrier scatter tests; `tests/test_cpu.py::test_cpu_data_contract_and_mutation`; `tests/test_tensor.py::test_tensor_version_is_shared_by_views_and_same_data_tensors`; evictable version tests |
| `RT004` | Carrier factories preserve dtype unless explicitly overridden, honor their public mutability argument, and allocate fresh storage of the requested physical size. | Test | Generic, CPU, FileBacked, Evictable `new_like` and `empty_like` tests in `tests/test_carrier.py`, `tests/test_cpu.py`, `tests/test_file_backed.py`, and `tests/test_evictable.py` |
| `RT005` | Public Python exports satisfy the repository docstring contract. Layout-command APIs additionally document StrideWeave syntax, semantics, and tensor mode assumptions. | Test | `AGENTS.md`; `tests/test_docstrings.py` |
| `RT006` | Runtime public exports and their `.pyi` declarations stay aligned; pybind signatures and defaults must agree with the stubs. | Test, Build, Review | `tests/test_docstrings.py::test_operation_runtime_exports_match_stub_declarations`; `uv run pyright`; relevant `.pyi` and binding files |
| `RT007` | Saved autograd inputs retain their carrier version, and backward rejects mutation that would invalidate the saved forward state. | Test | tensor mutation-after-forward tests in `tests/test_tensor.py` and `tests/test_evictable.py` |
| `RT008` | Carrier ownership is exclusive: owned tiers cannot be mutated, released, or moved through external aliases, while mutation through the owning carrier remains available. | Test | ownership and nested hierarchy tests in `tests/test_evictable.py` |
| `RT009` | Moving storage preserves tensor layout, dtype, values, and autograd semantics; failures leave the source usable and do not silently release it. | Test | `tests/test_move.py` |
| `RT010` | DLPack exports preserve dtype, offset, and flattened hierarchical shape/strides; unsupported carriers and requests fail explicitly. | Test | `tests/test_dlpack.py` |
| `RT011` | Composite adapters own the visible autograd graph. Delegated computation and internal storage transitions use the framework-owned sealed `execute_lowered_operation` helper, never dynamically dispatch another operation's `forward`, `_forward`, or `_execute_lowered` attribute; lowered execution shares framework hooks and Tensor-result validation without attaching an inner autograd node or clearing delegated backward state. | AST, Test, Review | `tools/lint_invariants.py`; `tests/test_lint_invariants.py`; lowered-execution tests in `tests/test_operation.py` and `tests/test_evictable.py`; Evictable transition tests |

## Native C++ Invariants

| ID | Invariant and canonical form | Enforcement | Evidence |
| --- | --- | --- | --- |
| `CPP001` | CPU hot loops release the Python GIL only after all required Python objects and cached layout metadata have been acquired. The released region must not call Python APIs. | Review | `py::gil_scoped_release` regions in `src/strideweave/carriers/cpu/native` |
| `CPP002` | Kernels iterate cached expanded layout keys and cached layout indices. They do not reconstruct hierarchical coordinates through Python inside hot loops. | Review | `_cpu_carrier.hpp`, `_cpu_operation.hpp`, `_cpu.cpp` |
| `CPP003` | Logical sizes, storage offsets, strides, and index arithmetic use signed `Index`. Container positions and lengths use `std::size_t`. | Build, Review | strict-warning build; native layout and carrier headers |
| `CPP004` | Crossings among Python sequence sizes, `std::size_t`, and signed `Index` use validated conversion helpers. Explicit casts are allowed only after range or non-negativity validation. | Build, Review | `layout_index::as_size`, `layout_index::size_as_index`; strict `-Wconversion` and `-Wsign-conversion` build |
| `CPP005` | Mutating native carrier entry points update carrier version state exactly as their Python-visible mutation contract requires. | Test, Review | native CPU mutation/scatter implementation; version tests |
| `CPP006` | Native operation registries store factories, not reusable operation objects; dispatch produces a fresh instance. | Test, Review | CPU operation registry and dispatch; `tests/test_cpu.py::test_cpu_dispatch_op_returns_supported_operations` |
| `CPP007` | Native code compiles cleanly with the project strict-warning set. Third-party pybind diagnostics receive only narrow, documented treatment; unfiltered `-Wpedantic` is not fatal. | Build | `STRIDEWEAVE_STRICT_WARNINGS` in `CMakeLists.txt`; `native-strict-warnings` CI job |
| `CPP008` | Every checked-in `.cpp` and `.hpp` conforms to the pinned project ClangFormat configuration. | Build | `.clang-format`; CI native-format step |
| `CPP009` | A Linux debug-oriented build runs the complete Python test suite with AddressSanitizer and UndefinedBehaviorSanitizer, halting on the first native diagnostic. | Build | `native-sanitizers` CI job; `STRIDEWEAVE_SANITIZERS` in `CMakeLists.txt` |

ThreadSanitizer is intentionally not an invariant check yet. Broad Clang-Tidy profiles
are also deferred. If introduced, evaluate diagnostic families individually and record
new enforced invariants here before making them fatal.

## Registry Maintenance

Use the existing prefix for the relevant layer (`SW`, `PY`, `RT`, or `CPP`) and the
next unused number. An entry must describe:

- the invariant in positive, implementation-guiding language;
- the canonical choice when multiple implementations are possible;
- every enforcement category currently providing coverage;
- stable evidence locations;
- any deliberately permitted exception.

Do not claim **Test** or **Build** enforcement unless a failing example would actually
make that check fail. Mark partially automated constraints as **Review** as well. When
removing an invariant, remove or revise its enforcement in the same change rather than
leaving a stale registry entry.
