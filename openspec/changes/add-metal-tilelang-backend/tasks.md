## 1. Confirm the external Metal ABI

- [x] 1.1 Run a focused TileLang Metal ABI spike that proves explicit PyTorch MPS input and output tensors, Float32/Int32/Bool arguments, device synchronization, and access to the generated host/device source and any exposed runtime artifact; record and pin the working dependency versions.
- [x] 1.2 Add TileLang and its Metal-side runtime dependencies as lazy optional dependencies, with actionable capability errors and regression tests proving that importing and using the CPU backend remains unaffected when they are absent.

## 2. Generalize kernel verification and provenance

- [x] 2.1 Implement provider-neutral profile, logical-kernel, manifest, and plan-classification interfaces and add fail-closed completeness tests using CPU and a synthetic JIT profile; remove prototype CPU wrapper APIs.
- [x] 2.2 Generalize Stage Two execution, synchronization, result decoding, and Stage One oracle authorization so a target plan may consume an exact CPU certificate matched by operation, plan identity, and test class without sharing the target kernel identifier; prove existing CPU behavior remains unchanged.
- [x] 2.3 Implement `verify_backend(target, *, output=None)` and its public records, enums, stubs, docstrings, profile selection, atomic optional output, and errors as a clean replacement with no prototype API compatibility.
- [x] 2.4 Implement direct schema-v3 compilation bundles and discriminated compiled-executable/JIT-specialization receipts, binding complete declared inputs and ordered typed generated artifacts without wrapping v2 records.
- [x] 2.5 Implement strict offline v3 loading and fresh v3-only storage, recording, status/stale/todo, publication, and refresh; reject v2 reports, stores, and snapshots before mutation with actionable recreate guidance and no migration.

## 3. Add the TileLang Metal carrier and dispatch track

- [x] 3.1 Implement the closed `Metal` carrier with PyTorch MPS-backed typed storage, construction and factory helpers, mutability/version/release behavior, public exports and stubs, and focused lifecycle, dtype, error, and CPU-only-environment tests.
- [x] 3.2 Implement exact-class bulk CPU-to-Metal, Metal-to-CPU, and Metal-to-Metal movement plus shared structural layout operations, with layout/cosize validation, unsupported-carrier failures, and autograd boundary tests.
- [x] 3.3 Implement immutable logical-to-physical address plans for hierarchical layouts and a TileLang Metal JIT cache with stable logical kernel identities and specialization keys suitable for provenance binding.
- [x] 3.4 Implement at least one faithful serial Metal plan for every registered pointwise unary, binary, weak-scalar, ternary, and predicate operation, with temporary ordinary forward and backward conformance tests against the CPU reference where differentiable.
- [x] 3.5 Implement at least one faithful serial Metal plan for every registered reduction and scan operation, with temporary ordinary edge-case and conformance tests.
- [x] 3.6 Implement at least one faithful Metal plan for every registered indexing, computational scatter, sort, and top-k operation, preserving observable bounds, duplicate-index, ordering, value, index, and conformance behavior without freezing internal dispatch names.
- [x] 3.7 Implement at least one faithful serial Metal plan for every registered matrix multiplication and convolution operation, with temporary ordinary layout, shape, accumulation, and conformance tests.
- [x] 3.8 Derive and seal the Metal capability declaration from the implemented exact dispatch surface, and prove both that every registered dispatch name has at least one executable Metal plan and that every advertised Metal capability executes successfully.

## 4. Integrate Metal with provenance-backed testing

- [x] 4.1 Register the Metal backend descriptor, manifest, plan classifications, synchronization hook, and TileLang JIT provider with the generic verification interface, binding each execution to its exact target specialization receipt and exact authorized CPU oracle certificate.
- [x] 4.2 Build the Metal Stage Two case catalog across every advertised capability, including movement and structural behavior, and verify complete passed/failed/error/blocked/deferred evidence with explicit reasons for any legitimate deferral.
- [x] 4.3 Remove only those temporary ordinary operation-matrix tests whose scenarios are equivalently covered by the provenance-backed Metal suite; retain focused carrier, public API, error-path, dispatch-boundary, and autograd tests.
- [x] 4.4 Generate a deterministic Metal verification report and validate strict offline round-trip, provenance-store ingestion/query/status, and publication paths without compiling or executing kernels during report loading.
- [x] 4.5 Decode every Metal specialization family through immutable typed reconstruction recipes, validating exact axes, plan correlations, address cardinalities, and dimensional equations before caches, optional runtime loading, compilation, or store mutation; normalize malformed canonical floats through provider, recording, and CLI boundaries.

## 5. Reconcile documentation, invariants, and release validation

- [x] 5.1 Update `llms.md` to document the Metal carrier, PyTorch MPS storage boundary, TileLang JIT dispatch, supported dtypes and operation surface, generic backend testing, mixed AOT/JIT provenance, development commands, and the initial implementation's limitations.
- [x] 5.2 Update `INVARIANTS.md` and enforcement evidence for RT007, RT012, RT023 including RT023e, and RT024, adding a new invariant only if synchronization or specialization identity is a distinct cross-cutting rule.
- [x] 5.3 Update public documentation, type stubs, docstrings, and optional-installation guidance for Metal and `verify_backend`, while identifying direct Metal allocation, PyTorch-free interop, optimization, autotuning, and non-Metal accelerators as follow-on work.
- [x] 5.4 Run strict OpenSpec validation, the full CPU test suite, provenance/store suites, temporary and provenance-backed Metal suites on supported hardware, Ruff, pyright, duplication checks, and applicable strict-warning or sanitizer jobs; document any hardware-gated validations that cannot run in the PR environment.
- [ ] 5.5 Review the delivered result against the proposal, delta specs, design, repository invariants, and acceptance criteria before preparing one pull request.
