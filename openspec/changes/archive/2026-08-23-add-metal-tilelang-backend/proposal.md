## Why

StrideWeave has no accelerator carrier, and its kernel verification entry point is
hard-coded to CPU execution. Introduce intended Metal behavior through TileLang
while first making the existing staged verification and provenance machinery
backend-neutral, so the new backend is judged through the same fail-closed evidence
path rather than a separate test system.

## What Changes

- Add a Metal-only carrier that executes through TileLang and initially uses
  PyTorch MPS tensors as its storage and TileLang argument boundary.
- Cover every registered operation dispatch name with at least one faithfully
  executable Metal plan; complete CPU plan-shape parity is not required.
- Preserve carrier-neutral structural views and add explicit bulk CPU-to-Metal,
  Metal-to-CPU, and Metal-to-Metal movement.
- Generalize local backend testing, kernel classification, Stage Two execution,
  and report binding so CPU remains the certified Stage One oracle and Metal is a
  selectable target.
- Extend provenance-complete evidence to bind TileLang JIT specializations and
  their generated Metal artifacts.
- Use temporary ordinary Metal tests until provenance-backed cases cover the
  same kernel semantics; retain focused lifecycle, API, error, and autograd tests.
- Keep direct Metal allocation, non-PyTorch argument interop, optimized kernels,
  autotuning, and non-Metal accelerator targets for follow-on changes.

This proposal introduces intended behavior rather than capturing an existing
implementation.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `carrier-storage`: Add the Metal carrier's storage, allocation, mutation,
  release, dtype, and host-access behavior.
- `carrier-dispatch`: Add Metal operation dispatch and its use of the existing
  carrier-neutral structural operations.
- `backend-capabilities`: Make Metal a sealed shipped independent carrier whose
  exact executable plan set covers every registered dispatch name.
- `kernel-verification`: Generalize staged verification to selectable target
  backends and add provenance-backed Metal execution.
- `verification-report`: Bind JIT compilation and Metal specialization provenance
  in deterministic, strictly validated reports.
- `kernel-evidence-tracking`: Store, query, publish, and refresh schema-v3
  selected-target evidence and reject v2 state without migration.
- `interop-movement`: Add exact Metal bulk-movement pairs, synchronization,
  failure atomicity, and the accelerator interop boundary.

## Impact

The change affects the carrier package, public exports and stubs, movement,
operation adapters, optional dependency packaging, tests, verification stages,
classification, compilation provenance, report parsing and persistence, and the
clean-replacement public `verify_backend` API. `llms.md` must update Core Model, Carriers,
Interoperability and Movement, Current Boundaries, Local Kernel Verification, raw
evidence persistence, and development guidance.

Relevant invariants are `RT001`-`RT004`, `RT007`, `RT009`, `RT011`-`RT013`, `RT016`,
`RT017`, `RT020`, `RT022`, `RT023`, and `RT024`. `RT012`, `RT023`, and `RT024`
require substantive revisions, including `RT023e`; the Metal storage/JIT boundary may warrant a
new cross-cutting invariant once its canonical form is fixed.
