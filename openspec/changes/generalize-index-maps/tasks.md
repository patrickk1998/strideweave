## 1. Shared IndexMap and Layout Foundation

- [ ] 1.1 Add immutable `Shape.encode` and `Shape.decode` behavior with recursive first-mode-fastest scalar normalization at every hierarchy node, rank-zero support, tuple/list compatibility, inverse-property tests, and exact type/value failure tests.
- [ ] 1.2 Add the public `IndexMap` contract with read-only `shape`, `size`, `codomain_size`, tri-state `is_injective`, `index`, direct-call evaluation, abstract-construction failure, and non-variadic composition validation, covered by focused return and error contract tests.
- [ ] 1.3 Make Shape, Stride, and Layout semantic state mechanically immutable, and test that rejected reassignment preserves every public value and Layout indexing result.
- [ ] 1.4 Migrate Layout under IndexMap, define `codomain_size == cosize`, route coordinate semantics through the shared Shape authority, and preserve native Layout indexing plus existing static/class-qualified composition, Shape, Tiler, broadcasting, complement, and tiling regressions.

## 2. Specialized Sibling Maps

- [ ] 2.1 Implement immutable `Permutation(values, codomain_size)` with copied explicit values, constructor validation, exact injectivity, sparse lookup evaluation, and constructor/evaluation/failure tests.
- [ ] 2.2 Implement public immutable `SwizzleStage` and `Swizzle(shape, *stages)` with the specified disjoint XOR fields, bit-width validation, zero-stage identity, exact injectivity, and positive/negative-shift evaluation and failure tests.
- [ ] 2.3 Implement variadic immutable `Product(*children)` without automatic flattening, including domain and target hierarchy construction, first-mode-fastest child-result packing, tri-state injectivity, explicit nesting, and constructor/evaluation/failure tests.

## 3. Composition Integration

- [ ] 3.1 Implement immutable generic IndexMap composition with declared codomain containment, correct result metadata and evaluation, metadata-preserving identity behavior, and conservative tri-state injectivity tests.
- [ ] 3.2 Implement Permutation lookup closure, equal-size Swizzle stage closure and adjacent cancellation, and structurally aligned Product componentwise closure, with tests for specialized result types and generic fallbacks when closure conditions do not hold.
- [ ] 3.3 Integrate Layout with sibling composition and test `outer.compose(inner)` orientation, smaller-codomain containment, incompatible bounds, arity and argument-type failures, mixed Layout/sibling generic results, Layout/Layout closure, and success and failure behavior for every Shape/Tiler convenience syntax.

## 4. Public Surface and Architecture Documentation

- [ ] 4.1 Export `IndexMap`, `Permutation`, `Product`, `Swizzle`, and `SwizzleStage` from the runtime, layout facade, top-level package, and matching stubs; add contract-complete public docstrings and examples satisfying RT005 and RT006.
- [ ] 4.2 Update `llms.md` Core Model and Current Boundaries for the generalized map hierarchy, flat codomain bounds, coordinate conversion, composition, Layout-only Tensor roles, and deferred dynamic forms.
- [ ] 4.3 Update `INVARIANTS.md` so RT002 preserves `Layout.cosize`-based Tensor storage, RT014 preserves Layout-only Tensor placement, expanded RT015 governs shared immutable IndexMap/composition/injectivity semantics, and the enforcement evidence names the new tests while retaining CPP002's cached native Layout path.

## 5. Validation and Acceptance

- [ ] 5.1 Run focused Shape, Layout, IndexMap, sibling-map, composition, export, docstring, and native Layout-index tests, including regression cases for all existing Layout composition inputs.
- [ ] 5.2 Run the non-Dolt test suite, Ruff formatting and lint, invariant lint, pyright, native formatting, strict-warning build, distribution build, and duplication check required by repository guidance; resolve every failure attributable to the change.
- [ ] 5.3 Run `openspec validate generalize-index-maps --strict`, verify the implementation against both delta specs and the design decisions, and leave the change ready for independent review rather than direct OpenSpec apply.
