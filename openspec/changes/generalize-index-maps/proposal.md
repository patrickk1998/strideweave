## Why

StrideWeave's layout algebra can express affine hierarchical mappings, but it
cannot represent sparse selections, bit swizzles, or Cartesian products as
first-class mapping values. Generalizing the mapping contract now gives those
forms one composable vocabulary while preserving `Layout` as the affine map
used by tensors.

## What Changes

- Introduce a public immutable `IndexMap` abstraction with a hierarchical
  `shape`, flat `codomain_size`, logical `size`, coordinate evaluation,
  non-variadic composition, and tri-state injectivity.
- Make `Layout` an `IndexMap` while preserving its existing shape/stride
  algebra, `cosize` meaning, indexing inputs, and `Layout`, `Shape`, and
  `Tiler` composition forms.
- Add public immutable `Permutation`, `Swizzle`, and `Product` sibling maps,
  plus public structured `SwizzleStage` values.
- Give every map recursive first-mode-fastest coordinate normalization, including
  scalar coordinates at any hierarchical node, and add the inverse Shape
  coordinate encoding used to pack Product results.
- Preserve closed same-kind compositions as their specialized public types and
  return other compatible compositions through the public `IndexMap` type.
- Enforce logical immutability across the mapping value graph.
- Defer callable maps, Tensor-backed dynamic maps, partial carriers, residency
  policy, gather/scatter, and autograd behavior.

This proposal introduces intended behavior. It extends the confirmed `Layout`
contract without treating uncaptured implementation details as normative.

## Capabilities

### New Capabilities

- `index-maps`: The common map contract, coordinate encoding, Permutation,
  Swizzle, Product, generic composition, injectivity, and immutability.

### Modified Capabilities

- `core-layout`: `Layout` becomes an `IndexMap`, gains the common flat-codomain
  and composition surface, and preserves its existing specialized behavior and
  `cosize` semantics.

## Impact

- Public Python API and typing exports gain `IndexMap`, `Permutation`,
  `Product`, `Swizzle`, and `SwizzleStage`; runtime exports and stubs must stay
  aligned under RT005 and RT006.
- Core layout and public API modules must gain the common map surface without
  changing Tensor placement behavior.
- `llms.md` Core Model and Current Boundaries must describe the generalized map
  hierarchy and its deferred dynamic forms.
- Relevant invariant IDs are RT002, RT005, RT006, RT014, RT015, and CPP002.
  RT014 continues to require ordinary `Layout` values for tensor placement and
  adjacent grouping, while RT015 expands to cover the shared map-domain,
  composition, immutability, and injectivity contract while retaining its exact
  Layout broadcasting and complement rules.
