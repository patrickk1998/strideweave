## ADDED Requirements

### Requirement: Metal movement uses exact registered bulk pairs

The shipped movement registry SHALL include exact-class CPU-to-Metal,
Metal-to-CPU, and Metal-to-Metal bulk operations. These registrations SHALL not
apply to subclasses. Every unregistered exact pair SHALL retain the ordinary
elementwise fallback or documented refusal rather than inheriting a nearby
registration.

Metal MAY serve as an Evictable tier only when every exact movement edge needed
by the hierarchy's construction, promotion, eviction, results, and gradients
is registered and supports the hierarchy's dtype. An unavailable edge SHALL
make construction fail before tier ownership transfers.

#### Scenario: Select each Metal bulk pair

- **WHEN** movement uses CPU-to-Metal, Metal-to-CPU, or Metal-to-Metal exact classes
- **THEN** the corresponding registered bulk operation performs the transfer

#### Scenario: Refuse an unusable Metal hierarchy

- **WHEN** an Evictable hierarchy would require a missing Metal movement edge
- **THEN** construction fails before either supplied tier becomes owned

### Requirement: Metal movement preserves the complete Tensor value boundary

A successful Metal movement SHALL allocate fresh destination storage of the
same dtype and physical `layout.cosize`, copy the complete physical span
addressed from the Tensor offset including holes, preserve the exact
hierarchical Layout and logical values, return a Tensor over the destination,
and release the source only after transfer completion. Stride-zero broadcast
layouts SHALL remain layouts rather than being materialized by logical size.

The implementation SHALL synchronize Metal work before CPU-visible values are
decoded, before a source whose work may still be pending is released, and before
the move is reported complete to a caller that observes host values. Metal-to-
Metal movement MAY remain asynchronous internally only when destination use and
source release preserve the same completion ordering.

Validation, allocation, copy, or synchronization failure SHALL leave the source
unreleased and usable, expose no partially published destination Tensor, and
preserve both carrier versions. A successful move SHALL not count as an in-place
value mutation of either carrier.

#### Scenario: Move a hierarchical physical span to Metal

- **WHEN** a Tensor with holes or nested modes moves from CPU to Metal
- **THEN** destination storage has the same dtype and `cosize`, and the returned
  Tensor has the exact source offset-relative values and Layout

#### Scenario: Synchronize before host decode and release

- **WHEN** a Metal-to-CPU move follows pending Metal work
- **THEN** synchronization completes before host values are decoded or Metal
  source storage is released

#### Scenario: Preserve the source after failure

- **WHEN** any Metal move step fails
- **THEN** the source remains unreleased and usable and no destination is published

### Requirement: Metal movement preserves reverse-mode carrier boundaries

A differentiable successful move SHALL attach the ordinary move autograd
boundary. Backward SHALL require an injective same-shape cotangent, use the
registered inverse exact carrier pair, and return a fresh gradient Tensor in
the source carrier class with the source layout semantics. A missing inverse
pair or failed transfer SHALL raise without substituting another carrier.

#### Scenario: Move a gradient back from Metal

- **WHEN** backward crosses a CPU-to-Metal forward move
- **THEN** the gradient synchronizes as needed and returns through Metal-to-CPU
  into fresh CPU storage

### Requirement: Metal exposes no new public device-interchange surface

Metal SHALL not opt into DLPack export in this change. Public APIs SHALL not
expose the private PyTorch MPS storage object or promise adoption of PyTorch,
Metal, or another device buffer. Friendly tensor factories SHALL remain
CPU-backed. These restrictions SHALL not prevent registered `move` operations.

Operation profiling of Metal SHALL measure the existing synchronous host
dispatch or launch boundary and SHALL not require device synchronization merely
to close an event. Device work MAY outlive that event; operations whose public
result semantics require host observation SHALL synchronize at their own
boundary.

#### Scenario: Reject Metal DLPack export

- **WHEN** a caller requests DLPack from a Metal Tensor
- **THEN** export fails through the documented non-exporting-carrier path

#### Scenario: Profile without forcing completion

- **WHEN** a Metal operation launches asynchronously inside a profiling context
- **THEN** event completion does not itself synchronize the device
