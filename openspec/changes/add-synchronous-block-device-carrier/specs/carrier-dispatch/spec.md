## MODIFIED Requirements

### Requirement: Carrier is open while shipped implementations are closed

`Carrier` SHALL remain open for new sibling implementations, including Python
and native subclasses that implement its storage and dispatch contract.
`DependentCarrier` SHALL remain open for dependent implementations.

Generic, CPU, FileBacked, BlockDeviceCarrier, and Evictable SHALL be closed
implementations at runtime and SHALL be declared final on their public typed
import paths. An attempt to subclass any of those five SHALL fail with
`TypeError` identifying it as a closed carrier implementation and directing
extension to a sibling Carrier.

#### Scenario: Implement a sibling carrier

- **WHEN** a caller subclasses Carrier and supplies the required storage and
  dispatch behavior
- **THEN** instances participate in the ordinary dispatch contract

#### Scenario: Reject specialization of a shipped carrier

- **WHEN** a caller attempts to subclass Generic, CPU, FileBacked,
  BlockDeviceCarrier, or Evictable
- **THEN** class creation fails with the common closed-carrier `TypeError`

## ADDED Requirements

### Requirement: BlockDeviceCarrier refuses computational dispatch

`BlockDeviceCarrier` SHALL be a storage-only carrier. Calling
`dispatch_op(operation_name)` on it SHALL apply the ordinary string validation
to `operation_name` and SHALL fail with `NotImplementedError` for every valid
computational dispatch name, including shared representation-preserving names.
The failure SHALL occur before allocation, device I/O, value mutation, version
change, or source release.

Cross-carrier `move` SHALL remain available through the separate exact-pair move
registry defined by `interop-movement`; it SHALL not make `move` a computational
dispatch name of `BlockDeviceCarrier`.

#### Scenario: Refuse block-device computation

- **WHEN** a caller asks a live `BlockDeviceCarrier` to dispatch `add`, `view`,
  or another computational name
- **THEN** dispatch fails with `NotImplementedError` without device I/O or state
  change

#### Scenario: Keep movement outside carrier dispatch

- **WHEN** a CPU tensor is moved to a `BlockDeviceCarrier`
- **THEN** the move registry selects the carrier-pair operation without calling
  the block carrier's computational dispatch hook
