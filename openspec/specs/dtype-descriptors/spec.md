---
title: Dtype Descriptors
publish: true
status: stable
order: 20
summary: Registered dtype identity, hierarchy, discovery, immutability, extension, and serialization.
---

# dtype-descriptors Specification

## Purpose

Define the public descriptor model that names StrideWeave dtypes, relates them
through categories, preserves their process identity, and permits cooperative
extensions without changing the built-in namespace.

## Terminology

| Term | Meaning |
| --- | --- |
| descriptor | An immutable, canonically registered `DType` singleton that is the process-local identity of one dtype category or representation and is compared by object identity. |
| registered name | The unique non-empty string by which a descriptor is discovered. |
| category | A `DTypeCategory` descriptor that places dtypes in a supertype hierarchy and may carry the legacy opaque-storage disposition, but defines neither a fixed-width scalar encoding nor a compound physical representation. |
| simple dtype | A `SimpleDType` descriptor for one fixed-width scalar encoding; being simple classifies the encoding and does not imply that any carrier can store or execute it. |
| structure | The immutable canonical identity record captured at descriptor finalization, recursively incorporating every representation-defining field, referenced descriptor structure, and extension contribution; it determines pickle compatibility and any kind-specific structural uniqueness. |
| extension descriptor | A non-built-in descriptor registered after the built-in graph is installed, discoverable through registry APIs without adding an attribute to the frozen `DType` namespace. |

## Requirements

### Requirement: Every dtype has one registered descriptor identity

Constructing a concrete descriptor SHALL atomically register that descriptor
under its `name`. `name` names the descriptor's unique registry key. `name`
SHALL be a non-empty string and SHALL be unique within the process. When
`name` is empty or is not a string, construction SHALL fail with `ValueError`;
when `name` is already registered, construction SHALL fail with `ValueError`.
A failed construction SHALL publish neither a registry entry nor a partially
usable descriptor.

Descriptors SHALL compare and hash by object identity. `name` SHALL return the
registered string, and the read-only compatibility property `value` SHALL
return that same string.

#### Scenario: Register a descriptor

- **WHEN** a caller constructs a valid concrete descriptor under an unused name
- **THEN** the returned object is the sole descriptor registered under that
  name and its `name` and `value` properties return the supplied string

#### Scenario: Reject a duplicate name atomically

- **WHEN** a caller constructs a descriptor under a name that is already
  registered
- **THEN** construction fails with `ValueError` and the original descriptor
  remains the registry identity for that name

### Requirement: Descriptor kinds and hierarchy are explicit

`DType` and `CompoundDType` SHALL be abstract, as SHALL a descriptor subclass
that does not declare `abstract=False`. In a descriptor subclass declaration,
`abstract` states whether the subclass remains non-constructible. `abstract`
SHALL be optional and SHALL default to `True`; `abstract=False` SHALL declare
the subclass concrete. Attempting to instantiate an abstract descriptor class
SHALL fail with `TypeError`. Every constructible descriptor SHALL be a
`DTypeCategory`, `SimpleDType`, or concrete `CompoundDType`.

For every descriptor, `is_category()`, `is_simple()`, and `is_compound()` SHALL
identify its representation kind. `supertype` SHALL return its immediately
enclosing category or `None`; `supertypes()` SHALL return all enclosing
categories from nearest to outermost. `is_subtype_of(other)` SHALL return
`True` when the descriptor is `other` or reaches `other` through that category
chain, and SHALL fail with `TypeError` when `other` is not a `DType`.

`is_opaque_storage()` SHALL report only a category's legacy opaque-storage
disposition. It SHALL NOT imply support by any carrier; carrier-specific
storage support is defined by `carrier-storage`.

#### Scenario: Query a hierarchy

- **WHEN** a caller queries `DType.Float32`
- **THEN** it is simple, its immediate supertype is `DType.Floating`, its
  supertypes include `DType.Floating` followed by `DType.Any`, and it is a
  subtype of each of those identities

#### Scenario: Reject a non-descriptor subtype query

- **WHEN** `is_subtype_of` receives an object that is not a `DType`
- **THEN** it fails with `TypeError`

### Requirement: Categories describe relationships and legacy disposition

`DTypeCategory(name, *, supertype=None, opaque_storage=False)` SHALL construct
and return a category descriptor. `name` names the category's unique registry
key and SHALL satisfy the registered-name contract. `supertype` names the
category's immediately enclosing category; it SHALL be optional, SHALL default
to `None` for a root category, and when provided SHALL be a `DTypeCategory`.
When `supertype` is neither `None` nor a `DTypeCategory`, construction SHALL
fail with `TypeError`. `opaque_storage` states whether the category has the
legacy opaque-storage disposition; it SHALL be optional, SHALL default to
`False`, and SHALL determine the result of `is_opaque_storage()` without making
the category simple or compound.

#### Scenario: Construct a category extension

- **WHEN** a caller constructs a category with a unique name and a registered
  category as its supertype
- **THEN** the result is registered, reports `is_category() == True`, and joins
  the supplied category chain

### Requirement: Simple dtypes declare one exact positive width

`SimpleDType(name, *, bits, supertype)` SHALL construct and return a simple
descriptor. `name` names the simple dtype's unique registry key and SHALL
satisfy the registered-name contract. `bits` names the encoding width in bits
and SHALL be an integer other than `bool` greater than zero. `supertype` names
the category immediately enclosing the encoding and SHALL be a
`DTypeCategory`. When `bits` is invalid, construction SHALL fail with
`ValueError`; when `supertype` is invalid, construction SHALL fail with
`TypeError`. Either failure SHALL occur before `name` is registered.

The `bits` property SHALL return the exact integer width. `is_simple()` SHALL
describe the representation and SHALL remain `True` even when no carrier
supports that encoding.

#### Scenario: Construct a simple extension

- **WHEN** a caller constructs `SimpleDType("Float16", bits=16,
  supertype=DType.Floating)` under an unused name
- **THEN** the result is a registered 16-bit simple dtype beneath the Floating
  and Any categories

#### Scenario: Reject an invalid width

- **WHEN** `bits` is a boolean, non-integer, zero, or negative
- **THEN** construction fails with `ValueError` and registers nothing

### Requirement: The built-in descriptor graph is stable

The built-in categories SHALL be `Any`, `Floating`, and `Integer`. `Any` SHALL
be the root; `Floating` and `Integer` SHALL have `Any` as their supertype.
`Any` and `Floating` SHALL report the legacy opaque-storage disposition, while
`Integer` SHALL not.

The built-in simple dtypes and widths SHALL be:

| Supertype | Simple dtype widths |
| --- | --- |
| `DType.Any` | `Bool`: 8 bits |
| `DType.Integer` | `Int32`: 32 bits; `Int8`: 8 bits |
| `DType.Floating` | `Float32`: 32 bits; `Float64`: 64 bits; `E8M0`: 8 bits; `E5M2`: 8 bits; `E4M3`: 8 bits; `E3M2`: 6 bits; `E2M3`: 6 bits; `E2M1`: 4 bits |

Each built-in SHALL be available as `DType.<name>` and SHALL be identical to
the object returned by `DType.from_name(<name>)`.

#### Scenario: Resolve a built-in through both surfaces

- **WHEN** a caller reads `DType.Float32` and calls
  `DType.from_name("Float32")`
- **THEN** both expressions return the same descriptor object

### Requirement: Registry discovery is class-narrowed

`DType.registered()` SHALL return an immutable snapshot of all registered
descriptors. Calling `registered()` on a descriptor subclass SHALL return only
registered instances of that subclass.

`DType.from_name(name)` SHALL return the registered identity whose exact name
matches `name`; calling it on a subclass SHALL additionally require the result
to be an instance of that subclass. A non-string name SHALL fail with
`TypeError`. An unknown name or a name registered to a different descriptor
kind SHALL fail with `LookupError`.

#### Scenario: Discover extension dtypes

- **WHEN** a caller registers a new `SimpleDType`
- **THEN** it appears in both `DType.registered()` and
  `SimpleDType.registered()` and resolves from both classes by name

#### Scenario: Reject a class-mismatched lookup

- **WHEN** `SimpleDType.from_name` receives the name of a category
- **THEN** it fails with `LookupError`

### Requirement: Published descriptors are immutable

After successful construction, assigning or deleting descriptor attributes
SHALL fail with `AttributeError`. Shallow copy and deep copy SHALL return the
same descriptor identity. `repr(descriptor)` SHALL identify its descriptor
kind and registered name.

The descriptor model SHALL own the state and accessors that determine its
kind, hierarchy, structure, planes, and representation. A descriptor subclass
that shadows model-owned members directly or through a base class SHALL fail
with `TypeError` when the class is defined. Assigning or deleting a model-owned
class member later SHALL fail with `AttributeError`.

#### Scenario: Preserve identity through copying

- **WHEN** a caller shallow-copies or deep-copies a registered descriptor
- **THEN** each result is the original object

#### Scenario: Refuse an extension that shadows descriptor identity

- **WHEN** a descriptor subclass defines a model-owned accessor such as
  `name`, `bits`, `simple_types`, or `structure`
- **THEN** class creation fails with `TypeError`

### Requirement: Descriptor structure is canonical and extensible

`structure()` SHALL return the immutable structure recorded once at successful
finalization. It SHALL include the descriptor contracts, the complete
structures of every referenced descriptor, and the result of
`structure_extension()`. Referenced descriptors SHALL already be canonical
registered identities.

`structure_extension()` SHALL default to `()`. An extension implementation MAY
override it to return a tuple of exact strings, numbers, `None`, `Whole`, and
nested tuples of those values. A non-tuple result SHALL fail with `TypeError`;
an unsupported value within the tuple SHALL fail with `TypeError`. A structure
extension SHALL participate in pickle compatibility and in every structural
uniqueness rule imposed by the descriptor kind.

#### Scenario: Distinguish extension representations

- **WHEN** two extension descriptors have different permitted values in their
  `structure_extension()` tuples
- **THEN** their recorded structures are different even if their inherited
  descriptor fields otherwise agree

#### Scenario: Reject an unregistered reference

- **WHEN** a descriptor attempts to use an unfinished or rejected descriptor
  as its supertype or representation component
- **THEN** construction fails with `ValueError` and registers nothing

### Requirement: Pickle reconstruction preserves receiving-process identity

Serializing a descriptor SHALL record its registered name and complete
structure, not a new descriptor definition. Deserializing SHALL return the
receiving process's already registered descriptor with that name when its
complete structure matches.

If the name is not registered in the receiving process, deserialization SHALL
fail with `LookupError`. If the name is registered to a descriptor with a
different structure, deserialization SHALL fail with `ValueError` rather than
substituting that descriptor.

#### Scenario: Unpickle a registered extension

- **WHEN** the receiving process has registered a structurally matching
  extension under the serialized name
- **THEN** deserialization returns that receiving-process descriptor identity

#### Scenario: Reject a mismatched receiving definition

- **WHEN** the receiving process registered the serialized name with a
  different width, hierarchy, plane, rule, scale, or extension structure
- **THEN** deserialization fails with `ValueError`

### Requirement: The DType class namespace contains only built-ins

Successful extension registration SHALL leave the attributes of `DType`
unchanged; extensions SHALL be reached through `registered()` and
`from_name()`. Reassigning or deleting a built-in `DType` binding SHALL fail
with `AttributeError`. Assigning an extension descriptor as a new `DType`
attribute SHALL fail with `AttributeError`.

#### Scenario: Register without extending the class namespace

- **WHEN** a caller registers an extension named `Float16`
- **THEN** `DType.from_name("Float16")` returns it and `DType.Float16` remains
  unavailable
