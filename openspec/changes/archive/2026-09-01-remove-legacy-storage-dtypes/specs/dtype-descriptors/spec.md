## ADDED Requirements

### Requirement: Categories describe hierarchy relationships

Within the dtype descriptor model, a category SHALL mean a `DTypeCategory`
that organizes descriptors in an abstract supertype hierarchy. A category
SHALL be a relationship identity rather than a physical storage
representation; `DType.Any`, `DType.Floating`, `DType.Integer`, and registered
extension categories SHALL all follow that same definition.

`DTypeCategory(name, *, supertype=None)` SHALL construct and return a category
descriptor. `name` names the category's unique registry key and SHALL satisfy
the registered-name contract. `supertype` names the category's immediately
enclosing category; it SHALL be optional, SHALL default to `None` for a root
category, and when provided SHALL be a registered `DTypeCategory`. When
`supertype` is neither `None` nor a `DTypeCategory`, construction SHALL fail
with `TypeError` before registering `name`. When `supertype` is a
`DTypeCategory` object but is unfinished, was rejected during registration, or
is not the canonical identity registered under its name, construction SHALL
fail with `ValueError` before registering `name`.

The constructor SHALL accept `supertype` only as a keyword argument. Supplying
the removed `opaque_storage` keyword SHALL fail with `TypeError` identifying an
unexpected keyword and SHALL register nothing.

#### Scenario: Construct a category extension

- **WHEN** a caller constructs a category with a unique name and a registered
  category as its `supertype`
- **THEN** the result is registered, reports `is_category() == True`, and joins
  the supplied category chain

#### Scenario: Reject the removed storage disposition

- **WHEN** a caller supplies `opaque_storage` to `DTypeCategory`
- **THEN** construction fails with `TypeError` before the requested name is
  registered

#### Scenario: Reject a non-category supertype

- **WHEN** a caller supplies a non-category descriptor or another non-category
  value as `supertype`
- **THEN** construction fails with `TypeError` before the requested name is
  registered

#### Scenario: Reject a noncanonical category supertype

- **WHEN** a caller supplies an unfinished, rejected, or noncanonical
  `DTypeCategory` object as `supertype`
- **THEN** construction fails with `ValueError` before the requested name is
  registered

## MODIFIED Requirements

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

The public descriptor surface SHALL provide no `is_opaque_storage` attribute.
Reading or calling that removed attribute SHALL fail with `AttributeError`.
Carrier-specific storage support remains defined by `carrier-storage` rather
than by a descriptor kind or category property.

#### Scenario: Query a hierarchy

- **WHEN** a caller queries `DType.Float32`
- **THEN** it is simple, its immediate supertype is `DType.Floating`, its
  supertypes include `DType.Floating` followed by `DType.Any`, and it is a
  subtype of each of those identities

#### Scenario: Reject a non-descriptor subtype query

- **WHEN** `is_subtype_of` receives an object that is not a `DType`
- **THEN** it fails with `TypeError`

#### Scenario: Observe removal of the opaque-storage query

- **WHEN** a caller reads `is_opaque_storage` from any descriptor
- **THEN** the read fails with `AttributeError`

### Requirement: The built-in descriptor graph is stable

The built-in categories SHALL be `Any`, `Floating`, and `Integer`. `Any` SHALL
be the root; `Floating` and `Integer` SHALL have `Any` as their supertype. All
three SHALL remain relationship-only categories that may serve as supertypes
for registered simple or compound descriptors.

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

#### Scenario: Retain categories as extension supertypes

- **WHEN** a caller registers a valid `SimpleDType` whose supertype is
  `DType.Floating`
- **THEN** the descriptor joins the Floating and Any hierarchy even though
  those categories are not physical storage representations

### Requirement: Descriptor structure is canonical and extensible

`structure()` SHALL return the immutable structure recorded once at successful
finalization. It SHALL include the descriptor contracts, the complete
structures of every referenced descriptor, and the result of
`structure_extension()`. Referenced descriptors SHALL already be canonical
registered identities. A category's canonical contract structure SHALL
describe its relationship kind and supertype and SHALL contain no storage
disposition field or value.

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

#### Scenario: Record a category without storage disposition

- **WHEN** a caller reads `structure()` from `DType.Any`, `DType.Floating`, or
  an extension category
- **THEN** the returned canonical structure contains its category hierarchy
  and no opaque-storage field or value

### Requirement: Pickle reconstruction preserves receiving-process identity

Serializing a descriptor SHALL record its registered name and complete
structure, not a new descriptor definition. Deserializing SHALL return the
receiving process's already registered descriptor with that name when its
complete structure matches.

If the name is not registered in the receiving process, deserialization SHALL
fail with `LookupError`. If the name is registered to a descriptor with a
different structure, deserialization SHALL fail with `ValueError` rather than
substituting that descriptor. A payload whose category structure contains the
removed opaque-storage disposition SHALL therefore be incompatible with the
new relationship-only category structure and SHALL fail with `ValueError`.

#### Scenario: Unpickle a registered extension

- **WHEN** the receiving process has registered a structurally matching
  extension under the serialized name
- **THEN** deserialization returns that receiving-process descriptor identity

#### Scenario: Reject a mismatched receiving definition

- **WHEN** the receiving process registered the serialized name with a
  different width, hierarchy, plane, rule, scale, or extension structure
- **THEN** deserialization fails with `ValueError`

#### Scenario: Reject a legacy category structure

- **WHEN** a serialized descriptor recursively contains a category structure
  with the removed opaque-storage disposition
- **THEN** deserialization against the relationship-only registered graph
  fails with `ValueError` rather than substituting a descriptor

## REMOVED Requirements

### Requirement: Categories describe relationships and legacy disposition

**Reason**: Categories remain public hierarchy identities but no longer carry a
physical-storage disposition.

**Migration**: Construct categories with `DTypeCategory(name, *,
supertype=None)`, query carrier storage through
`carrier.supports_storage_dtype(dtype)`, and use a supported `SimpleDType` for
physical storage.
