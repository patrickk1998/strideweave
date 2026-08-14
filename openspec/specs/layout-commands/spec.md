---
title: Layout Commands
publish: true
status: stable
order: 5
summary: Hierarchical layout command lexing, layout-reference grammar, and the rearrange, reduce, and contraction lowerings.
---

# layout-commands Specification

## Purpose

Define the StrideWeave layout command language exported by the
`strideweave.einops` submodule: the lexical grammar, the layout-reference
grammar, the three command forms, the compiled command specifications they
produce, and the lowering each command executes.

A layout command is a string that names hierarchical layout structure. This
specification owns the grammar, its diagnostics, and the compiled lowering
plan. The layout, view, dtype, dispatch, and autograd contracts the lowering
executes remain owned by `core-layout`, `core-tensor-views`,
`operation-dtype-policy`, `carrier-dispatch`, and `autograd`, and are
referenced here rather than restated.

## Scope

This specification covers the layout command language as one contract rather
than splitting the parser surface from the executing operations. The lexer and
the layout-reference parser are public and independently callable, but a
layout reference is only half of a command: the arrow and comma structure, the
output-reference rules, and the symbol classification that gives a symbol its
meaning all live in the command layer. Splitting the language across two
documents would force a reimplementer to read both to implement any single
command string, so the grammar and the commands it serves are specified
together.

## Terminology

| Term | Meaning |
| --- | --- |
| layout command | A string describing a StrideWeave layout transformation, such as `"a (b c) -> c a"`. The executing operations name this input `description`; the compiling functions name it `command`. |
| dimension symbol | A named leaf in a layout reference, matching `[A-Za-z][A-Za-z0-9_]*`. A symbol names one extracted leaf; it never names a size, and it may not repeat within one layout reference. |
| layout reference | A whitespace-separated sequence of dimension symbols, singleton `1` markers, and parenthesized groups describing one hierarchical layout. |
| input reference | The layout reference that names the leaves extracted from an operand's layout. |
| output reference | The layout reference after `->` that names which extracted leaves appear in the result and how they are grouped. |
| extracted leaf | One sublayout identified by a leaf of the input reference's selection tree. An extracted leaf may itself be hierarchical. Every dimension symbol names an extracted leaf, but a `1` marker also produces one, so extracted leaves outnumber dimension symbols whenever the input reference contains a `1`. |
| infix parse order | Left-to-right order over a layout reference with each parenthesized group's contents visited where the group appears, so `"a (b c) d"` orders `a`, `b`, `c`, `d`. |
| source id | The zero-based position an extracted leaf occupies in infix parse order over its input reference. |
| selection tree | A `Tree` whose items are `Node.Leaf` markers and nested `Tree`s, congruent with the operand layout's mode hierarchy. Each `Node.Leaf` selects one source subtree as one extracted leaf; each nested `Tree` descends into a source mode without extracting it. |
| output tree | A `Tree` whose items are `Node.id` markers, unidentified `Node.Leaf` markers, and nested `Tree`s. A `Node.id` places the extracted leaf carrying that source id, an unidentified `Node.Leaf` inserts a new mode, and a nested `Tree` constructs one nested output mode. |
| logical size | The count of logical coordinates a layout addresses, as defined by `core-layout`. |
| shared symbol | In a contraction command, a dimension symbol naming a leaf in both input references. |
| contraction symbol | A shared symbol omitted from the output reference. |
| batch symbol | A shared symbol retained in the output reference. |
| free symbol | A symbol naming a leaf in exactly one input reference. |
| kept structure | In a reduce command, the output tree naming the extracted leaves the result retains. |
| reduced structure | In a reduce command, the tree naming the extracted leaves the result sums away. |
| union symbol order | For a contraction command, every left-reference symbol in left order followed by every right-reference symbol absent from the left reference, in right order. A symbol's union position is its index in that order. |

## Requirements

### Requirement: The layout command surface is an explicit submodule export

The `strideweave.einops` submodule's `__all__` SHALL list exactly `EinsumSpec`,
`LayoutReference`, `RearrangeSpec`, `ReduceSpec`, `Token`, `TokenKind`,
`einsum`, `lex`, `parse_einsum`, `parse_layout_ref`, `parse_rearrange`,
`parse_reduce`, `rearrange`, and `reduce`.

`lex`, `parse_layout_ref`, `parse_rearrange`, `parse_reduce`, and
`parse_einsum` compile command text without touching a tensor. `rearrange`,
`reduce`, and `einsum` compile command text and then execute the compiled
lowering against tensor operands; each successful call MUST return an ordinary
public `Tensor`. `Token`, `TokenKind`, `LayoutReference`, `RearrangeSpec`,
`ReduceSpec`, and `EinsumSpec` are the value types those functions return.

For `lex(command)`, `parse_rearrange(command)`, `parse_reduce(command)`, and
`parse_einsum(command)`, `command` SHALL mean the layout command text to
compile. For `rearrange(tensor, description)`, `reduce(tensor, description)`,
and `einsum(lhs, rhs, description)`, `description` SHALL mean that same
command text. Each SHALL be a `str`. A non-`str` `command` SHALL raise
`TypeError` with `command must be a str`; a non-`str` `description` SHALL
raise `TypeError` with `description must be a str`. Both rejections MUST
occur before any lexing, compilation, or operation dispatch.

#### Scenario: Import the layout command surface

- **WHEN** a caller imports `strideweave.einops`
- **THEN** the fourteen listed names are available from that submodule
- **AND** each successful `rearrange`, `reduce`, or `einsum` call returns a `Tensor`

#### Scenario: Reject a non-string command

- **WHEN** `lex`, `parse_rearrange`, `parse_reduce`, or `parse_einsum` receives a non-`str` command
- **THEN** it raises `TypeError` with `command must be a str`
- **AND** no token, compiled specification, or operation dispatch is produced

#### Scenario: Reject a non-string description

- **WHEN** `rearrange`, `reduce`, or `einsum` receives a non-`str` description
- **THEN** it raises `TypeError` with `description must be a str`
- **AND** no operation is dispatched

### Requirement: Layout commands describe hierarchical layouts

A layout command names StrideWeave layout structure. A dimension symbol names
one extracted leaf, which may itself be hierarchical; parentheses in an input
reference select a source subtree as one unit, and parentheses in an output
reference construct a nested output mode. A command SHALL therefore be
interpreted as a hierarchy-preserving transformation over the operand's
existing mode structure.

Because a command names only structure the operand already carries, splitting
one mode into several output leaves SHALL require that the operand layout
already carry that hierarchy, named by a parenthesized group in the input
reference. The language provides no axis-size argument, no ellipsis, no repeat
or tile form, and no contraction over more than two operands. A symbol
repeated within one layout reference SHALL be rejected as a duplicate, so the
language expresses neither a diagonal nor a trace.

#### Scenario: Select an existing hierarchical mode

- **WHEN** a command names a parenthesized group such as `"a (b k), c k -> a b c"` and the operand layout carries that nested mode
- **THEN** the group selects the existing source subtree and its leaves are addressable as `b` and `k`

#### Scenario: Refuse to divide a flat mode

- **WHEN** a rearrange command such as `"a (b c) -> a b c"` is applied to a tensor whose second mode is a flat integer extent
- **THEN** the command fails with `ValueError` with `layout and tree profile do not match`
- **AND** no extent is inferred for `b` or `c`

### Requirement: The lexer emits ASCII tokens carrying source offsets

`lex(command)` SHALL scan `command` from left to right and return a `list` of
`Token` in source order. ASCII whitespace — space, tab, newline, carriage
return, form feed, and vertical tab — separates tokens and produces none. An
empty command SHALL produce an empty list.

The lexer SHALL emit `left_paren` for `(`, `right_paren` for `)`, `comma` for
`,`, `arrow` for `->`, `one` for a singleton `1`, and `symbol` for a run
matching `[A-Za-z][A-Za-z0-9_]*`. Each token's `value` is its exact source
text, `start` is its zero-based first offset, and `end` is the offset one past
its last character.

Every failure SHALL raise `ValueError` naming the offset at which it was
detected:

| Condition | Message |
| --- | --- |
| A byte at or above 128 | `Layout command must contain only ASCII characters at offset {offset}` |
| A `-` not followed by `>` | `Expected '->' arrow token at offset {offset}` |
| A `>` not preceded by `-` | `Unexpected '>' at offset {offset}` |
| A `1` immediately followed by a letter, digit, or underscore | `Invalid singleton token at offset {offset}` |
| A digit other than a standalone `1` | `Invalid dimension symbol at offset {offset}` |
| Any other ASCII character | `Unexpected character at offset {offset}` |

The ASCII check SHALL apply to every scanned character, including a character
examined only to decide whether a symbol or singleton run continues.
`Expected '->' arrow token` and `Invalid singleton token` are each decided by
examining the character after the run's first character, and SHALL report the
offset of that first character. Every other condition SHALL report the offset
of the character that triggered it.

#### Scenario: Tokenize a grouped command

- **WHEN** a caller lexes `"a (b c) -> c a"`
- **THEN** the token kinds are `symbol`, `left_paren`, `symbol`, `symbol`, `right_paren`, `arrow`, `symbol`, `symbol`

#### Scenario: Report the offset of a trailing non-ASCII character

- **WHEN** a caller lexes `"aé"`
- **THEN** it raises `ValueError` with `Layout command must contain only ASCII characters at offset 1`

#### Scenario: Reject a digit that is not a standalone singleton

- **WHEN** a caller lexes `"11"`
- **THEN** it raises `ValueError` with `Invalid singleton token at offset 0`

#### Scenario: Reject a numeric extent

- **WHEN** a caller lexes `"2"`
- **THEN** it raises `ValueError` with `Invalid dimension symbol at offset 0`

### Requirement: Tokens are immutable value records

`Token` SHALL be a frozen, slotted value type whose attribute set is exactly
`kind`, `value`, `start`, and `end`, compared and hashed by those fields.
Assigning to any field SHALL raise `AttributeError`.

`TokenKind` SHALL be the string literal type whose members are exactly
`left_paren`, `right_paren`, `arrow`, `comma`, `one`, and `symbol`. Every
emitted token's `kind` SHALL be one of those six strings.

#### Scenario: Refuse to mutate a token

- **WHEN** a caller assigns to the `kind` field of a lexed token
- **THEN** the assignment raises `AttributeError`
- **AND** the token's attribute set remains exactly `kind`, `value`, `start`, and `end`

### Requirement: A layout reference parses into a selection tree and infix symbol ids

For `parse_layout_ref(tokens)`, `tokens` SHALL mean a sequence of `Token`
describing exactly one layout reference, without commas or arrows. The call
SHALL return a `LayoutReference` with the fields `tree` and `symbol_ids`. A
`tokens` value that is not iterable SHALL raise `TypeError`; a sequence
element that is not a `Token` SHALL raise `TypeError` with `layout reference
tokens must be Token objects`.

`tree` is the selection tree: each `symbol` or `one` token becomes one
`Node.Leaf`, and each parenthesized group becomes one nested `Tree`. Every
leaf consumes the next source id in infix parse order, so the leaf count
equals the number of `symbol` and `one` tokens. `symbol_ids` SHALL pair each
dimension symbol with its source id, ordered by the symbol's position in the
reference. A `one` token consumes a source id and produces an anonymous
extracted leaf that `symbol_ids` omits; because an output reference can name
only a symbol, an anonymous extracted leaf is always omitted from the output.

Every failure SHALL raise `ValueError`:

| Condition | Message |
| --- | --- |
| The reference has no token, or a level has no item and no unconsumed token remains | `Layout reference must not be empty` |
| A parenthesized group encloses no item | `Empty parenthesized group at offset {offset}` |
| A `)` has no open group | `Unmatched right parenthesis at offset {offset}` |
| A `(` is never closed | `Unclosed left parenthesis at offset {offset}` |
| A `,` appears in the reference | `Commas are not valid in layout references at offset {offset}` |
| A `->` appears in the reference | `Arrows are not valid in layout references at offset {offset}` |
| A symbol repeats within the reference | `Duplicate dimension symbol '{symbol}' at offset {offset}` |

The unclosed-group message SHALL name the offset of the unmatched `(`; every
other offset names the offending token.

#### Scenario: Assign source ids in infix order

- **WHEN** a caller parses the reference `"a (b c)"`
- **THEN** `tree` is `Tree(Node.Leaf, Tree(Node.Leaf, Node.Leaf))`
- **AND** `symbol_ids` is `(('a', 0), ('b', 1), ('c', 2))`

#### Scenario: Keep an input singleton anonymous

- **WHEN** a caller parses the reference `"1 a"`
- **THEN** the selection tree has two leaves
- **AND** `symbol_ids` is `(('a', 1),)`

#### Scenario: Reject a repeated symbol

- **WHEN** a caller parses the reference `"a a"`
- **THEN** it raises `ValueError` with `Duplicate dimension symbol 'a' at offset 2`

#### Scenario: Reject an empty group

- **WHEN** a caller parses the reference `"a ()"`
- **THEN** it raises `ValueError` with `Empty parenthesized group at offset 3`

### Requirement: An output reference names selected leaves and inserts singleton modes

An output reference SHALL use the same bracketing grammar as an input
reference and SHALL produce an output tree. Each dimension symbol becomes
`Node.id` bound to the source id that symbol carries in the surrounding
command. Each `one` token becomes an unidentified `Node.Leaf`, which the
rearrangement contract in `core-layout` materializes as an inserted
extent-one, stride-zero mode. Each parenthesized group becomes one nested
output mode.

An output reference SHALL raise `ValueError` with `Unknown dimension symbol
'{symbol}' at offset {offset}` for a symbol the command's input references do
not bind, and `Duplicate output dimension symbol '{symbol}' at offset
{offset}` for a symbol that already appears in the output. The empty,
parenthesis, comma, and arrow diagnostics of a layout reference apply
unchanged.

#### Scenario: Group and insert output modes

- **WHEN** a rearrange command `"a b c -> a (b c)"` is compiled
- **THEN** the output tree is `Tree(Node.id(0), Tree(Node.id(1), Node.id(2)))`

#### Scenario: Insert a singleton output mode

- **WHEN** a rearrange command `"a -> a 1"` is compiled
- **THEN** the output tree is `Tree(Node.id(0), Node.Leaf)`

#### Scenario: Reject an unbound output symbol

- **WHEN** a rearrange command `"a -> b"` is compiled
- **THEN** it raises `ValueError` with `Unknown dimension symbol 'b' at offset 5`

#### Scenario: Reject a repeated output symbol

- **WHEN** a rearrange command `"a b -> a a"` is compiled
- **THEN** it raises `ValueError` with `Duplicate output dimension symbol 'a' at offset 9`

### Requirement: Command forms are separated by one arrow and, for contraction, one comma

A rearrange or reduce command SHALL have the form `input -> output`, and a
contraction command SHALL have the form `lhs, rhs -> output`. The separators
SHALL be located before either side is parsed.

A command with no arrow SHALL raise `ValueError` with `{Name} command must
contain one '->' arrow`, and a command with more than one arrow SHALL raise
`{Name} command must contain only one '->' arrow`, where `{Name}` is
`Rearrange` for `parse_rearrange`, `Reduce` for `parse_reduce`, and `Einsum`
for `parse_einsum`.

A contraction command SHALL additionally require exactly one comma before the
arrow: none SHALL raise `ValueError` with `Contraction command must contain
one comma before '->'`, and more than one SHALL raise `Contraction command
must contain only one comma before '->'`. Only commas before the arrow are
counted; a comma after the arrow is rejected by the output reference as
`Commas are not valid in layout references at offset {offset}`.

#### Scenario: Reject a rearrange command without an arrow

- **WHEN** a caller compiles the rearrange command `"a b"`
- **THEN** it raises `ValueError` with `Rearrange command must contain one '->' arrow`

#### Scenario: Reject a contraction command without a comma

- **WHEN** a caller compiles the contraction command `"a b c -> a c"`
- **THEN** it raises `ValueError` with `Contraction command must contain one comma before '->'`

#### Scenario: Reject a comma after the arrow

- **WHEN** a caller compiles the contraction command `"a b, c b -> a, c"`
- **THEN** it raises `ValueError` with `Commas are not valid in layout references at offset 13`

### Requirement: A rearrange command compiles into a selection and output tree

`parse_rearrange(command)` SHALL return a `RearrangeSpec` whose `selection` is
the input reference's selection tree, whose `output` is the output tree parsed
against that reference's symbol ids, and whose `symbol_ids` are that
reference's symbol ids.

Compilation SHALL accept an output reference naming any subset of the
extracted leaves. An omitted extracted leaf is admissible only when its
logical size is one, and that condition is enforced by the rearrangement
contract in `core-layout` when the compiled spec is applied to a layout,
rather than at compile time.

#### Scenario: Compile a grouped rearrange command

- **WHEN** a caller compiles `"a (b c) -> a c"`
- **THEN** `selection` is `Tree(Node.Leaf, Tree(Node.Leaf, Node.Leaf))`
- **AND** `output` is `Tree(Node.id(0), Node.id(2))`
- **AND** `symbol_ids` is `(('a', 0), ('b', 1), ('c', 2))`

#### Scenario: Compile an omission and defer its size condition

- **WHEN** a caller compiles `"1 a -> a"`
- **THEN** compilation succeeds with `output` `Tree(Node.id(1))`
- **AND** applying it to a layout whose first extracted leaf has logical size greater than one raises `ValueError` with `Layout rearrange ids must include every extracted layout`

### Requirement: A reduce command lowers omitted dimensions into a second mode

`parse_reduce(command)` SHALL return a `ReduceSpec` whose `selection` and
`symbol_ids` come from the input reference and whose `output` is the kept
structure. `reduced` SHALL be the reduced structure: a `Tree` of `Node.id`
markers naming every source id absent from `output`, in ascending source-id
order. `rearrange_output` SHALL be `Tree(output, reduced)`, the two-mode
intermediate whose first mode holds the kept structure and whose second mode
holds the reduced structure.

A command whose output retains every source id SHALL raise `ValueError` with
`Reduce command must omit at least one dimension`. An anonymous extracted leaf
introduced by a `1` token cannot be named in the output and therefore always
joins the reduced structure, carrying whatever logical size the operand layout
gives it; a `1` in an input reference SHALL NOT be read as an assertion that
the leaf has logical size one.

#### Scenario: Compile a hierarchical reduce command

- **WHEN** a caller compiles `"a (c d) b -> a c"`
- **THEN** `output` is `Tree(Node.id(0), Node.id(1))`
- **AND** `reduced` is `Tree(Node.id(2), Node.id(3))`
- **AND** `rearrange_output` is `Tree(Tree(Node.id(0), Node.id(1)), Tree(Node.id(2), Node.id(3)))`

#### Scenario: Reject a reduce command that omits nothing

- **WHEN** a caller compiles `"a b -> a b"`
- **THEN** it raises `ValueError` with `Reduce command must omit at least one dimension`

### Requirement: A contraction command classifies every symbol by output presence

`parse_einsum(command)` SHALL parse both input references, classify each
symbol uniformly, and return an `EinsumSpec`.

`lhs_selection` and `rhs_selection` SHALL be the two input references'
selection trees, and `lhs_symbol_ids` and `rhs_symbol_ids` their symbol ids.
`common_symbols` SHALL list the symbols bound by both input references,
ordered by their position in the left reference. A command whose inputs share
no symbol SHALL raise `ValueError` with `Contraction command must include at
least one shared dimension`.

`union_symbols` SHALL be the union symbol order. The output reference SHALL be
parsed against union positions to produce `general_output`. Regardless of
symbol classification, `lhs_union_output` and `rhs_union_output` SHALL each
hold one top-level marker per union symbol in union symbol order — `Node.id`
bound to that side's source id when the side binds the symbol, and an
unidentified `Node.Leaf` otherwise — and `union_selection` SHALL be a `Tree`
of one `Node.Leaf` per union symbol.

A shared symbol absent from the output is a contraction symbol; a shared
symbol present in the output is a batch symbol. `contraction_symbols` and
`batch_symbols` SHALL each preserve `common_symbols` order. Every free symbol
MUST appear in the output; omitting one SHALL raise `ValueError` with `Einsum
output must include every non-shared input symbol: ` followed by the missing
symbols sorted and joined by `, `.

#### Scenario: Classify batch and contraction symbols

- **WHEN** a caller compiles `"b i k, b j k -> b i j"`
- **THEN** `union_symbols` is `('b', 'i', 'k', 'j')`
- **AND** `batch_symbols` is `('b',)` and `contraction_symbols` is `('k',)`

#### Scenario: Populate the union fields without a batch symbol

- **WHEN** a caller compiles `"a b, c b -> a c"`, whose `batch_symbols` is empty
- **THEN** `lhs_union_output` is `Tree(Node.id(0), Node.id(1), Node.Leaf)`
- **AND** `rhs_union_output` is `Tree(Node.Leaf, Node.id(1), Node.id(0))`
- **AND** `union_selection` is `Tree(Node.Leaf, Node.Leaf, Node.Leaf)`

#### Scenario: Reject an omitted free symbol

- **WHEN** a caller compiles `"a b, c b -> a b"`
- **THEN** it raises `ValueError` with `Einsum output must include every non-shared input symbol: c`

#### Scenario: Reject a contraction with no shared symbol

- **WHEN** a caller compiles `"a, b -> a b"`
- **THEN** it raises `ValueError` with `Contraction command must include at least one shared dimension`

### Requirement: A contraction without batch symbols compiles to the two-mode matmul lowering

When `batch_symbols` is empty, `lhs_rearrange_output` and
`rhs_rearrange_output` SHALL each be `Tree(outer, inner)`. `outer` holds that
side's free symbols as `Node.id` markers in that reference's order, or a
single `Node.Leaf` when the side has no free symbol. `inner` holds the shared
symbols as `Node.id` markers in `common_symbols` order, so both operands
present their shared modes in the left reference's order regardless of the
order the right reference wrote them.

`matmul_output_selection` SHALL be `Tree(lhs_selection_modes,
rhs_selection_modes)`, where each side is a `Tree` of `Node.Leaf` markers, one
per free symbol on that side, or a single `Node.Leaf` when that side has none.
`output` SHALL be the output reference parsed against ids that number the
leaves of `matmul_output_selection`: the left free symbols first, then the
right free symbols. Because a side with no free symbol still contributes one
`Node.Leaf` for its inserted singleton, that side still consumes one id.

#### Scenario: Compile a matrix contraction

- **WHEN** a caller compiles `"a b, c b -> a c"`
- **THEN** `lhs_rearrange_output` and `rhs_rearrange_output` are both `Tree(Tree(Node.id(0)), Tree(Node.id(1)))`
- **AND** `matmul_output_selection` is `Tree(Tree(Node.Leaf), Tree(Node.Leaf))`
- **AND** `output` is `Tree(Node.id(0), Node.id(1))`

#### Scenario: Reorder right shared modes to left order

- **WHEN** a caller compiles `"a b c, d c b -> a d"`
- **THEN** `lhs_rearrange_output` is `Tree(Tree(Node.id(0)), Tree(Node.id(1), Node.id(2)))`
- **AND** `rhs_rearrange_output` is `Tree(Tree(Node.id(0)), Tree(Node.id(2), Node.id(1)))`

#### Scenario: Number output ids past an inserted singleton

- **WHEN** a caller compiles `"b, c b -> c"`, whose left side has no free symbol
- **THEN** `matmul_output_selection` is `Tree(Tree(Node.Leaf), Tree(Node.Leaf))`
- **AND** `output` is `Tree(Node.id(1))`, because id `0` addresses the left side's inserted singleton

#### Scenario: Compile a contraction with a one-sided singleton

- **WHEN** a caller compiles `"b, b -> 1"`
- **THEN** both rearrange outputs are `Tree(Tree(Node.Leaf), Tree(Node.id(0)))`
- **AND** `output` is `Tree(Node.Leaf)`

### Requirement: A contraction with batch symbols compiles to the union-aligned lowering

When `batch_symbols` is non-empty, `lhs_rearrange_output` and
`rhs_rearrange_output` SHALL equal `lhs_union_output` and `rhs_union_output`,
`matmul_output_selection` SHALL equal `union_selection`, and `output` SHALL
equal `general_output`. Each operand therefore presents one top-level mode per
union symbol, with an extent-one, stride-zero mode wherever that operand does
not bind the symbol. Symbol positions are preserved: each operand keeps its
own union-ordered modes, and the lowering derives result structure from union
symbol order alone.

#### Scenario: Compile a batched contraction

- **WHEN** a caller compiles `"b i k, b j k -> b i j"`
- **THEN** `lhs_union_output` is `Tree(Node.id(0), Node.id(1), Node.id(2), Node.Leaf)`
- **AND** `rhs_union_output` is `Tree(Node.id(0), Node.Leaf, Node.id(2), Node.id(1))`
- **AND** `general_output` is `Tree(Node.id(0), Node.id(1), Node.id(3))`

### Requirement: Compiled command specifications are cached per command kind

`parse_rearrange`, `parse_reduce`, and `parse_einsum` SHALL each memoize
successful compilations by exact command string and SHALL return the identical
specification object for a repeated string. The three memoizations are
independent, so the same string compiled as two different command kinds SHALL
yield two distinct specifications.

A compilation that raises SHALL NOT be memoized; recompiling the same invalid
command SHALL raise the same diagnostic again. Compiled specifications SHALL
be immutable, so sharing one object across callers is observable only through
identity.

`parse_layout_ref` and `lex` are not memoized and SHALL compile on every call.

#### Scenario: Reuse a compiled specification

- **WHEN** a caller compiles the same rearrange command twice
- **THEN** both calls return the identical specification object

#### Scenario: Keep command kinds separate

- **WHEN** a caller compiles one string as both a rearrange and a reduce command
- **THEN** each kind returns its own specification
- **AND** repeating either call returns that kind's identical object

#### Scenario: Do not reuse a failed compilation

- **WHEN** a caller compiles an invalid command twice
- **THEN** both calls raise the same `ValueError`

### Requirement: Layout command operands are public tensors the input reference describes

For `rearrange(tensor, description)` and `reduce(tensor, description)`,
`tensor` SHALL mean the operand `Tensor` whose layout the command's input
reference describes. For `einsum(lhs, rhs, description)`, `lhs` and `rhs`
SHALL mean the operand `Tensor` values described by the left and right input
references respectively.

Each operand's layout MUST be congruent with its input reference: the
reference's selection tree SHALL identify one extracted leaf per `symbol` or
`one` token, and each such leaf SHALL correspond to an existing mode or
subtree of that operand's layout. An operand layout that does not match its
reference SHALL raise `ValueError` with `layout and tree profile do not
match`. Operand layouts may be hierarchical to any depth, and an extracted
leaf may be a whole subtree.

The command SHALL compile before any operand is inspected, so a command
diagnostic SHALL precede an operand diagnostic. `rearrange` and `reduce` SHALL
reject an operand that is not a `Tensor` with `TypeError` and the message
`tensor must be a Tensor`. An `einsum` operand that does not provide the
required Tensor interface SHALL raise `AttributeError` at its first missing
Tensor attribute.

#### Scenario: Accept a hierarchical operand

- **WHEN** a caller applies `"a (c d) b -> a c"` to a tensor with layout `Shape([2, [3, 4], 5])`
- **THEN** the parenthesized group selects the existing nested mode as extracted leaves `c` and `d`

#### Scenario: Reject a reference that does not match the operand layout

- **WHEN** a caller applies `"a b c -> a"` through `rearrange` or `reduce`, or `"a b c, d b -> a c d"` through `einsum`, to a two-mode operand
- **THEN** it raises `ValueError` with `layout and tree profile do not match`

#### Scenario: Report a command diagnostic before an operand diagnostic

- **WHEN** a caller passes both an invalid command and a non-Tensor operand
- **THEN** the command diagnostic is raised and no operand is inspected

### Requirement: Rearrange executes the compiled trees through the core rearrangement contract

`rearrange(tensor, description)` SHALL compile `description` as a rearrange
command and apply the compiled `selection` and `output` trees through the
public tensor rearrangement operation. The result SHALL be a view governed by
`core-tensor-views`: it shares the operand's backing carrier and Tensor offset
and copies no value.

An omitted extracted leaf whose logical size is greater than one SHALL raise
`ValueError` with `Layout rearrange ids must include every extracted layout`,
before any view is returned.

#### Scenario: Rearrange without copying

- **WHEN** a caller rearranges a two-mode tensor with `"a b -> b a"`
- **THEN** the result shares the operand's carrier and Tensor offset and carries the transposed layout

#### Scenario: Reject an omitted non-singleton leaf

- **WHEN** a caller rearranges a tensor with `"a b -> a"` and `b` has logical size greater than one
- **THEN** it raises `ValueError` with `Layout rearrange ids must include every extracted layout`

### Requirement: Reduce sums the omitted dimensions through the two-mode reduction

`reduce(tensor, description, *, accumulator_dtype=None)` SHALL compile
`description` as a reduce command, rearrange the operand into the
`rearrange_output` two-mode intermediate, and sum that intermediate's second
mode using public `reduce_sum` semantics. The operand may carry any layout
congruent with its input reference; the command lowers it to a two-mode
intermediate whose first mode holds the kept structure and whose second mode
holds the reduced structure, and the reduction primitive consumes that
two-mode intermediate and reduces its second top-level mode.

The result layout SHALL be the compiled kept structure applied to the
operand's extracted leaves, and each result element SHALL be the sum over the
complete reduced-structure fiber addressed by that kept coordinate.

`accumulator_dtype` SHALL mean the dtype the reduction accumulates in. It is
optional and SHALL default to `None`, which selects the reduction's default
accumulator. A supplied value SHALL be forwarded to the reduction as a typed
execution option governed by `operation-dtype-policy`. Requesting an
accumulator that the operand's dtype disposition does not support SHALL retain
that operation's diagnostic.

#### Scenario: Reduce a hierarchical operand

- **WHEN** a caller reduces a tensor with layout `Shape([2, [3, 4], 5])` using `"a (c d) b -> a c"`
- **THEN** the result has two kept modes with extents `2` and `3`
- **AND** each result element sums the twenty elements of its `d` and `b` fiber

#### Scenario: Default the reduction accumulator

- **WHEN** a caller reduces a tensor without supplying `accumulator_dtype`
- **THEN** the reduction uses its default accumulator

#### Scenario: Widen the reduction accumulator

- **WHEN** a caller reduces a Float32 tensor with `accumulator_dtype` set to `DType.Float64`
- **THEN** accumulation widens under the operation dtype policy and the result dtype is unchanged

### Requirement: Contraction validates shared sizes before executing its compiled lowering

`einsum(lhs, rhs, description)` SHALL compile `description` as a contraction
command and then, before dispatching any operation, extract each operand's
leaves and compare the logical size each operand gives every shared symbol.
Shared symbols MUST have equal logical sizes on both operands; a mismatch
SHALL raise `ValueError` with `Einsum shared dimension '{symbol}' has
mismatched logical size`. Because that comparison extracts leaves, an operand
layout incongruent with its reference SHALL fail first with `layout and tree
profile do not match`.

With no batch symbol, `einsum` SHALL rearrange each operand into its
`Tree(outer, inner)` two-mode intermediate, contract those intermediates with
the public two-mode `matmul`, and rearrange that result with `output` and
`matmul_output_selection`.

With at least one batch symbol, `einsum` SHALL rearrange each operand into its
union layout, multiply the two union-shaped operands elementwise, and then
either rearrange the product with `general_output` and `union_selection` when
there is no contraction symbol, or rearrange it into `Tree(general_output,
contracted)` — where `contracted` names the contraction symbols by union
position — and sum that second mode using public `reduce_sum` semantics.

`einsum` takes no accumulator input, so both lowerings use their operations'
default accumulators. Each result element SHALL equal the sum, over every
combination of contraction symbol values, of the product of the two operand
elements at the matching batch and free coordinates.

#### Scenario: Contract two matrices

- **WHEN** a caller evaluates `"a b, c b -> a c"` on congruent operands
- **THEN** each result element is the sum over `b` of the operand products
- **AND** the compiled lowering contracts the two two-mode intermediates with the public two-mode matmul

#### Scenario: Contract with a batch symbol

- **WHEN** a caller evaluates `"b i k, b j k -> b i j"` on congruent operands
- **THEN** each result element sums over `k` at its own `b` coordinate

#### Scenario: Contract a batch symbol with no contraction symbol

- **WHEN** a caller evaluates `"b i, b j -> b i j"` on congruent operands
- **THEN** each result element is the product of the two operand elements at its own `b` coordinate

#### Scenario: Reject mismatched shared sizes

- **WHEN** a caller evaluates `"a b, c b -> a c"` where `b` has different logical sizes on the two operands
- **THEN** it raises `ValueError` with `Einsum shared dimension 'b' has mismatched logical size`
- **AND** no operation is dispatched

### Requirement: Layout command results differentiate through their lowered operations

`rearrange`, `reduce`, and `einsum` SHALL each record the autograd nodes of
the public operations their lowering dispatches, and no additional node of
their own. Those operations are rearrangement, elementwise multiplication,
broadcast, two-mode `matmul`, and two-mode `reduce_sum`. The gradient
behavior of a layout command result SHALL be whatever the `autograd`
specification gives for that recorded graph.

The union alignment a batched contraction performs inserts extent-one,
stride-zero modes, which the elementwise multiplication expands with the
public broadcast operation. The gradients of those broadcast operands SHALL
follow the `autograd` requirement covering gradient buffers that aggregate
broadcast aliases; this specification adds no gradient rule of its own.

#### Scenario: Backpropagate through a rearranged view

- **WHEN** a caller calls `backward` on the result of a rearrange command
- **THEN** the operand gradient is produced by the recorded rearrangement node under the autograd contract

#### Scenario: Backpropagate through a batched contraction

- **WHEN** a caller calls `backward` on the result of a contraction command carrying a batch symbol
- **THEN** the recorded graph includes the broadcast nodes the union alignment introduced
- **AND** each operand gradient aggregates the contributions of its broadcast modes under the autograd contract

### Requirement: Public operations that take a description accept this command language

The command language specified here is the grammar of every public
`strideweave` entry point that accepts a description string. `rearrange`
accepts a rearrange command as an alternative to an explicit output `Tree`,
and a string description combined with an explicit selection tree SHALL raise
`TypeError`. `einsum` accepts a contraction command. `reduce_sum`,
`reduce_prod`, `reduce_max`, `reduce_min`, `argmax`, and `argmin` accept a
reduce command, lowering it through the same two-mode intermediate and then
dispatching their own reduction rather than summation.

Those entry points contribute their own operation semantics, which this
specification does not restate. A command string valid for one of them SHALL
compile identically here, and every diagnostic in this specification SHALL
surface unchanged through them.

#### Scenario: Reuse a reduce command for a non-sum reduction

- **WHEN** a caller passes `"a (c d) b -> a c"` to `reduce_max`
- **THEN** the command compiles to the same two-mode intermediate
- **AND** the maximum, rather than the sum, is taken over the reduced structure

#### Scenario: Reject a string description combined with a selection tree

- **WHEN** a caller passes a rearrange command and an explicit selection tree to `rearrange`
- **THEN** it raises `TypeError`
