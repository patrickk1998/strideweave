"""StrideWeave hierarchical layout command parsing and tensor operation helpers.

This module exposes a native lexer, Python parsers for layout references, and
string-based rearrange, reduce, and einsum wrappers built on StrideWeave layouts.

Description strings are built from layout references. A layout reference is a
whitespace-separated sequence of dimension symbols, literal ``1`` singleton
dimensions, and parenthesized groups such as ``"a (b c)"``. Dimension symbols
name extracted layout leaves in infix order. Parenthesized groups preserve or
create hierarchical layout modes. ``->`` separates input references from output
references, and the two-input contraction form uses a comma before the arrow:
``"lhs, rhs -> output"``.

The tensor operations lower string descriptions into existing layout operations.
Rearrange reorders, groups, drops singleton dimensions, or inserts literal
``1`` dimensions without copying values. Reduce rearranges the tensor into a
two-mode intermediate ``(kept, reduced)`` layout and sums the second mode.
The contraction helper keeps the two-mode matmul lowering for contractions
without batch symbols. When a shared symbol is retained in the output, it
aligns both inputs over their union index space, multiplies elementwise, and
reduces only the shared symbols omitted from the output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, cast

from ..core.layout import Layout, Node, Tree

TokenKind = Literal[
    "left_paren",
    "right_paren",
    "arrow",
    "comma",
    "one",
    "symbol",
]
# Typing aliases do not support statement docstrings.
cast(
    Any, TokenKind
).__doc__ = "String literal token kinds emitted by the StrideWeave layout lexer."


@dataclass(frozen=True, slots=True)
class Token:
    """Lexed token from a StrideWeave layout command string."""

    kind: TokenKind
    value: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class LayoutReference:
    """Parsed layout-reference tree plus symbol-to-source-id bindings."""

    tree: Tree
    symbol_ids: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class RearrangeSpec:
    """Compiled rearrange command expressed as selection and output trees."""

    selection: Tree
    output: Tree
    symbol_ids: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ReduceSpec:
    """Compiled reduce command lowered through rearrange then sum-reduce."""

    selection: Tree
    output: Tree
    reduced: Tree
    rearrange_output: Tree
    symbol_ids: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class EinsumSpec:
    """Compiled two-input contraction with fast and general lowering metadata."""

    lhs_selection: Tree
    rhs_selection: Tree
    lhs_rearrange_output: Tree
    rhs_rearrange_output: Tree
    matmul_output_selection: Tree
    output: Tree
    lhs_symbol_ids: tuple[tuple[str, int], ...]
    rhs_symbol_ids: tuple[tuple[str, int], ...]
    common_symbols: tuple[str, ...]
    union_symbols: tuple[str, ...]
    contraction_symbols: tuple[str, ...]
    batch_symbols: tuple[str, ...]
    lhs_union_output: Tree
    rhs_union_output: Tree
    union_selection: Tree
    general_output: Tree


_einops = import_module("strideweave._einops")


def lex(command: str) -> list[Token]:
    """Tokenize a StrideWeave layout command string.

    The lexer scans ASCII command text, skips whitespace, and emits tokens with
    zero-based source offsets for parser construction.

    Syntax:
        Recognized tokens are ``(``, ``)``, ``->``, ``,``; the singleton
        literal ``1``; and dimension symbols matching ``[A-Za-z][A-Za-z0-9_]*``.
        Non-ASCII characters, malformed arrows, and numbers other than the
        singleton ``1`` are invalid.

    Args:
        command: StrideWeave layout command string to tokenize.

    Returns:
        A list of tokens in source order.

    Examples:
        >>> from sw.einops import lex
        >>> [token.kind for token in lex("a (b c) -> c a")]
        ['symbol', 'left_paren', 'symbol', 'symbol', 'right_paren', 'arrow', 'symbol', 'symbol']
    """

    if not isinstance(command, str):
        raise TypeError("command must be a str")
    return cast(list[Token], _einops.lex(command, Token))


def parse_layout_ref(tokens: Sequence[Token]) -> LayoutReference:
    """Parse a layout-reference token sequence.

    Symbols and singleton markers are converted into a selection tree whose
    leaves correspond to extracted source-layout positions.

    Syntax:
        A layout reference is a non-empty sequence of symbols, singleton ``1``
        markers, and non-empty parenthesized layout references. Commas and
        arrows are not valid inside an individual layout reference.

    Semantics:
        Leaves are assigned source ids in infix parse order. Named symbols are
        recorded in ``symbol_ids``; literal ``1`` leaves consume ids but remain
        anonymous.

    Args:
        tokens: Tokens making up one layout reference, without commas or arrows.

    Returns:
        Parsed layout reference containing the tree and named symbol ids.

    Examples:
        >>> from sw.einops import lex, parse_layout_ref
        >>> ref = parse_layout_ref(lex("a (b c)"))
        >>> ref.symbol_ids
        (('a', 0), ('b', 1), ('c', 2))
    """

    parser = _SelectionParser(tokens)
    return parser.parse()


def parse_rearrange(command: str) -> RearrangeSpec:
    """Compile a rearrange command.

    The compiled spec maps the input layout reference to an output tree suitable
    for the lower-level Tree-based rearrange operation.

    Syntax:
        Rearrange descriptions have the form ``"input -> output"``. The left
        side names each selected source layout leaf once. The right side may
        reorder those names, group them with parentheses, omit dimensions whose
        logical size is ``1``, or insert singleton dimensions with literal ``1``.

    Semantics:
        The left side becomes the selection tree passed to ``Layout.rearrange``.
        The right side becomes the output tree. The operation changes only the
        tensor layout and offset relationship; it does not copy tensor values.

    Args:
        command: Rearrange command containing exactly one ``->`` separator.

    Returns:
        Cached rearrange spec for the command string.

    Examples:
        >>> from strideweave import Node, Tree
        >>> from sw.einops import parse_rearrange
        >>> spec = parse_rearrange("a b -> b a")
        >>> spec.output == Tree(Node.id(1), Node.id(0))
        True
    """

    if not isinstance(command, str):
        raise TypeError("command must be a str")
    return cast(
        RearrangeSpec,
        _einops._cached_rearrange_spec(command, _parse_rearrange_uncached),
    )


def parse_reduce(command: str) -> ReduceSpec:
    """Compile a sum-reduce command.

    Omitted left-hand-side dimensions are treated as reduced dimensions and are
    lowered into the second mode consumed by the tensor reduce operation.

    Syntax:
        Reduce descriptions have the form ``"input -> kept"``. The left side
        names source layout leaves. The right side names the dimensions to keep,
        may group kept dimensions with parentheses, and may include literal
        ``1`` singleton dimensions. At least one left-side dimension must be
        omitted.

    Semantics:
        The omitted dimensions are reduced by summation. The command is lowered
        by rearranging the input into a two-mode layout ``(kept, omitted)`` and
        then applying the tensor reduce primitive, which assumes a two-mode
        tensor and reduces its second top-level mode.

    Args:
        command: Reduce command containing exactly one ``->`` separator.

    Returns:
        Cached reduce spec with selection, output, and intermediate layout trees.

    Examples:
        >>> from sw.einops import parse_reduce
        >>> spec = parse_reduce("a b -> a")
        >>> spec.symbol_ids
        (('a', 0), ('b', 1))
    """

    if not isinstance(command, str):
        raise TypeError("command must be a str")
    return cast(
        ReduceSpec,
        _einops._cached_reduce_spec(command, _parse_reduce_uncached),
    )


def parse_einsum(command: str) -> EinsumSpec:
    """Compile a two-input contraction command.

    Shared symbols omitted from the output become contraction dimensions.
    Shared symbols retained in the output are batch dimensions, while
    one-sided symbols are free dimensions and must be retained.

    Syntax:
        Contraction descriptions have the form ``"lhs, rhs -> output"``.
        Symbols on both inputs may be omitted for contraction or retained as
        batch dimensions. Every one-sided input symbol must appear exactly once
        in the output. The output may group dimensions with parentheses and may
        insert literal ``1`` singleton dimensions.

    Semantics:
        With no batch symbols, each input is rearranged into a two-mode layout
        ``(outer, shared_inner)`` and matmul remains the fast path. With batch
        symbols, both inputs are aligned over the union symbol space,
        elementwise-multiplied, and reduced over shared symbols omitted from the
        output. Shared dimensions must have equal logical sizes.

    Args:
        command: Contraction command in ``lhs, rhs -> output`` form.

    Returns:
        Cached contraction spec describing both fast and general lowering.

    Examples:
        >>> from sw.einops import parse_einsum
        >>> spec = parse_einsum("a b, c b -> a c")
        >>> spec.contraction_symbols
        ('b',)
    """

    if not isinstance(command, str):
        raise TypeError("command must be a str")
    return cast(
        EinsumSpec,
        _einops._cached_einsum_spec(command, _parse_einsum_uncached),
    )


def rearrange(tensor: Any, description: str) -> Any:
    """Rearrange a tensor using a StrideWeave layout description.

    The description is parsed, cached, and executed through the existing
    autograd-aware Tree rearrange operation.

    Syntax:
        ``description`` must be ``"input -> output"``. Input symbols name the
        tensor layout leaves in infix order. Output symbols select those leaves,
        parentheses create hierarchical output modes, and literal ``1`` inserts
        singleton dimensions.

    Semantics:
        The operation returns a view over the same backing carrier with the layout
        transformed according to the parsed output tree.

    Mode assumptions:
        The input reference must describe the tensor's layout structure. Modes
        may be hierarchical; each named leaf corresponds to one extracted
        subtree selected by the left-hand reference. Omitted source dimensions
        are valid only when their logical size is ``1``.

    Args:
        tensor: Tensor whose layout should be rearranged.
        description: Rearrange command such as ``"a b -> b a"``.

    Returns:
        Tensor view with the rearranged layout.

    Examples:
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> from sw.einops import rearrange
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> rearrange(x, "a b -> b a")[2, 1]
        6
    """

    if not isinstance(description, str):
        raise TypeError("description must be a str")
    spec = parse_rearrange(description)

    from ..functional.api import _rearrange_tree

    return _rearrange_tree(tensor, spec.output, spec.selection)


def reduce(tensor: Any, description: str) -> Any:
    """Sum-reduce a tensor using a StrideWeave layout description.

    Dimensions omitted from the output reference are grouped into the reduction
    mode and summed by the tensor reduce operation.

    Syntax:
        ``description`` must be ``"input -> kept"``. Symbols omitted from
        ``kept`` are summed away. Parentheses preserve hierarchical kept modes,
        and literal ``1`` inserts singleton output dimensions.

    Semantics:
        The operation sums every logical element in each omitted-dimension
        fiber and returns a tensor whose layout matches the kept reference.

    Mode assumptions:
        The described tensor may have any compatible hierarchical input layout,
        but the command is lowered into a two-mode intermediate. The first mode
        contains kept dimensions, and the second mode contains omitted
        dimensions. The underlying tensor reduce primitive assumes that
        two-mode intermediate and reduces the second top-level mode.

    Args:
        tensor: Tensor to reduce.
        description: Reduce command such as ``"a b -> a"``.

    Returns:
        Tensor containing the kept dimensions after summing omitted dimensions.

    Examples:
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> from sw.einops import reduce
        >>> x = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> reduce(x, "a b -> a")[1]
        12
    """

    if not isinstance(description, str):
        raise TypeError("description must be a str")
    spec = parse_reduce(description)

    from ..functional.api import _rearrange_tree, _reduce_second_mode

    intermediate = _rearrange_tree(tensor, spec.rearrange_output, spec.selection)
    return _reduce_second_mode(intermediate)


def einsum(lhs: Any, rhs: Any, description: str) -> Any:
    """Contract two tensors using a StrideWeave contraction description.

    Shared input symbols omitted from the output are contracted. Shared symbols
    retained in the output are batch dimensions, and one-sided output symbols
    are free dimensions broadcast into the other operand.

    Syntax:
        ``description`` must be ``"lhs, rhs -> output"``. Every one-sided input
        symbol must appear exactly once in ``output``. A shared symbol is
        contracted when omitted and retained as a batch dimension when present.
        Parentheses preserve output grouping, and literal ``1`` inserts a
        singleton output dimension.

    Semantics:
        Symbols are classified uniformly: shared-and-omitted means contraction,
        shared-and-retained means batch, and one-sided-and-retained means free.
        Batch/free cases align both tensors over the union index space,
        multiply elementwise, and sum only contraction dimensions.

    Mode assumptions:
        Each input reference must describe the corresponding hierarchical
        tensor layout, and same-named symbols must have equal logical sizes.
        The flat no-batch contraction keeps the two-mode matmul fast path.
        Batched contractions preserve symbol positions without flattening or
        implicit rank alignment and use differentiable stride-zero views for
        one-sided free dimensions.

    Args:
        lhs: Left input tensor.
        rhs: Right input tensor.
        description: Contraction command in ``lhs, rhs -> output`` form.

    Returns:
        Tensor with the requested output layout and contracted values.

    Examples:
        >>> from strideweave import Generic, Layout, Shape, Stride, Tensor
        >>> from sw.einops import einsum
        >>> lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> rhs = Tensor(Generic([1, 1, 1, 2, 2, 2]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
        >>> einsum(lhs, rhs, "a b, c b -> a c")[1, 1]
        22
    """

    if not isinstance(description, str):
        raise TypeError("description must be a str")
    spec = parse_einsum(description)
    _validate_einsum_shared_symbol_sizes(lhs, rhs, spec)

    from ..functional.api import (
        _matmul_2mode,
        _rearrange_tree,
        _reduce_second_mode,
        elementwise_mul,
    )

    if not spec.batch_symbols:
        lhs_intermediate = _rearrange_tree(
            lhs, spec.lhs_rearrange_output, spec.lhs_selection
        )
        rhs_intermediate = _rearrange_tree(
            rhs, spec.rhs_rearrange_output, spec.rhs_selection
        )
        result = _matmul_2mode(lhs_intermediate, rhs_intermediate)
        return _rearrange_tree(result, spec.output, spec.matmul_output_selection)

    lhs_union = _rearrange_tree(lhs, spec.lhs_union_output, spec.lhs_selection)
    rhs_union = _rearrange_tree(rhs, spec.rhs_union_output, spec.rhs_selection)
    product = elementwise_mul(lhs_union, rhs_union)
    if not spec.contraction_symbols:
        return _rearrange_tree(product, spec.general_output, spec.union_selection)

    union_positions = {symbol: index for index, symbol in enumerate(spec.union_symbols)}
    contracted = Tree(
        *(Node.id(union_positions[symbol]) for symbol in spec.contraction_symbols)
    )
    reduction_layout = Tree(spec.general_output, contracted)
    intermediate = _rearrange_tree(product, reduction_layout, spec.union_selection)
    return _reduce_second_mode(intermediate)


def _parse_rearrange_uncached(command: str) -> RearrangeSpec:
    tokens, arrow_position = _split_command(command, "Rearrange")
    layout_ref = parse_layout_ref(tokens[:arrow_position])
    output = _OutputParser(
        tokens[arrow_position + 1 :], dict(layout_ref.symbol_ids)
    ).parse()
    return RearrangeSpec(layout_ref.tree, output, layout_ref.symbol_ids)


def _parse_reduce_uncached(command: str) -> ReduceSpec:
    tokens, arrow_position = _split_command(command, "Reduce")
    layout_ref = parse_layout_ref(tokens[:arrow_position])
    output = _OutputParser(
        tokens[arrow_position + 1 :], dict(layout_ref.symbol_ids)
    ).parse()
    used_ids = _source_ids(output)
    reduced_ids = [
        source_id
        for source_id in range(layout_ref.tree.size)
        if source_id not in used_ids
    ]
    if len(reduced_ids) == 0:
        raise ValueError("Reduce command must omit at least one dimension")

    reduced = Tree(*(Node.id(source_id) for source_id in reduced_ids))
    rearrange_output = Tree(output, reduced)
    return ReduceSpec(
        layout_ref.tree, output, reduced, rearrange_output, layout_ref.symbol_ids
    )


def _parse_einsum_uncached(command: str) -> EinsumSpec:
    tokens, comma_position, arrow_position = _split_einsum_command(command)
    lhs_ref = parse_layout_ref(tokens[:comma_position])
    rhs_ref = parse_layout_ref(tokens[comma_position + 1 : arrow_position])

    rhs_symbol_ids = dict(rhs_ref.symbol_ids)
    common_symbols = tuple(
        symbol for symbol, _source_id in lhs_ref.symbol_ids if symbol in rhs_symbol_ids
    )
    if len(common_symbols) == 0:
        raise ValueError(
            "Contraction command must include at least one shared dimension"
        )

    lhs_symbols = tuple(symbol for symbol, _source_id in lhs_ref.symbol_ids)
    rhs_symbols = tuple(symbol for symbol, _source_id in rhs_ref.symbol_ids)
    lhs_symbol_set = set(lhs_symbols)
    union_symbols = (*lhs_symbols, *(s for s in rhs_symbols if s not in lhs_symbol_set))
    union_symbol_ids = {symbol: index for index, symbol in enumerate(union_symbols)}
    general_parser = _OutputParser(tokens[arrow_position + 1 :], union_symbol_ids)
    general_output = general_parser.parse()
    common_set = set(common_symbols)
    required_free_symbols = set(union_symbols) - common_set
    if not required_free_symbols <= general_parser.used_symbols:
        missing = sorted(required_free_symbols - general_parser.used_symbols)
        raise ValueError(
            "Einsum output must include every non-shared input symbol: "
            + ", ".join(missing)
        )

    contraction_symbols = tuple(
        symbol for symbol in common_symbols if symbol not in general_parser.used_symbols
    )
    batch_symbols = tuple(
        symbol for symbol in common_symbols if symbol in general_parser.used_symbols
    )
    lhs_ids = dict(lhs_ref.symbol_ids)
    rhs_ids = dict(rhs_ref.symbol_ids)
    lhs_union_output = Tree(
        *(
            Node.id(lhs_ids[symbol]) if symbol in lhs_ids else Node.Leaf
            for symbol in union_symbols
        )
    )
    rhs_union_output = Tree(
        *(
            Node.id(rhs_ids[symbol]) if symbol in rhs_ids else Node.Leaf
            for symbol in union_symbols
        )
    )
    union_selection = Tree(*(Node.Leaf for _symbol in union_symbols))

    if not batch_symbols:
        output_symbol_ids, matmul_output_selection = _einsum_output_symbol_ids(
            lhs_ref.symbol_ids, rhs_ref.symbol_ids, common_set
        )
        output_parser = _OutputParser(tokens[arrow_position + 1 :], output_symbol_ids)
        output = output_parser.parse()
    else:
        output = general_output
        matmul_output_selection = union_selection

    return EinsumSpec(
        lhs_ref.tree,
        rhs_ref.tree,
        (
            _einsum_input_rearrange_output(lhs_ref.symbol_ids, common_symbols)
            if not batch_symbols
            else lhs_union_output
        ),
        (
            _einsum_input_rearrange_output(rhs_ref.symbol_ids, common_symbols)
            if not batch_symbols
            else rhs_union_output
        ),
        matmul_output_selection,
        output,
        lhs_ref.symbol_ids,
        rhs_ref.symbol_ids,
        common_symbols,
        tuple(union_symbols),
        contraction_symbols,
        batch_symbols,
        lhs_union_output,
        rhs_union_output,
        union_selection,
        general_output,
    )


def _split_command(command: str, command_name: str) -> tuple[list[Token], int]:
    tokens = lex(command)
    arrow_positions = [
        index for index, token in enumerate(tokens) if token.kind == "arrow"
    ]
    if len(arrow_positions) == 0:
        raise ValueError(f"{command_name} command must contain one '->' arrow")
    if len(arrow_positions) > 1:
        raise ValueError(f"{command_name} command must contain only one '->' arrow")
    return tokens, arrow_positions[0]


def _split_einsum_command(command: str) -> tuple[list[Token], int, int]:
    tokens, arrow_position = _split_command(command, "Einsum")
    comma_positions = [
        index
        for index, token in enumerate(tokens[:arrow_position])
        if token.kind == "comma"
    ]
    if len(comma_positions) == 0:
        raise ValueError("Contraction command must contain one comma before '->'")
    if len(comma_positions) > 1:
        raise ValueError("Contraction command must contain only one comma before '->'")
    return tokens, comma_positions[0], arrow_position


def _einsum_input_rearrange_output(
    symbol_ids: tuple[tuple[str, int], ...], common_symbols: tuple[str, ...]
) -> Tree:
    common_set = set(common_symbols)
    outer_ids = [
        source_id for symbol, source_id in symbol_ids if symbol not in common_set
    ]
    inner_ids = [dict(symbol_ids)[symbol] for symbol in common_symbols]
    outer = _tree_from_source_ids(outer_ids, singleton_if_empty=True)
    inner = _tree_from_source_ids(inner_ids, singleton_if_empty=False)
    return Tree(outer, inner)


def _einsum_output_symbol_ids(
    lhs_symbol_ids: tuple[tuple[str, int], ...],
    rhs_symbol_ids: tuple[tuple[str, int], ...],
    common_set: set[str],
) -> tuple[dict[str, int], Tree]:
    output_symbol_ids: dict[str, int] = {}
    next_id = 0

    lhs_outer_symbols = [
        symbol for symbol, _id in lhs_symbol_ids if symbol not in common_set
    ]
    lhs_selection = _selection_tree(len(lhs_outer_symbols), singleton_if_empty=True)
    for symbol in lhs_outer_symbols:
        output_symbol_ids[symbol] = next_id
        next_id += 1
    if len(lhs_outer_symbols) == 0:
        next_id += 1

    rhs_outer_symbols = [
        symbol for symbol, _id in rhs_symbol_ids if symbol not in common_set
    ]
    rhs_selection = _selection_tree(len(rhs_outer_symbols), singleton_if_empty=True)
    for symbol in rhs_outer_symbols:
        output_symbol_ids[symbol] = next_id
        next_id += 1
    if len(rhs_outer_symbols) == 0:
        next_id += 1

    return output_symbol_ids, Tree(lhs_selection, rhs_selection)


def _tree_from_source_ids(source_ids: list[int], *, singleton_if_empty: bool) -> Tree:
    if len(source_ids) == 0:
        if not singleton_if_empty:
            raise ValueError("Einsum inner dimension must not be empty")
        return Tree(Node.Leaf)
    return Tree(*(Node.id(source_id) for source_id in source_ids))


def _selection_tree(size: int, *, singleton_if_empty: bool) -> Tree:
    if size == 0:
        if not singleton_if_empty:
            raise ValueError("Einsum selection must not be empty")
        return Tree(Node.Leaf)
    return Tree(*(Node.Leaf for _ in range(size)))


def _validate_einsum_shared_symbol_sizes(lhs: Any, rhs: Any, spec: EinsumSpec) -> None:
    lhs_layouts = Layout.extract_profile(lhs.layout, spec.lhs_selection)
    rhs_layouts = Layout.extract_profile(rhs.layout, spec.rhs_selection)
    lhs_symbol_ids = dict(spec.lhs_symbol_ids)
    rhs_symbol_ids = dict(spec.rhs_symbol_ids)

    for symbol in spec.common_symbols:
        lhs_size = lhs_layouts[lhs_symbol_ids[symbol]].shape.logical_size
        rhs_size = rhs_layouts[rhs_symbol_ids[symbol]].shape.logical_size
        if lhs_size != rhs_size:
            raise ValueError(
                f"Einsum shared dimension '{symbol}' has mismatched logical size"
            )


def _source_ids(tree: Tree) -> set[int]:
    source_ids: set[int] = set()
    for marker in tree:
        if isinstance(marker, Tree):
            source_ids.update(_source_ids(marker))
        elif marker == Node.Leaf:
            continue
        elif isinstance(getattr(marker, "id", None), int):
            source_ids.add(marker.id)
        else:
            raise ValueError("output tree contains an invalid marker")
    return source_ids


def _token_error(token: Token, message: str) -> ValueError:
    return ValueError(f"{message} at offset {token.start}")


def _as_tokens(tokens: Sequence[Token]) -> tuple[Token, ...]:
    normalized = tuple(tokens)
    for token in normalized:
        if not isinstance(token, Token):
            raise TypeError("layout reference tokens must be Token objects")
    return normalized


class _BaseParser(ABC):
    def __init__(self, tokens: Sequence[Token]):
        self.tokens = _as_tokens(tokens)
        self.position = 0

    def parse_tree(self) -> Tree:
        tree = self._parse_level(stop_at_right_paren=False)
        if self.position != len(self.tokens):
            token = self.tokens[self.position]
            raise _token_error(token, "Unexpected token")
        return tree

    def _parse_level(self, *, stop_at_right_paren: bool) -> Tree:
        items: list[Any] = []
        while self.position < len(self.tokens):
            token = self.tokens[self.position]
            if token.kind == "right_paren":
                if stop_at_right_paren:
                    break
                raise _token_error(token, "Unmatched right parenthesis")
            if token.kind == "left_paren":
                self.position += 1
                child = self._parse_level(stop_at_right_paren=True)
                if self.position >= len(self.tokens):
                    raise _token_error(token, "Unclosed left parenthesis")
                right_paren = self.tokens[self.position]
                if right_paren.kind != "right_paren":
                    raise _token_error(right_paren, "Expected right parenthesis")
                self.position += 1
                items.append(child)
                continue
            if token.kind in {"symbol", "one"}:
                items.append(self._leaf_for_token(token))
                self.position += 1
                continue
            if token.kind == "comma":
                raise _token_error(token, "Commas are not valid in layout references")
            if token.kind == "arrow":
                raise _token_error(token, "Arrows are not valid in layout references")
            raise _token_error(token, "Unexpected token")

        if len(items) == 0:
            if stop_at_right_paren and self.position < len(self.tokens):
                raise _token_error(
                    self.tokens[self.position], "Empty parenthesized group"
                )
            raise ValueError("Layout reference must not be empty")
        return Tree(*items)

    @abstractmethod
    def _leaf_for_token(self, token: Token) -> Any:
        raise NotImplementedError


class _SelectionParser(_BaseParser):
    def __init__(self, tokens: Sequence[Token]):
        super().__init__(tokens)
        self.symbol_ids: dict[str, int] = {}
        self.next_id = 0

    def parse(self) -> LayoutReference:
        tree = self.parse_tree()
        return LayoutReference(tree, tuple(self.symbol_ids.items()))

    def _leaf_for_token(self, token: Token) -> Node:
        if token.kind == "symbol":
            if token.value in self.symbol_ids:
                raise _token_error(token, f"Duplicate dimension symbol '{token.value}'")
            self.symbol_ids[token.value] = self.next_id
        self.next_id += 1
        return Node.Leaf


class _OutputParser(_BaseParser):
    def __init__(self, tokens: Sequence[Token], symbol_ids: dict[str, int]):
        super().__init__(tokens)
        self.symbol_ids = symbol_ids
        self.used_symbols: set[str] = set()

    def parse(self) -> Tree:
        return self.parse_tree()

    def _leaf_for_token(self, token: Token) -> Any:
        if token.kind == "one":
            return Node.Leaf
        if token.value not in self.symbol_ids:
            raise _token_error(token, f"Unknown dimension symbol '{token.value}'")
        if token.value in self.used_symbols:
            raise _token_error(
                token, f"Duplicate output dimension symbol '{token.value}'"
            )
        self.used_symbols.add(token.value)
        return Node.id(self.symbol_ids[token.value])


__all__ = [
    "EinsumSpec",
    "LayoutReference",
    "RearrangeSpec",
    "ReduceSpec",
    "Token",
    "TokenKind",
    "einsum",
    "lex",
    "parse_einsum",
    "parse_layout_ref",
    "parse_rearrange",
    "parse_reduce",
    "rearrange",
    "reduce",
]
