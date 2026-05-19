from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, cast

from .layout import Layout, Node, Tree

TokenKind = Literal[
    "left_paren",
    "right_paren",
    "arrow",
    "comma",
    "one",
    "symbol",
]


@dataclass(frozen=True, slots=True)
class Token:
    kind: TokenKind
    value: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class LayoutReference:
    tree: Tree
    symbol_ids: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class RearrangeSpec:
    selection: Tree
    output: Tree
    symbol_ids: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ReduceSpec:
    selection: Tree
    output: Tree
    reduced: Tree
    rearrange_output: Tree
    symbol_ids: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class EinsumSpec:
    lhs_selection: Tree
    rhs_selection: Tree
    lhs_rearrange_output: Tree
    rhs_rearrange_output: Tree
    matmul_output_selection: Tree
    output: Tree
    lhs_symbol_ids: tuple[tuple[str, int], ...]
    rhs_symbol_ids: tuple[tuple[str, int], ...]
    common_symbols: tuple[str, ...]


_einops = import_module("neotorch._einops")


def lex(command: str) -> list[Token]:
    if not isinstance(command, str):
        raise TypeError("command must be a str")
    return cast(list[Token], _einops.lex(command, Token))


def parse_layout_ref(tokens: Sequence[Token]) -> LayoutReference:
    parser = _SelectionParser(tokens)
    return parser.parse()


def parse_rearrange(command: str) -> RearrangeSpec:
    if not isinstance(command, str):
        raise TypeError("command must be a str")
    return cast(
        RearrangeSpec,
        _einops._cached_rearrange_spec(command, _parse_rearrange_uncached),
    )


def parse_reduce(command: str) -> ReduceSpec:
    if not isinstance(command, str):
        raise TypeError("command must be a str")
    return cast(
        ReduceSpec,
        _einops._cached_reduce_spec(command, _parse_reduce_uncached),
    )


def parse_einsum(command: str) -> EinsumSpec:
    if not isinstance(command, str):
        raise TypeError("command must be a str")
    return cast(
        EinsumSpec,
        _einops._cached_einsum_spec(command, _parse_einsum_uncached),
    )


def rearrange(tensor: Any, description: str) -> Any:
    if not isinstance(description, str):
        raise TypeError("description must be a str")
    spec = parse_rearrange(description)

    from .operation import rearrange as tree_rearrange

    return tree_rearrange(tensor, spec.output, spec.selection)


def reduce(tensor: Any, description: str) -> Any:
    if not isinstance(description, str):
        raise TypeError("description must be a str")
    spec = parse_reduce(description)

    from .operation import rearrange as tree_rearrange
    from .operation import reduce as tensor_reduce

    intermediate = tree_rearrange(tensor, spec.rearrange_output, spec.selection)
    return tensor_reduce(intermediate)


def einsum(lhs: Any, rhs: Any, description: str) -> Any:
    if not isinstance(description, str):
        raise TypeError("description must be a str")
    spec = parse_einsum(description)
    _validate_einsum_shared_symbol_sizes(lhs, rhs, spec)

    from .operation import matmul
    from .operation import rearrange as tree_rearrange

    lhs_intermediate = tree_rearrange(
        lhs, spec.lhs_rearrange_output, spec.lhs_selection
    )
    rhs_intermediate = tree_rearrange(
        rhs, spec.rhs_rearrange_output, spec.rhs_selection
    )
    result = matmul(lhs_intermediate, rhs_intermediate)
    return tree_rearrange(result, spec.output, spec.matmul_output_selection)


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
        raise ValueError("Einsum command must include at least one shared dimension")

    common_set = set(common_symbols)
    output_symbol_ids, matmul_output_selection = _einsum_output_symbol_ids(
        lhs_ref.symbol_ids, rhs_ref.symbol_ids, common_set
    )
    output_parser = _OutputParser(tokens[arrow_position + 1 :], output_symbol_ids)
    output = output_parser.parse()
    required_output_symbols = set(output_symbol_ids)
    if output_parser.used_symbols != required_output_symbols:
        missing = sorted(required_output_symbols - output_parser.used_symbols)
        raise ValueError(
            "Einsum output must include every non-shared input symbol: "
            + ", ".join(missing)
        )

    return EinsumSpec(
        lhs_ref.tree,
        rhs_ref.tree,
        _einsum_input_rearrange_output(lhs_ref.symbol_ids, common_symbols),
        _einsum_input_rearrange_output(rhs_ref.symbol_ids, common_symbols),
        matmul_output_selection,
        output,
        lhs_ref.symbol_ids,
        rhs_ref.symbol_ids,
        common_symbols,
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
        raise ValueError("Einsum command must contain one comma before '->'")
    if len(comma_positions) > 1:
        raise ValueError("Einsum command must contain only one comma before '->'")
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
    "lex",
    "einsum",
    "parse_layout_ref",
    "parse_einsum",
    "parse_rearrange",
    "parse_reduce",
    "rearrange",
    "reduce",
]
