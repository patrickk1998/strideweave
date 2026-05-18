from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, cast

from .layout import Node, Tree

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
    "LayoutReference",
    "RearrangeSpec",
    "ReduceSpec",
    "Token",
    "TokenKind",
    "lex",
    "parse_layout_ref",
    "parse_rearrange",
    "parse_reduce",
    "rearrange",
    "reduce",
]
