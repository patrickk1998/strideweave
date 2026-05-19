from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .layout import Tree
from .tensor import Tensor

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

def lex(command: str) -> list[Token]: ...
def parse_layout_ref(tokens: Sequence[Token]) -> LayoutReference: ...
def parse_rearrange(command: str) -> RearrangeSpec: ...
def parse_reduce(command: str) -> ReduceSpec: ...
def parse_einsum(command: str) -> EinsumSpec: ...
def rearrange(tensor: Tensor, description: str) -> Tensor: ...
def reduce(tensor: Tensor, description: str) -> Tensor: ...
def einsum(lhs: Tensor, rhs: Tensor, description: str) -> Tensor: ...
