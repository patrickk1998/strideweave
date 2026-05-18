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

def lex(command: str) -> list[Token]: ...
def parse_layout_ref(tokens: Sequence[Token]) -> LayoutReference: ...
def parse_rearrange(command: str) -> RearrangeSpec: ...
def rearrange(tensor: Tensor, description: str) -> Tensor: ...
