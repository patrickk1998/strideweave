from dataclasses import dataclass
from typing import Literal

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

def lex(command: str) -> list[Token]: ...
