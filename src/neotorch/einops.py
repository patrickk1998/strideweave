from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Literal, cast

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


_einops = import_module("neotorch._einops")


def lex(command: str) -> list[Token]:
    if not isinstance(command, str):
        raise TypeError("command must be a str")
    return cast(list[Token], _einops.lex(command, Token))


__all__ = ["Token", "TokenKind", "lex"]
