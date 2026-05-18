import random
import string
from typing import Any

import pytest
from neotorch.einops import Token, TokenKind, lex

VALID_FUZZ_SEED = 1103515245
INVALID_FUZZ_SEED = 8675309
ASCII_WHITESPACE = ["", " ", "\t", "\n", "\r", "\f", "\v", " \t"]
PUNCTUATION_TOKENS: list[tuple[TokenKind, str]] = [
    ("left_paren", "("),
    ("right_paren", ")"),
    ("arrow", "->"),
    ("comma", ","),
    ("one", "1"),
]


def token_values(tokens: list[Token]) -> list[tuple[str, str, int, int]]:
    return [(token.kind, token.value, token.start, token.end) for token in tokens]


def reference_lex(command: str) -> list[tuple[str, str, int, int]]:
    tokens: list[tuple[str, str, int, int]] = []
    position = 0
    while position < len(command):
        value = command[position]
        if not value.isascii():
            raise ValueError("non-ASCII character")
        if value in " \t\n\r\f\v":
            position += 1
            continue

        start = position
        if value == "(":
            tokens.append(("left_paren", value, start, start + 1))
            position += 1
        elif value == ")":
            tokens.append(("right_paren", value, start, start + 1))
            position += 1
        elif value == ",":
            tokens.append(("comma", value, start, start + 1))
            position += 1
        elif value == "-":
            if command[position : position + 2] != "->":
                raise ValueError("malformed arrow")
            tokens.append(("arrow", "->", start, start + 2))
            position += 2
        elif value == ">":
            raise ValueError("malformed arrow")
        elif value == "1":
            if position + 1 < len(command) and (
                command[position + 1].isalnum() or command[position + 1] == "_"
            ):
                raise ValueError("invalid singleton")
            tokens.append(("one", value, start, start + 1))
            position += 1
        elif value.isdigit():
            raise ValueError("invalid symbol")
        elif value.isalpha():
            position += 1
            while position < len(command) and (
                command[position].isalnum() or command[position] == "_"
            ):
                position += 1
            tokens.append(("symbol", command[start:position], start, position))
        else:
            raise ValueError("unexpected character")
    return tokens


def fuzz_symbol(random_source: random.Random) -> str:
    head = random_source.choice(string.ascii_letters)
    tail_alphabet = string.ascii_letters + string.digits + "_"
    tail = "".join(
        random_source.choice(tail_alphabet) for _ in range(random_source.randint(0, 8))
    )
    return head + tail


def fuzz_valid_command(random_source: random.Random) -> str:
    parts = [random_source.choice(ASCII_WHITESPACE)]
    previous_token = ""
    for _ in range(random_source.randint(0, 40)):
        if random_source.random() < 0.45:
            token = fuzz_symbol(random_source)
        else:
            token = random_source.choice(PUNCTUATION_TOKENS)[1]

        separator = random_source.choice(ASCII_WHITESPACE)
        if (
            previous_token == "1"
            and separator == ""
            and (token[0].isalnum() or token[0] == "_")
        ):
            separator = " "
        parts.append(separator)
        parts.append(token)
        previous_token = token

    parts.append(random_source.choice(ASCII_WHITESPACE))
    return "".join(parts)


def test_einops_lexes_nested_reduce_command():
    tokens = lex("a (b c) -> a c")

    assert token_values(tokens) == [
        ("symbol", "a", 0, 1),
        ("left_paren", "(", 2, 3),
        ("symbol", "b", 3, 4),
        ("symbol", "c", 5, 6),
        ("right_paren", ")", 6, 7),
        ("arrow", "->", 8, 10),
        ("symbol", "a", 11, 12),
        ("symbol", "c", 13, 14),
    ]


def test_einops_lexes_comma_and_singleton():
    tokens = lex("a, 1 -> a")

    assert token_values(tokens) == [
        ("symbol", "a", 0, 1),
        ("comma", ",", 1, 2),
        ("one", "1", 3, 4),
        ("arrow", "->", 5, 7),
        ("symbol", "a", 8, 9),
    ]


def test_einops_lex_skips_ascii_whitespace():
    tokens = lex("\ta\n(\rb\f)\v-> c")

    assert token_values(tokens) == [
        ("symbol", "a", 1, 2),
        ("left_paren", "(", 3, 4),
        ("symbol", "b", 5, 6),
        ("right_paren", ")", 7, 8),
        ("arrow", "->", 9, 11),
        ("symbol", "c", 12, 13),
    ]


def test_einops_lexes_symbol_forms():
    tokens = lex("a a1 batch_size")

    assert token_values(tokens) == [
        ("symbol", "a", 0, 1),
        ("symbol", "a1", 2, 4),
        ("symbol", "batch_size", 5, 15),
    ]


@pytest.mark.parametrize(
    "command",
    [
        "2",
        "11",
        "1a",
        "1_",
        "-",
        ">",
        "+",
        "é",
    ],
)
def test_einops_lex_rejects_invalid_commands(command: str):
    with pytest.raises(ValueError):
        lex(command)


def test_einops_lex_rejects_non_string_input():
    command: Any = 1

    with pytest.raises(TypeError):
        lex(command)


def test_einops_token_is_frozen_and_slotted():
    token = lex("a")[0]

    assert not hasattr(token, "__dict__")
    with pytest.raises(AttributeError):
        token.kind = "symbol"  # type: ignore[misc]


def test_einops_lex_matches_reference_for_deterministic_valid_fuzz_cases():
    random_source = random.Random(VALID_FUZZ_SEED)

    for _ in range(500):
        command = fuzz_valid_command(random_source)

        assert token_values(lex(command)) == reference_lex(command)


def test_einops_lex_rejects_deterministic_invalid_fuzz_cases():
    random_source = random.Random(INVALID_FUZZ_SEED)
    invalid_fragments = [
        "2",
        "3",
        "01",
        "11",
        "1a",
        "1A",
        "1_",
        "-",
        "--",
        ">",
        "+",
        "[",
        "]",
        "é",
        "aé",
    ]

    for _ in range(200):
        command = " ".join(
            [
                fuzz_valid_command(random_source),
                random_source.choice(invalid_fragments),
                fuzz_valid_command(random_source),
            ]
        )

        with pytest.raises(ValueError):
            lex(command)
