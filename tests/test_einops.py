import random
import string
from importlib import import_module
from typing import Any, cast

import neotorch
import pytest
from neotorch import (
    Generic,
    Layout,
    Node,
    RearrangeOperation,
    Shape,
    Stride,
    Tensor,
    Tree,
)
from neotorch.einops import (
    Token,
    TokenKind,
    lex,
    parse_layout_ref,
    parse_rearrange,
    rearrange,
)

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
native_einops = cast(Any, import_module("neotorch._einops"))


def token_values(tokens: list[Token]) -> list[tuple[str, str, int, int]]:
    return [(token.kind, token.value, token.start, token.end) for token in tokens]


def tensor_values(tensor: Tensor) -> list[Any]:
    return [tensor[i] for i in range(tensor.size())]


def require_grad(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


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


def test_einops_parse_layout_ref_builds_tree_and_infix_symbol_ids():
    reference = parse_layout_ref(lex("a (b c)"))

    assert reference.tree == Tree(Node.Leaf, Tree(Node.Leaf, Node.Leaf))
    assert reference.symbol_ids == (("a", 0), ("b", 1), ("c", 2))


def test_einops_parse_rearrange_builds_selection_and_output_trees():
    spec = parse_rearrange("a (b c) -> a c")

    assert spec.selection == Tree(Node.Leaf, Tree(Node.Leaf, Node.Leaf))
    assert spec.output == Tree(Node.id(0), Node.id(2))
    assert spec.symbol_ids == (("a", 0), ("b", 1), ("c", 2))


def test_einops_parse_rearrange_preserves_nested_output_structure():
    spec = parse_rearrange("a b c -> a (b c)")

    assert spec.selection == Tree(Node.Leaf, Node.Leaf, Node.Leaf)
    assert spec.output == Tree(Node.id(0), Tree(Node.id(1), Node.id(2)))


def test_einops_parse_rearrange_tracks_anonymous_left_singletons():
    spec = parse_rearrange("1 a -> a")

    assert spec.selection == Tree(Node.Leaf, Node.Leaf)
    assert spec.output == Tree(Node.id(1))
    assert spec.symbol_ids == (("a", 1),)


def test_einops_parse_rearrange_inserts_output_singletons():
    spec = parse_rearrange("a -> a 1")

    assert spec.selection == Tree(Node.Leaf)
    assert spec.output == Tree(Node.id(0), Node.Leaf)


def test_einops_parse_rearrange_output_works_with_layout_rearrange():
    spec = parse_rearrange("a (b c) -> (b c) a")
    layout = Layout(Shape([2, [3, 4]]), Stride([1, [2, 6]]))

    result = Layout.rearrange(layout, spec.output, spec.selection)

    assert result == Layout(Shape([[3, 4], 2]), Stride([[2, 6], 1]))


def test_einops_parse_rearrange_left_singleton_compatibility_uses_layout_validation():
    spec = parse_rearrange("1 a -> a")

    assert Layout.rearrange(
        Layout(Shape([1, 3]), Stride([5, 1])),
        spec.output,
        spec.selection,
    ) == Layout(Shape(3), Stride(1))
    with pytest.raises(ValueError):
        Layout.rearrange(
            Layout(Shape([2, 3]), Stride([1, 2])),
            spec.output,
            spec.selection,
        )


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        lex("a ()"),
        lex("a )"),
        lex("a (b"),
        lex("a, b"),
        lex("a -> b"),
    ],
)
def test_einops_parse_layout_ref_rejects_invalid_syntax(tokens: list[Token]):
    with pytest.raises(ValueError):
        parse_layout_ref(tokens)


@pytest.mark.parametrize(
    "command",
    [
        "a a -> a",
        "a b -> a a",
        "a -> b",
        "a ->",
        "-> a",
        "a b",
        "a -> b -> c",
        "a () -> a",
        "a (b -> a",
        "a b) -> a",
        "a, b -> a",
        "a -> a, b",
    ],
)
def test_einops_parse_rearrange_rejects_invalid_syntax(command: str):
    with pytest.raises(ValueError):
        parse_rearrange(command)


def test_einops_rearrange_string_api_returns_rearranged_tensor_view():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))

    result = rearrange(tensor, "a b -> b a")

    assert result.data is tensor.data
    assert result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert result[2, 1] == tensor[1, 2]
    assert isinstance(result.autograd_ctx, RearrangeOperation)


def test_top_level_rearrange_accepts_einops_string_descriptions():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))

    result = neotorch.rearrange(tensor, "a b -> b a")

    assert result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert result[2, 1] == tensor[1, 2]


def test_top_level_rearrange_preserves_existing_tree_api():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))

    result = neotorch.rearrange(tensor, Tree(Node.id(1), Node.id(0)))

    assert result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert result[2, 1] == tensor[1, 2]


def test_top_level_rearrange_rejects_string_description_with_explicit_selection():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))

    with pytest.raises(TypeError):
        neotorch.rearrange(
            tensor,
            cast(Any, "a b -> b a"),
            Tree(Node.Leaf, Node.Leaf),
        )


def test_einops_rearrange_string_api_backpropagates_through_existing_operation():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    result = rearrange(tensor, "a b -> b a")
    gradient = Tensor(Generic([10, 40, 20, 50, 30, 60]), 0, result.layout)

    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert result.grad is None
    assert tensor_grad.layout == tensor.layout
    assert tensor_values(tensor_grad) == [10, 40, 20, 50, 30, 60]
    assert type(tensor_grad.data) is type(tensor.data)


def test_einops_parse_rearrange_reuses_cached_spec_objects():
    first = parse_rearrange("cache_a cache_b -> cache_b cache_a")
    second = parse_rearrange("cache_a cache_b -> cache_b cache_a")

    assert first is second


def test_native_rearrange_spec_cache_does_not_cache_failed_compilations():
    calls = 0
    cached_sentinel = object()
    uncached_sentinel = object()

    def compiler(_command: str) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("compile failed")
        if calls == 2:
            return cached_sentinel
        return uncached_sentinel

    with pytest.raises(ValueError):
        native_einops._cached_rearrange_spec("failing-cache-key", compiler)

    # The cache is private native state, so call count is the observable proxy:
    # failure must miss, the retry must compile, and the final call must hit.
    assert (
        native_einops._cached_rearrange_spec("failing-cache-key", compiler)
        is cached_sentinel
    )
    assert calls == 2
    cached_result = native_einops._cached_rearrange_spec("failing-cache-key", compiler)
    assert cached_result is cached_sentinel
    assert cached_result is not uncached_sentinel
    assert calls == 2


def test_einops_parse_rearrange_invalid_descriptions_still_raise_after_retries():
    for _ in range(2):
        with pytest.raises(ValueError):
            parse_rearrange("invalid_a -> invalid_b")


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
