import random
import string
from importlib import import_module
from typing import Any, cast

import pytest

import strideweave as sw
from strideweave import (
    CPU,
    Generic,
    GenericMatmulOperation,
    GenericReduceSumOperation,
    Layout,
    Node,
    RearrangeOperation,
    Shape,
    Stride,
    Tensor,
    Tree,
)
from strideweave.einops import (
    Token,
    TokenKind,
    lex,
    parse_einsum,
    parse_layout_ref,
    parse_rearrange,
    parse_reduce,
    rearrange,
)
from strideweave.einops import (
    einsum as einops_einsum,
)
from strideweave.einops import (
    reduce as einops_reduce,
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
native_einops = cast(Any, import_module("strideweave._einops"))


def token_values(tokens: list[Token]) -> list[tuple[str, str, int, int]]:
    return [(token.kind, token.value, token.start, token.end) for token in tokens]


def tensor_values(tensor: Tensor) -> list[Any]:
    return [tensor[i] for i in range(tensor.size())]


def require_grad(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


def make_cpu_tensor(values: list[float], layout: Layout) -> Tensor:
    carrier = CPU(len(values))
    for index, value in enumerate(values):
        carrier[index] = value
    return Tensor(carrier, 0, layout)


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


def test_einops_parse_reduce_builds_two_mode_rearrange_spec():
    spec = parse_reduce("a (c d) b -> a c")

    assert spec.selection == Tree(Node.Leaf, Tree(Node.Leaf, Node.Leaf), Node.Leaf)
    assert spec.output == Tree(Node.id(0), Node.id(1))
    assert spec.reduced == Tree(Node.id(2), Node.id(3))
    assert spec.rearrange_output == Tree(
        Tree(Node.id(0), Node.id(1)), Tree(Node.id(2), Node.id(3))
    )
    assert spec.symbol_ids == (("a", 0), ("c", 1), ("d", 2), ("b", 3))


def test_einops_parse_reduce_preserves_nested_output_structure():
    spec = parse_reduce("a b c -> (a c)")

    assert spec.selection == Tree(Node.Leaf, Node.Leaf, Node.Leaf)
    assert spec.output == Tree(Tree(Node.id(0), Node.id(2)))
    assert spec.reduced == Tree(Node.id(1))
    assert spec.rearrange_output == Tree(spec.output, spec.reduced)


def test_einops_parse_reduce_inserts_output_singletons():
    spec = parse_reduce("a b -> a 1")

    assert spec.output == Tree(Node.id(0), Node.Leaf)
    assert spec.reduced == Tree(Node.id(1))
    assert spec.rearrange_output == Tree(spec.output, spec.reduced)


def test_einops_parse_einsum_builds_matmul_rearrange_spec():
    spec = parse_einsum("a b, c b -> a c")

    assert spec.lhs_selection == Tree(Node.Leaf, Node.Leaf)
    assert spec.rhs_selection == Tree(Node.Leaf, Node.Leaf)
    assert spec.lhs_rearrange_output == Tree(Tree(Node.id(0)), Tree(Node.id(1)))
    assert spec.rhs_rearrange_output == Tree(Tree(Node.id(0)), Tree(Node.id(1)))
    assert spec.matmul_output_selection == Tree(Tree(Node.Leaf), Tree(Node.Leaf))
    assert spec.output == Tree(Node.id(0), Node.id(1))
    assert spec.lhs_symbol_ids == (("a", 0), ("b", 1))
    assert spec.rhs_symbol_ids == (("c", 0), ("b", 1))
    assert spec.common_symbols == ("b",)


def test_einops_parse_einsum_reorders_rhs_inner_to_lhs_shared_order():
    spec = parse_einsum("a b c, d c b -> a d")

    assert spec.lhs_rearrange_output == Tree(
        Tree(Node.id(0)), Tree(Node.id(1), Node.id(2))
    )
    assert spec.rhs_rearrange_output == Tree(
        Tree(Node.id(0)), Tree(Node.id(2), Node.id(1))
    )
    assert spec.common_symbols == ("b", "c")


def test_einops_parse_einsum_preserves_nested_output_structure():
    spec = parse_einsum("a b, c b -> (c a)")

    assert spec.output == Tree(Tree(Node.id(1), Node.id(0)))


def test_einops_parse_einsum_supports_one_sided_outer_singleton():
    spec = parse_einsum("a b, b -> a")

    assert spec.lhs_rearrange_output == Tree(Tree(Node.id(0)), Tree(Node.id(1)))
    assert spec.rhs_rearrange_output == Tree(Tree(Node.Leaf), Tree(Node.id(0)))
    assert spec.matmul_output_selection == Tree(Tree(Node.Leaf), Tree(Node.Leaf))
    assert spec.output == Tree(Node.id(0))


def test_einops_parse_einsum_supports_dot_product_singleton_output():
    spec = parse_einsum("b, b -> 1")

    assert spec.lhs_rearrange_output == Tree(Tree(Node.Leaf), Tree(Node.id(0)))
    assert spec.rhs_rearrange_output == Tree(Tree(Node.Leaf), Tree(Node.id(0)))
    assert spec.matmul_output_selection == Tree(Tree(Node.Leaf), Tree(Node.Leaf))
    assert spec.output == Tree(Node.Leaf)
    assert spec.common_symbols == ("b",)


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
        "a b -> a b",
    ],
)
def test_einops_parse_reduce_rejects_invalid_syntax(command: str):
    with pytest.raises(ValueError):
        parse_reduce(command)


@pytest.mark.parametrize(
    "command",
    [
        "a b c -> a c",
        "a, b, c -> a",
        "a b, c b",
        "a b, c b -> a -> c",
        "a a, b a -> b",
        "a b, c b -> d",
        "a b, c b -> a b",
        "a b, c b -> a",
        "a, b -> a b",
        "a b, c b -> a a",
        "a (), c -> a",
        "a b, c b -> a, c",
    ],
)
def test_einops_parse_einsum_rejects_invalid_syntax(command: str):
    with pytest.raises(ValueError):
        parse_einsum(command)


def test_einops_rearrange_string_api_returns_rearranged_tensor_view():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))

    result = rearrange(tensor, "a b -> b a")

    assert result.carrier is tensor.carrier
    assert result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert result[2, 1] == tensor[1, 2]
    assert isinstance(result.autograd_ctx, RearrangeOperation)


def test_top_level_rearrange_accepts_einops_string_descriptions():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))

    result = sw.rearrange(tensor, "a b -> b a")

    assert result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert result[2, 1] == tensor[1, 2]


def test_top_level_rearrange_preserves_existing_tree_api():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))

    result = sw.rearrange(tensor, Tree(Node.id(1), Node.id(0)))

    assert result.layout == Layout(Shape([3, 2]), Stride([2, 1]))
    assert result[2, 1] == tensor[1, 2]


def test_top_level_rearrange_rejects_string_description_with_explicit_selection():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))

    with pytest.raises(TypeError):
        sw.rearrange(
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
    assert type(tensor_grad.carrier) is type(tensor.carrier)


def test_einops_reduce_string_api_reduces_omitted_dimensions():
    layout = Layout(Shape([2, [3, 4], 5]), Stride([1, [2, 6], 24]))
    tensor = Tensor(Generic(range(layout.shape.logical_size)), 0, layout)

    result = einops_reduce(tensor, "a (c d) b -> a c")

    assert result.layout == Layout(Shape([2, 3]), Stride([1, 2]))
    assert tensor_values(result) == [
        sum(tensor[a, [c, d], b] for d in range(4) for b in range(5))
        for c in range(3)
        for a in range(2)
    ]
    assert isinstance(result.autograd_ctx, GenericReduceSumOperation)


def test_top_level_reduce_accepts_einops_string_descriptions():
    layout = Layout(Shape([2, [3, 4], 5]), Stride([1, [2, 6], 24]))
    tensor = Tensor(Generic(range(layout.shape.logical_size)), 0, layout)

    result = sw.reduce(tensor, "a (c d) b -> a c")

    assert result.layout == Layout(Shape([2, 3]), Stride([1, 2]))
    assert result[1, 2] == sum(tensor[1, [2, d], b] for d in range(4) for b in range(5))


def test_top_level_reduce_rejects_non_string_description():
    tensor = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    description: Any = object()

    with pytest.raises(TypeError):
        sw.reduce(tensor, description)
    with pytest.raises(TypeError):
        einops_reduce(tensor, description)


def test_einops_reduce_string_api_preserves_nested_output_structure():
    tensor = Tensor(Generic(range(24)), 0, Layout(Shape([2, 3, 4]), Stride([1, 2, 6])))

    result = einops_reduce(tensor, "a b c -> (a c)")

    assert result.layout == Layout(Shape([[2, 4]]), Stride([[1, 2]]))
    assert tensor_values(result) == [
        sum(tensor[a, b, c] for b in range(3)) for c in range(4) for a in range(2)
    ]


def test_einops_reduce_string_api_backpropagates_through_existing_operations():
    layout = Layout(Shape([2, [3, 4], 5]), Stride([1, [2, 6], 24]))
    tensor = Tensor(Generic(range(layout.shape.logical_size)), 0, layout)
    result = einops_reduce(tensor, "a (c d) b -> a c")
    gradient = Tensor(Generic([10, 20, 30, 40, 50, 60]), 0, result.layout)

    result.backward(gradient)
    tensor_grad = require_grad(tensor)

    assert result.grad is None
    assert tensor_grad.layout == tensor.layout
    assert type(tensor_grad.carrier) is type(tensor.carrier)
    for a in range(2):
        for c in range(3):
            for d in range(4):
                for b in range(5):
                    assert tensor_grad[a, [c, d], b] == gradient[a, c]


def test_einops_reduce_string_api_works_with_cpu_tensors():
    layout = Layout(Shape([2, [3, 4], 5]), Stride([1, [2, 6], 24]))
    tensor = make_cpu_tensor(
        [float(i) for i in range(layout.shape.logical_size)], layout
    )

    result = sw.reduce(tensor, "a (c d) b -> a c")

    assert result.layout == Layout(Shape([2, 3]), Stride([1, 2]))
    assert tensor_values(result) == pytest.approx(
        [
            sum(tensor[a, [c, d], b] for d in range(4) for b in range(5))
            for c in range(3)
            for a in range(2)
        ]
    )
    assert type(result.carrier) is CPU


def test_einops_einsum_string_api_matches_manual_dot_products():
    lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    rhs = Tensor(
        Generic([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]),
        0,
        Layout(Shape([4, 3]), Stride([1, 4])),
    )

    result = einops_einsum(lhs, rhs, "a b, c b -> a c")

    assert result.layout == Layout(Shape([2, 4]), Stride([1, 2]))
    assert tensor_values(result) == [
        sum(lhs[a, b] * rhs[c, b] for b in range(3)) for c in range(4) for a in range(2)
    ]
    assert isinstance(result.autograd_ctx, RearrangeOperation)
    assert isinstance(
        result.autograd_ctx.inputs()[0].autograd_ctx, GenericMatmulOperation
    )


def test_top_level_einsum_accepts_einops_string_descriptions():
    lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    rhs = Tensor(
        Generic([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]),
        0,
        Layout(Shape([4, 3]), Stride([1, 4])),
    )

    result = sw.einsum(lhs, rhs, "a b, c b -> a c")

    assert result.layout == Layout(Shape([2, 4]), Stride([1, 2]))
    assert result[1, 2] == sum(lhs[1, b] * rhs[2, b] for b in range(3))


def test_top_level_einsum_rejects_non_string_description():
    lhs = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    rhs = Tensor(Generic(range(12)), 0, Layout(Shape([4, 3]), Stride([1, 4])))
    description: Any = object()

    with pytest.raises(TypeError):
        sw.einsum(lhs, rhs, description)
    with pytest.raises(TypeError):
        einops_einsum(lhs, rhs, description)


def test_einops_einsum_reorders_rhs_shared_dimensions():
    lhs = Tensor(
        Generic(range(24)),
        0,
        Layout(Shape([2, 3, 4]), Stride([1, 2, 6])),
    )
    rhs = Tensor(
        Generic(range(60)),
        0,
        Layout(Shape([5, 4, 3]), Stride([1, 5, 20])),
    )

    result = einops_einsum(lhs, rhs, "a b c, d c b -> a d")

    assert result.layout == Layout(Shape([2, 5]), Stride([1, 2]))
    assert tensor_values(result) == [
        sum(lhs[a, b, c] * rhs[d, c, b] for c in range(4) for b in range(3))
        for d in range(5)
        for a in range(2)
    ]


def test_einops_einsum_preserves_nested_output_structure():
    lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    rhs = Tensor(
        Generic([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]),
        0,
        Layout(Shape([4, 3]), Stride([1, 4])),
    )

    result = einops_einsum(lhs, rhs, "a b, c b -> (c a)")

    assert result.layout == Layout(Shape([[4, 2]]), Stride([[2, 1]]))
    assert tensor_values(result) == [
        sum(lhs[a, b] * rhs[c, b] for b in range(3)) for a in range(2) for c in range(4)
    ]


def test_einops_einsum_supports_one_sided_outer_singleton():
    lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    rhs = Tensor(Generic([10, 20, 30]), 0, Layout(Shape(3), Stride(1)))

    result = einops_einsum(lhs, rhs, "a b, b -> a")

    assert result.layout == Layout(Shape(2), Stride(1))
    assert tensor_values(result) == [
        sum(lhs[a, b] * rhs[b] for b in range(3)) for a in range(2)
    ]


def test_einops_einsum_supports_dot_product_singleton_output():
    lhs = Tensor(Generic([1, 2, 3]), 0, Layout(Shape(3), Stride(1)))
    rhs = Tensor(Generic([10, 20, 30]), 0, Layout(Shape(3), Stride(1)))

    result = einops_einsum(lhs, rhs, "b, b -> 1")

    assert result.layout == Layout(Shape(1), Stride(0))
    assert tensor_values(result) == [sum(lhs[b] * rhs[b] for b in range(3))]


def test_einops_einsum_supports_hierarchical_layouts():
    lhs = Tensor(
        Generic(range(24)),
        0,
        Layout(Shape([2, [3, 4]]), Stride([1, [2, 6]])),
    )
    rhs = Tensor(Generic(range(20)), 0, Layout(Shape([5, 4]), Stride([1, 5])))

    result = einops_einsum(lhs, rhs, "a (b k), c k -> a b c")

    assert result.layout == Layout(Shape([2, 3, 5]), Stride([1, 2, 6]))
    assert tensor_values(result) == [
        sum(lhs[a, [b, k]] * rhs[c, k] for k in range(4))
        for c in range(5)
        for b in range(3)
        for a in range(2)
    ]


def test_einops_einsum_rejects_mismatched_shared_dimension_sizes():
    lhs = Tensor(Generic(range(6)), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    rhs = Tensor(Generic(range(8)), 0, Layout(Shape([4, 2]), Stride([1, 4])))

    with pytest.raises(ValueError):
        einops_einsum(lhs, rhs, "a b, c b -> a c")


def test_einops_einsum_string_api_works_with_cpu_tensors():
    lhs = make_cpu_tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Layout(Shape([2, 3]), Stride([1, 2])),
    )
    rhs = make_cpu_tensor(
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        Layout(Shape([4, 3]), Stride([1, 4])),
    )

    result = sw.einsum(lhs, rhs, "a b, c b -> a c")

    assert result.layout == Layout(Shape([2, 4]), Stride([1, 2]))
    assert tensor_values(result) == pytest.approx(
        [
            sum(lhs[a, b] * rhs[c, b] for b in range(3))
            for c in range(4)
            for a in range(2)
        ]
    )
    assert type(result.carrier) is CPU


def test_einops_einsum_backpropagates_through_existing_operations():
    lhs = Tensor(Generic([1, 2, 3, 4, 5, 6]), 0, Layout(Shape([2, 3]), Stride([1, 2])))
    rhs = Tensor(
        Generic([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]),
        0,
        Layout(Shape([4, 3]), Stride([1, 4])),
    )
    result = einops_einsum(lhs, rhs, "a b, c b -> a c")
    gradient = Tensor(Generic([10, 20, 30, 40, 50, 60, 70, 80]), 0, result.layout)

    result.backward(gradient)
    lhs_grad = require_grad(lhs)
    rhs_grad = require_grad(rhs)

    assert result.grad is None
    assert tensor_values(lhs_grad) == [
        sum(gradient[a, c] * rhs[c, b] for c in range(4))
        for b in range(3)
        for a in range(2)
    ]
    assert tensor_values(rhs_grad) == [
        sum(gradient[a, c] * lhs[a, b] for a in range(2))
        for b in range(3)
        for c in range(4)
    ]


def test_einops_parse_rearrange_reuses_cached_spec_objects():
    first = parse_rearrange("cache_a cache_b -> cache_b cache_a")
    second = parse_rearrange("cache_a cache_b -> cache_b cache_a")

    assert first is second


def test_einops_parse_reduce_reuses_cached_spec_objects():
    first = parse_reduce("reduce_cache_a reduce_cache_b -> reduce_cache_b")
    second = parse_reduce("reduce_cache_a reduce_cache_b -> reduce_cache_b")

    assert first is second


def test_einops_parse_einsum_reuses_cached_spec_objects():
    first = parse_einsum(
        "einsum_cache_a einsum_cache_b, einsum_cache_c einsum_cache_b -> einsum_cache_a einsum_cache_c"
    )
    second = parse_einsum(
        "einsum_cache_a einsum_cache_b, einsum_cache_c einsum_cache_b -> einsum_cache_a einsum_cache_c"
    )

    assert first is second


def test_einops_reduce_and_rearrange_spec_caches_do_not_collide():
    command = "shared_cache_a shared_cache_b -> shared_cache_b"

    rearrange_spec = parse_rearrange(command)
    reduce_spec = parse_reduce(command)

    assert parse_rearrange(command) is rearrange_spec
    assert parse_reduce(command) is reduce_spec
    assert not hasattr(rearrange_spec, "reduced")
    assert reduce_spec.reduced == Tree(Node.id(0))


def test_native_spec_caches_do_not_collide_for_the_same_key():
    key = "shared-native-cache-key"
    rearrange_sentinel = object()
    reduce_sentinel = object()
    einsum_sentinel = object()

    assert (
        native_einops._cached_rearrange_spec(key, lambda _command: rearrange_sentinel)
        is rearrange_sentinel
    )
    assert (
        native_einops._cached_reduce_spec(key, lambda _command: reduce_sentinel)
        is reduce_sentinel
    )
    assert (
        native_einops._cached_einsum_spec(key, lambda _command: einsum_sentinel)
        is einsum_sentinel
    )


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


def test_native_reduce_spec_cache_does_not_cache_failed_compilations():
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
        native_einops._cached_reduce_spec("failing-reduce-cache-key", compiler)

    # The cache is private native state, so call count is the observable proxy:
    # failure must miss, the retry must compile, and the final call must hit.
    assert (
        native_einops._cached_reduce_spec("failing-reduce-cache-key", compiler)
        is cached_sentinel
    )
    assert calls == 2
    cached_result = native_einops._cached_reduce_spec(
        "failing-reduce-cache-key", compiler
    )
    assert cached_result is cached_sentinel
    assert cached_result is not uncached_sentinel
    assert calls == 2


def test_native_einsum_spec_cache_does_not_cache_failed_compilations():
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
        native_einops._cached_einsum_spec("failing-einsum-cache-key", compiler)

    # The cache is private native state, so call count is the observable proxy:
    # failure must miss, the retry must compile, and the final call must hit.
    assert (
        native_einops._cached_einsum_spec("failing-einsum-cache-key", compiler)
        is cached_sentinel
    )
    assert calls == 2
    cached_result = native_einops._cached_einsum_spec(
        "failing-einsum-cache-key", compiler
    )
    assert cached_result is cached_sentinel
    assert cached_result is not uncached_sentinel
    assert calls == 2


def test_einops_parse_rearrange_invalid_descriptions_still_raise_after_retries():
    for _ in range(2):
        with pytest.raises(ValueError):
            parse_rearrange("invalid_a -> invalid_b")


def test_einops_parse_reduce_invalid_descriptions_still_raise_after_retries():
    for _ in range(2):
        with pytest.raises(ValueError):
            parse_reduce("invalid_a -> invalid_b")


def test_einops_parse_einsum_invalid_descriptions_still_raise_after_retries():
    for _ in range(2):
        with pytest.raises(ValueError):
            parse_einsum("invalid_a, invalid_b -> invalid_a invalid_b")


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
