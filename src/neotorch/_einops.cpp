#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>

namespace py = pybind11;

namespace {

bool is_ascii_alpha(char value) {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z');
}

bool is_ascii_digit(char value) { return value >= '0' && value <= '9'; }

bool is_symbol_tail(char value) {
    return is_ascii_alpha(value) || is_ascii_digit(value) || value == '_';
}

bool is_ascii_whitespace(char value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r' ||
           value == '\f' || value == '\v';
}

void ensure_ascii(char value, std::size_t position) {
    if (static_cast<unsigned char>(value) >= 128) {
        throw py::value_error(
            "Einops command must contain only ASCII characters at offset " +
            std::to_string(position)
        );
    }
}

void append_token(
    py::list& tokens,
    const py::object& token_type,
    const char* kind,
    const std::string& value,
    std::size_t start,
    std::size_t end
) {
    tokens.append(token_type(
        py::str(kind), py::str(value), py::int_(start), py::int_(end)
    ));
}

py::list lex(py::str command, py::object token_type) {
    const std::string input = py::cast<std::string>(command);
    py::list tokens;
    std::size_t position = 0;

    while (position < input.size()) {
        const char value = input[position];
        ensure_ascii(value, position);

        if (is_ascii_whitespace(value)) {
            ++position;
            continue;
        }

        const std::size_t start = position;
        if (value == '(') {
            append_token(tokens, token_type, "left_paren", "(", start, start + 1);
            ++position;
            continue;
        }
        if (value == ')') {
            append_token(tokens, token_type, "right_paren", ")", start, start + 1);
            ++position;
            continue;
        }
        if (value == ',') {
            append_token(tokens, token_type, "comma", ",", start, start + 1);
            ++position;
            continue;
        }
        if (value == '-') {
            if (position + 1 < input.size() && input[position + 1] == '>') {
                append_token(tokens, token_type, "arrow", "->", start, start + 2);
                position += 2;
                continue;
            }
            throw py::value_error(
                "Expected '->' arrow token at offset " + std::to_string(start)
            );
        }
        if (value == '>') {
            throw py::value_error(
                "Unexpected '>' at offset " + std::to_string(start)
            );
        }
        if (value == '1') {
            if (position + 1 < input.size()) {
                ensure_ascii(input[position + 1], position + 1);
                if (is_symbol_tail(input[position + 1])) {
                    throw py::value_error(
                        "Invalid singleton token at offset " +
                        std::to_string(start)
                    );
                }
            }
            append_token(tokens, token_type, "one", "1", start, start + 1);
            ++position;
            continue;
        }
        if (is_ascii_digit(value)) {
            throw py::value_error(
                "Invalid dimension symbol at offset " + std::to_string(start)
            );
        }
        if (is_ascii_alpha(value)) {
            ++position;
            while (position < input.size()) {
                ensure_ascii(input[position], position);
                if (!is_symbol_tail(input[position])) {
                    break;
                }
                ++position;
            }
            append_token(
                tokens,
                token_type,
                "symbol",
                input.substr(start, position - start),
                start,
                position
            );
            continue;
        }

        throw py::value_error(
            "Unexpected character at offset " + std::to_string(start)
        );
    }

    return tokens;
}

}  // namespace

PYBIND11_MODULE(_einops, module) {
    module.doc() = "Native lexer for neotorch einops commands";
    module.def("lex", &lex, py::arg("command"), py::arg("token_type"));
}
