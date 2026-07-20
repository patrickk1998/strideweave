#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>
#include <unordered_map>

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
            "Layout command must contain only ASCII characters at offset " +
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

std::unordered_map<std::string, py::object>& rearrange_spec_cache() {
    static auto* cache = new std::unordered_map<std::string, py::object>();
    return *cache;
}

std::unordered_map<std::string, py::object>& reduce_spec_cache() {
    static auto* cache = new std::unordered_map<std::string, py::object>();
    return *cache;
}

std::unordered_map<std::string, py::object>& einsum_spec_cache() {
    static auto* cache = new std::unordered_map<std::string, py::object>();
    return *cache;
}

py::object cached_spec(
    py::str command,
    py::function compiler,
    std::unordered_map<std::string, py::object>& cache
) {
    const std::string key = py::cast<std::string>(command);
    const auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }

    py::object spec = compiler(command);
    cache.emplace(key, spec);
    return spec;
}

py::object cached_rearrange_spec(py::str command, py::function compiler) {
    return cached_spec(command, compiler, rearrange_spec_cache());
}

py::object cached_reduce_spec(py::str command, py::function compiler) {
    return cached_spec(command, compiler, reduce_spec_cache());
}

py::object cached_einsum_spec(py::str command, py::function compiler) {
    return cached_spec(command, compiler, einsum_spec_cache());
}

}  // namespace

PYBIND11_MODULE(_einops, module) {
    module.doc() = "Native lexer for StrideWeave hierarchical layout commands";
    module.def("lex", &lex, py::arg("command"), py::arg("token_type"));
    module.def(
        "_cached_rearrange_spec",
        &cached_rearrange_spec,
        py::arg("command"),
        py::arg("compiler")
    );
    module.def(
        "_cached_reduce_spec",
        &cached_reduce_spec,
        py::arg("command"),
        py::arg("compiler")
    );
    module.def(
        "_cached_einsum_spec",
        &cached_einsum_spec,
        py::arg("command"),
        py::arg("compiler")
    );
}
