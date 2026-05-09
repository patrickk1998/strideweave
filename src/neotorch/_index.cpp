#include <pybind11/pybind11.h>

#include <vector>

namespace py = pybind11;

namespace {

using Index = long long;

bool is_int(py::handle value) { return py::isinstance<py::int_>(value); }

Index as_index(py::handle value) { return py::cast<Index>(value); }

Index logical_size(py::handle shape_level) {
    return as_index(shape_level.attr("logical_size"));
}

Index python_mod(Index value, Index divisor) {
    Index result = value % divisor;
    if (result < 0) {
        result += divisor;
    }
    return result;
}

std::vector<py::object> expand_int(py::handle key_object, py::handle shape_object) {
    Index key = as_index(key_object);
    if (key >= logical_size(shape_object)) {
        throw py::value_error("key is to large!");
    }

    py::sequence shape = py::reinterpret_borrow<py::sequence>(shape_object);
    std::vector<py::object> coordinate;
    coordinate.reserve(static_cast<std::size_t>(py::len(shape)));

    for (py::handle element : shape) {
        Index level_size = is_int(element) ? as_index(element) : logical_size(element);
        Index coordinate_value = python_mod(key, level_size);
        coordinate.emplace_back(py::int_(coordinate_value));
        key -= coordinate_value;
        key /= level_size;
    }

    return coordinate;
}

Index get_index_levels(py::handle shape_object, py::handle stride_object, py::handle key) {
    py::sequence shape = py::reinterpret_borrow<py::sequence>(shape_object);
    py::sequence stride = py::reinterpret_borrow<py::sequence>(stride_object);

    if (py::len(shape) != py::len(stride)) {
        throw py::value_error("Shape and Stride Lengths do not match");
    }

    std::vector<py::object> expanded_key;
    py::iterator key_iterator;
    const bool key_is_int = is_int(key);
    if (key_is_int) {
        expanded_key = expand_int(key, shape);
    } else {
        key_iterator = py::iter(key);
    }

    Index index = 0;
    for (py::ssize_t i = 0; i < py::len(shape); ++i) {
        py::object coordinate;
        if (key_is_int) {
            coordinate = expanded_key[static_cast<std::size_t>(i)];
        } else {
            if (key_iterator == py::iterator::sentinel()) {
                break;
            }
            coordinate = py::reinterpret_borrow<py::object>(*key_iterator);
            ++key_iterator;
        }

        py::object shape_value = shape[i];
        py::object stride_value = stride[i];
        if (is_int(shape_value)) {
            Index coordinate_value = as_index(coordinate);
            if (coordinate_value >= as_index(shape_value)) {
                throw py::value_error("Key is not in domain of shape");
            }
            index += as_index(stride_value) * coordinate_value;
        } else {
            index += get_index_levels(shape_value, stride_value, coordinate);
        }
    }

    return index;
}

Index get_index(py::object layout, py::object key) {
    py::object shape = layout.attr("shape").attr("top_level");
    py::object stride = layout.attr("stride").attr("top_level");
    return get_index_levels(shape, stride, key);
}

}  // namespace

PYBIND11_MODULE(_index, module) {
    module.doc() = "Native indexing helpers for neotorch layouts";
    module.def("get_index", &get_index, py::arg("layout"), py::arg("key"));
}
