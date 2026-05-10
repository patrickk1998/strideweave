#include <pybind11/pybind11.h>

#include "_layout_index.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_index, module) {
    module.doc() = "Native indexing helpers for neotorch layouts";
    module.def(
        "get_index", &neotorch::layout_index::get_index, py::arg("layout"), py::arg("key")
    );
}
