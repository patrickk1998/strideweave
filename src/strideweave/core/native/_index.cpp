#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "_layout_index.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_index, module) {
    module.doc() = "Native indexing helpers for strideweave layouts";
    py::class_<strideweave::layout_index::LayoutCache>(module, "_LayoutCache")
        .def(py::init<py::object>(), py::arg("layout"))
        .def("expand_key", &strideweave::layout_index::LayoutCache::expand_key)
        .def("get_index", &strideweave::layout_index::LayoutCache::get_index)
        .def(
            "index_expanded",
            [](const strideweave::layout_index::LayoutCache& cache,
               const std::vector<strideweave::layout_index::Index>& key) {
                return cache.index_expanded(key);
            },
            py::arg("key")
        )
        .def(
            "increment_key",
            [](const strideweave::layout_index::LayoutCache& cache,
               std::vector<strideweave::layout_index::Index> key) {
                cache.increment_key(key.data(), key.size());
                return key;
            },
            py::arg("key")
        )
        .def(
            "increment_mode",
            [](const strideweave::layout_index::LayoutCache& cache,
               std::vector<strideweave::layout_index::Index> key,
               strideweave::layout_index::Index mode) {
                cache.increment_mode(key.data(), key.size(), mode);
                return key;
            },
            py::arg("key"),
            py::arg("mode")
        )
        .def_property_readonly(
            "logical_size", &strideweave::layout_index::LayoutCache::logical_size
        )
        .def_property_readonly("cosize", &strideweave::layout_index::LayoutCache::cosize)
        .def_property_readonly("rank", &strideweave::layout_index::LayoutCache::rank)
        .def_property_readonly(
            "leaf_rank", &strideweave::layout_index::LayoutCache::leaf_rank
        )
        .def_property_readonly(
            "leaf_shapes", &strideweave::layout_index::LayoutCache::leaf_shapes
        )
        .def_property_readonly(
            "leaf_strides", &strideweave::layout_index::LayoutCache::leaf_strides
        );

    module.def(
        "get_index", &strideweave::layout_index::get_index, py::arg("layout"), py::arg("key")
    );
}
