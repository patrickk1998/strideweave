#include <pybind11/pybind11.h>

#include <utility>

#include "_layout_index.hpp"

namespace py = pybind11;

namespace {

using Index = neotorch::layout_index::Index;

bool is_tensor_key(py::handle key) {
    if (neotorch::layout_index::is_int(key)) {
        return true;
    }
    if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
        py::sequence sequence = py::reinterpret_borrow<py::sequence>(key);
        for (py::handle value : sequence) {
            if (!is_tensor_key(value)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

class Tensor {
public:
    Tensor(py::object data, Index offset, py::object layout)
        : data_(std::move(data)), offset_(offset), layout_(std::move(layout)) {
        if (offset_ < 0) {
            throw py::value_error("Tensor offset must be non-negative");
        }

        const Index data_size = py::cast<Index>(data_.attr("size")());
        const Index storage_size = neotorch::layout_index::cosize(layout_);
        const bool storage_exceeds_data =
            data_size < 0 || offset_ > data_size || storage_size > data_size - offset_;
        if (storage_exceeds_data) {
            throw py::value_error("Tensor storage exceeds data size");
        }
    }

    py::object data() const { return data_; }

    Index offset() const { return offset_; }

    py::object layout() const { return layout_; }

    py::object get_item(py::object key) const {
        if (!is_tensor_key(key)) {
            throw py::type_error(
                "Tensor indices must be integers or tuples/lists of integers"
            );
        }

        return data_.attr("__getitem__")(py::int_(data_index(key)));
    }

    void set_item(py::object key, py::object value) const {
        if (!is_tensor_key(key)) {
            throw py::type_error(
                "Tensor indices must be integers or tuples/lists of integers"
            );
        }

        data_.attr("__setitem__")(py::int_(data_index(key)), value);
    }

    Index size() const {
        return py::cast<Index>(layout_.attr("shape").attr("logical_size"));
    }

    bool is_mutable() const {
        return py::cast<bool>(data_.attr("is_mutable")());
    }

    py::object dtype() const { return data_.attr("type")(); }

    py::object device() const {
        return py::module_::import("builtins").attr("type")(data_);
    }

private:
    Index data_index(py::object key) const {
        const Index layout_index = neotorch::layout_index::get_index(layout_, key);
        return offset_ + layout_index;
    }

    py::object data_;
    Index offset_;
    py::object layout_;
};

}  // namespace

PYBIND11_MODULE(_tensor, module) {
    module.doc() = "Native tensor type for neotorch";

    py::class_<Tensor>(module, "Tensor")
        .def(
            py::init<py::object, Index, py::object>(),
            py::arg("data"),
            py::arg("offset"),
            py::arg("layout")
        )
        .def_property_readonly("data", &Tensor::data)
        .def_property_readonly("offset", &Tensor::offset)
        .def_property_readonly("layout", &Tensor::layout)
        .def("__getitem__", &Tensor::get_item, py::arg("key"))
        .def("__setitem__", &Tensor::set_item, py::arg("key"), py::arg("value"))
        .def("size", &Tensor::size)
        .def("is_mutable", &Tensor::is_mutable)
        .def("dtype", &Tensor::dtype)
        .def("device", &Tensor::device);
}
