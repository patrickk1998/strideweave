#include <pybind11/pybind11.h>

#include "_data.hpp"

namespace py = pybind11;

namespace {

using neotorch::data::Data;
using neotorch::data::Index;

class PyData : public Data {
public:
    using Data::Data;

    Index size() const override { PYBIND11_OVERRIDE_PURE(Index, Data, size); }

    py::object type() const override {
        PYBIND11_OVERRIDE_PURE(py::object, Data, type);
    }

    py::object get_value(Index index) const override {
        PYBIND11_OVERRIDE_PURE(py::object, Data, get_value, index);
    }

    py::object new_like(py::iterable values, bool is_mutable) const override {
        PYBIND11_OVERRIDE_PURE(py::object, Data, new_like, values, is_mutable);
    }

    void scatter(
        py::object to_scatter,
        py::object scatter_onto,
        py::object mapping,
        Index mapping_offset
    ) override {
        PYBIND11_OVERRIDE_PURE(
            void, Data, scatter, to_scatter, scatter_onto, mapping, mapping_offset
        );
    }

    bool is_mutable() const override {
        PYBIND11_OVERRIDE(bool, Data, is_mutable);
    }

    py::dict dlpack_info() const override {
        PYBIND11_OVERRIDE(py::dict, Data, dlpack_info);
    }

protected:
    void set_value(Index index, py::object value) override {
        PYBIND11_OVERRIDE(void, Data, set_value, index, value);
    }

    void _release() override { PYBIND11_OVERRIDE(void, Data, _release); }
};

class VectorDataForTest : public Data {
public:
    explicit VectorDataForTest(py::iterable values) : values_(py::list(values)) {}

    Index size() const override { return static_cast<Index>(py::len(values_)); }

    py::object type() const override {
        return py::module_::import("neotorch.data").attr("DataType").attr("Any");
    }

    py::object get_value(Index index) const override {
        return py::reinterpret_borrow<py::object>(values_[index]);
    }

    py::object new_like(py::iterable values, bool) const override {
        return py::cast(VectorDataForTest(values));
    }

    void scatter(py::object, py::object, py::object, Index) override {
        throw py::type_error("_VectorDataForTest does not implement scatter");
    }

private:
    py::list values_;
};

}  // namespace

PYBIND11_MODULE(_data, module) {
    module.doc() = "Native data base classes for neotorch";

    py::class_<Data, PyData>(module, "Data")
        .def(py::init<>())
        .def("size", &Data::size)
        .def("type", &Data::type)
        .def("get_value", &Data::get_value, py::arg("index"))
        .def(
            "new_like",
            &Data::new_like,
            py::arg("values"),
            py::kw_only(),
            py::arg("mutable") = true
        )
        .def(
            "scatter",
            &Data::scatter,
            py::arg("to_scatter"),
            py::arg("scatter_onto"),
            py::arg("mapping"),
            py::arg("mapping_offset") = 0
        )
        .def("is_mutable", &Data::is_mutable)
        .def("dlpack_info", &Data::dlpack_info)
        .def_property_readonly("version", &Data::version)
        .def("_increment_version", &Data::increment_version)
        .def("is_released", &Data::is_released)
        .def(
            "release",
            &Data::release,
            "Release the data's storage; further element access raises.\n\n"
            "Contract: ``new_like`` must remain usable after ``release()`` --\n"
            "it constructs fresh storage and reads nothing from the released\n"
            "instance. Move's backward pass relies on this to materialize\n"
            "gradients in a released source data class."
        )
        .def_static("dispatch_op", &Data::dispatch_op, py::arg("operation_name"))
        .def("__getitem__", &Data::get_item, py::arg("index"))
        .def("__setitem__", &Data::set_item, py::arg("index"), py::arg("value"));

    py::class_<VectorDataForTest, Data>(module, "_VectorDataForTest")
        .def(py::init<py::iterable>(), py::arg("values"));

    py::module_::import("neotorch._operation");
    neotorch::data::bind_cpu(module);
}
