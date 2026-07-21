#include <pybind11/pybind11.h>

#include "_operation.hpp"

namespace py = pybind11;

namespace {

thread_local bool grad_enabled = true;

bool is_grad_enabled() {
    return grad_enabled;
}

void set_grad_enabled(bool enabled) {
    grad_enabled = enabled;
}

using strideweave::operation::Operation;

class PyOperation : public Operation {
public:
    using Operation::Operation;

    py::object _forward(py::args inputs) override {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(this, "_forward");
        if (!override) {
            throw py::type_error("Operation._forward must be implemented");
        }

        PyObject* result = PyObject_CallObject(override.ptr(), inputs.ptr());
        if (result == nullptr) {
            throw py::error_already_set();
        }
        return py::reinterpret_steal<py::object>(result);
    }

    py::object backward(py::object gradient) override {
        PYBIND11_OVERRIDE_PURE(py::object, Operation, backward, gradient);
    }
};

}  // namespace

PYBIND11_MODULE(_operation, module) {
    module.doc() = "Native operation base class for strideweave autograd";

    module.def("is_grad_enabled", &is_grad_enabled);
    module.def("set_grad_enabled", &set_grad_enabled, py::arg("enabled"));

    py::class_<Operation, PyOperation>(module, "Operation")
        .def(py::init<>())
        .def("forward", [](Operation& operation,
                           py::args inputs) { return operation.forward(inputs); })
        .def("_forward", [](Operation& operation,
                            py::args inputs) { return operation._forward(inputs); })
        .def("backward", &Operation::backward, py::arg("gradient"))
        .def_property_readonly("ctx", &Operation::ctx)
        .def("store_inputs", [](Operation& operation,
                                py::args inputs) { operation.store_inputs(inputs); })
        .def("inputs", &Operation::inputs)
        .def("input_versions", &Operation::input_versions)
        .def("validate_input_versions", &Operation::validate_input_versions);
}
