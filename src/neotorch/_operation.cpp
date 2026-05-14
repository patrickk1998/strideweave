#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

class Operation {
public:
    Operation() : ctx_(py::dict()), inputs_(py::tuple()) {}

    virtual ~Operation() = default;

    virtual py::object forward(py::args inputs) = 0;
    virtual py::object backward(py::object gradient) = 0;

    py::dict ctx() const { return ctx_; }

    void store_inputs(py::args inputs) {
        inputs_ = py::reinterpret_borrow<py::tuple>(inputs);
    }

    py::tuple inputs() const { return inputs_; }

private:
    py::dict ctx_;
    py::tuple inputs_;
};

class PyOperation : public Operation {
public:
    using Operation::Operation;

    py::object forward(py::args inputs) override {
        py::gil_scoped_acquire gil;
        py::function override = py::get_override(this, "forward");
        if (!override) {
            throw py::type_error("Operation.forward must be implemented");
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
    module.doc() = "Native operation base class for neotorch autograd";

    py::class_<Operation, PyOperation>(module, "Operation")
        .def(py::init<>())
        .def(
            "forward",
            [](Operation& operation, py::args inputs) {
                return operation.forward(inputs);
            }
        )
        .def("backward", &Operation::backward, py::arg("gradient"))
        .def_property_readonly("ctx", &Operation::ctx)
        .def(
            "store_inputs",
            [](Operation& operation, py::args inputs) {
                operation.store_inputs(inputs);
            }
        )
        .def("inputs", &Operation::inputs);
}
