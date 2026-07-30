#pragma once

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace strideweave::operation {

inline py::object tensor_type() {
    return py::module_::import("strideweave.tensor").attr("Tensor");
}

inline bool objects_equal(py::handle left, py::handle right) {
    const int result = PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_EQ);
    if (result < 0) {
        throw py::error_already_set();
    }
    return result == 1;
}

inline py::object dtype_object(const char* name) {
    return py::module_::import("strideweave.carriers").attr("DType").attr(name);
}

inline bool is_differentiable_dtype(py::handle dtype) {
    return objects_equal(dtype, dtype_object("Float32")) ||
           objects_equal(dtype, dtype_object("Floating"));
}

inline bool is_differentiable_tensor(py::handle tensor) {
    return is_differentiable_dtype(tensor.attr("dtype")());
}

inline bool is_grad_enabled() {
    return py::cast<bool>(
        py::module_::import("strideweave._operation").attr("is_grad_enabled")());
}

class Operation {
public:
    Operation()
        : ctx_(py::dict()), inputs_(py::tuple()), input_versions_(py::tuple()),
          autograd_state_freed_(false), is_dispatched_(false),
          dispatch_carrier_class_(py::none()) {}

    virtual ~Operation() = default;

    py::object forward(py::args inputs) {
        require_single_subtensor_inputs(inputs);
        const bool should_store_inputs =
            is_grad_enabled() && has_differentiable_tensor_input(inputs);
        if (should_store_inputs) {
            store_tensor_inputs(inputs);
        } else {
            clear_inputs();
        }

        py::object result = execute(inputs);
        const bool build_autograd_graph =
            should_store_inputs && is_differentiable_tensor(result);
        if (build_autograd_graph) {
            result.attr("autograd_ctx") =
                py::cast(this, py::return_value_policy::reference);
        } else {
            clear_inputs();
        }
        return result;
    }

    py::object execute_lowered(py::args inputs) {
        require_single_subtensor_inputs(inputs);
        return execute(inputs);
    }

    virtual py::object _forward(py::args inputs) = 0;
    virtual py::object backward(py::object gradient) = 0;

    py::dict ctx() const { return ctx_; }

    void store_inputs(py::args inputs) { store_tensor_inputs(inputs); }

    py::tuple inputs() const { return inputs_; }

    py::tuple input_versions() const { return input_versions_; }

    bool autograd_state_freed() const { return autograd_state_freed_; }

    void release_autograd_state() {
        clear_inputs();
        ctx_.clear();
        autograd_state_freed_ = true;
    }

    void set_dispatch_metadata(std::string operation_name, py::object carrier_class) {
        operation_name_ = std::move(operation_name);
        is_dispatched_ = true;
        dispatch_carrier_class_ = std::move(carrier_class);
    }

    bool is_dispatched() const { return is_dispatched_; }

    const std::string& operation_name_value() const { return operation_name_; }

    py::object operation_name() const {
        if (operation_name_.empty()) {
            return py::none();
        }
        return py::str(operation_name_);
    }

    py::object dispatch_carrier_class() const { return dispatch_carrier_class_; }

    void validate_input_versions() const {
        for (std::size_t i = 0; i < py::len(inputs_); ++i) {
            py::object input = py::reinterpret_borrow<py::object>(inputs_[i]);
            py::object current_version = input.attr("_version_token")();
            py::object expected_version =
                py::reinterpret_borrow<py::object>(input_versions_[i]);
            if (!objects_equal(current_version, expected_version)) {
                throw std::runtime_error(
                    "A tensor needed for gradient computation was modified "
                    "in-place: its representation version token changed");
            }
        }
    }

protected:
    py::dict ctx_;

private:
    py::object execute(py::args inputs);

    void clear_inputs() {
        inputs_ = py::tuple();
        input_versions_ = py::tuple();
    }

    bool has_differentiable_tensor_input(py::args inputs) {
        py::object tensor = tensor_type();
        for (py::handle input : inputs) {
            if (py::isinstance(input, tensor) && is_differentiable_tensor(input)) {
                return true;
            }
        }
        return false;
    }

    void store_tensor_inputs(py::args inputs) {
        py::object tensor = tensor_type();
        std::vector<py::object> tensors;
        for (py::handle input : inputs) {
            if (py::isinstance(input, tensor)) {
                tensors.push_back(py::reinterpret_borrow<py::object>(input));
            }
        }

        py::tuple stored(tensors.size());
        py::tuple versions(tensors.size());
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            stored[i] = tensors[i];
            versions[i] = tensors[i].attr("_version_token")();
        }
        inputs_ = std::move(stored);
        input_versions_ = std::move(versions);
    }

    void require_single_subtensor_inputs(py::args inputs) {
        py::object tensor = tensor_type();
        for (py::handle input : inputs) {
            if (py::isinstance(input, tensor)) {
                input.attr("_require_single_subtensor")("operation execution");
            }
        }
    }

    py::tuple inputs_;
    py::tuple input_versions_;
    bool autograd_state_freed_;
    bool is_dispatched_;
    std::string operation_name_;
    py::object dispatch_carrier_class_;
};

}  // namespace strideweave::operation
