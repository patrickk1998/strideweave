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
          dispatch_carrier_class_(py::none()), execution_options_(py::none()) {}

    virtual ~Operation() = default;

    py::object forward(py::args inputs, py::kwargs kwargs) {
        execution_options_ = validated_execution_options(kwargs);
        require_single_subtensor_inputs(inputs);
        const bool should_store_inputs =
            is_grad_enabled() && has_differentiable_tensor_input(inputs);
        if (should_store_inputs) {
            store_tensor_inputs(inputs);
        } else {
            clear_inputs();
        }

        try {
            py::object result = execute(inputs);
            std::vector<py::object> results = validated_results(result);
            bool built_autograd_graph = false;
            if (should_store_inputs && _allows_autograd()) {
                for (std::size_t index = 0; index < results.size(); ++index) {
                    if (!is_differentiable_tensor(results[index])) {
                        continue;
                    }
                    results[index].attr("autograd_ctx") =
                        _autograd_context_for_result(index);
                    built_autograd_graph = true;
                }
            }
            if (!built_autograd_graph) {
                clear_inputs();
            }
            return result;
        } catch (...) {
            clear_inputs();
            throw;
        }
    }

    py::object execute_lowered(py::args inputs, py::kwargs kwargs) {
        execution_options_ = validated_execution_options(kwargs);
        require_single_subtensor_inputs(inputs);
        return execute(inputs);
    }

    virtual py::object _forward(py::args inputs) = 0;
    virtual py::object backward(py::object gradient) = 0;

    virtual bool _accepts_multiple_results() const { return false; }

    virtual bool _allows_autograd() const { return true; }

    virtual py::object _autograd_context_for_result(std::size_t index) {
        if (index != 0U) {
            throw std::out_of_range("single-result Operation context index is invalid");
        }
        return py::cast(this, py::return_value_policy::reference);
    }

    py::dict ctx() const { return ctx_; }

    void store_inputs(py::args inputs) { store_tensor_inputs(inputs); }

    py::tuple inputs() const { return inputs_; }

    py::tuple input_versions() const { return input_versions_; }

    bool autograd_state_freed() const { return autograd_state_freed_; }

    py::object execution_options() const { return execution_options_; }

    void release_autograd_state() {
        clear_inputs();
        ctx_.clear();
        execution_options_ = py::none();
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

    std::vector<py::object> validated_results(py::handle result) {
        py::object tensor = tensor_type();
        if (py::isinstance(result, tensor)) {
            return {py::reinterpret_borrow<py::object>(result)};
        }
        if (!_accepts_multiple_results() || !py::isinstance<py::tuple>(result)) {
            throw py::type_error("Operation._forward must return a Tensor");
        }
        py::tuple results = py::reinterpret_borrow<py::tuple>(result);
        if (results.empty()) {
            throw py::type_error(
                "multi-result Operation._forward must return a non-empty Tensor tuple");
        }
        std::vector<py::object> tensors;
        tensors.reserve(results.size());
        for (py::handle item : results) {
            if (!py::isinstance(item, tensor)) {
                throw py::type_error(
                    "multi-result Operation._forward must return only Tensors");
            }
            tensors.push_back(py::reinterpret_borrow<py::object>(item));
        }
        return tensors;
    }

    py::object validated_execution_options(py::kwargs kwargs) const {
        if (kwargs.empty()) {
            return py::none();
        }
        const py::str options_name("options");
        if (kwargs.size() != 1 || !kwargs.contains(options_name)) {
            for (auto item : kwargs) {
                py::str name = py::reinterpret_borrow<py::str>(item.first);
                if (!name.equal(options_name)) {
                    throw py::type_error("unknown operation execution option '" +
                                         py::cast<std::string>(name) + "'");
                }
            }
            throw py::type_error("operation execution accepts only options=");
        }
        py::object options = py::reinterpret_borrow<py::object>(kwargs[options_name]);
        if (options.is_none()) {
            return options;
        }
        py::object options_type =
            py::module_::import("strideweave.carriers.operation_policy")
                .attr("OperationExecutionOptions");
        if (!py::isinstance(options, options_type)) {
            throw py::type_error("options must be OperationExecutionOptions or None");
        }
        if (!operation_name_.empty()) {
            const std::string options_operation =
                py::cast<std::string>(options.attr("operation"));
            if (options_operation != operation_name_) {
                throw py::value_error("execution options for '" + options_operation +
                                      "' cannot be used for operation '" +
                                      operation_name_ + "'");
            }
        }
        return options;
    }

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
        // Pure c0 layout views preserve every plane and are the only v0
        // operations allowed to execute on a generalized representation.
        // Arithmetic, movement, and coordinate indexing retain the
        // one-subtensor preflight until their per-plane semantics exist.
        const bool layout_view =
            operation_name_ == "broadcast_to" || operation_name_ == "permute" ||
            operation_name_ == "reshape" || operation_name_ == "as_strided" ||
            operation_name_ == "squeeze" || operation_name_ == "unsqueeze" ||
            operation_name_ == "view";
        if (layout_view) {
            return;
        }
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
    py::object execution_options_;
};

}  // namespace strideweave::operation
