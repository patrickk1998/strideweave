#include <pybind11/pybind11.h>

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "_layout_index.hpp"

namespace py = pybind11;

namespace {

using Index = neotorch::layout_index::Index;

py::object add_python_objects(py::handle left, py::handle right) {
    PyObject* result = PyNumber_Add(left.ptr(), right.ptr());
    if (result == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(result);
}

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

bool contains_slice(py::handle key) {
    if (PySlice_Check(key.ptr())) {
        return true;
    }
    if (py::isinstance<py::tuple>(key) || py::isinstance<py::list>(key)) {
        py::sequence sequence = py::reinterpret_borrow<py::sequence>(key);
        for (py::handle value : sequence) {
            if (contains_slice(value)) {
                return true;
            }
        }
    }
    return false;
}

void validate_tensor_key(py::handle key) {
    if (!is_tensor_key(key)) {
        throw py::type_error(
            "Tensor indices must be integers or tuples/lists of integers"
        );
    }
}

bool layouts_equal(py::handle left, py::handle right) {
    const int result = PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_EQ);
    if (result < 0) {
        throw py::error_already_set();
    }
    return result == 1;
}

bool objects_equal(py::handle left, py::handle right) {
    const int result = PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_EQ);
    if (result < 0) {
        throw py::error_already_set();
    }
    return result == 1;
}

py::object tensor_type() {
    return py::module_::import("neotorch.tensor").attr("Tensor");
}

py::object data_type(const char* name) {
    return py::module_::import("neotorch.data").attr("DataType").attr(name);
}

bool is_differentiable_dtype(py::handle dtype) {
    return objects_equal(dtype, data_type("Float32")) ||
           objects_equal(dtype, data_type("Floating"));
}

class Tensor {
public:
    Tensor(py::object data, Index offset, py::object layout)
        : data_(std::move(data)),
          offset_(offset),
          layout_(std::move(layout)),
          autograd_ctx_(py::none()),
          grad_(py::none()),
          retain_grad_(false) {
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

    py::object autograd_ctx() const { return autograd_ctx_; }

    void set_autograd_ctx(py::object autograd_ctx) {
        if (!autograd_ctx.is_none()) {
            require_differentiable("autograd_ctx is not available for non-differentiable tensors");
        }
        autograd_ctx_ = std::move(autograd_ctx);
    }

    py::object grad() const {
        require_differentiable("grad is not available for non-differentiable tensors");
        return grad_;
    }

    void set_grad(py::object grad) {
        if (!grad.is_none()) {
            require_differentiable("grad is not available for non-differentiable tensors");
        }
        grad_ = std::move(grad);
    }

    void retain_grad(bool retain) {
        require_differentiable("retain_grad is not available for non-differentiable tensors");
        retain_grad_ = retain;
    }

    py::object get_item(py::object key) const {
        validate_tensor_key(key);

        return data_.attr("__getitem__")(py::int_(data_index(key)));
    }

    void set_item(py::object key, py::object value) const {
        validate_tensor_key(key);

        data_.attr("__setitem__")(py::int_(data_index(key)), value);
    }

    Index size() const {
        return py::cast<Index>(layout_.attr("shape").attr("logical_size"));
    }

    bool is_mutable() const {
        return py::cast<bool>(data_.attr("is_mutable")());
    }

    bool is_evictable() const {
        return py::cast<bool>(data_.attr("is_evictable")());
    }

    bool is_evicted() const {
        return py::cast<bool>(data_.attr("is_evicted")());
    }

    void evict() const { data_.attr("evict")(); }

    void promote() const { data_.attr("promote")(); }

    py::object dtype() const { return data_.attr("type")(); }

    bool is_differentiable() const { return is_differentiable_dtype(dtype()); }

    py::object device() const {
        return py::module_::import("builtins").attr("type")(data_);
    }

    void backward(py::object gradient) {
        require_differentiable("backward is not available for non-differentiable tensors");
        py::object effective_gradient = normalize_backward_gradient(std::move(gradient));
        if (should_accumulate_grad()) {
            accumulate_grad(effective_gradient);
        }
        backwards_traversal(std::move(effective_gradient), autograd_ctx_);
    }

    static void backwards_traversal(py::object gradient, py::object operation) {
        if (operation.is_none()) {
            return;
        }

        // Phase 1: discover the reachable operation graph and count, for each
        // differentiable tensor, the number of reachable operations consuming
        // it. The counts allow phase 2 to run every operation's backward
        // exactly once, with the summed gradient of its output, instead of
        // re-traversing shared subgraphs once per consumer.
        std::unordered_map<PyObject*, Index> remaining_consumers;
        std::unordered_set<PyObject*> visited_operations;
        std::vector<py::object> keepalive;
        std::vector<py::object> operation_stack;
        visited_operations.insert(operation.ptr());
        operation_stack.push_back(operation);
        while (!operation_stack.empty()) {
            py::object current = std::move(operation_stack.back());
            operation_stack.pop_back();
            py::sequence inputs =
                py::reinterpret_borrow<py::sequence>(current.attr("inputs")());
            std::unordered_set<PyObject*> seen_inputs;
            for (py::ssize_t i = 0; i < py::len(inputs); ++i) {
                py::object input = py::reinterpret_borrow<py::object>(inputs[i]);
                if (!seen_inputs.insert(input.ptr()).second) {
                    continue;
                }
                if (!py::cast<bool>(input.attr("is_differentiable")())) {
                    continue;
                }
                ++remaining_consumers[input.ptr()];
                keepalive.push_back(input);
                py::object input_ctx = input.attr("autograd_ctx");
                if (!input_ctx.is_none() &&
                    visited_operations.insert(input_ctx.ptr()).second) {
                    operation_stack.push_back(std::move(input_ctx));
                }
            }
            keepalive.push_back(std::move(current));
        }

        // Phase 2: propagate gradients in topological order. A tensor is
        // finalized once every consuming operation has reported its
        // contribution; only then are its gradient accumulated and its own
        // producing operation scheduled. Operations whose output received no
        // gradient are still visited (with a none gradient) so that consumer
        // counts keep decrementing across skipped branches.
        std::unordered_map<PyObject*, py::object> pending_gradients;
        std::vector<std::pair<py::object, py::object>> ready;
        ready.emplace_back(operation, std::move(gradient));
        while (!ready.empty()) {
            auto [current, current_gradient] = std::move(ready.back());
            ready.pop_back();

            py::sequence inputs =
                py::reinterpret_borrow<py::sequence>(current.attr("inputs")());

            if (!current_gradient.is_none()) {
                py::object input_gradients_object =
                    current.attr("backward")(current_gradient);
                py::sequence input_gradients =
                    py::reinterpret_borrow<py::sequence>(input_gradients_object);
                if (py::len(input_gradients) != py::len(inputs)) {
                    throw py::value_error(
                        "Operation backward returned wrong number of gradients"
                    );
                }

                for (py::ssize_t i = 0; i < py::len(inputs); ++i) {
                    py::object input = py::reinterpret_borrow<py::object>(inputs[i]);
                    py::object input_gradient =
                        py::reinterpret_borrow<py::object>(input_gradients[i]);
                    if (input_gradient.is_none()) {
                        continue;
                    }
                    if (!py::cast<bool>(input.attr("is_differentiable")())) {
                        continue;
                    }
                    Tensor& input_tensor = py::cast<Tensor&>(input);
                    input_tensor.validate_gradient(input_gradient);
                    auto found = pending_gradients.find(input.ptr());
                    if (found == pending_gradients.end()) {
                        pending_gradients.emplace(
                            input.ptr(), std::move(input_gradient)
                        );
                    } else {
                        found->second = input_tensor.combined_gradient(
                            found->second, input_gradient
                        );
                    }
                }
            }

            std::unordered_set<PyObject*> seen_inputs;
            for (py::ssize_t i = 0; i < py::len(inputs); ++i) {
                py::object input = py::reinterpret_borrow<py::object>(inputs[i]);
                if (!seen_inputs.insert(input.ptr()).second) {
                    continue;
                }
                auto consumer = remaining_consumers.find(input.ptr());
                if (consumer == remaining_consumers.end()) {
                    continue;
                }
                if (--consumer->second != 0) {
                    continue;
                }

                py::object total_gradient = py::none();
                auto found = pending_gradients.find(input.ptr());
                if (found != pending_gradients.end()) {
                    total_gradient = std::move(found->second);
                    pending_gradients.erase(found);
                }

                Tensor& input_tensor = py::cast<Tensor&>(input);
                if (!total_gradient.is_none() &&
                    input_tensor.should_accumulate_grad()) {
                    input_tensor.accumulate_grad(total_gradient);
                }
                py::object input_ctx = input.attr("autograd_ctx");
                if (!input_ctx.is_none()) {
                    ready.emplace_back(
                        std::move(input_ctx), std::move(total_gradient)
                    );
                }
            }
        }
    }

private:
    void require_differentiable(const char* message) const {
        if (!is_differentiable()) {
            throw std::runtime_error(message);
        }
    }

    Index data_index(py::object key) const {
        const Index layout_index = neotorch::layout_index::get_index(layout_, key);
        return offset_ + layout_index;
    }

    py::object normalize_backward_gradient(py::object gradient) const {
        if (gradient.is_none()) {
            return implicit_scalar_gradient();
        }
        validate_gradient(gradient);
        return gradient;
    }

    py::object implicit_scalar_gradient() const {
        if (!is_scalar()) {
            throw py::value_error(
                "Tensor.backward requires a gradient for non-scalar tensors"
            );
        }

        py::list values;
        values.append(py::int_(1));
        py::object grad_data = data_.attr("new_like")(values);
        return tensor_type()(grad_data, py::int_(0), layout_);
    }

    bool is_scalar() const {
        return py::len(layout_) == 1 &&
               py::cast<bool>(layout_.attr("is_leaf")) && size() == 1;
    }

    void validate_gradient(py::handle gradient) const {
        if (!py::isinstance(gradient, tensor_type())) {
            throw py::type_error("Tensor.backward requires a Tensor gradient");
        }
        py::object gradient_layout = gradient.attr("layout");
        if (!layouts_equal(layout_, gradient_layout)) {
            throw py::value_error("Tensor gradient layout must match tensor layout");
        }
    }

    py::object detached_gradient_copy(py::handle gradient) const {
        validate_gradient(gradient);

        const Index storage_size = neotorch::layout_index::cosize(layout_);
        py::list values;
        for (Index i = 0; i < storage_size; ++i) {
            values.append(py::none());
        }

        const Index tensor_size = size();
        for (Index i = 0; i < tensor_size; ++i) {
            values[neotorch::layout_index::get_index(layout_, py::int_(i))] =
                gradient.attr("__getitem__")(py::int_(i));
        }

        py::object grad_data = data_.attr("new_like")(values);
        return tensor_type()(grad_data, py::int_(0), layout_);
    }

    py::object combined_gradient(py::handle accumulated, py::handle addition) const {
        py::object combined = detached_gradient_copy(accumulated);
        for (Index i = 0; i < size(); ++i) {
            py::object key = py::int_(i);
            py::object combined_value = add_python_objects(
                combined.attr("__getitem__")(key), addition.attr("__getitem__")(key)
            );
            combined.attr("__setitem__")(key, combined_value);
        }
        return combined;
    }

    void accumulate_grad(py::handle gradient) {
        if (grad_.is_none()) {
            grad_ = detached_gradient_copy(gradient);
            return;
        }

        for (Index i = 0; i < size(); ++i) {
            py::object key = py::int_(i);
            py::object accumulated_value = add_python_objects(
                grad_.attr("__getitem__")(key), gradient.attr("__getitem__")(key)
            );
            grad_.attr("__setitem__")(key, accumulated_value);
        }
    }

    bool should_accumulate_grad() const {
        return autograd_ctx_.is_none() || retain_grad_;
    }

    py::object data_;
    Index offset_;
    py::object layout_;
    py::object autograd_ctx_;
    py::object grad_;
    bool retain_grad_;
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
        .def_property("autograd_ctx", &Tensor::autograd_ctx, &Tensor::set_autograd_ctx)
        .def_property("grad", &Tensor::grad, &Tensor::set_grad)
        .def("retain_grad", &Tensor::retain_grad, py::arg("retain") = true)
        .def(
            "__getitem__",
            [](py::object self, py::object key) {
                if (contains_slice(key)) {
                    return py::module_::import("neotorch.operation").attr("view")(
                        self, key
                    );
                }
                Tensor& tensor = py::cast<Tensor&>(self);
                return tensor.get_item(key);
            },
            py::arg("key")
        )
        .def("__setitem__", &Tensor::set_item, py::arg("key"), py::arg("value"))
        .def(
            "__add__",
            [](py::object self, py::object other) {
                return py::module_::import("neotorch.operation").attr("add")(self, other);
            },
            py::is_operator()
        )
        .def(
            "__mul__",
            [](py::object self, py::object other) {
                return py::module_::import("neotorch.operation").attr("mul")(self, other);
            },
            py::is_operator()
        )
        .def(
            "__rmul__",
            [](py::object self, py::object other) {
                return py::module_::import("neotorch.operation").attr("mul")(self, other);
            },
            py::is_operator()
        )
        .def(
            "__truediv__",
            [](py::object self, py::object other) {
                return py::module_::import("neotorch.operation").attr("div")(self, other);
            },
            py::is_operator()
        )
        .def(
            "__pow__",
            [](py::object self, py::object exponent) {
                return py::module_::import("neotorch.operation").attr("pow")(
                    self, exponent
                );
            },
            py::is_operator()
        )
        .def(
            "__matmul__",
            [](py::object self, py::object other) {
                return py::module_::import("neotorch.operation").attr("matmul")(
                    self, other
                );
            },
            py::is_operator()
        )
        .def("size", &Tensor::size)
        .def("is_mutable", &Tensor::is_mutable)
        .def("is_evictable", &Tensor::is_evictable)
        .def("is_evicted", &Tensor::is_evicted)
        .def("evict", &Tensor::evict)
        .def("promote", &Tensor::promote)
        .def("dtype", &Tensor::dtype)
        .def("is_differentiable", &Tensor::is_differentiable)
        .def("device", &Tensor::device)
        .def("backward", &Tensor::backward, py::arg("gradient") = py::none())
        .def_static(
            "backwards_traversal",
            &Tensor::backwards_traversal,
            py::arg("gradient"),
            py::arg("operation")
        );
}
