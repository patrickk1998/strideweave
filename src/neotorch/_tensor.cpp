#include <pybind11/pybind11.h>

#include <utility>

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

py::object tensor_type() {
    return py::module_::import("neotorch.tensor").attr("Tensor");
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
        autograd_ctx_ = std::move(autograd_ctx);
    }

    py::object grad() const { return grad_; }

    void set_grad(py::object grad) {
        grad_ = std::move(grad);
    }

    void retain_grad(bool retain) {
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

    py::object device() const {
        return py::module_::import("builtins").attr("type")(data_);
    }

    void backward(py::object gradient) {
        validate_gradient(gradient);
        if (should_accumulate_grad()) {
            accumulate_grad(gradient);
        }
        backwards_traversal(std::move(gradient), autograd_ctx_);
    }

    static void backwards_traversal(py::object gradient, py::object operation) {
        if (operation.is_none()) {
            return;
        }

        py::object input_gradients_object = operation.attr("backward")(gradient);
        py::object inputs_object = operation.attr("inputs")();
        py::sequence input_gradients =
            py::reinterpret_borrow<py::sequence>(input_gradients_object);
        py::sequence inputs = py::reinterpret_borrow<py::sequence>(inputs_object);

        if (py::len(input_gradients) != py::len(inputs)) {
            throw py::value_error("Operation backward returned wrong number of gradients");
        }

        for (py::ssize_t i = 0; i < py::len(inputs); ++i) {
            py::object input = py::reinterpret_borrow<py::object>(inputs[i]);
            py::object input_gradient =
                py::reinterpret_borrow<py::object>(input_gradients[i]);
            input.attr("backward")(input_gradient);
        }
    }

private:
    Index data_index(py::object key) const {
        const Index layout_index = neotorch::layout_index::get_index(layout_, key);
        return offset_ + layout_index;
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
        .def("device", &Tensor::device)
        .def("backward", &Tensor::backward, py::arg("gradient"))
        .def_static(
            "backwards_traversal",
            &Tensor::backwards_traversal,
            py::arg("gradient"),
            py::arg("operation")
        );
}
