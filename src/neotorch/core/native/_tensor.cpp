#include <pybind11/pybind11.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "_data.hpp"
#include "_layout_index.hpp"

namespace py = pybind11;

namespace {

using Index = neotorch::layout_index::Index;

enum DLDeviceType : std::int32_t {
    kDLCPU = 1,
};

enum DLDataTypeCode : std::uint8_t {
    kDLInt = 0U,
    kDLFloat = 2U,
};

struct DLPackVersion {
    std::uint32_t major;
    std::uint32_t minor;
};

struct DLDevice {
    DLDeviceType device_type;
    std::int32_t device_id;
};

struct DLDataType {
    std::uint8_t code;
    std::uint8_t bits;
    std::uint16_t lanes;
};

struct DLTensor {
    void* data;
    DLDevice device;
    std::int32_t ndim;
    DLDataType dtype;
    std::int64_t* shape;
    std::int64_t* strides;
    std::uint64_t byte_offset;
};

struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(DLManagedTensor* self);
};

struct DLManagedTensorVersioned {
    DLPackVersion version;
    void* manager_ctx;
    void (*deleter)(DLManagedTensorVersioned* self);
    std::uint64_t flags;
    DLTensor dl_tensor;
};

constexpr std::uint64_t dlpack_flag_read_only = 1UL << 0UL;
constexpr const char* dlpack_capsule_name = "dltensor";
constexpr const char* used_dlpack_capsule_name = "used_dltensor";
constexpr const char* versioned_dlpack_capsule_name = "dltensor_versioned";
constexpr const char* used_versioned_dlpack_capsule_name =
    "used_dltensor_versioned";

struct DLPackDTypeInfo {
    DLDataType dtype;
    std::uint64_t item_size;
};

struct DLPackStorageInfo {
    std::uintptr_t pointer;
    DLDevice device;
};

struct LegacyDLPackTensor {
    DLManagedTensor managed;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    PyObject* owner = nullptr;
};

struct VersionedDLPackTensor {
    DLManagedTensorVersioned managed;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    PyObject* owner = nullptr;
};

[[noreturn]] void throw_buffer_error(const char* message) {
    PyErr_SetString(PyExc_BufferError, message);
    throw py::error_already_set();
}

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

DLPackDTypeInfo dlpack_dtype_info(py::handle dtype) {
    if (objects_equal(dtype, data_type("Float32"))) {
        return {{kDLFloat, 32, 1}, sizeof(float)};
    }
    if (objects_equal(dtype, data_type("Int32"))) {
        return {{kDLInt, 32, 1}, sizeof(std::int32_t)};
    }
    throw_buffer_error("DLPack export supports only Float32 and Int32 tensors");
}

std::vector<std::int64_t> to_int64_vector(const std::vector<Index>& values) {
    std::vector<std::int64_t> result;
    result.reserve(values.size());
    for (Index value : values) {
        result.push_back(static_cast<std::int64_t>(value));
    }
    return result;
}

void decref_owner(PyObject* owner) {
    if (owner == nullptr) {
        return;
    }
    if (!Py_IsInitialized()) {
        return;
    }
    PyGILState_STATE state = PyGILState_Ensure();
    Py_DECREF(owner);
    PyGILState_Release(state);
}

void legacy_dlpack_managed_deleter(DLManagedTensor* self) {
    auto* holder = reinterpret_cast<LegacyDLPackTensor*>(self);
    decref_owner(holder->owner);
    delete holder;
}

void versioned_dlpack_managed_deleter(DLManagedTensorVersioned* self) {
    auto* holder = reinterpret_cast<VersionedDLPackTensor*>(self);
    decref_owner(holder->owner);
    delete holder;
}

void legacy_dlpack_capsule_deleter(PyObject* capsule) {
    if (PyCapsule_IsValid(capsule, used_dlpack_capsule_name)) {
        return;
    }
    auto* managed = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule, dlpack_capsule_name)
    );
    if (managed == nullptr) {
        PyErr_WriteUnraisable(capsule);
        return;
    }
    if (managed->deleter != nullptr) {
        managed->deleter(managed);
    }
}

void versioned_dlpack_capsule_deleter(PyObject* capsule) {
    if (PyCapsule_IsValid(capsule, used_versioned_dlpack_capsule_name)) {
        return;
    }
    auto* managed = static_cast<DLManagedTensorVersioned*>(
        PyCapsule_GetPointer(capsule, versioned_dlpack_capsule_name)
    );
    if (managed == nullptr) {
        PyErr_WriteUnraisable(capsule);
        return;
    }
    if (managed->deleter != nullptr) {
        managed->deleter(managed);
    }
}

bool should_export_versioned_dlpack(py::handle max_version) {
    if (max_version.is_none()) {
        return false;
    }
    if (!py::isinstance<py::tuple>(max_version) &&
        !py::isinstance<py::list>(max_version)) {
        throw py::type_error("DLPack max_version must be a tuple or list");
    }
    py::sequence version = py::reinterpret_borrow<py::sequence>(max_version);
    if (py::len(version) != 2) {
        throw py::value_error("DLPack max_version must have two elements");
    }
    const int major = py::cast<int>(version[0]);
    const int minor = py::cast<int>(version[1]);
    if (major < 0 || minor < 0) {
        throw py::value_error("DLPack max_version elements must be non-negative");
    }
    return major >= 1;
}

std::uint64_t byte_offset_for(Index offset, std::uint64_t item_size) {
    const auto offset_u = static_cast<std::uint64_t>(offset);
    if (offset_u > std::numeric_limits<std::uint64_t>::max() / item_size) {
        throw_buffer_error("Tensor offset is too large for DLPack export");
    }
    return offset_u * item_size;
}

std::int32_t sequence_device_component(py::sequence device, py::ssize_t index) {
    return py::cast<std::int32_t>(device[index]);
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

    ::neotorch::data::Version version() const {
        return py::cast<::neotorch::data::Version>(data_.attr("version"));
    }

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

    py::object dtype() const { return data_.attr("type")(); }

    bool is_differentiable() const { return is_differentiable_dtype(dtype()); }

    py::object device() const {
        return py::module_::import("builtins").attr("type")(data_);
    }

    py::tuple dlpack_device() const {
        DLPackStorageInfo storage = dlpack_storage_info();
        return py::make_tuple(
            static_cast<std::int32_t>(storage.device.device_type),
            storage.device.device_id
        );
    }

    py::object dlpack(
        py::object self,
        py::object stream,
        py::object max_version,
        py::object dl_device,
        py::object copy
    ) const {
        (void)stream;
        if (!copy.is_none() && py::cast<bool>(copy)) {
            throw_buffer_error("DLPack copy exports are not supported");
        }

        DLPackStorageInfo storage = dlpack_storage_info();
        validate_dlpack_device_request(dl_device, storage.device);
        DLPackDTypeInfo dtype_info = dlpack_dtype_info(dtype());
        const bool versioned = should_export_versioned_dlpack(max_version);
        if (versioned) {
            return make_versioned_dlpack_capsule(
                std::move(self), storage, dtype_info
            );
        }
        return make_legacy_dlpack_capsule(std::move(self), storage, dtype_info);
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
                current.attr("validate_input_versions")();
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

    DLPackStorageInfo dlpack_storage_info() const {
        py::dict info = py::cast<py::dict>(data_.attr("dlpack_info")());
        const auto pointer = py::cast<std::uintptr_t>(info["pointer"]);
        const auto device_type_int = py::cast<std::int32_t>(info["device_type"]);
        const auto device_id = py::cast<std::int32_t>(info["device_id"]);
        if (pointer == 0 && size() != 0) {
            throw_buffer_error("DLPack data pointer must be non-null");
        }
        return {
            pointer,
            {static_cast<DLDeviceType>(device_type_int), device_id},
        };
    }

    void validate_dlpack_device_request(
        py::handle requested_device, DLDevice actual_device
    ) const {
        if (requested_device.is_none()) {
            return;
        }
        if (!py::isinstance<py::tuple>(requested_device) &&
            !py::isinstance<py::list>(requested_device)) {
            throw py::type_error("DLPack dl_device must be a tuple or list");
        }
        py::sequence device = py::reinterpret_borrow<py::sequence>(requested_device);
        if (py::len(device) != 2) {
            throw py::value_error("DLPack dl_device must have two elements");
        }
        const auto requested_type = sequence_device_component(device, 0);
        const auto requested_id = sequence_device_component(device, 1);
        if (requested_type != static_cast<std::int32_t>(actual_device.device_type) ||
            requested_id != actual_device.device_id) {
            throw_buffer_error(
                "DLPack cross-device exports are not supported for this tensor"
            );
        }
    }

    void populate_dlpack_tensor(
        DLTensor& dl_tensor,
        std::vector<std::int64_t>& shape,
        std::vector<std::int64_t>& strides,
        DLPackStorageInfo storage,
        DLPackDTypeInfo dtype_info
    ) const {
        const neotorch::layout_index::LayoutCache& cache =
            neotorch::layout_index::cache_from_layout(layout_);
        shape = to_int64_vector(cache.leaf_shapes());
        strides = to_int64_vector(cache.leaf_strides());

        dl_tensor.data = reinterpret_cast<void*>(storage.pointer);
        dl_tensor.device = storage.device;
        dl_tensor.ndim = static_cast<std::int32_t>(shape.size());
        dl_tensor.dtype = dtype_info.dtype;
        dl_tensor.shape = shape.empty() ? nullptr : shape.data();
        dl_tensor.strides = strides.empty() ? nullptr : strides.data();
        dl_tensor.byte_offset = byte_offset_for(offset_, dtype_info.item_size);
    }

    py::object make_legacy_dlpack_capsule(
        py::object self,
        DLPackStorageInfo storage,
        DLPackDTypeInfo dtype_info
    ) const {
        auto holder = std::make_unique<LegacyDLPackTensor>();
        populate_dlpack_tensor(
            holder->managed.dl_tensor,
            holder->shape,
            holder->strides,
            storage,
            dtype_info
        );
        holder->managed.manager_ctx = holder.get();
        holder->managed.deleter = legacy_dlpack_managed_deleter;
        holder->owner = self.ptr();
        Py_INCREF(holder->owner);

        DLManagedTensor* managed = &holder->managed;
        holder.release();
        return py::capsule(
            managed, dlpack_capsule_name, legacy_dlpack_capsule_deleter
        );
    }

    py::object make_versioned_dlpack_capsule(
        py::object self,
        DLPackStorageInfo storage,
        DLPackDTypeInfo dtype_info
    ) const {
        auto holder = std::make_unique<VersionedDLPackTensor>();
        populate_dlpack_tensor(
            holder->managed.dl_tensor,
            holder->shape,
            holder->strides,
            storage,
            dtype_info
        );
        holder->managed.version = {1, 0};
        holder->managed.manager_ctx = holder.get();
        holder->managed.deleter = versioned_dlpack_managed_deleter;
        holder->managed.flags = is_mutable() ? 0 : dlpack_flag_read_only;
        holder->owner = self.ptr();
        Py_INCREF(holder->owner);

        DLManagedTensorVersioned* managed = &holder->managed;
        holder.release();
        return py::capsule(
            managed,
            versioned_dlpack_capsule_name,
            versioned_dlpack_capsule_deleter
        );
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
        .def_property_readonly("version", &Tensor::version)
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
        .def("dtype", &Tensor::dtype)
        .def("is_differentiable", &Tensor::is_differentiable)
        .def("device", &Tensor::device)
        .def("__dlpack_device__", &Tensor::dlpack_device)
        .def(
            "__dlpack__",
            [](py::object self,
               py::object stream,
               py::object max_version,
               py::object dl_device,
               py::object copy) {
                const Tensor& tensor = py::cast<const Tensor&>(self);
                return tensor.dlpack(
                    std::move(self),
                    std::move(stream),
                    std::move(max_version),
                    std::move(dl_device),
                    std::move(copy)
                );
            },
            py::arg("stream") = py::none(),
            py::kw_only(),
            py::arg("max_version") = py::none(),
            py::arg("dl_device") = py::none(),
            py::arg("copy") = py::none()
        )
        .def("backward", &Tensor::backward, py::arg("gradient") = py::none())
        .def_static(
            "backwards_traversal",
            &Tensor::backwards_traversal,
            py::arg("gradient"),
            py::arg("operation")
        );
}
