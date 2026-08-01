#include <pybind11/pybind11.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "_carrier.hpp"
#include "_layout_index.hpp"

namespace py = pybind11;

namespace {

using Index = strideweave::layout_index::Index;

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
constexpr const char* used_versioned_dlpack_capsule_name = "used_dltensor_versioned";

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
    if (strideweave::layout_index::is_int(key)) {
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
            "Tensor indices must be integers or tuples/lists of integers");
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
    return py::module_::import("strideweave.tensor").attr("Tensor");
}

py::object dtype_object(const char* name) {
    return py::module_::import("strideweave.carriers").attr("DType").attr(name);
}

bool is_differentiable_dtype(py::handle dtype) {
    return objects_equal(dtype, dtype_object("Float32")) ||
           objects_equal(dtype, dtype_object("Floating"));
}

DLPackDTypeInfo dlpack_dtype_info(py::handle dtype) {
    if (objects_equal(dtype, dtype_object("Float32"))) {
        return {{kDLFloat, 32, 1}, sizeof(float)};
    }
    if (objects_equal(dtype, dtype_object("Int32"))) {
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
        PyCapsule_GetPointer(capsule, dlpack_capsule_name));
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
        PyCapsule_GetPointer(capsule, versioned_dlpack_capsule_name));
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

std::int32_t sequence_device_component(py::sequence device, std::size_t index) {
    return py::cast<std::int32_t>(device[index]);
}

py::object representation_module() {
    return py::module_::import("strideweave.core._representation");
}

py::object make_tensor_representation(py::object carrier, Index offset,
                                      py::object layout) {
    py::object representation = representation_module();
    py::object dtype = carrier.attr("dtype")();
    py::object subtensor =
        representation.attr("Subtensor")(dtype, carrier, py::int_(offset), layout);
    return representation.attr("TensorRepresentation")(
        dtype, py::make_tuple(std::move(subtensor)), py::tuple());
}

struct FromRepresentationTag {};

class Tensor {
public:
    Tensor(py::object carrier, Index offset, py::object layout)
        : Tensor(
              make_tensor_representation(std::move(carrier), offset, std::move(layout)),
              FromRepresentationTag{}) {}

    static Tensor from_representation(py::object representation) {
        py::object expected = representation_module().attr("TensorRepresentation");
        if (!py::isinstance(representation, expected)) {
            throw py::type_error(
                "_from_representation requires a TensorRepresentation");
        }
        return Tensor(std::move(representation), FromRepresentationTag{});
    }

    py::object representation() const { return representation_; }

    py::object carrier() const { return primary().attr("carrier"); }

    Index offset() const { return py::cast<Index>(primary().attr("offset")); }

    py::object layout() const { return primary().attr("layout"); }

    ::strideweave::carrier::Version version() const {
        return py::cast<::strideweave::carrier::Version>(carrier().attr("version"));
    }

    py::object version_token() const {
        return representation_.attr("_version_token")();
    }

    void require_single_subtensor(const std::string& reason) const {
        if (!py::cast<bool>(representation_.attr("is_single_subtensor"))) {
            const std::string message =
                "Multi-subtensor Tensor " + reason +
                " is not implemented; the operation was rejected before "
                "changing any constituent carrier";
            PyErr_SetString(PyExc_NotImplementedError, message.c_str());
            throw py::error_already_set();
        }
    }

    py::object autograd_ctx() const { return autograd_ctx_; }

    void set_autograd_ctx(py::object autograd_ctx) {
        if (!autograd_ctx.is_none()) {
            require_differentiable(
                "autograd_ctx is not available for non-differentiable tensors");
        }
        autograd_ctx_ = std::move(autograd_ctx);
    }

private:
    Tensor(py::object representation, FromRepresentationTag)
        : representation_(std::move(representation)), autograd_ctx_(py::none()),
          grad_(py::none()), retain_grad_(false) {}

    py::object primary() const { return representation_.attr("primary"); }

public:
    py::object grad() const {
        require_differentiable("grad is not available for non-differentiable tensors");
        return grad_;
    }

    void set_grad(py::object grad) {
        if (!grad.is_none()) {
            require_differentiable(
                "grad is not available for non-differentiable tensors");
        }
        grad_ = std::move(grad);
    }

    void retain_grad(bool retain) {
        require_differentiable(
            "retain_grad is not available for non-differentiable tensors");
        retain_grad_ = retain;
    }

    py::object get_item(py::object key) const {
        require_single_subtensor("indexing");
        validate_tensor_key(key);

        return carrier().attr("__getitem__")(py::int_(carrier_index(key)));
    }

    void set_item(py::object key, py::object value) const {
        require_single_subtensor("mutation");
        validate_tensor_key(key);

        carrier().attr("__setitem__")(py::int_(carrier_index(key)), value);
    }

    Index size() const {
        return py::cast<Index>(layout().attr("shape").attr("logical_size"));
    }

    bool is_mutable() const {
        return py::cast<bool>(representation_.attr("is_single_subtensor")) &&
               py::cast<bool>(carrier().attr("is_mutable")());
    }

    py::object dtype() const { return representation_.attr("logical_dtype"); }

    bool is_differentiable() const { return is_differentiable_dtype(dtype()); }

    py::object carrier_type() const {
        return py::module_::import("builtins").attr("type")(carrier());
    }

    py::tuple dlpack_device() const {
        require_single_subtensor("DLPack export");
        DLPackStorageInfo storage = dlpack_storage_info();
        return py::make_tuple(static_cast<std::int32_t>(storage.device.device_type),
                              storage.device.device_id);
    }

    py::object dlpack(py::object self, py::object stream, py::object max_version,
                      py::object dl_device, py::object copy) const {
        (void)stream;
        require_single_subtensor("DLPack export");
        if (!copy.is_none() && py::cast<bool>(copy)) {
            throw_buffer_error("DLPack copy exports are not supported");
        }

        DLPackStorageInfo storage = dlpack_storage_info();
        validate_dlpack_device_request(dl_device, storage.device);
        DLPackDTypeInfo dtype_info = dlpack_dtype_info(dtype());
        const bool versioned = should_export_versioned_dlpack(max_version);
        if (versioned) {
            return make_versioned_dlpack_capsule(std::move(self), storage, dtype_info);
        }
        return make_legacy_dlpack_capsule(std::move(self), storage, dtype_info);
    }

    void backward(py::object gradient, bool retain_graph) {
        require_single_subtensor("backward");
        require_differentiable(
            "backward is not available for non-differentiable tensors");
        py::object effective_gradient =
            normalize_backward_gradient(std::move(gradient));
        if (should_accumulate_grad()) {
            accumulate_grad(effective_gradient);
        }
        backwards_traversal(std::move(effective_gradient), autograd_ctx_, retain_graph);
    }

    static void backwards_traversal(py::object gradient, py::object operation,
                                    bool retain_graph) {
        if (operation.is_none()) {
            return;
        }

        TraversalGraph graph = discover_graph(operation);
        propagate_gradient(std::move(gradient), operation, graph, nullptr);
        if (!retain_graph) {
            release_graph(graph);
        }
    }

    static py::tuple functional_grad(py::object output, py::sequence inputs,
                                     py::object cotangents, bool batched,
                                     bool retain_graph) {
        if (!py::isinstance(output, tensor_type())) {
            throw py::type_error("grad output must be a Tensor");
        }
        Tensor& output_tensor = py::cast<Tensor&>(output);
        output_tensor.require_single_subtensor("grad");
        output_tensor.require_differentiable(
            "grad is not available for non-differentiable outputs");

        std::vector<py::object> requested_inputs;
        requested_inputs.reserve(static_cast<std::size_t>(py::len(inputs)));
        std::unordered_set<PyObject*> requested;
        for (py::handle input_handle : inputs) {
            if (!py::isinstance(input_handle, tensor_type())) {
                throw py::type_error("grad inputs must contain only Tensors");
            }
            py::object input = py::reinterpret_borrow<py::object>(input_handle);
            if (!py::cast<bool>(input.attr("is_differentiable")())) {
                throw py::value_error(
                    "grad inputs must contain only differentiable Tensors");
            }
            requested.insert(input.ptr());
            requested_inputs.push_back(std::move(input));
        }

        std::vector<py::object> cotangent_slices =
            output_tensor.normalize_functional_cotangents(cotangents, batched);
        py::object operation = output.attr("autograd_ctx");
        TraversalGraph graph = discover_graph(operation);

        std::vector<std::vector<py::object>> collected_by_input(
            requested_inputs.size());
        for (const py::object& cotangent_slice : cotangent_slices) {
            py::object cotangent = cotangent_slice;
            GradientMap collected =
                propagate_gradient(cotangent, operation, graph, &requested);
            if (requested.find(output.ptr()) != requested.end()) {
                collected.insert_or_assign(
                    output.ptr(), output_tensor.detached_gradient_copy(cotangent));
            }
            for (std::size_t i = 0; i < requested_inputs.size(); ++i) {
                auto found = collected.find(requested_inputs[i].ptr());
                if (found == collected.end()) {
                    collected_by_input[i].push_back(py::none());
                } else {
                    collected_by_input[i].push_back(found->second);
                }
            }
        }

        if (!retain_graph) {
            release_graph(graph);
        }

        py::tuple result(requested_inputs.size());
        for (std::size_t i = 0; i < requested_inputs.size(); ++i) {
            if (!batched || collected_by_input[i][0].is_none()) {
                result[i] = collected_by_input[i][0];
            } else {
                result[i] = stack_gradients(collected_by_input[i]);
            }
        }
        return result;
    }

private:
    using ConsumerMap = std::unordered_map<PyObject*, Index>;
    using GradientMap = std::unordered_map<PyObject*, py::object>;

    struct TraversalGraph {
        ConsumerMap consumers;
        std::vector<py::object> operations;
        std::vector<py::object> keepalive;
    };

    static TraversalGraph discover_graph(py::handle operation) {
        TraversalGraph graph;
        if (operation.is_none()) {
            return graph;
        }

        // Phase 1 is topology-only and is deliberately reusable across every
        // cotangent in a functional batched VJP.
        std::unordered_set<PyObject*> visited_operations;
        std::vector<py::object> operation_stack;
        py::object root = py::reinterpret_borrow<py::object>(operation);
        visited_operations.insert(root.ptr());
        graph.operations.push_back(root);
        operation_stack.push_back(std::move(root));
        while (!operation_stack.empty()) {
            py::object current = std::move(operation_stack.back());
            operation_stack.pop_back();
            py::sequence inputs =
                py::reinterpret_borrow<py::sequence>(current.attr("inputs")());
            std::unordered_set<PyObject*> seen_inputs;
            for (py::handle input_handle : inputs) {
                py::object input = py::reinterpret_borrow<py::object>(input_handle);
                if (!seen_inputs.insert(input.ptr()).second ||
                    !py::cast<bool>(input.attr("is_differentiable")())) {
                    continue;
                }
                ++graph.consumers[input.ptr()];
                graph.keepalive.push_back(input);
                py::object input_ctx = input.attr("autograd_ctx");
                if (!input_ctx.is_none() &&
                    visited_operations.insert(input_ctx.ptr()).second) {
                    graph.operations.push_back(input_ctx);
                    operation_stack.push_back(std::move(input_ctx));
                }
            }
            graph.keepalive.push_back(std::move(current));
        }
        return graph;
    }

    static GradientMap
    propagate_gradient(py::object gradient, py::handle operation,
                       const TraversalGraph& graph,
                       const std::unordered_set<PyObject*>* collector) {
        GradientMap collected;
        if (operation.is_none()) {
            return collected;
        }

        ConsumerMap remaining_consumers = graph.consumers;
        GradientMap pending_gradients;
        std::vector<std::pair<py::object, py::object>> ready;
        ready.emplace_back(py::reinterpret_borrow<py::object>(operation),
                           std::move(gradient));
        while (!ready.empty()) {
            auto [current, current_gradient] = std::move(ready.back());
            ready.pop_back();

            py::sequence inputs =
                py::reinterpret_borrow<py::sequence>(current.attr("inputs")());
            if (!current_gradient.is_none()) {
                if (py::cast<bool>(current.attr("_autograd_state_freed"))) {
                    throw std::runtime_error(
                        "Trying to backward through the graph a second time "
                        "after its saved tensors have already been freed. "
                        "Specify retain_graph=True if you need to backward "
                        "through the graph a second time.");
                }
                current.attr("validate_input_versions")();
                py::object input_gradients_object =
                    current.attr("backward")(current_gradient);
                py::sequence input_gradients =
                    py::reinterpret_borrow<py::sequence>(input_gradients_object);
                if (py::len(input_gradients) != py::len(inputs)) {
                    throw py::value_error(
                        "Operation backward returned wrong number of gradients");
                }

                for (std::size_t i = 0; i < py::len(inputs); ++i) {
                    py::object input = py::reinterpret_borrow<py::object>(inputs[i]);
                    py::object input_gradient =
                        py::reinterpret_borrow<py::object>(input_gradients[i]);
                    if (input_gradient.is_none() ||
                        !py::cast<bool>(input.attr("is_differentiable")())) {
                        continue;
                    }
                    Tensor& input_tensor = py::cast<Tensor&>(input);
                    input_tensor.validate_gradient(input_gradient);
                    auto found = pending_gradients.find(input.ptr());
                    if (found == pending_gradients.end()) {
                        pending_gradients.emplace(input.ptr(),
                                                  std::move(input_gradient));
                    } else {
                        found->second = input_tensor.combined_gradient(found->second,
                                                                       input_gradient);
                    }
                }
            }

            std::unordered_set<PyObject*> seen_inputs;
            for (py::handle input_handle : inputs) {
                py::object input = py::reinterpret_borrow<py::object>(input_handle);
                if (!seen_inputs.insert(input.ptr()).second) {
                    continue;
                }
                auto consumer = remaining_consumers.find(input.ptr());
                if (consumer == remaining_consumers.end() || --consumer->second != 0) {
                    continue;
                }

                py::object total_gradient = py::none();
                auto found = pending_gradients.find(input.ptr());
                if (found != pending_gradients.end()) {
                    total_gradient = std::move(found->second);
                    pending_gradients.erase(found);
                }

                Tensor& input_tensor = py::cast<Tensor&>(input);
                if (collector != nullptr) {
                    if (!total_gradient.is_none() &&
                        collector->find(input.ptr()) != collector->end()) {
                        collected.insert_or_assign(
                            input.ptr(),
                            input_tensor.detached_gradient_copy(total_gradient));
                    }
                } else if (!total_gradient.is_none() &&
                           input_tensor.should_accumulate_grad()) {
                    input_tensor.accumulate_grad(total_gradient);
                }
                py::object input_ctx = input.attr("autograd_ctx");
                if (!input_ctx.is_none()) {
                    ready.emplace_back(std::move(input_ctx), std::move(total_gradient));
                }
            }
        }
        return collected;
    }

    static void release_graph(const TraversalGraph& graph) {
        for (py::handle operation : graph.operations) {
            operation.attr("_release_autograd_state")();
        }
    }

    std::vector<py::object> normalize_functional_cotangents(py::handle cotangents,
                                                            bool batched) const {
        if (!py::isinstance(cotangents, tensor_type())) {
            throw py::type_error("grad cotangents must be a Tensor");
        }
        py::object cotangent = py::reinterpret_borrow<py::object>(cotangents);
        if (!batched) {
            validate_gradient(cotangent);
            return {std::move(cotangent)};
        }

        constexpr const char* layout_error =
            "A batched cotangent layout must contain one prepended leaf batch "
            "mode followed by the output layout exactly";
        py::object cotangent_layout = cotangent.attr("layout");
        const std::size_t output_modes = static_cast<std::size_t>(py::len(layout()));
        if (static_cast<std::size_t>(py::len(cotangent_layout)) != output_modes + 1U) {
            throw py::value_error(layout_error);
        }
        py::object batch_layout = cotangent_layout.attr("__getitem__")(py::int_(0));
        if (!py::cast<bool>(batch_layout.attr("is_leaf"))) {
            throw py::value_error(layout_error);
        }
        for (std::size_t i = 0; i < output_modes; ++i) {
            py::object cotangent_mode = cotangent_layout.attr("__getitem__")(
                py::int_(static_cast<Index>(i + 1U)));
            py::object output_mode =
                layout().attr("__getitem__")(py::int_(static_cast<Index>(i)));
            if (!layouts_equal(cotangent_mode, output_mode)) {
                throw py::value_error(layout_error);
            }
        }

        const Index batch_extent =
            py::cast<Index>(batch_layout.attr("shape").attr("__int__")());
        const Index batch_stride =
            py::cast<Index>(batch_layout.attr("stride").attr("__int__")());
        const Index cotangent_offset = py::cast<Index>(cotangent.attr("offset"));
        py::object cotangent_carrier = cotangent.attr("carrier");
        std::vector<py::object> slices;
        slices.reserve(static_cast<std::size_t>(batch_extent));
        for (Index i = 0; i < batch_extent; ++i) {
            py::object slice =
                tensor_type()(cotangent_carrier,
                              py::int_(cotangent_offset + i * batch_stride), layout());
            validate_gradient(slice);
            slices.push_back(std::move(slice));
        }
        return slices;
    }

    static py::object stack_gradients(const std::vector<py::object>& gradients) {
        py::object first = gradients.front();
        py::object slice_layout = first.attr("layout");
        const Index slice_cosize = strideweave::layout_index::cosize(slice_layout);
        py::list values;
        for (const py::object& gradient : gradients) {
            if (gradient.is_none() ||
                !layouts_equal(slice_layout, gradient.attr("layout"))) {
                throw std::runtime_error(
                    "Batched gradient slices must share one layout");
            }
            py::object carrier = gradient.attr("carrier");
            const Index offset = py::cast<Index>(gradient.attr("offset"));
            for (Index i = 0; i < slice_cosize; ++i) {
                values.append(carrier.attr("__getitem__")(py::int_(offset + i)));
            }
        }

        py::object carrier = first.attr("carrier").attr("new_like")(values);
        py::object layout_type =
            py::module_::import("strideweave.layout").attr("Layout");
        py::object layout_module = py::module_::import("strideweave.layout");
        py::object batch_layout = layout_type(
            layout_module.attr("Shape")(py::int_(static_cast<Index>(gradients.size()))),
            layout_module.attr("Stride")(py::int_(slice_cosize)));
        py::object stacked_layout =
            layout_type.attr("concat")(batch_layout, slice_layout);
        return tensor_type()(std::move(carrier), py::int_(0),
                             std::move(stacked_layout));
    }

    void require_differentiable(const char* message) const {
        if (!is_differentiable()) {
            throw std::runtime_error(message);
        }
    }

    Index carrier_index(py::object key) const {
        const Index layout_index = strideweave::layout_index::get_index(layout(), key);
        return offset() + layout_index;
    }

    DLPackStorageInfo dlpack_storage_info() const {
        py::dict info = py::cast<py::dict>(carrier().attr("dlpack_info")());
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

    void validate_dlpack_device_request(py::handle requested_device,
                                        DLDevice actual_device) const {
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
                "DLPack cross-device exports are not supported for this tensor");
        }
    }

    void populate_dlpack_tensor(DLTensor& dl_tensor, std::vector<std::int64_t>& shape,
                                std::vector<std::int64_t>& strides,
                                DLPackStorageInfo storage,
                                DLPackDTypeInfo dtype_info) const {
        const strideweave::layout_index::LayoutCache& cache =
            strideweave::layout_index::cache_from_layout(layout());
        shape = to_int64_vector(cache.leaf_shapes());
        strides = to_int64_vector(cache.leaf_strides());

        dl_tensor.data = reinterpret_cast<void*>(storage.pointer);
        dl_tensor.device = storage.device;
        dl_tensor.ndim = static_cast<std::int32_t>(shape.size());
        dl_tensor.dtype = dtype_info.dtype;
        dl_tensor.shape = shape.empty() ? nullptr : shape.data();
        dl_tensor.strides = strides.empty() ? nullptr : strides.data();
        dl_tensor.byte_offset = byte_offset_for(offset(), dtype_info.item_size);
    }

    py::object make_legacy_dlpack_capsule(py::object self, DLPackStorageInfo storage,
                                          DLPackDTypeInfo dtype_info) const {
        auto holder = std::make_unique<LegacyDLPackTensor>();
        populate_dlpack_tensor(holder->managed.dl_tensor, holder->shape,
                               holder->strides, storage, dtype_info);
        holder->managed.manager_ctx = holder.get();
        holder->managed.deleter = legacy_dlpack_managed_deleter;
        holder->owner = self.ptr();
        Py_INCREF(holder->owner);

        DLManagedTensor* managed = &holder->managed;
        holder.release();
        return py::capsule(managed, dlpack_capsule_name, legacy_dlpack_capsule_deleter);
    }

    py::object make_versioned_dlpack_capsule(py::object self, DLPackStorageInfo storage,
                                             DLPackDTypeInfo dtype_info) const {
        auto holder = std::make_unique<VersionedDLPackTensor>();
        populate_dlpack_tensor(holder->managed.dl_tensor, holder->shape,
                               holder->strides, storage, dtype_info);
        holder->managed.version = {1, 0};
        holder->managed.manager_ctx = holder.get();
        holder->managed.deleter = versioned_dlpack_managed_deleter;
        holder->managed.flags = is_mutable() ? 0 : dlpack_flag_read_only;
        holder->owner = self.ptr();
        Py_INCREF(holder->owner);

        DLManagedTensorVersioned* managed = &holder->managed;
        holder.release();
        return py::capsule(managed, versioned_dlpack_capsule_name,
                           versioned_dlpack_capsule_deleter);
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
                "Tensor.backward requires a gradient for non-scalar tensors");
        }

        py::list values;
        values.append(py::int_(1));
        py::object grad_carrier = carrier().attr("new_like")(values);
        return tensor_type()(grad_carrier, py::int_(0), layout());
    }

    bool is_scalar() const {
        py::object tensor_layout = layout();
        return py::len(tensor_layout) == 1 &&
               py::cast<bool>(tensor_layout.attr("is_leaf")) && size() == 1;
    }

    void validate_gradient(py::handle gradient) const {
        if (!py::isinstance(gradient, tensor_type())) {
            throw py::type_error("Tensor.backward requires a Tensor gradient");
        }
        py::object gradient_layout = gradient.attr("layout");
        const bool target_is_injective = py::cast<bool>(layout().attr("is_injective"));
        if (target_is_injective && !layouts_equal(layout(), gradient_layout)) {
            throw py::value_error("Tensor gradient layout must match tensor layout");
        }
        if (target_is_injective) {
            return;
        }
        if (!py::cast<bool>(layout().attr("_has_only_broadcast_aliasing"))) {
            throw py::value_error(
                "Autograd does not support non-injective layouts whose "
                "aliasing is not caused only by stride-zero broadcast modes");
        }
        if (!layouts_equal(layout().attr("shape"), gradient_layout.attr("shape")) ||
            !py::cast<bool>(gradient_layout.attr("is_injective"))) {
            throw py::value_error(
                "A gradient for a broadcast tensor must have the same shape "
                "in an injective layout");
        }
    }

    py::object copy_gradient_storage(py::handle gradient,
                                     bool aggregate_aliases) const {
        validate_gradient(gradient);

        const bool target_is_injective = py::cast<bool>(layout().attr("is_injective"));
        py::object output_layout = layout();
        if (!target_is_injective) {
            output_layout = py::reinterpret_borrow<py::object>(gradient.attr("layout"));
        }
        const Index storage_size = strideweave::layout_index::cosize(output_layout);
        py::list values;
        for (Index i = 0; i < storage_size; ++i) {
            values.append(py::int_(0));
        }

        const Index tensor_size = size();
        if (aggregate_aliases && !target_is_injective) {
            std::unordered_map<Index, py::object> totals;
            for (Index i = 0; i < tensor_size; ++i) {
                py::object key = py::int_(i);
                const Index target_index =
                    strideweave::layout_index::get_index(layout(), key);
                py::object contribution = gradient.attr("__getitem__")(key);
                auto found = totals.find(target_index);
                if (found == totals.end()) {
                    totals.emplace(target_index, std::move(contribution));
                } else {
                    found->second = add_python_objects(found->second, contribution);
                }
            }
            for (Index i = 0; i < tensor_size; ++i) {
                py::object key = py::int_(i);
                const Index target_index =
                    strideweave::layout_index::get_index(layout(), key);
                const Index output_index =
                    strideweave::layout_index::get_index(output_layout, key);
                values[strideweave::layout_index::as_size(output_index)] =
                    totals.at(target_index);
            }
        } else {
            for (Index i = 0; i < tensor_size; ++i) {
                py::object key = py::int_(i);
                const Index output_index =
                    strideweave::layout_index::get_index(output_layout, key);
                values[strideweave::layout_index::as_size(output_index)] =
                    gradient.attr("__getitem__")(key);
            }
        }

        py::object grad_carrier = carrier().attr("new_like")(values);
        return tensor_type()(grad_carrier, py::int_(0), output_layout);
    }

    py::object detached_gradient_copy(py::handle gradient) const {
        return copy_gradient_storage(gradient, true);
    }

    py::object detached_logical_gradient_copy(py::handle gradient) const {
        return copy_gradient_storage(gradient, false);
    }

    py::object combined_gradient(py::handle accumulated, py::handle addition) const {
        py::object combined = detached_logical_gradient_copy(accumulated);
        for (Index i = 0; i < size(); ++i) {
            py::object key = py::int_(i);
            py::object combined_value = add_python_objects(
                combined.attr("__getitem__")(key), addition.attr("__getitem__")(key));
            combined.attr("__setitem__")(key, combined_value);
        }
        return combined;
    }

    void accumulate_grad(py::handle gradient) {
        py::object contribution = detached_gradient_copy(gradient);
        if (grad_.is_none()) {
            grad_ = std::move(contribution);
            return;
        }

        for (Index i = 0; i < size(); ++i) {
            py::object key = py::int_(i);
            py::object accumulated_value = add_python_objects(
                grad_.attr("__getitem__")(key), contribution.attr("__getitem__")(key));
            grad_.attr("__setitem__")(key, accumulated_value);
        }
    }

    bool should_accumulate_grad() const {
        return autograd_ctx_.is_none() || retain_grad_;
    }

    py::object representation_;
    py::object autograd_ctx_;
    py::object grad_;
    bool retain_grad_;
};

}  // namespace

PYBIND11_MODULE(_tensor, module) {
    module.doc() = "Native tensor type for StrideWeave";

    py::class_<Tensor>(module, "Tensor")
        .def(py::init<py::object, Index, py::object>(), py::arg("carrier"),
             py::arg("offset"), py::arg("layout"))
        .def_static("_from_representation", &Tensor::from_representation,
                    py::arg("representation"))
        .def_property_readonly("_representation", &Tensor::representation)
        .def_property_readonly("carrier", &Tensor::carrier)
        .def_property_readonly("offset", &Tensor::offset)
        .def_property_readonly("layout", &Tensor::layout)
        .def_property_readonly("version", &Tensor::version)
        .def("_version_token", &Tensor::version_token)
        .def("_require_single_subtensor", &Tensor::require_single_subtensor,
             py::arg("reason"))
        .def_property("autograd_ctx", &Tensor::autograd_ctx, &Tensor::set_autograd_ctx)
        .def_property("grad", &Tensor::grad, &Tensor::set_grad)
        .def("retain_grad", &Tensor::retain_grad, py::arg("retain") = true)
        .def(
            "__getitem__",
            [](py::object self, py::object key) {
                if (contains_slice(key)) {
                    return py::module_::import("strideweave.functional.api")
                        .attr("_view")(self, key);
                }
                Tensor& tensor = py::cast<Tensor&>(self);
                return tensor.get_item(key);
            },
            py::arg("key"))
        .def("__setitem__", &Tensor::set_item, py::arg("key"), py::arg("value"))
        .def(
            "__add__",
            [](py::object self, py::object other) {
                return py::module_::import("strideweave.operation")
                    .attr("add")(self, other);
            },
            py::is_operator())
        .def(
            "__sub__",
            [](py::object self, py::object other) {
                return py::module_::import("strideweave.operation")
                    .attr("sub")(self, other);
            },
            py::is_operator())
        .def(
            "__neg__",
            [](py::object self) {
                return py::module_::import("strideweave.operation").attr("neg")(self);
            },
            py::is_operator())
        .def(
            "__mul__",
            [](py::object self, py::object other) {
                return py::module_::import("strideweave.operation")
                    .attr("mul")(self, other);
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](py::object self, py::object other) {
                return py::module_::import("strideweave.operation")
                    .attr("mul")(self, other);
            },
            py::is_operator())
        .def(
            "__truediv__",
            [](py::object self, py::object other) {
                return py::module_::import("strideweave.operation")
                    .attr("div")(self, other);
            },
            py::is_operator())
        .def(
            "__pow__",
            [](py::object self, py::object exponent) {
                return py::module_::import("strideweave.operation")
                    .attr("pow")(self, exponent);
            },
            py::is_operator())
        .def(
            "__matmul__",
            [](py::object self, py::object other) {
                return py::module_::import("strideweave.operation")
                    .attr("matmul")(self, other);
            },
            py::is_operator())
        .def("size", &Tensor::size)
        .def("is_mutable", &Tensor::is_mutable)
        .def("dtype", &Tensor::dtype)
        .def("is_differentiable", &Tensor::is_differentiable)
        .def("carrier_type", &Tensor::carrier_type)
        .def("__dlpack_device__", &Tensor::dlpack_device)
        .def(
            "__dlpack__",
            [](py::object self, py::object stream, py::object max_version,
               py::object dl_device, py::object copy) {
                const Tensor& tensor = py::cast<const Tensor&>(self);
                return tensor.dlpack(std::move(self), std::move(stream),
                                     std::move(max_version), std::move(dl_device),
                                     std::move(copy));
            },
            py::arg("stream") = py::none(), py::kw_only(),
            py::arg("max_version") = py::none(), py::arg("dl_device") = py::none(),
            py::arg("copy") = py::none())
        .def("backward", &Tensor::backward, py::arg("gradient") = py::none(),
             py::arg("retain_graph") = false)
        .def_static("backwards_traversal", &Tensor::backwards_traversal,
                    py::arg("gradient"), py::arg("operation"),
                    py::arg("retain_graph") = false)
        .def_static("_functional_grad", &Tensor::functional_grad, py::arg("output"),
                    py::arg("inputs"), py::arg("cotangents"), py::kw_only(),
                    py::arg("batched") = false, py::arg("retain_graph") = false);
}
