#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "_data.hpp"
#include "_layout_index.hpp"

namespace py = pybind11;

namespace neotorch::data {
namespace {

py::object tensor_type() {
    return py::module_::import("neotorch.tensor").attr("Tensor");
}

bool layouts_equal(py::handle left, py::handle right) {
    const int result = PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_EQ);
    if (result < 0) {
        throw py::error_already_set();
    }
    return result == 1;
}

class CPU : public Data {
public:
    CPU(Index size, py::object pointer, bool is_mutable)
        : size_(size), is_mutable_(is_mutable) {
        if (size_ < 0) {
            throw py::value_error("CPU size must be non-negative");
        }

        if (pointer.is_none()) {
            owned_data_ = std::make_unique<float[]>(static_cast<std::size_t>(size_));
            data_ = reinterpret_cast<std::byte*>(owned_data_.get());
            std::fill_n(data_as<float>(), static_cast<std::size_t>(size_), 0.0f);
            return;
        }

        if (!py::isinstance<py::int_>(pointer)) {
            throw py::type_error("CPU pointer must be an integer address or None");
        }

        const auto pointer_value = py::cast<long long>(pointer);
        if (pointer_value <= 0) {
            throw py::value_error("CPU pointer must be a positive integer address");
        }
        data_ = reinterpret_cast<std::byte*>(
            static_cast<std::uintptr_t>(pointer_value)
        );
    }

    Index size() const override { return size_; }

    py::object type() const override {
        return py::module_::import("neotorch.data").attr("DataType").attr("Float32");
    }

    py::object get_value(Index index) const override {
        return py::float_(value_at(index));
    }

    py::object new_like(py::iterable values, bool is_mutable) const override {
        py::list materialized_values(values);
        py::object cpu_type = py::module_::import("neotorch._data").attr("CPU");
        py::object result = cpu_type(
            py::int_(py::len(materialized_values)),
            py::none(),
            py::arg("mutable") = is_mutable
        );
        CPU& result_data = py::cast<CPU&>(result);
        for (py::ssize_t i = 0; i < py::len(materialized_values); ++i) {
            if (materialized_values[i].is_none()) {
                result_data.element<float>(static_cast<Index>(i)) = 0.0f;
                continue;
            }
            result_data.element<float>(static_cast<Index>(i)) =
                py::cast<float>(materialized_values[i]);
        }
        return result;
    }

    void scatter(
        py::object to_scatter,
        py::object scatter_onto,
        py::object mapping,
        Index mapping_offset
    ) override;

    bool is_mutable() const override { return is_mutable_; }

    std::uintptr_t pointer() const {
        return reinterpret_cast<std::uintptr_t>(data_);
    }

    void set_value_public(Index index, py::object value) {
        if (!is_mutable()) {
            throw std::runtime_error("Data is not mutable");
        }
        set_value_at(index, py::cast<float>(value));
    }

    float value_at(Index index) const {
        return element<float>(index);
    }

    void set_value_at(Index index, float value) {
        element<float>(index) = value;
    }

    template <typename T>
    T* data_as() {
        return reinterpret_cast<T*>(data_);
    }

    template <typename T>
    const T* data_as() const {
        return reinterpret_cast<const T*>(data_);
    }

    template <typename T>
    T& element(Index index) {
        validate_index(index);
        return unchecked_element<T>(index);
    }

    template <typename T>
    const T& element(Index index) const {
        validate_index(index);
        return unchecked_element<T>(index);
    }

    template <typename T>
    T& unchecked_element(Index index) {
        return *reinterpret_cast<T*>(data_ + index * static_cast<Index>(sizeof(T)));
    }

    template <typename T>
    const T& unchecked_element(Index index) const {
        return *reinterpret_cast<const T*>(
            data_ + index * static_cast<Index>(sizeof(T))
        );
    }

    static py::object dispatch_op(const std::string& operation_name) {
        py::object native_data = py::module_::import("neotorch._data");
        if (operation_name == "add") {
            return native_data.attr("_CPUAddOperation")();
        }
        if (operation_name == "matmul") {
            return native_data.attr("_CPUMatmulOperation")();
        }
        if (operation_name == "mul") {
            return native_data.attr("_CPUScalarMulOperation")();
        }
        if (operation_name == "reduce") {
            return native_data.attr("_CPUReduceSumOperation")();
        }
        if (operation_name == "permute") {
            return py::module_::import("neotorch.operation").attr("PermuteOperation")();
        }
        if (operation_name == "rearrange") {
            return py::module_::import("neotorch.operation").attr(
                "RearrangeOperation"
            )();
        }
        if (operation_name == "view") {
            return py::module_::import("neotorch.operation").attr(
                "GenericViewOperation"
            )();
        }

        PyErr_Format(
            PyExc_NotImplementedError,
            "CPU data does not support operation '%s'",
            operation_name.c_str()
        );
        throw py::error_already_set();
    }

protected:
    void set_value(Index index, py::object value) override {
        set_value_at(index, py::cast<float>(value));
    }

private:
    void validate_index(Index index) const {
        if (index < 0 || index >= size_) {
            throw py::index_error("Data index out of range");
        }
    }

    Index size_;
    bool is_mutable_;
    std::unique_ptr<float[]> owned_data_;
    std::byte* data_ = nullptr;
};

float require_float(py::handle value, const char* name) {
    PyObject* value_ptr = value.ptr();
    const double result = PyFloat_AsDouble(value_ptr);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw py::type_error(std::string(name) + " must be a numerical scalar");
    }
    return static_cast<float>(result);
}

CPU& cpu_data_from_tensor(py::handle tensor, const char* name) {
    if (!py::isinstance(tensor, tensor_type())) {
        throw py::type_error(std::string(name) + " must be a Tensor");
    }
    py::object data = tensor.attr("data");
    if (!py::isinstance<CPU>(data)) {
        throw py::type_error(std::string(name) + " must be backed by CPU data");
    }
    if (py::cast<bool>(data.attr("is_evicted")())) {
        throw std::runtime_error(std::string(name) + " data is evicted");
    }
    return py::cast<CPU&>(data);
}

Index tensor_offset(py::handle tensor) {
    return py::cast<Index>(tensor.attr("offset"));
}

py::object tensor_layout(py::handle tensor) { return tensor.attr("layout"); }

Index tensor_size(py::handle tensor) {
    return py::cast<Index>(tensor.attr("size")());
}

template <typename T>
struct CpuTensorView {
    CPU* data;
    Index offset;
    py::object cache_owner;
    const neotorch::layout_index::LayoutCache* cache;
    Index logical_size;

    std::size_t leaf_rank() const {
        return static_cast<std::size_t>(cache->leaf_rank());
    }

    T read_logical(Index key) const {
        const Index layout_index = cache->index_logical(key);
        return data->unchecked_element<T>(offset + layout_index);
    }

    void write_logical(Index key, T value) const {
        const Index layout_index = cache->index_logical(key);
        data->unchecked_element<T>(offset + layout_index) = value;
    }

    T read_coords(const Index* key, std::size_t size) const {
        const Index layout_index = cache->index(key, size);
        return data->unchecked_element<T>(offset + layout_index);
    }

    void write_coords(const Index* key, std::size_t size, T value) const {
        const Index layout_index = cache->index(key, size);
        data->unchecked_element<T>(offset + layout_index) = value;
    }

    T read_expanded(const Index* key, std::size_t size) const {
        const Index layout_index = cache->index_expanded(key, size);
        return data->unchecked_element<T>(offset + layout_index);
    }

    T read_expanded(const std::vector<Index>& key) const {
        return read_expanded(key.data(), key.size());
    }

    void write_expanded(const Index* key, std::size_t size, T value) const {
        const Index layout_index = cache->index_expanded(key, size);
        data->unchecked_element<T>(offset + layout_index) = value;
    }

    void write_expanded(const std::vector<Index>& key, T value) const {
        write_expanded(key.data(), key.size(), value);
    }
};

template <typename T>
CpuTensorView<T> cpu_tensor_view(py::handle tensor, const char* name) {
    CPU& data = cpu_data_from_tensor(tensor, name);
    py::object layout = tensor_layout(tensor);
    py::object cache_owner = layout.attr("_cache");
    auto& cache = py::cast<neotorch::layout_index::LayoutCache&>(cache_owner);
    return CpuTensorView<T>{
        &data,
        tensor_offset(tensor),
        std::move(cache_owner),
        &cache,
        cache.logical_size(),
    };
}

void CPU::scatter(
    py::object to_scatter,
    py::object scatter_onto,
    py::object mapping,
    Index mapping_offset
) {
    if (mapping_offset < 0) {
        throw py::value_error("mapping_offset must be non-negative");
    }
    CPU& destination_data = cpu_data_from_tensor(scatter_onto, "scatter_onto");
    if (&destination_data != this) {
        throw py::value_error("scatter_onto must be backed by this data object");
    }
    if (!is_mutable()) {
        throw std::runtime_error("Data is not mutable");
    }

    CpuTensorView<float> source = cpu_tensor_view<float>(to_scatter, "to_scatter");
    py::object source_layout = tensor_layout(to_scatter);
    py::object source_shape = source_layout.attr("shape");
    py::object mapping_shape = mapping.attr("shape");
    if (!layouts_equal(mapping_shape, source_shape)) {
        throw py::value_error("mapping shape must match to_scatter layout shape");
    }

    py::object mapping_cache_owner = mapping.attr("_cache");
    auto& mapping_cache =
        py::cast<neotorch::layout_index::LayoutCache&>(mapping_cache_owner);
    const Index destination_offset = tensor_offset(scatter_onto);
    const Index mapped_storage_size = mapping_cache.cosize();
    const bool storage_exceeds_data =
        destination_offset > size_ ||
        mapping_offset > size_ - destination_offset ||
        mapped_storage_size > size_ - destination_offset - mapping_offset;
    if (storage_exceeds_data) {
        throw py::value_error("scatter mapping exceeds destination data size");
    }

    {
        py::gil_scoped_release release;
        std::vector<Index> key(mapping_cache.leaf_rank(), 0);
        for (Index i = 0; i < mapping_cache.logical_size(); ++i) {
            const Index source_index = source.cache->index_expanded(
                key.data(), key.size()
            );
            const Index destination_index =
                destination_offset + mapping_offset +
                mapping_cache.index_expanded(key.data(), key.size());
            unchecked_element<float>(destination_index) =
                source.data->unchecked_element<float>(source.offset + source_index);
            mapping_cache.increment_key(key.data(), key.size());
        }
    }
}

void require_same_layout(py::handle lhs, py::handle rhs) {
    if (!layouts_equal(tensor_layout(lhs), tensor_layout(rhs))) {
        throw py::value_error("Tensor layouts must match");
    }
}

void require_layout(py::handle tensor, py::handle layout) {
    if (!layouts_equal(tensor_layout(tensor), layout)) {
        throw py::value_error("Tensor layouts must match");
    }
}

py::object mode_shape(py::handle layout, Index mode) {
    return layout.attr("shape").attr("top_level")[py::int_(mode)];
}

Index mode_logical_size(py::handle layout, Index mode) {
    py::object shape = mode_shape(layout, mode);
    if (neotorch::layout_index::is_int(shape)) {
        return py::cast<Index>(shape);
    }
    return py::cast<Index>(shape.attr("logical_size"));
}

void require_two_mode_tensor(py::handle tensor, const char* name) {
    cpu_data_from_tensor(tensor, name);
    if (py::len(tensor_layout(tensor)) != 2) {
        throw py::value_error(std::string(name) + " must have a two-mode layout");
    }
}

std::pair<py::object, Index> canonical_stride_level(
    py::handle shape_level, Index stride
) {
    if (neotorch::layout_index::is_int(shape_level)) {
        return {py::int_(stride), stride * py::cast<Index>(shape_level)};
    }

    py::list stride_level;
    Index next_stride = stride;
    for (py::handle shape : py::reinterpret_borrow<py::sequence>(shape_level)) {
        auto [child_stride, child_next_stride] =
            canonical_stride_level(shape, next_stride);
        stride_level.append(child_stride);
        next_stride = child_next_stride;
    }
    return {std::move(stride_level), next_stride};
}

py::object canonical_layout_from_modes(std::initializer_list<py::object> modes) {
    py::object layout_module = py::module_::import("neotorch.layout");
    py::object shape_type = layout_module.attr("Shape");
    py::object stride_type = layout_module.attr("Stride");
    py::object layout_type = layout_module.attr("Layout");

    py::object shape;
    if (modes.size() == 1) {
        shape = shape_type(*modes.begin());
    } else {
        py::list mode_list;
        for (const py::object& mode : modes) {
            mode_list.append(mode);
        }
        shape = shape_type(mode_list);
    }

    auto [stride_level, _] = canonical_stride_level(shape.attr("top_level"), 1);
    return layout_type(shape, stride_type(stride_level));
}

py::object make_cpu_data(Index size) {
    py::object cpu_type = py::module_::import("neotorch._data").attr("CPU");
    return cpu_type(py::int_(size));
}

py::object make_tensor(py::object data, py::object layout) {
    return tensor_type()(std::move(data), py::int_(0), std::move(layout));
}

template <typename T>
struct CpuTensorAllocation {
    py::object data_object;
    py::object layout_object;
    CpuTensorView<T> view;
};

template <typename T>
CpuTensorAllocation<T> allocate_cpu_tensor(py::object layout) {
    py::object cache_owner = layout.attr("_cache");
    auto& cache = py::cast<neotorch::layout_index::LayoutCache&>(cache_owner);
    py::object data = make_cpu_data(cache.cosize());
    CPU& cpu = py::cast<CPU&>(data);
    return CpuTensorAllocation<T>{
        std::move(data),
        std::move(layout),
        CpuTensorView<T>{&cpu, 0, std::move(cache_owner), &cache, cache.logical_size()},
    };
}

template <typename T>
py::object copy_gradient_for(py::handle target, py::handle gradient) {
    require_same_layout(target, gradient);
    CpuTensorView<T> gradient_view = cpu_tensor_view<T>(gradient, "gradient");
    py::object output_layout = tensor_layout(target);
    CpuTensorAllocation<T> output = allocate_cpu_tensor<T>(output_layout);
    {
        py::gil_scoped_release release;
        std::vector<Index> key(output.view.leaf_rank(), 0);
        for (Index i = 0; i < output.view.logical_size; ++i) {
            output.view.write_expanded(key, gradient_view.read_expanded(key));
            output.view.cache->increment_key(key.data(), key.size());
        }
    }
    return make_tensor(std::move(output.data_object), std::move(output.layout_object));
}

class CpuOperation {
public:
    CpuOperation() : ctx_(py::dict()), inputs_(py::tuple()) {}

    py::dict ctx() const { return ctx_; }

    py::tuple inputs() const { return inputs_; }

protected:
    bool begin_forward(py::args inputs) {
        const bool build_autograd_graph = py::cast<bool>(
            py::module_::import("neotorch.operation").attr("is_grad_enabled")()
        );
        if (!build_autograd_graph) {
            inputs_ = py::tuple();
            return false;
        }

        py::object tensor = tensor_type();
        py::ssize_t tensor_count = 0;
        for (py::handle input : inputs) {
            if (py::isinstance(input, tensor)) {
                ++tensor_count;
            }
        }

        py::tuple stored(tensor_count);
        py::ssize_t stored_index = 0;
        for (py::handle input : inputs) {
            if (py::isinstance(input, tensor)) {
                stored[stored_index] = py::reinterpret_borrow<py::object>(input);
                ++stored_index;
            }
        }
        inputs_ = std::move(stored);
        return true;
    }

    template <typename Operation>
    py::object finish_forward(Operation* operation, py::object result, bool build_graph) {
        if (!py::isinstance(result, tensor_type())) {
            throw py::type_error("CPU operation forward must return a Tensor");
        }
        if (build_graph) {
            result.attr("autograd_ctx") =
                py::cast(operation, py::return_value_policy::reference);
        }
        return result;
    }

    py::dict ctx_;

private:
    py::tuple inputs_;
};

class CpuAddOperation : public CpuOperation {
public:
    py::object forward(py::args inputs) {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU add requires lhs and rhs tensors");
        }
        const bool build_graph = begin_forward(inputs);
        py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(inputs[1]);
        CpuTensorView<float> lhs_view = cpu_tensor_view<float>(lhs, "lhs");
        CpuTensorView<float> rhs_view = cpu_tensor_view<float>(rhs, "rhs");
        require_same_layout(lhs, rhs);

        CpuTensorAllocation<float> result =
            allocate_cpu_tensor<float>(tensor_layout(lhs));
        {
            py::gil_scoped_release release;
            std::vector<Index> key(lhs_view.leaf_rank(), 0);
            for (Index i = 0; i < lhs_view.logical_size; ++i) {
                result.view.write_expanded(
                    key,
                    lhs_view.read_expanded(key) + rhs_view.read_expanded(key)
                );
                lhs_view.cache->increment_key(key.data(), key.size());
            }
        }
        return finish_forward(
            this,
            make_tensor(std::move(result.data_object), std::move(result.layout_object)),
            build_graph
        );
    }

    py::object backward(py::object gradient) {
        py::tuple input_tensors = inputs();
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
        return py::make_tuple(
            copy_gradient_for<float>(lhs, gradient),
            copy_gradient_for<float>(rhs, gradient)
        );
    }
};

class CpuScalarMulOperation : public CpuOperation {
public:
    py::object forward(py::args inputs) {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU scalar multiply requires a tensor and scalar");
        }
        const bool build_graph = begin_forward(inputs);
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        CpuTensorView<float> tensor_view = cpu_tensor_view<float>(tensor, "tensor");
        scalar_ = require_float(inputs[1], "scalar");
        ctx_["scalar"] = py::float_(scalar_);

        CpuTensorAllocation<float> result =
            allocate_cpu_tensor<float>(tensor_layout(tensor));
        {
            py::gil_scoped_release release;
            std::vector<Index> key(tensor_view.leaf_rank(), 0);
            for (Index i = 0; i < tensor_view.logical_size; ++i) {
                result.view.write_expanded(
                    key, tensor_view.read_expanded(key) * scalar_
                );
                tensor_view.cache->increment_key(key.data(), key.size());
            }
        }
        return finish_forward(
            this,
            make_tensor(std::move(result.data_object), std::move(result.layout_object)),
            build_graph
        );
    }

    py::object backward(py::object gradient) {
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_same_layout(tensor, gradient);
        CpuTensorView<float> gradient_view =
            cpu_tensor_view<float>(gradient, "gradient");

        CpuTensorAllocation<float> result =
            allocate_cpu_tensor<float>(tensor_layout(tensor));
        {
            py::gil_scoped_release release;
            std::vector<Index> key(gradient_view.leaf_rank(), 0);
            for (Index i = 0; i < gradient_view.logical_size; ++i) {
                result.view.write_expanded(
                    key, gradient_view.read_expanded(key) * scalar_
                );
                gradient_view.cache->increment_key(key.data(), key.size());
            }
        }
        return py::make_tuple(
            make_tensor(std::move(result.data_object), std::move(result.layout_object))
        );
    }

private:
    float scalar_ = 0.0f;
};

class CpuReduceSumOperation : public CpuOperation {
public:
    py::object forward(py::args inputs) {
        if (py::len(inputs) != 1) {
            throw py::type_error("CPU reduce requires a tensor");
        }
        const bool build_graph = begin_forward(inputs);
        py::object tensor = py::reinterpret_borrow<py::object>(inputs[0]);
        require_two_mode_tensor(tensor, "tensor");
        CpuTensorView<float> tensor_view = cpu_tensor_view<float>(tensor, "tensor");

        const Index n_size = mode_logical_size(tensor_layout(tensor), 0);
        const Index m_size = mode_logical_size(tensor_layout(tensor), 1);
        output_layout_ = canonical_layout_from_modes(
            {mode_shape(tensor_layout(tensor), 0)}
        );
        ctx_["output_layout"] = output_layout_;

        CpuTensorAllocation<float> result =
            allocate_cpu_tensor<float>(output_layout_);
        {
            py::gil_scoped_release release;
            std::vector<Index> row_key(tensor_view.leaf_rank(), 0);
            std::vector<Index> input_key(tensor_view.leaf_rank(), 0);
            std::vector<Index> output_key(result.view.leaf_rank(), 0);
            for (Index i = 0; i < n_size; ++i) {
                float sum = 0.0f;
                input_key = row_key;
                for (Index j = 0; j < m_size; ++j) {
                    sum += tensor_view.read_expanded(input_key);
                    tensor_view.cache->increment_mode(
                        input_key.data(), input_key.size(), 1
                    );
                }
                result.view.write_expanded(output_key, sum);
                tensor_view.cache->increment_mode(row_key.data(), row_key.size(), 0);
                result.view.cache->increment_key(
                    output_key.data(), output_key.size()
                );
            }
        }
        return finish_forward(
            this,
            make_tensor(std::move(result.data_object), std::move(result.layout_object)),
            build_graph
        );
    }

    py::object backward(py::object gradient) {
        py::tuple input_tensors = inputs();
        py::object tensor = py::reinterpret_borrow<py::object>(input_tensors[0]);
        require_layout(gradient, output_layout_);
        CpuTensorView<float> gradient_view =
            cpu_tensor_view<float>(gradient, "gradient");

        const Index n_size = mode_logical_size(tensor_layout(tensor), 0);
        const Index m_size = mode_logical_size(tensor_layout(tensor), 1);
        CpuTensorAllocation<float> result =
            allocate_cpu_tensor<float>(tensor_layout(tensor));
        {
            py::gil_scoped_release release;
            std::vector<Index> row_key(result.view.leaf_rank(), 0);
            std::vector<Index> input_key(result.view.leaf_rank(), 0);
            std::vector<Index> gradient_key(gradient_view.leaf_rank(), 0);
            for (Index i = 0; i < n_size; ++i) {
                const float gradient_value = gradient_view.read_expanded(gradient_key);
                input_key = row_key;
                for (Index j = 0; j < m_size; ++j) {
                    result.view.write_expanded(input_key, gradient_value);
                    result.view.cache->increment_mode(
                        input_key.data(), input_key.size(), 1
                    );
                }
                result.view.cache->increment_mode(row_key.data(), row_key.size(), 0);
                gradient_view.cache->increment_key(
                    gradient_key.data(), gradient_key.size()
                );
            }
        }
        return py::make_tuple(
            make_tensor(std::move(result.data_object), std::move(result.layout_object))
        );
    }

private:
    py::object output_layout_ = py::none();
};

class CpuMatmulOperation : public CpuOperation {
public:
    py::object forward(py::args inputs) {
        if (py::len(inputs) != 2) {
            throw py::type_error("CPU matmul requires lhs and rhs tensors");
        }
        const bool build_graph = begin_forward(inputs);
        py::object lhs = py::reinterpret_borrow<py::object>(inputs[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(inputs[1]);
        require_two_mode_tensor(lhs, "lhs");
        require_two_mode_tensor(rhs, "rhs");
        CpuTensorView<float> lhs_view = cpu_tensor_view<float>(lhs, "lhs");
        CpuTensorView<float> rhs_view = cpu_tensor_view<float>(rhs, "rhs");

        const Index n_size = mode_logical_size(tensor_layout(lhs), 0);
        const Index lhs_k_size = mode_logical_size(tensor_layout(lhs), 1);
        const Index m_size = mode_logical_size(tensor_layout(rhs), 0);
        const Index rhs_k_size = mode_logical_size(tensor_layout(rhs), 1);
        if (lhs_k_size != rhs_k_size) {
            throw py::value_error("Matmul inner dimensions must match");
        }

        output_layout_ = canonical_layout_from_modes(
            {mode_shape(tensor_layout(lhs), 0), mode_shape(tensor_layout(rhs), 0)}
        );
        ctx_["output_layout"] = output_layout_;

        CpuTensorAllocation<float> result =
            allocate_cpu_tensor<float>(output_layout_);
        {
            py::gil_scoped_release release;
            std::vector<Index> output_key(result.view.leaf_rank(), 0);
            std::vector<Index> rhs_j_base(rhs_view.leaf_rank(), 0);
            std::vector<Index> lhs_i_base(lhs_view.leaf_rank(), 0);
            std::vector<Index> lhs_key(lhs_view.leaf_rank(), 0);
            std::vector<Index> rhs_key(rhs_view.leaf_rank(), 0);
            for (Index j = 0; j < m_size; ++j) {
                std::fill(lhs_i_base.begin(), lhs_i_base.end(), 0);
                for (Index i = 0; i < n_size; ++i) {
                    float sum = 0.0f;
                    lhs_key = lhs_i_base;
                    rhs_key = rhs_j_base;
                    for (Index k = 0; k < lhs_k_size; ++k) {
                        sum += lhs_view.read_expanded(lhs_key) *
                               rhs_view.read_expanded(rhs_key);
                        lhs_view.cache->increment_mode(
                            lhs_key.data(), lhs_key.size(), 1
                        );
                        rhs_view.cache->increment_mode(
                            rhs_key.data(), rhs_key.size(), 1
                        );
                    }
                    result.view.write_expanded(output_key, sum);
                    result.view.cache->increment_key(
                        output_key.data(), output_key.size()
                    );
                    lhs_view.cache->increment_mode(
                        lhs_i_base.data(), lhs_i_base.size(), 0
                    );
                }
                rhs_view.cache->increment_mode(
                    rhs_j_base.data(), rhs_j_base.size(), 0
                );
            }
        }
        return finish_forward(
            this,
            make_tensor(std::move(result.data_object), std::move(result.layout_object)),
            build_graph
        );
    }

    py::object backward(py::object gradient) {
        py::tuple input_tensors = inputs();
        py::object lhs = py::reinterpret_borrow<py::object>(input_tensors[0]);
        py::object rhs = py::reinterpret_borrow<py::object>(input_tensors[1]);
        require_layout(gradient, output_layout_);
        CpuTensorView<float> lhs_view = cpu_tensor_view<float>(lhs, "lhs");
        CpuTensorView<float> rhs_view = cpu_tensor_view<float>(rhs, "rhs");
        CpuTensorView<float> gradient_view =
            cpu_tensor_view<float>(gradient, "gradient");

        const Index n_size = mode_logical_size(tensor_layout(lhs), 0);
        const Index k_size = mode_logical_size(tensor_layout(lhs), 1);
        const Index m_size = mode_logical_size(tensor_layout(rhs), 0);

        CpuTensorAllocation<float> lhs_result =
            allocate_cpu_tensor<float>(tensor_layout(lhs));
        CpuTensorAllocation<float> rhs_result =
            allocate_cpu_tensor<float>(tensor_layout(rhs));
        {
            py::gil_scoped_release release;
            std::vector<Index> lhs_k_base(lhs_result.view.leaf_rank(), 0);
            std::vector<Index> rhs_k_base(rhs_view.leaf_rank(), 0);
            std::vector<Index> lhs_output_key(lhs_result.view.leaf_rank(), 0);
            std::vector<Index> gradient_i_base(gradient_view.leaf_rank(), 0);
            std::vector<Index> gradient_key(gradient_view.leaf_rank(), 0);
            std::vector<Index> rhs_key(rhs_view.leaf_rank(), 0);
            for (Index k = 0; k < k_size; ++k) {
                lhs_output_key = lhs_k_base;
                std::fill(gradient_i_base.begin(), gradient_i_base.end(), 0);
                for (Index i = 0; i < n_size; ++i) {
                    float sum = 0.0f;
                    gradient_key = gradient_i_base;
                    rhs_key = rhs_k_base;
                    for (Index j = 0; j < m_size; ++j) {
                        sum += gradient_view.read_expanded(gradient_key) *
                               rhs_view.read_expanded(rhs_key);
                        gradient_view.cache->increment_mode(
                            gradient_key.data(), gradient_key.size(), 1
                        );
                        rhs_view.cache->increment_mode(
                            rhs_key.data(), rhs_key.size(), 0
                        );
                    }
                    lhs_result.view.write_expanded(lhs_output_key, sum);
                    lhs_result.view.cache->increment_mode(
                        lhs_output_key.data(), lhs_output_key.size(), 0
                    );
                    gradient_view.cache->increment_mode(
                        gradient_i_base.data(), gradient_i_base.size(), 0
                    );
                }
                lhs_result.view.cache->increment_mode(
                    lhs_k_base.data(), lhs_k_base.size(), 1
                );
                rhs_view.cache->increment_mode(
                    rhs_k_base.data(), rhs_k_base.size(), 1
                );
            }

            std::vector<Index> rhs_k_output_base(rhs_result.view.leaf_rank(), 0);
            std::vector<Index> lhs_k_base_for_rhs(lhs_view.leaf_rank(), 0);
            std::vector<Index> rhs_output_key(rhs_result.view.leaf_rank(), 0);
            std::vector<Index> gradient_j_base(gradient_view.leaf_rank(), 0);
            std::vector<Index> lhs_key(lhs_view.leaf_rank(), 0);
            for (Index k = 0; k < k_size; ++k) {
                rhs_output_key = rhs_k_output_base;
                std::fill(gradient_j_base.begin(), gradient_j_base.end(), 0);
                for (Index j = 0; j < m_size; ++j) {
                    float sum = 0.0f;
                    gradient_key = gradient_j_base;
                    lhs_key = lhs_k_base_for_rhs;
                    for (Index i = 0; i < n_size; ++i) {
                        sum += gradient_view.read_expanded(gradient_key) *
                               lhs_view.read_expanded(lhs_key);
                        gradient_view.cache->increment_mode(
                            gradient_key.data(), gradient_key.size(), 0
                        );
                        lhs_view.cache->increment_mode(
                            lhs_key.data(), lhs_key.size(), 0
                        );
                    }
                    rhs_result.view.write_expanded(rhs_output_key, sum);
                    rhs_result.view.cache->increment_mode(
                        rhs_output_key.data(), rhs_output_key.size(), 0
                    );
                    gradient_view.cache->increment_mode(
                        gradient_j_base.data(), gradient_j_base.size(), 1
                    );
                }
                rhs_result.view.cache->increment_mode(
                    rhs_k_output_base.data(), rhs_k_output_base.size(), 1
                );
                lhs_view.cache->increment_mode(
                    lhs_k_base_for_rhs.data(), lhs_k_base_for_rhs.size(), 1
                );
            }
        }

        return py::make_tuple(
            make_tensor(
                std::move(lhs_result.data_object), std::move(lhs_result.layout_object)
            ),
            make_tensor(
                std::move(rhs_result.data_object), std::move(rhs_result.layout_object)
            )
        );
    }

private:
    py::object output_layout_ = py::none();
};

}  // namespace

void bind_cpu(py::module_& module) {
    py::class_<CPU, Data>(module, "CPU")
        .def(
            py::init<Index, py::object, bool>(),
            py::arg("size"),
            py::arg("pointer") = py::none(),
            py::kw_only(),
            py::arg("mutable") = true
        )
        .def("pointer", &CPU::pointer)
        .def("set_value", &CPU::set_value_public, py::arg("index"), py::arg("value"))
        .def_static("dispatch_op", &CPU::dispatch_op, py::arg("operation_name"));

    py::class_<CpuAddOperation>(module, "_CPUAddOperation")
        .def(py::init<>())
        .def(
            "forward",
            [](CpuAddOperation& operation, py::args inputs) {
                return operation.forward(inputs);
            }
        )
        .def("backward", &CpuAddOperation::backward, py::arg("gradient"))
        .def_property_readonly("ctx", &CpuAddOperation::ctx)
        .def("inputs", &CpuAddOperation::inputs);

    py::class_<CpuScalarMulOperation>(module, "_CPUScalarMulOperation")
        .def(py::init<>())
        .def(
            "forward",
            [](CpuScalarMulOperation& operation, py::args inputs) {
                return operation.forward(inputs);
            }
        )
        .def("backward", &CpuScalarMulOperation::backward, py::arg("gradient"))
        .def_property_readonly("ctx", &CpuScalarMulOperation::ctx)
        .def("inputs", &CpuScalarMulOperation::inputs);

    py::class_<CpuReduceSumOperation>(module, "_CPUReduceSumOperation")
        .def(py::init<>())
        .def(
            "forward",
            [](CpuReduceSumOperation& operation, py::args inputs) {
                return operation.forward(inputs);
            }
        )
        .def("backward", &CpuReduceSumOperation::backward, py::arg("gradient"))
        .def_property_readonly("ctx", &CpuReduceSumOperation::ctx)
        .def("inputs", &CpuReduceSumOperation::inputs);

    py::class_<CpuMatmulOperation>(module, "_CPUMatmulOperation")
        .def(py::init<>())
        .def(
            "forward",
            [](CpuMatmulOperation& operation, py::args inputs) {
                return operation.forward(inputs);
            }
        )
        .def("backward", &CpuMatmulOperation::backward, py::arg("gradient"))
        .def_property_readonly("ctx", &CpuMatmulOperation::ctx)
        .def("inputs", &CpuMatmulOperation::inputs);
}

}  // namespace neotorch::data
